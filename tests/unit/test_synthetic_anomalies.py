"""Tests for the synthetic-anomaly generator and its image/latent cache."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from ijepa_ad.caching.synthetic_cache import (
    SyntheticCacheError,
    encode_synthetic_set,
    load_synthetic_cache,
    load_synthetic_manifest,
    save_synthetic_cache,
    write_synthetic_set,
)
from ijepa_ad.data.synthetic_anomalies import (
    SyntheticConfig,
    estimate_foreground,
    generate_synthetic_anomaly,
    perlin_noise,
)
from ijepa_ad.data.transforms import PreprocessConfig


def _object_image(seed: int = 0, size: int = 160) -> Image.Image:
    """A textured disc on a flat background, like an MVTec object."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.35) ** 2
    pixels = np.full((size, size, 3), 40, dtype=np.uint8)
    texture = rng.integers(120, 200, size=(size, size, 3), dtype=np.uint8)
    pixels[disc] = texture[disc]
    return Image.fromarray(pixels)


def _sources(tmp_path: Path, count: int, offset: int = 0) -> list[SimpleNamespace]:
    out = []
    for index in range(count):
        path = tmp_path / f"src_{offset + index}.png"
        _object_image(offset + index).save(path)
        out.append(SimpleNamespace(image_path=path, image_id=f"id{offset + index}"))
    return out


CONFIG = SyntheticConfig(seed=3, anomalies_per_image=4)


def test_perlin_noise_shape_range_and_determinism() -> None:
    first = perlin_noise(64, 48, 4, 3, np.random.default_rng(0))
    again = perlin_noise(64, 48, 4, 3, np.random.default_rng(0))
    other = perlin_noise(64, 48, 4, 3, np.random.default_rng(1))

    assert first.shape == (64, 48)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert np.abs(first).max() <= 1.5 and first.std() > 0.05
    with pytest.raises(ValueError, match="positive"):
        perlin_noise(0, 4, 1, 1, np.random.default_rng(0))


def test_foreground_finds_the_object_not_the_background() -> None:
    foreground = estimate_foreground(_object_image())

    assert foreground[80, 80]
    assert not foreground[3, 3] and not foreground[156, 156]
    assert 0.2 < foreground.mean() < 0.6


@pytest.mark.parametrize("variant", [0, 1, 2, 3])  # type: ignore[misc]
def test_defect_changes_only_pixels_inside_its_mask(
    tmp_path: Path, variant: int
) -> None:
    image = _object_image()
    result = generate_synthetic_anomaly(
        image,
        image_id="base",
        variant=variant,
        texture_paths=[s.image_path for s in _sources(tmp_path, 2)],
        config=CONFIG,
    )
    before, after = np.asarray(image), np.asarray(result.image)

    changed = (before != after).any(axis=-1)
    assert result.mask.shape == before.shape[:2]
    assert not (changed & ~result.mask).any()  # nothing outside the mask moved
    assert result.mask.sum() > 0 and (changed & result.mask).sum() > 0
    assert result.kind == ("cutpaste", "perlin")[variant % 2]


def test_defects_stay_on_the_object_and_have_requested_scale(tmp_path: Path) -> None:
    image = _object_image()
    foreground = estimate_foreground(image)
    textures = [s.image_path for s in _sources(tmp_path, 2)]
    for variant in range(6):
        result = generate_synthetic_anomaly(
            image,
            image_id="scale",
            variant=variant,
            texture_paths=textures,
            config=CONFIG,
        )
        if result.kind == "perlin":
            assert not (result.mask & ~foreground).any()
        assert (
            0.5 * CONFIG.min_area_realised * CONFIG.min_area_fraction
            <= result.area_fraction
            <= 2.0 * CONFIG.max_area_fraction
        )


def test_generation_is_deterministic_and_depends_on_inputs(tmp_path: Path) -> None:
    image = _object_image()
    textures = [s.image_path for s in _sources(tmp_path, 2)]

    def make(
        image_id: str, variant: int, config: SyntheticConfig = CONFIG
    ) -> NDArray[np.uint8]:
        result = generate_synthetic_anomaly(
            image,
            image_id=image_id,
            variant=variant,
            texture_paths=textures,
            config=config,
        )
        return np.asarray(result.image, dtype=np.uint8)

    assert np.array_equal(make("a", 1), make("a", 1))
    assert not np.array_equal(make("a", 1), make("b", 1))
    assert not np.array_equal(make("a", 1), make("a", 3))
    assert not np.array_equal(make("a", 1), make("a", 1, SyntheticConfig(seed=4)))


def test_fingerprint_changes_with_every_setting() -> None:
    base = SyntheticConfig()
    assert base.fingerprint() == SyntheticConfig().fingerprint()
    for override in (
        {"seed": 1},
        {"anomalies_per_image": 2},
        {"min_area_fraction": 0.004},
        {"max_aspect_ratio": 3.0},
        {"min_opacity": 0.5},
        {"max_attempts": 3},
    ):
        assert SyntheticConfig(**override).fingerprint() != base.fingerprint()
    with pytest.raises(ValueError, match="min_area_fraction"):
        SyntheticConfig(min_area_fraction=0.5, max_area_fraction=0.6)
    with pytest.raises(ValueError, match="empty"):
        generate_synthetic_anomaly(
            _object_image(), image_id="x", variant=0, texture_paths=[], config=CONFIG
        )


def _write(tmp_path: Path):  # type: ignore[no-untyped-def]
    base, textures = _sources(tmp_path, 3, 0), _sources(tmp_path, 3, 10)
    root = tmp_path / "out"
    samples = write_synthetic_set(
        base, textures, category="demo", config=CONFIG, root=root
    )
    return base, textures, root, samples


def test_written_set_has_hashes_masks_and_a_manifest(tmp_path: Path) -> None:
    base, _, root, samples = _write(tmp_path)

    assert len(samples) == 3 * CONFIG.anomalies_per_image
    assert len({s.image_id for s in samples}) == len(samples)
    assert {s.base_image_id for s in samples} == {b.image_id for b in base}
    for sample in samples:
        mask = np.asarray(Image.open(sample.mask_path)) > 0
        assert mask.sum() > 0 and Image.open(sample.image_path).size == (160, 160)
    assert load_synthetic_manifest(root, CONFIG, "demo") == samples


def test_rewriting_is_idempotent_but_never_replaces_different_files(
    tmp_path: Path,
) -> None:
    base, textures, root, samples = _write(tmp_path)
    again = write_synthetic_set(
        base, textures, category="demo", config=CONFIG, root=root
    )
    assert again == samples

    samples[0].image_path.write_bytes(b"corrupt")
    with pytest.raises(SyntheticCacheError, match="different content"):
        write_synthetic_set(base, textures, category="demo", config=CONFIG, root=root)
    with pytest.raises(SyntheticCacheError, match="changed since generation"):
        load_synthetic_manifest(root, CONFIG, "demo")


def test_a_texture_source_may_not_also_be_a_base_image(tmp_path: Path) -> None:
    base = _sources(tmp_path, 2)
    with pytest.raises(SyntheticCacheError, match="also a base image"):
        write_synthetic_set(
            base, base[:1], category="demo", config=CONFIG, root=tmp_path / "o"
        )


class _MeanEncoder:
    """Encoder stand-in: 196 tokens whose first channels are the patch mean colour."""

    def encode(self, image_batch: Tensor) -> Tensor:
        count = image_batch.shape[0]
        patches = image_batch.unfold(2, 16, 16).unfold(3, 16, 16)
        colours = patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1).reshape(count, 196, 3)
        return torch.cat([colours, torch.zeros(count, 196, 5)], dim=-1)


def test_cache_round_trip_and_provenance_checks(tmp_path: Path) -> None:
    _, _, _, samples = _write(tmp_path)
    preprocess = PreprocessConfig()
    latents = encode_synthetic_set(
        samples, _MeanEncoder(), preprocess=preprocess, batch_size=5, embedding_dim=8
    )
    path = tmp_path / "cache" / "demo.pt"
    kwargs = {"config": CONFIG, "encoder_name": "mock", "preprocess": preprocess}
    save_synthetic_cache(path, latents, samples, checkpoint_sha256="a" * 64, **kwargs)  # type: ignore[arg-type]

    loaded = load_synthetic_cache(path, samples, checkpoint_sha256="a" * 64, **kwargs)  # type: ignore[arg-type]
    assert torch.equal(loaded, latents) and loaded.shape == (len(samples), 196, 8)
    with pytest.raises(SyntheticCacheError, match="checkpoint_sha256"):
        load_synthetic_cache(path, samples, checkpoint_sha256="b" * 64, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(SyntheticCacheError, match="sample_ids"):
        load_synthetic_cache(path, samples[::-1], checkpoint_sha256="a" * 64, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(SyntheticCacheError, match="fingerprint"):
        load_synthetic_cache(
            path,
            samples,
            checkpoint_sha256="a" * 64,
            config=SyntheticConfig(seed=9),
            encoder_name="mock",
            preprocess=preprocess,
        )
    with pytest.raises(SyntheticCacheError, match="overwrite"):
        save_synthetic_cache(
            path, latents, samples, checkpoint_sha256="a" * 64, **kwargs  # type: ignore[arg-type]
        )
