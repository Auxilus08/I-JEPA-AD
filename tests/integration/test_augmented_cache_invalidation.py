"""Latent caches must be rebuilt, never reused, when their inputs change."""

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from torch import Tensor

from ijepa_ad.caching.builder import build_latent_cache
from ijepa_ad.data.augmentation import (
    AugmentationConfig,
    materialize_augmented_records,
)
from ijepa_ad.data.transforms import PreprocessConfig
from ijepa_ad.data.validators import SampleMetadata

PREPROCESS = PreprocessConfig(image_size=32)
CHECKPOINT = "c" * 64


class _CountingEncoder:
    def __init__(self) -> None:
        self.images_encoded = 0

    def encode(self, image_batch: Tensor) -> Tensor:
        self.images_encoded += len(image_batch)
        pooled = image_batch.mean(dim=(1, 2, 3)).view(-1, 1, 1)
        return pooled.expand(-1, 4, 8).clone()


def _originals(tmp_path: Path) -> list[SampleMetadata]:
    directory = tmp_path / "src"
    directory.mkdir()
    records = []
    for index in range(2):
        pixels = np.random.default_rng(index).integers(0, 255, (48, 48, 3), np.uint8)
        path = directory / f"{index}.png"
        Image.fromarray(pixels).save(path)
        records.append(SampleMetadata(path, _sha(index), "train", "good"))
    return records


def _sha(index: int) -> str:
    return hashlib.sha256(str(index).encode()).hexdigest()


def _config() -> AugmentationConfig:
    return AugmentationConfig(2, 0, 8.0, 0.5, 0.1, 0.1, 0.0)


def _build(
    records: Sequence[SampleMetadata],
    encoder: _CountingEncoder,
    root: Path,
    *,
    checkpoint: str = CHECKPOINT,
    config: PreprocessConfig = PREPROCESS,
) -> int:
    report = build_latent_cache(
        records,
        encoder,
        cache_root=root,
        category="bottle",
        split="train",
        encoder_name="fake",
        checkpoint_sha256=checkpoint,
        patch_size=16,
        embedding_dim=8,
        num_tokens=4,
        config=config,
    )
    return report.encoded


def test_cache_is_reused_only_for_identical_augmented_samples(tmp_path: Path) -> None:
    originals = _originals(tmp_path)
    images = tmp_path / "aug"
    cache = tmp_path / "cache"
    encoder = _CountingEncoder()
    records = materialize_augmented_records(
        originals, _config(), category="bottle", output_root=images
    )

    assert _build(records, encoder, cache) == len(records)
    assert _build(records, encoder, cache) == 0  # identical inputs: reused

    changed = materialize_augmented_records(
        originals,
        replace(_config(), max_rotation_degrees=3.0),
        category="bottle",
        output_root=images,
    )
    assert _build(changed, encoder, cache) == len(changed)  # whole entry rebuilt


def test_cache_rebuilds_when_encoder_checkpoint_or_preprocessing_changes(
    tmp_path: Path,
) -> None:
    records = _originals(tmp_path)
    cache = tmp_path / "cache"
    encoder = _CountingEncoder()
    assert _build(records, encoder, cache) == 2

    assert _build(records, encoder, cache, checkpoint="d" * 64) == 2
    assert (
        _build(
            records,
            encoder,
            cache,
            config=PreprocessConfig(image_size=32, mean=(0.5, 0.5, 0.5)),
        )
        == 2
    )


def test_cache_rebuilds_when_image_content_changes_under_same_path(
    tmp_path: Path,
) -> None:
    from ijepa_ad.data.mvtec_dataset import MVTecADDataset

    root = tmp_path / "mvtec" / "bottle" / "train" / "good"
    root.mkdir(parents=True)
    path = root / "000.png"
    Image.fromarray(np.zeros((48, 48, 3), np.uint8)).save(path)
    cache = tmp_path / "cache"
    encoder = _CountingEncoder()

    first = list(MVTecADDataset(tmp_path / "mvtec", "bottle", "train").samples)
    assert _build(first, encoder, cache) == 1
    Image.fromarray(np.full((48, 48, 3), 200, np.uint8)).save(path)
    second = list(MVTecADDataset(tmp_path / "mvtec", "bottle", "train").samples)

    assert first[0].image_id != second[0].image_id
    assert _build(second, encoder, cache) == 1  # new content id => re-encoded


def test_augmentation_seed_changes_cache_sample_ids(tmp_path: Path) -> None:
    originals = _originals(tmp_path)
    one = materialize_augmented_records(
        originals, _config(), category="bottle", output_root=tmp_path / "a"
    )
    other = materialize_augmented_records(
        originals,
        replace(_config(), seed=1),
        category="bottle",
        output_root=tmp_path / "a",
    )

    assert {r.image_id for r in one[2:]}.isdisjoint({r.image_id for r in other[2:]})


def test_ids_follow_pixel_content_so_unchanged_copies_share_cache_rows(
    tmp_path: Path,
) -> None:
    originals = _originals(tmp_path)
    two = materialize_augmented_records(
        originals, _config(), category="bottle", output_root=tmp_path / "a"
    )
    three = materialize_augmented_records(
        originals,
        replace(_config(), copies_per_image=3),
        category="bottle",
        output_root=tmp_path / "a",
    )

    # Copies 0-1 are rendered from the same per-copy seeds, hence identical pixels
    # and ids; only copy 2 is new. Content identity decides reuse.
    assert {r.image_id for r in two} < {r.image_id for r in three}
    assert len(three) - len(two) == len(originals)
