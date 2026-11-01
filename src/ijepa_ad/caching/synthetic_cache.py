"""Images, masks and latent cache for the synthetic-anomaly validation set.

Kept apart from ``latent_cache`` on purpose: that module only accepts official
nominal and test records (its checks keep the test split evaluation-only). Synthetic
images are none of those, so they get their own tiny, equally strict format:

* ``<image_root>/<fingerprint>/<category>/*.png``  defect images and ``*_mask.png``
* ``<image_root>/<fingerprint>/<category>/manifest.json``  provenance of every sample
* ``<cache_root>/<fingerprint>/<category>.pt``  ``{"latents", "metadata"}``

The cache is rejected unless the generator fingerprint, encoder checkpoint SHA-256,
preprocessing and the ordered image ids all match what the caller expects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ijepa_ad.caching.cache_schema import PreprocessingSpec
from ijepa_ad.data.synthetic_anomalies import (
    SyntheticConfig,
    generate_synthetic_anomaly,
)
from ijepa_ad.data.transforms import PreprocessConfig, load_image, preprocess_image

CACHE_FORMAT_VERSION = 1


class SyntheticCacheError(ValueError):
    """The synthetic images or latents do not match what the caller expects."""


class _Source(Protocol):
    image_path: Path
    image_id: str


@dataclass(frozen=True)
class SyntheticSample:
    """One synthetic defect image on disk and where it came from."""

    image_id: str  # SHA-256 of the PNG bytes
    base_image_id: str  # the nominal validation image it was made from
    category: str
    variant: int
    kind: str
    image_path: Path
    mask_path: Path
    area_fraction: float
    details: dict[str, Any]


def synthetic_dir(root: str | Path, config: SyntheticConfig, category: str) -> Path:
    return Path(root) / config.fingerprint() / category


def synthetic_cache_path(
    root: str | Path, config: SyntheticConfig, category: str
) -> Path:
    return Path(root) / config.fingerprint() / f"{category}.pt"


def write_synthetic_set(
    base_samples: Sequence[_Source],
    texture_samples: Sequence[_Source],
    *,
    category: str,
    config: SyntheticConfig,
    root: str | Path,
) -> tuple[SyntheticSample, ...]:
    """Generate and write ``config.anomalies_per_image`` defects per base image.

    ``base_samples`` must be nominal validation images and ``texture_samples``
    nominal training images; the caller owns that guarantee (it comes from
    ``split_nominal_train_val``). Existing files are verified, never overwritten.
    """
    if not base_samples or not texture_samples:
        raise ValueError("base_samples and texture_samples must not be empty")
    base_ids = {s.image_id for s in base_samples}
    if base_ids & {s.image_id for s in texture_samples}:
        raise SyntheticCacheError("a texture source is also a base image")
    target = synthetic_dir(root, config, category)
    target.mkdir(parents=True, exist_ok=True)
    texture_paths = [s.image_path for s in texture_samples]
    samples: list[SyntheticSample] = []
    for base in sorted(base_samples, key=lambda s: s.image_id):
        image = load_image(base.image_path)
        for variant in range(config.anomalies_per_image):
            anomaly = generate_synthetic_anomaly(
                image,
                image_id=base.image_id,
                variant=variant,
                texture_paths=texture_paths,
                config=config,
            )
            stem = f"{base.image_id[:16]}_v{variant}"
            image_path, mask_path = target / f"{stem}.png", target / f"{stem}_mask.png"
            _write_png(anomaly.image, image_path)
            _write_png(Image.fromarray(anomaly.mask.astype(np.uint8) * 255), mask_path)
            samples.append(
                SyntheticSample(
                    image_id=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    base_image_id=base.image_id,
                    category=category,
                    variant=variant,
                    kind=anomaly.kind,
                    image_path=image_path,
                    mask_path=mask_path,
                    area_fraction=anomaly.area_fraction,
                    details={
                        **anomaly.details,
                        "attempts": anomaly.attempts,
                        "foreground_fallback": anomaly.foreground_fallback,
                    },
                )
            )
    manifest = {
        "fingerprint": config.fingerprint(),
        "config": asdict(config),
        "category": category,
        "samples": [
            {
                **asdict(s),
                "image_path": s.image_path.name,
                "mask_path": s.mask_path.name,
            }
            for s in samples
        ],
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8"
    )
    return tuple(samples)


def _write_png(image: Image.Image, path: Path) -> None:
    """Write deterministic PNG bytes; refuse to replace a different file."""
    if path.exists():
        before = path.read_bytes()
        tmp = path.with_suffix(".tmp.png")
        image.save(tmp, format="PNG", optimize=False)
        same = tmp.read_bytes() == before
        tmp.unlink()
        if not same:
            raise SyntheticCacheError(f"{path} exists with different content")
        return
    image.save(path, format="PNG", optimize=False)


def load_synthetic_manifest(
    root: str | Path, config: SyntheticConfig, category: str
) -> tuple[SyntheticSample, ...]:
    """Read back the samples of a written set, checking every file's hash."""
    target = synthetic_dir(root, config, category)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest["fingerprint"] != config.fingerprint():
        raise SyntheticCacheError("manifest fingerprint does not match the config")
    samples = []
    for row in manifest["samples"]:
        sample = SyntheticSample(
            **{
                **row,
                "image_path": target / row["image_path"],
                "mask_path": target / row["mask_path"],
            }
        )
        digest = hashlib.sha256(sample.image_path.read_bytes()).hexdigest()
        if digest != sample.image_id:
            raise SyntheticCacheError(f"{sample.image_path} changed since generation")
        samples.append(sample)
    return tuple(samples)


class _Encoder(Protocol):
    def encode(self, image_batch: Tensor) -> Tensor: ...


def encode_synthetic_set(
    samples: Sequence[SyntheticSample],
    encoder: _Encoder,
    *,
    preprocess: PreprocessConfig,
    batch_size: int = 8,
    num_tokens: int = 196,
    embedding_dim: int = 1408,
) -> Tensor:
    """Encode the written PNGs (not the in-memory images) into ``[N, T, D]``."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    chunks: list[Tensor] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        images = torch.stack(
            [preprocess_image(s.image_path, preprocess) for s in batch]
        )
        latents = encoder.encode(images).detach().to("cpu", torch.float32)
        if latents.shape != (len(batch), num_tokens, embedding_dim):
            raise SyntheticCacheError(
                f"unexpected encoder output {tuple(latents.shape)}"
            )
        if not bool(torch.isfinite(latents).all()):
            raise SyntheticCacheError("encoder returned non-finite latents")
        chunks.append(latents)
    return torch.cat(chunks)


def save_synthetic_cache(
    path: str | Path,
    latents: Tensor,
    samples: Sequence[SyntheticSample],
    *,
    config: SyntheticConfig,
    encoder_name: str,
    checkpoint_sha256: str,
    preprocess: PreprocessConfig,
) -> None:
    if latents.shape[0] != len(samples):
        raise SyntheticCacheError("latents and samples differ in length")
    metadata = _metadata(
        config,
        encoder_name,
        checkpoint_sha256,
        preprocess,
        [s.image_id for s in samples],
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SyntheticCacheError(f"refusing to overwrite {destination}")
    torch.save({"latents": latents.cpu(), "metadata": metadata}, destination)


def load_synthetic_cache(
    path: str | Path,
    samples: Sequence[SyntheticSample],
    *,
    config: SyntheticConfig,
    encoder_name: str,
    checkpoint_sha256: str,
    preprocess: PreprocessConfig,
) -> Tensor:
    """Load latents ``[N, T, D]``; any provenance mismatch raises."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = _metadata(
        config,
        encoder_name,
        checkpoint_sha256,
        preprocess,
        [s.image_id for s in samples],
    )
    if payload["metadata"] != expected:
        diff = sorted(k for k in expected if payload["metadata"].get(k) != expected[k])
        raise SyntheticCacheError(f"synthetic cache metadata mismatch in: {diff}")
    latents = payload["latents"]
    if latents.shape[0] != len(samples) or not bool(torch.isfinite(latents).all()):
        raise SyntheticCacheError("synthetic cache latents are malformed")
    return latents


def _metadata(
    config: SyntheticConfig,
    encoder_name: str,
    checkpoint_sha256: str,
    preprocess: PreprocessConfig,
    ids: list[str],
) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "fingerprint": config.fingerprint(),
        "encoder_name": encoder_name,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing": asdict(PreprocessingSpec.from_config(preprocess)),
        "sample_ids": ids,
    }
