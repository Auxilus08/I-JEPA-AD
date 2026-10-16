"""Deterministic image-space augmentation of nominal training images.

Augmented images are written under a directory named after a fingerprint of the
resolved augmentation settings, and each augmented sample is identified by the
SHA-256 of its encoded file. Changing any setting therefore produces new files and
new sample ids, so neither old images nor old latent caches can be reused silently
(the cache builder compares ``sample_ids``). Old fingerprint directories are left
untouched.

Augmentation never touches validation or test images; callers pass only the nominal
training side of ``split_nominal_train_val``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .transforms import load_image
from .validators import SampleMetadata

# Bump when ``augment_image`` changes behavior for the same settings.
AUGMENTATION_VERSION = 2

_CONFIG_FIELDS = (
    "copies_per_image",
    "seed",
    "max_rotation_degrees",
    "horizontal_flip_probability",
    "brightness_factor",
    "contrast_factor",
    "blur_probability",
)


@dataclass(frozen=True)
class AugmentationConfig:
    """Resolved augmentation settings for one category."""

    copies_per_image: int
    seed: int
    max_rotation_degrees: float
    horizontal_flip_probability: float
    brightness_factor: float
    contrast_factor: float
    blur_probability: float

    def __post_init__(self) -> None:
        if self.copies_per_image < 1:
            raise ValueError("copies_per_image must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_rotation_degrees < 0:
            raise ValueError("max_rotation_degrees must be non-negative")
        for name in ("horizontal_flip_probability", "blur_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("brightness_factor", "contrast_factor"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

    def fingerprint(self) -> str:
        """Short stable hash of every setting that affects the produced pixels."""
        payload = {"version": AUGMENTATION_VERSION, **asdict(self)}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_augmentation_config(path: str | Path, category: str) -> AugmentationConfig:
    """Resolve ``augmentation`` defaults plus ``category_overrides[category]``."""
    import yaml  # type: ignore[import-untyped]

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("augmentation"), dict
    ):
        raise ValueError(f"invalid augmentation configuration: {path}")
    values: dict[str, Any] = dict(document["augmentation"])
    overrides = (document.get("category_overrides") or {}).get(category, {})
    unknown = sorted(set(overrides) - set(_CONFIG_FIELDS))
    if unknown:
        raise ValueError(
            f"unknown augmentation override fields for {category}: {unknown}"
        )
    values.update(overrides)
    missing = sorted(set(_CONFIG_FIELDS) - set(values))
    if missing:
        raise ValueError(f"augmentation configuration is missing fields: {missing}")
    return AugmentationConfig(**{name: values[name] for name in _CONFIG_FIELDS})


def augment_image(
    image: Image.Image, config: AugmentationConfig, rng: random.Random
) -> Image.Image:
    """Mirror, rotate, jitter and blur one RGB image using ``rng`` only."""
    if rng.random() < config.horizontal_flip_probability:
        image = ImageOps.mirror(image)
    angle = rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees)
    # Fill the rotated-in corners with the image's own border colour; a constant
    # black fill would paint dark wedges onto bright MVTec backgrounds that never
    # occur in nominal or test images.
    image = image.rotate(
        angle, resample=Image.Resampling.BILINEAR, fillcolor=_border_color(image)
    )
    brightness = 1.0 + rng.uniform(-config.brightness_factor, config.brightness_factor)
    contrast = 1.0 + rng.uniform(-config.contrast_factor, config.contrast_factor)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    if rng.random() < config.blur_probability:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    return image


def materialize_augmented_records(
    samples: Sequence[SampleMetadata],
    config: AugmentationConfig,
    *,
    category: str,
    output_root: str | Path,
) -> list[SampleMetadata]:
    """Write augmented copies of ``samples`` and return originals plus copies.

    Files go to ``<output_root>/<category>/train/good/<fingerprint>/``. A file that
    already exists in that directory was produced by identical settings, so it is
    reused; writes are atomic so an interrupted run never leaves a half-written PNG.
    """
    directory = Path(output_root) / category / "train" / "good" / config.fingerprint()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"version": AUGMENTATION_VERSION, "category": category, **asdict(config)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    records = list(samples)
    for sample in samples:
        source: Image.Image | None = None
        for copy_index in range(config.copies_per_image):
            output = directory / f"{sample.image_id[:16]}_aug{copy_index:02d}.png"
            if not output.is_file():
                if source is None:
                    source = load_image(sample.image_path)
                rng = random.Random(
                    f"{config.seed}:{category}:{sample.image_id}:{copy_index}"
                )
                _save_png_atomically(augment_image(source.copy(), config, rng), output)
            records.append(
                SampleMetadata(output, _file_sha256(output), "train", "good")
            )
    return records


def _border_color(image: Image.Image) -> tuple[int, int, int]:
    pixels = np.asarray(image.convert("RGB"))
    border = np.concatenate(
        [pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]], axis=0
    )
    red, green, blue = (int(value) for value in np.median(border, axis=0))
    return red, green, blue


def _save_png_atomically(image: Image.Image, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AUGMENTATION_VERSION",
    "AugmentationConfig",
    "augment_image",
    "load_augmentation_config",
    "materialize_augmented_records",
]
