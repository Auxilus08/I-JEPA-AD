"""Deterministic synthetic defects pasted onto *nominal validation* images.

Plan item 12 (``docs/METRICS_IMPROVEMENT_PLAN.md``): only nominal validation data
exists, so nothing can be chosen for *detection* quality without touching the test
split. This module makes a labelled proxy: CutPaste-style scars and DRAEM-style
Perlin-noise texture blends, generated with a fixed seed.

Data rules (enforced by the callers' types and by the tests):

* Base images are nominal **validation** images; texture sources are nominal
  **training** images of the same category. Official test images are never read.
* Every defect is a pure function of ``(config, image_id, variant)``: the same
  inputs always give the same pixels, whatever the call order.
* Pixels outside the returned mask are bit-identical to the base image.
* The defect mask is the exact set of pasted pixels, at the base image resolution.

Changing any field of :class:`SyntheticConfig` (or ``SYNTHETIC_VERSION``) changes
its fingerprint, so a stale image folder or latent cache is never reused silently.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageFilter

from .transforms import load_image

# Bump when generation changes behaviour for the same settings.
SYNTHETIC_VERSION = 1
KINDS = ("cutpaste", "perlin")

BoolArray: TypeAlias = NDArray[np.bool_]
UInt8Array: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True)
class SyntheticConfig:
    """Settings of the synthetic-defect generator (all declared, none tuned)."""

    seed: int = 0
    anomalies_per_image: int = 4  # even, so both kinds are equally frequent
    # Defect area as a fraction of the whole image, sampled log-uniformly. 0.3% is
    # about 0.6 of one 16 px token, 4% about 8 tokens: small defects dominate, as in
    # MVTec AD.
    min_area_fraction: float = 0.003
    max_area_fraction: float = 0.04
    # CutPaste patch aspect ratio (long side / short side), log-uniform in [1, max].
    max_aspect_ratio: float = 6.0
    max_rotation_degrees: float = 45.0
    # Perlin texture blend opacity, uniform.
    min_opacity: float = 0.4
    max_opacity: float = 1.0
    # A defect must change the pixels it covers by at least this many 8-bit levels
    # on average, otherwise it is invisible and the attempt is repeated.
    min_visible_difference: float = 10.0
    # Minimum fraction of the requested area that must land inside the foreground.
    min_area_realised: float = 0.5
    max_attempts: int = 12

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.anomalies_per_image < 1:
            raise ValueError("anomalies_per_image must be at least 1")
        if not 0 < self.min_area_fraction <= self.max_area_fraction < 0.5:
            raise ValueError("need 0 < min_area_fraction <= max_area_fraction < 0.5")
        if self.max_aspect_ratio < 1:
            raise ValueError("max_aspect_ratio must be at least 1")
        if not 0 < self.min_opacity <= self.max_opacity <= 1:
            raise ValueError("need 0 < min_opacity <= max_opacity <= 1")
        if not 0 < self.min_area_realised <= 1:
            raise ValueError("min_area_realised must be in (0, 1]")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def fingerprint(self) -> str:
        """Short stable hash of every setting that affects the produced pixels."""
        payload = {"version": SYNTHETIC_VERSION, **asdict(self)}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SyntheticAnomaly:
    """One generated defect: the edited image, its mask and how it was made."""

    image: Image.Image
    mask: BoolArray
    kind: str
    variant: int
    area_fraction: float
    attempts: int
    foreground_fallback: bool
    details: dict[str, Any]


def _rng_for(
    config: SyntheticConfig, image_id: str, variant: int
) -> np.random.Generator:
    key = f"ijepa-ad/synthetic/v{SYNTHETIC_VERSION}|{config.seed}|{image_id}|{variant}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def perlin_noise(
    height: int, width: int, cells_y: int, cells_x: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """2-D gradient (Perlin) noise with ``cells_y x cells_x`` lattice cells.

    Values lie in roughly ``[-1, 1]``. Uses the quintic fade ``6t^5 - 15t^4 + 10t^3``.
    """
    if min(height, width, cells_y, cells_x) < 1:
        raise ValueError("height, width and cell counts must be positive")
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(cells_y + 1, cells_x + 1))
    grad_x, grad_y = np.cos(angles), np.sin(angles)
    ys = np.linspace(0.0, cells_y, height, endpoint=False)
    xs = np.linspace(0.0, cells_x, width, endpoint=False)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]

    def corner(dy: int, dx: int) -> NDArray[np.float64]:
        iy, ix = (y0 + dy)[:, None], (x0 + dx)[None, :]
        return np.asarray(
            grad_x[iy, ix] * (fx - dx) + grad_y[iy, ix] * (fy - dy), dtype=np.float64
        )

    def fade(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(t**3 * (t * (t * 6 - 15) + 10), dtype=np.float64)

    sy, sx = fade(fy), fade(fx)
    top = corner(0, 0) * (1 - sx) + corner(0, 1) * sx
    bottom = corner(1, 0) * (1 - sx) + corner(1, 1) * sx
    return np.asarray(top * (1 - sy) + bottom * sy, dtype=np.float64)


def estimate_foreground(image: Image.Image) -> BoolArray:
    """Rough object mask: pixels far from the median colour of the image border.

    Works at 256 px and is upsampled to the image size. MVTec objects sit on a
    near-uniform background, which is all this heuristic assumes. It is a placement
    aid, not a segmentation: it only decides where defects may land.
    """
    rgb = image.convert("RGB")
    small = np.asarray(
        rgb.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float64
    )
    frame = np.zeros((256, 256), dtype=bool)
    frame[:8, :] = frame[-8:, :] = frame[:, :8] = frame[:, -8:] = True
    reference = np.median(small[frame], axis=0)
    distance = np.sqrt(((small - reference) ** 2).sum(axis=-1))
    threshold = max(25.0, 1.5 * float(np.percentile(distance[frame], 99)))
    mask = Image.fromarray(((distance > threshold) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    mask = mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    resized = mask.resize(rgb.size, Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


def _jitter(patch: Image.Image, rng: np.random.Generator, spread: float) -> Image.Image:
    for enhancer in (
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
    ):
        factor = float(rng.uniform(1.0 - spread, 1.0 + spread))
        patch = enhancer(patch).enhance(factor)
    return patch


def _pick_centre(region: BoolArray, rng: np.random.Generator) -> tuple[int, int]:
    rows, cols = np.nonzero(region)
    index = int(rng.integers(0, rows.size))
    return int(rows[index]), int(cols[index])


def _cutpaste(
    base: UInt8Array,
    foreground: BoolArray,
    area_pixels: float,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[UInt8Array, BoolArray, dict[str, Any]]:
    """Cut a patch from elsewhere on the object, jitter it, rotate it, paste it."""
    height, width = base.shape[:2]
    aspect = float(math.exp(rng.uniform(0.0, math.log(config.max_aspect_ratio))))
    if rng.random() < 0.5:
        aspect = 1.0 / aspect
    patch_h = int(np.clip(round(math.sqrt(area_pixels / aspect)), 3, height // 2))
    patch_w = int(np.clip(round(math.sqrt(area_pixels * aspect)), 3, width // 2))
    src_y, src_x = _pick_centre(foreground, rng)
    top = int(np.clip(src_y - patch_h // 2, 0, height - patch_h))
    left = int(np.clip(src_x - patch_w // 2, 0, width - patch_w))
    crop = Image.fromarray(base[top : top + patch_h, left : left + patch_w])
    crop = _jitter(crop, rng, 0.4).convert("RGBA")
    angle = float(
        rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees)
    )
    rotated = np.asarray(
        crop.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
    )
    alpha = rotated[..., 3] > 250
    rot_h, rot_w = alpha.shape
    dst_y, dst_x = _pick_centre(foreground, rng)
    y0, x0 = dst_y - rot_h // 2, dst_x - rot_w // 2
    y1, x1 = max(y0, 0), max(x0, 0)
    y2, x2 = min(y0 + rot_h, height), min(x0 + rot_w, width)
    mask = np.zeros((height, width), dtype=bool)
    out = base.copy()
    if y2 > y1 and x2 > x1:
        window = alpha[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0]
        window = window & foreground[y1:y2, x1:x2]  # never spill onto the background
        pixels = rotated[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0, :3]
        out[y1:y2, x1:x2][window] = pixels[window]
        mask[y1:y2, x1:x2] = window
    return out, mask, {"aspect": aspect, "angle": angle, "patch": [patch_h, patch_w]}


def _perlin_blend(
    base: UInt8Array,
    foreground: BoolArray,
    area_pixels: float,
    texture: UInt8Array,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[UInt8Array, BoolArray, dict[str, Any]]:
    """Blend a foreign texture into a compact Perlin-shaped blob (DRAEM style)."""
    height, width = base.shape[:2]
    cells = int(2 ** rng.integers(1, 4))
    noise = perlin_noise(height, width, cells, cells, rng)
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-8)
    centre_y, centre_x = _pick_centre(foreground, rng)
    sigma = 1.2 * math.sqrt(area_pixels)
    yy = np.arange(height)[:, None] - centre_y
    xx = np.arange(width)[None, :] - centre_x
    envelope = np.exp(-(yy**2 + xx**2) / (2.0 * sigma**2))
    field = noise + 2.0 * envelope
    candidates = field[foreground]
    keep = int(min(max(round(area_pixels), 1), candidates.size))
    cut = np.partition(candidates, -keep)[-keep]
    mask = (field >= cut) & foreground
    opacity = float(rng.uniform(config.min_opacity, config.max_opacity))
    blended = base.astype(np.float64)
    blended[mask] = (1 - opacity) * blended[mask] + opacity * texture[mask]
    out = base.copy()
    out[mask] = np.clip(np.rint(blended[mask]), 0, 255).astype(np.uint8)
    return out, mask, {"opacity": opacity, "cells": cells}


def _texture(path: Path, size: tuple[int, int], rng: np.random.Generator) -> UInt8Array:
    """A strongly altered nominal training image used as foreign texture."""
    image = load_image(path).resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image)
    array = np.rot90(array, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        array = array[:, ::-1]
    array = np.ascontiguousarray(array)
    if array.shape[:2] != (size[1], size[0]):
        array = np.asarray(
            Image.fromarray(array).resize(size, Image.Resampling.BILINEAR)
        )
    altered = _jitter(Image.fromarray(array), rng, 0.5)
    return np.asarray(altered)[..., rng.permutation(3)].copy()


def generate_synthetic_anomaly(
    image: Image.Image,
    *,
    image_id: str,
    variant: int,
    texture_paths: Sequence[Path],
    config: SyntheticConfig,
) -> SyntheticAnomaly:
    """Make defect ``variant`` of ``image``; the kind alternates with the variant.

    ``texture_paths`` must be nominal *training* images of the same category.
    """
    if variant < 0:
        raise ValueError("variant must be non-negative")
    if not texture_paths:
        raise ValueError("texture_paths must not be empty")
    rng = _rng_for(config, image_id, variant)
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    height, width = base.shape[:2]
    foreground = estimate_foreground(image)
    fallback = not 0.03 <= float(foreground.mean()) <= 0.97
    if fallback:
        foreground = np.ones((height, width), dtype=bool)
    kind = KINDS[variant % len(KINDS)]

    best: tuple[UInt8Array, BoolArray, dict[str, Any], int] | None = None
    for attempt in range(1, config.max_attempts + 1):
        fraction = float(
            math.exp(
                rng.uniform(
                    math.log(config.min_area_fraction),
                    math.log(config.max_area_fraction),
                )
            )
        )
        area = fraction * height * width
        if kind == "cutpaste":
            out, mask, details = _cutpaste(base, foreground, area, config, rng)
        else:
            path = texture_paths[int(rng.integers(0, len(texture_paths)))]
            texture = _texture(path, (width, height), rng)
            out, mask, details = _perlin_blend(
                base, foreground, area, texture, config, rng
            )
            details["texture"] = path.name
        count = int(mask.sum())
        if count < config.min_area_realised * area:
            continue
        difference = np.abs(out[mask].astype(np.int16) - base[mask].astype(np.int16))
        if float(difference.mean()) < config.min_visible_difference:
            continue
        best = (out, mask, details, attempt)
        break
    if best is None:
        raise RuntimeError(
            f"no valid {kind} defect for {image_id} variant {variant} after "
            f"{config.max_attempts} attempts"
        )
    out, mask, details, attempts = best
    return SyntheticAnomaly(
        image=Image.fromarray(out),
        mask=mask,
        kind=kind,
        variant=variant,
        area_fraction=float(mask.mean()),
        attempts=attempts,
        foreground_fallback=fallback,
        details=details,
    )
