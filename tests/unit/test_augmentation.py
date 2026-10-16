"""Tests for deterministic, fingerprinted nominal-image augmentation."""

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ijepa_ad.data.augmentation import (
    AugmentationConfig,
    augment_image,
    load_augmentation_config,
    materialize_augmented_records,
)
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.validators import SampleMetadata, validate_training_records

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "augmentation" / "base.yaml"


def _sha(index: int) -> str:
    return hashlib.sha256(str(index).encode()).hexdigest()


def _config(**overrides: object) -> AugmentationConfig:
    values: dict[str, object] = {
        "copies_per_image": 2,
        "seed": 0,
        "max_rotation_degrees": 8.0,
        "horizontal_flip_probability": 0.5,
        "brightness_factor": 0.1,
        "contrast_factor": 0.1,
        "blur_probability": 0.0,
    }
    values.update(overrides)
    return AugmentationConfig(**values)  # type: ignore[arg-type]


def _asymmetric_image() -> Image.Image:
    pixels = np.full((48, 48, 3), 230, dtype=np.uint8)  # bright background
    pixels[10:38, 6:20] = (10, 10, 10)  # dark block on the left only
    return Image.fromarray(pixels)


def _samples(tmp_path: Path, count: int = 3) -> list[SampleMetadata]:
    directory = tmp_path / "src"
    directory.mkdir()
    records = []
    for index in range(count):
        pixels = np.random.default_rng(index).integers(0, 255, (48, 48, 3), np.uint8)
        path = directory / f"{index}.png"
        Image.fromarray(pixels).save(path)
        records.append(SampleMetadata(path, _sha(index), "train", "good"))
    return records


def test_fingerprint_changes_with_every_setting() -> None:
    base = _config()
    changed = [
        replace(base, copies_per_image=3),
        replace(base, seed=1),
        replace(base, max_rotation_degrees=4.0),
        replace(base, horizontal_flip_probability=0.0),
        replace(base, brightness_factor=0.2),
        replace(base, contrast_factor=0.2),
        replace(base, blur_probability=0.5),
    ]

    assert len({base.fingerprint(), *(item.fingerprint() for item in changed)}) == 8
    assert base.fingerprint() == _config().fingerprint()


def test_capsule_override_disables_mirroring_but_other_categories_keep_it() -> None:
    assert load_augmentation_config(CONFIG, "capsule").horizontal_flip_probability == 0
    for category in ("bottle", "hazelnut"):
        assert (
            load_augmentation_config(CONFIG, category).horizontal_flip_probability
            == 0.5
        )
    assert (
        load_augmentation_config(CONFIG, "capsule").max_rotation_degrees
        == load_augmentation_config(CONFIG, "bottle").max_rotation_degrees
    )


def test_unknown_override_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "aug.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "horizontal_flip_probability: 0.0", "horizontal_flip_probabilty: 0.0"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown augmentation override"):
        load_augmentation_config(path, "capsule")


def test_no_mirroring_when_probability_is_zero_and_mirroring_when_one() -> None:
    image = _asymmetric_image()
    still = _config(
        max_rotation_degrees=0.0, brightness_factor=0.0, contrast_factor=0.0
    )

    kept = augment_image(
        image, replace(still, horizontal_flip_probability=0.0), random.Random(0)
    )
    flipped = augment_image(
        image, replace(still, horizontal_flip_probability=1.0), random.Random(0)
    )

    assert np.array_equal(np.asarray(kept), np.asarray(image))
    assert np.array_equal(np.asarray(flipped), np.asarray(image)[:, ::-1])


def test_rotation_fills_corners_with_background_not_black() -> None:
    config = _config(
        max_rotation_degrees=8.0,
        horizontal_flip_probability=0.0,
        brightness_factor=0.0,
        contrast_factor=0.0,
    )
    rotated = np.asarray(augment_image(_asymmetric_image(), config, random.Random(1)))

    assert rotated[:3, :3].min() > 200  # corners stay bright


def test_materialization_is_reproducible_and_ids_are_content_hashes(
    tmp_path: Path,
) -> None:
    samples = _samples(tmp_path)
    config = _config()

    first = materialize_augmented_records(
        samples, config, category="bottle", output_root=tmp_path / "out"
    )
    second = materialize_augmented_records(
        samples, config, category="bottle", output_root=tmp_path / "out"
    )

    assert first == second
    assert first[:3] == samples  # originals first, untouched
    assert len(first) == 3 + 3 * config.copies_per_image
    validate_training_records(first)  # nominal-only, unique ids
    assert all(len(record.image_id) == 64 for record in first[3:])


def test_changed_settings_never_reuse_old_images_or_ids(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    old = materialize_augmented_records(
        samples, _config(), category="bottle", output_root=tmp_path / "out"
    )
    old_files = sorted(record.image_path for record in old[3:])
    old_bytes = [path.read_bytes() for path in old_files]

    new = materialize_augmented_records(
        samples,
        _config(max_rotation_degrees=2.0),
        category="bottle",
        output_root=tmp_path / "out",
    )

    assert {r.image_id for r in old[3:]}.isdisjoint({r.image_id for r in new[3:]})
    assert not set(old_files) & {record.image_path for record in new[3:]}
    assert [path.read_bytes() for path in old_files] == old_bytes  # left in place


def test_augmentation_seed_does_not_change_the_nominal_split(tmp_path: Path) -> None:
    samples = _samples(tmp_path, count=10)
    before = split_nominal_train_val(samples, val_fraction=0.2, seed=0)

    for seed in (0, 7):
        materialize_augmented_records(
            before.train,
            _config(seed=seed),
            category="bottle",
            output_root=tmp_path / f"out{seed}",
        )
    after = split_nominal_train_val(samples, val_fraction=0.2, seed=0)

    assert before == after
