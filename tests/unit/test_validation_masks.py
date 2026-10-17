"""Validation target masks must be per-sample and independent of batching."""

import os
import subprocess
import sys

import pytest
import torch

from ijepa_ad.predictor.target_masks import sample_mask_seed
from ijepa_ad.predictor.training import TrainingConfig, build_validation_target_mask

NUM_TOKENS = 196
BASE_SEED = 10_003
IDS = tuple(f"bottle/train/good/{index:03d}.png" for index in range(37))


def _config(strategy: str) -> TrainingConfig:
    return TrainingConfig(
        batch_size=4,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        mask_ratio=0.4,
        loss_type="cosine",
        patience=0,
        min_delta=0.0,
        seed=3,
        device="cpu",
        mask_strategy=strategy,
    )


def _masks_by_batching(
    config: TrainingConfig,
    ids: tuple[str, ...],
    batch_size: int,
    epoch: int | None = None,
) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for start in range(0, len(ids), batch_size):  # last batch may be partial
        chunk = ids[start : start + batch_size]
        batch = build_validation_target_mask(
            config, chunk, NUM_TOKENS, base_seed=BASE_SEED, epoch=epoch
        )
        assert batch.shape == (len(chunk), NUM_TOKENS)
        masks.update(zip(chunk, batch, strict=True))
    return masks


@pytest.mark.parametrize("strategy", ["random", "inference_aligned"])  # type: ignore[misc]
def test_sample_masks_do_not_depend_on_batch_size(strategy: str) -> None:
    config = _config(strategy)
    reference = _masks_by_batching(config, IDS, 1)

    for batch_size in (4, 16):  # 37 samples -> partial final batches (1 and 5)
        masks = _masks_by_batching(config, IDS, batch_size)
        assert masks.keys() == reference.keys()
        for image_id in IDS:
            assert torch.equal(masks[image_id], reference[image_id]), image_id


@pytest.mark.parametrize("strategy", ["random", "inference_aligned"])  # type: ignore[misc]
def test_sample_masks_do_not_depend_on_order_or_neighbours(strategy: str) -> None:
    config = _config(strategy)
    reference = _masks_by_batching(config, IDS, 16)
    shuffled = tuple(IDS[index] for index in torch.randperm(len(IDS)).tolist())
    reversed_ids = IDS[::-1]

    for ids in (shuffled, reversed_ids, IDS[5:9]):
        for batch_size in (1, 4, 16):
            masks = _masks_by_batching(config, ids, batch_size)
            for image_id, mask in masks.items():
                assert torch.equal(mask, reference[image_id]), image_id


@pytest.mark.parametrize("strategy", ["random", "inference_aligned"])  # type: ignore[misc]
def test_validation_masks_are_reproducible_and_valid(strategy: str) -> None:
    config = _config(strategy)
    one = build_validation_target_mask(config, IDS, NUM_TOKENS, base_seed=BASE_SEED)
    two = build_validation_target_mask(config, IDS, NUM_TOKENS, base_seed=BASE_SEED)
    other = build_validation_target_mask(
        config, IDS, NUM_TOKENS, base_seed=BASE_SEED + 1
    )

    assert torch.equal(one, two)
    assert not torch.equal(one, other)
    assert one.dtype == torch.bool
    assert (one.sum(dim=1) >= 1).all() and (one.sum(dim=1) < NUM_TOKENS).all()
    assert len({tuple(row.tolist()) for row in one}) > 1  # samples differ


def test_epoch_is_part_of_the_seed_only_when_given() -> None:
    config = _config("inference_aligned")
    fixed = _masks_by_batching(config, IDS, 4)
    assert all(
        torch.equal(fixed[i], mask)
        for i, mask in _masks_by_batching(config, IDS, 16).items()
    )

    epoch_one = _masks_by_batching(config, IDS, 4, epoch=1)
    epoch_two = _masks_by_batching(config, IDS, 16, epoch=2)
    assert any(not torch.equal(epoch_one[i], epoch_two[i]) for i in IDS)
    for image_id, mask in _masks_by_batching(config, IDS, 1, epoch=1).items():
        assert torch.equal(mask, epoch_one[image_id])  # still batch independent


def test_validation_masks_leave_global_rng_untouched() -> None:
    config = _config("inference_aligned")
    torch.manual_seed(1234)
    expected = torch.rand(5)
    torch.manual_seed(1234)

    build_validation_target_mask(config, IDS, NUM_TOKENS, base_seed=BASE_SEED)

    assert torch.equal(torch.rand(5), expected)


def test_sample_seed_is_a_stable_sha256_derivation() -> None:
    # Golden values: change only if the documented derivation changes.
    assert sample_mask_seed(10_003, "bottle/train/good/000.png") == (
        7147443051857376430
    )
    assert sample_mask_seed(10_003, "bottle/train/good/000.png", 7) == (
        4956533692495183833
    )
    seeds = {sample_mask_seed(BASE_SEED, image_id) for image_id in IDS}
    assert len(seeds) == len(IDS)
    assert all(0 <= seed < 2**63 for seed in seeds)
    assert sample_mask_seed(1, "a") != sample_mask_seed(2, "a")
    assert sample_mask_seed(1, "a", 0) != sample_mask_seed(1, "a", 1)
    assert sample_mask_seed(1, "a") != sample_mask_seed(1, "a", 0)

    # Independent of Python's process-randomized hash(): differing PYTHONHASHSEED
    # values must give the same seed in fresh interpreters.
    code = (
        "from ijepa_ad.predictor.target_masks import sample_mask_seed;"
        "print(sample_mask_seed(10003, 'bottle/train/good/000.png', 7))"
    )
    outputs = set()
    for hash_seed in ("0", "1", "random"):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(completed.stdout.strip())
    assert outputs == {str(sample_mask_seed(10_003, "bottle/train/good/000.png", 7))}
