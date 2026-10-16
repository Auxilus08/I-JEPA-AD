"""Tests for inference-aligned predictor training masks."""

from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from ijepa_ad.masking import CoarseMasker, PatchGrid
from ijepa_ad.predictor.losses import cosine_prediction_loss
from ijepa_ad.predictor.target_masks import build_inference_aligned_target_mask
from ijepa_ad.predictor.training import TrainingConfig, build_target_mask
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)

ROOT = Path(__file__).resolve().parents[2]


def _sample(
    batch: int = 400,
    seed: int = 0,
    block_probability: float = 0.5,
    single_token_count: int = 8,
) -> torch.Tensor:
    return build_inference_aligned_target_mask(
        batch,
        196,
        torch.Generator().manual_seed(seed),
        block_probability=block_probability,
        single_token_count=single_token_count,
    )


def _neighbours(token: int, side: int = 14) -> set[int]:
    row, column = divmod(token, side)
    found = set()
    if row:
        found.add(token - side)
    if row + 1 < side:
        found.add(token + side)
    if column:
        found.add(token - 1)
    if column + 1 < side:
        found.add(token + 1)
    return found


def _has_adjacent_targets(row: torch.Tensor) -> bool:
    targets = set(row.nonzero().flatten().tolist())
    return any(_neighbours(token) & targets for token in targets)


def test_block_rows_are_exactly_the_inference_coarse_windows() -> None:
    windows = CoarseMasker(PatchGrid(), 4).masks()
    mask = _sample(block_probability=1.0)

    seen = set()
    for row in mask:
        matches = (windows == row).all(dim=1).nonzero().flatten()
        assert len(matches) == 1, "block row is not a coarse inference window"
        seen.add(int(matches))
    assert seen == set(range(len(windows)))  # all 16 windows get sampled


def test_single_rows_are_isolated_tokens_with_visible_neighbours() -> None:
    mask = _sample(block_probability=0.0, single_token_count=8)

    for row in mask:
        assert int(row.sum()) == 8
        assert not _has_adjacent_targets(row)


def test_mixture_ratio_and_context_always_remain() -> None:
    mask = _sample(batch=2000, block_probability=0.5)

    assert (mask.sum(dim=1) >= 1).all() and (mask.sum(dim=1) < 196).all()
    # Single rows never have adjacent targets; every window row (>= 2 tokens wide
    # in both directions) always does. So adjacency identifies the block share.
    block_share = sum(_has_adjacent_targets(row) for row in mask) / len(mask)
    assert 0.4 < block_share < 0.6


def test_sampler_is_deterministic_for_a_seed_and_varies_across_seeds() -> None:
    assert torch.equal(_sample(seed=3), _sample(seed=3))
    assert not torch.equal(_sample(seed=3), _sample(seed=4))


def test_sampler_validates_arguments() -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="square"):
        build_inference_aligned_target_mask(1, 15, generator)
    with pytest.raises(ValueError, match="block_size"):
        build_inference_aligned_target_mask(1, 16, generator, block_size=4)
    with pytest.raises(ValueError, match="block_probability"):
        build_inference_aligned_target_mask(1, 196, generator, block_probability=1.5)


def test_small_grid_keeps_target_and_context() -> None:
    mask = build_inference_aligned_target_mask(
        64, 16, torch.Generator().manual_seed(0), block_size=2, single_token_count=8
    )

    assert (mask.sum(dim=1) >= 1).all() and (mask.sum(dim=1) < 16).all()


def _config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "batch_size": 4,
        "max_epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "mask_ratio": 0.4,
        "loss_type": "cosine",
        "patience": 0,
        "min_delta": 0.0,
        "seed": 0,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def test_training_config_default_stays_random_and_dispatch_selects_sampler() -> None:
    assert _config().mask_strategy == "random"
    random_mask = build_target_mask(_config(), 3, 196, torch.Generator().manual_seed(0))
    assert torch.equal(random_mask.sum(dim=1), torch.full((3,), 78))

    aligned = build_target_mask(
        _config(mask_strategy="inference_aligned"),
        400,
        196,
        torch.Generator().manual_seed(0),
    )
    assert not (aligned.sum(dim=1) == 78).any()
    with pytest.raises(ValueError, match="mask_strategy"):
        _config(mask_strategy="bogus")


def test_configured_training_masks_match_inference_masking_config() -> None:
    predictor = yaml.safe_load(
        (ROOT / "configs" / "predictor" / "base.yaml").read_text(encoding="utf-8")
    )["training"]
    masking = yaml.safe_load(
        (ROOT / "configs" / "masking" / "coarse_to_fine.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert predictor["mask_strategy"] == "inference_aligned"
    assert predictor["block_size"] == masking["coarse_size"]
    TrainingConfig(**{**predictor, "device": "cpu"})  # every key is accepted


def test_targets_never_reach_predictor_and_loss_ignores_context_tokens() -> None:
    torch.manual_seed(0)
    model = LatentTransformerPredictor(
        PredictorConfig(
            embedding_dim=8,
            num_tokens=196,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
        )
    ).eval()
    latents = torch.randn(6, 196, 8)
    mask = _sample(batch=6, seed=1)

    with torch.no_grad():
        baseline = model(latents, mask)
        corrupted = latents.clone()
        corrupted[mask] = torch.randn(int(mask.sum()), 8) * 100  # rewrite targets
        assert torch.equal(model(corrupted, mask), baseline)  # no target leakage

        # The loss is exactly mean(1 - cos) over target tokens, nothing else.
        cosine = torch.nn.functional.cosine_similarity(
            baseline[mask], latents[mask], dim=-1
        )
        expected = (1 - cosine).mean()
        assert torch.allclose(cosine_prediction_loss(baseline, latents, mask), expected)
        changed = baseline.clone()
        changed[~mask] = torch.randn(int((~mask).sum()), 8)
        assert torch.equal(
            cosine_prediction_loss(changed, latents, mask),
            cosine_prediction_loss(baseline, latents, mask),
        )
