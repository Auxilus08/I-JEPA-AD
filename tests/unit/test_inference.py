"""Tests for predictor-backed coarse-to-fine inference."""

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ijepa_ad.inference import (
    InferenceMode,
    InferenceResult,
    MaskedLatentScorer,
    image_score_upper_bound,
    load_predictor_checkpoint,
    load_predictor_checkpoint_with_provenance,
)
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)


class _ControlledPredictor(nn.Module):
    def __init__(self, anomalous: bool = False) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.anomalous = anomalous
        self.forward_dtype: torch.dtype | None = None
        self.anomaly_indices = tuple(
            row * 14 + column for row in range(4) for column in range(4)
        )

    def forward(self, latents: Tensor, target_mask: Tensor) -> Tensor:
        self.forward_dtype = latents.dtype
        predicted = latents.clone()
        if self.anomalous:
            predicted[:, self.anomaly_indices] *= -1
        return predicted


class _CosinePredictor(nn.Module):
    """Predicts tokens in the top-left 4x4 block with a fixed cosine to the target."""

    def __init__(self, cosine: float) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.cosine = cosine
        self.block = tuple(row * 14 + column for row in range(4) for column in range(4))

    def forward(self, latents: Tensor, target_mask: Tensor) -> Tensor:
        predicted = latents.clone()
        sine = max(0.0, 1.0 - self.cosine**2) ** 0.5
        predicted[:, self.block] = 0
        predicted[:, self.block, 0] = self.cosine
        predicted[:, self.block, 1] = sine
        return predicted


def _unit_latents() -> Tensor:
    latents = torch.zeros(1, 196, 4)
    latents[..., 0] = 1.0
    return latents


def _image(mode: InferenceMode, cosine: float, lam: float) -> InferenceResult:
    scorer = MaskedLatentScorer(_CosinePredictor(cosine), lam=lam)
    return scorer.score(_unit_latents(), mode=mode)


def test_adaptive_inference_skips_fine_masks_for_nominal_latents() -> None:
    scorer = MaskedLatentScorer(_ControlledPredictor())
    result = scorer.score(torch.ones(1, 196, 4))

    assert result.coarse_scores.shape == (196,)
    assert result.fine_scores.shape == (196,)
    assert result.heatmap.shape == (1, 1, 224, 224)
    assert result.evaluated_regions == 16
    assert result.total_regions == 212
    assert result.image_score == 0.0


def test_adaptive_inference_refines_only_flagged_coarse_region() -> None:
    scorer = MaskedLatentScorer(_ControlledPredictor(anomalous=True))
    result = scorer.score(torch.ones(1, 196, 4))

    assert result.evaluated_regions == 32
    assert torch.count_nonzero(result.fine_scores) == 16
    assert torch.all(result.patch_scores[:4] == 4)
    assert result.image_score == 1.0


def test_dense_and_fixed_modes_report_actual_mask_counts() -> None:
    scorer = MaskedLatentScorer(_ControlledPredictor())
    latents = torch.ones(1, 196, 4)

    dense = scorer.score(latents, mode="dense")
    fixed = scorer.score(latents, mode="fixed")

    assert (dense.evaluated_regions, dense.total_regions) == (196, 196)
    assert (fixed.evaluated_regions, fixed.total_regions) == (16, 16)


def test_inference_rejects_non_finite_latents() -> None:
    scorer = MaskedLatentScorer(_ControlledPredictor())
    latents = torch.ones(1, 196, 4)
    latents[0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        scorer.score(latents)


@pytest.mark.parametrize(  # type: ignore[misc]
    "cache_dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_inference_converts_cached_latents_to_predictor_dtype(
    cache_dtype: torch.dtype,
) -> None:
    predictor = _ControlledPredictor()
    scorer = MaskedLatentScorer(predictor)

    result = scorer.score(torch.ones(1, 196, 4, dtype=cache_dtype))

    assert predictor.forward_dtype == torch.float32
    assert result.patch_scores.dtype == torch.float32


def test_predictor_checkpoint_requires_matching_encoder_provenance(
    tmp_path: Path,
) -> None:
    config = PredictorConfig(
        embedding_dim=1408,
        num_tokens=196,
        hidden_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    predictor = LatentTransformerPredictor(config)
    checkpoint = tmp_path / "predictor.pt"
    torch.save(
        {
            "model_state_dict": predictor.state_dict(),
            "model_config": config.to_dict(),
            "training_config": {"loss_type": "cosine"},
            "cache_provenance": {
                "category": "bottle",
                "checkpoint_sha256": "a" * 64,
            },
        },
        checkpoint,
    )

    loaded = load_predictor_checkpoint(
        checkpoint,
        category="bottle",
        encoder_sha256="a" * 64,
        device="cpu",
    )
    assert not loaded.training

    with pytest.raises(ValueError, match="different encoder"):
        load_predictor_checkpoint(
            checkpoint,
            category="bottle",
            encoder_sha256="b" * 64,
            device="cpu",
        )


def _write_checkpoint(path: Path, **overrides: object) -> None:
    config = PredictorConfig(
        embedding_dim=1408,
        num_tokens=196,
        hidden_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    payload: dict[str, object] = {
        "model_state_dict": LatentTransformerPredictor(config).state_dict(),
        "model_config": config.to_dict(),
        "training_config": {"loss_type": "cosine"},
        "cache_provenance": {"category": "bottle", "checkpoint_sha256": "a" * 64},
    }
    payload.update(overrides)
    torch.save(payload, path)


def _load(path: Path, **overrides: str) -> tuple[nn.Module, dict[str, object]]:
    arguments = {"category": "bottle", "encoder_sha256": "a" * 64, **overrides}
    return load_predictor_checkpoint_with_provenance(path, device="cpu", **arguments)


def test_current_checkpoint_metadata_is_preserved_verbatim(tmp_path: Path) -> None:
    training_config = {
        "loss_type": "cosine",
        "mask_strategy": "inference_aligned",
        "seed": 3,
        "block_size": 4,
    }
    reproducibility = {"seed": 3, "git_commit": "abc", "git_dirty": False}
    path = tmp_path / "current.pt"
    _write_checkpoint(
        path, training_config=training_config, reproducibility=reproducibility
    )

    predictor, metadata = _load(path)

    assert not predictor.training
    assert metadata == {
        "status": "complete",
        "mask_strategy": "inference_aligned",
        "training_config": training_config,
        "reproducibility": reproducibility,
    }


def test_legacy_checkpoint_does_not_fabricate_missing_metadata(
    tmp_path: Path,
) -> None:
    # Named like a current checkpoint on purpose: the filename must never be used.
    path = tmp_path / "bottle_inference_aligned_cosine.pt"
    _write_checkpoint(path)

    _, metadata = _load(path)

    assert metadata["status"] == "legacy"
    assert metadata["mask_strategy"] is None
    assert metadata["reproducibility"] is None
    assert metadata["training_config"] == {"loss_type": "cosine"}


def test_partial_checkpoint_is_flagged_not_completed(tmp_path: Path) -> None:
    path = tmp_path / "partial.pt"
    _write_checkpoint(path, reproducibility={"seed": 1})

    _, metadata = _load(path)

    assert metadata["status"] == "partial"
    assert metadata["mask_strategy"] is None
    assert metadata["reproducibility"] == {"seed": 1}


def test_metadata_loader_keeps_provenance_and_integrity_checks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictor.pt"
    _write_checkpoint(path, reproducibility={"seed": 1})

    with pytest.raises(ValueError, match="different encoder"):
        _load(path, encoder_sha256="b" * 64)
    with pytest.raises(ValueError, match="category"):
        _load(path, category="capsule")
    _write_checkpoint(path, reproducibility="not-a-mapping")
    with pytest.raises(ValueError, match="reproducibility"):
        _load(path)


def test_plain_loader_still_returns_only_the_predictor(tmp_path: Path) -> None:
    path = tmp_path / "predictor.pt"
    _write_checkpoint(path)

    loaded = load_predictor_checkpoint(
        path, category="bottle", encoder_sha256="a" * 64, device="cpu"
    )

    assert isinstance(loaded, LatentTransformerPredictor)


# --- image-level score scaling -------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    ("mode", "lam", "bound"),
    [
        ("fixed", 0.0, 2.0),
        ("fixed", 3.0, 2.0),
        ("dense", 0.0, 0.0),
        ("dense", 1.0, 2.0),
        ("dense", 2.5, 5.0),
        ("adaptive", 0.0, 2.0),
        ("adaptive", 1.0, 4.0),
        ("adaptive", 0.5, 3.0),
    ],
)
def test_image_score_upper_bound_per_mode(
    mode: InferenceMode, lam: float, bound: float
) -> None:
    assert image_score_upper_bound(mode, lam) == bound


def test_image_score_upper_bound_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        image_score_upper_bound("fixed", -0.1)
    with pytest.raises(ValueError, match="mode"):
        image_score_upper_bound("bogus", 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("mode", "lam"),
    [
        ("fixed", 1.0),
        ("dense", 1.0),
        ("dense", 2.0),
        ("adaptive", 1.0),
        ("adaptive", 0.5),
        ("adaptive", 2.0),
    ],
)
def test_image_score_bounds_hold_at_both_extremes(
    mode: InferenceMode, lam: float
) -> None:
    nominal = _image(mode, 1.0, lam)
    worst = _image(mode, -1.0, lam)  # discrepancy 2 everywhere in the block

    assert nominal.image_score == 0.0
    assert worst.image_score == pytest.approx(1.0)
    assert worst.raw_image_score == pytest.approx(image_score_upper_bound(mode, lam))
    assert worst.raw_image_score <= worst.score_upper_bound + 1e-6


def test_adaptive_scores_above_two_no_longer_tie() -> None:
    # Discrepancy 1 vs 1.5 -> fused (lambda = 1) raw maxima 2 and 3. The old
    # ``max / 2`` clamp mapped both to 1.0 and tied them.
    moderate = _image("adaptive", 0.0, 1.0)
    severe = _image("adaptive", -0.5, 1.0)

    assert moderate.raw_image_score == pytest.approx(2.0)
    assert severe.raw_image_score == pytest.approx(3.0)
    assert moderate.image_score == pytest.approx(0.5)
    assert severe.image_score == pytest.approx(0.75)
    assert moderate.image_score < severe.image_score < 1.0


def test_image_score_ranking_matches_raw_score_ranking() -> None:
    cosines = [1.0, 0.8, 0.3, 0.0, -0.4, -0.9, -1.0]
    modes: tuple[InferenceMode, ...] = ("fixed", "dense", "adaptive")
    for mode in modes:
        results = [_image(mode, cosine, 1.5) for cosine in cosines]
        raw = [result.raw_image_score for result in results]
        scaled = [result.image_score for result in results]
        assert raw == sorted(raw)
        assert scaled == sorted(scaled)
        assert len(set(scaled)) == len(set(raw)) == len(cosines)  # no new ties


def test_image_score_is_consistent_within_mode_and_depends_on_lambda() -> None:
    first = _image("adaptive", 0.0, 1.0)
    again = _image("adaptive", 0.0, 1.0)
    other_lambda = _image("adaptive", 0.0, 2.0)

    assert first.image_score == again.image_score
    assert first.score_upper_bound == 4.0 and other_lambda.score_upper_bound == 6.0
    # Different lambda -> different raw scores and a different divisor: scores are
    # not interchangeable across lambdas, so the bound is exposed on the result.
    assert other_lambda.raw_image_score == pytest.approx(3.0)
    assert other_lambda.image_score == pytest.approx(0.5)


def test_fixed_and_dense_scores_use_their_own_bounds() -> None:
    fixed = _image("fixed", 0.0, 2.0)  # coarse only: lambda is not applied
    dense = _image("dense", 0.0, 2.0)  # fine only, scaled by lambda

    assert (fixed.raw_image_score, fixed.score_upper_bound) == (1.0, 2.0)
    assert (dense.raw_image_score, dense.score_upper_bound) == (2.0, 4.0)
    assert fixed.image_score == dense.image_score == pytest.approx(0.5)


def test_dense_with_zero_lambda_scores_zero_without_dividing_by_zero() -> None:
    result = _image("dense", -1.0, 0.0)

    assert result.score_upper_bound == 0.0
    assert result.image_score == 0.0


def test_result_constructed_without_bound_keeps_legacy_two_range() -> None:
    patch_scores = torch.zeros(196)
    patch_scores[0] = 1.0
    result = InferenceResult(
        coarse_scores=patch_scores,
        fine_scores=patch_scores,
        patch_scores=patch_scores,
        heatmap=torch.zeros(1, 1, 224, 224),
        evaluated_regions=1,
        total_regions=1,
    )

    assert result.score_upper_bound == 2.0
    assert result.image_score == 0.5
