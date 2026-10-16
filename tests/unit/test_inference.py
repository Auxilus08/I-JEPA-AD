"""Tests for predictor-backed coarse-to-fine inference."""

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ijepa_ad.inference import MaskedLatentScorer, load_predictor_checkpoint
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
