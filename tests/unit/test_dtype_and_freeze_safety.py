"""dtype alignment and frozen-encoder invariants."""

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder
from ijepa_ad.inference import MaskedLatentScorer
from ijepa_ad.predictor.losses import cosine_prediction_loss
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3 * 16 * 16, 1408)

    def forward(self, image_batch: Tensor) -> Tensor:
        patches = image_batch.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(3)
        return self.projection(patches.flatten(0, 2)).view(
            image_batch.shape[0], 196, 1408
        )


def _encoder(tmp_path: Path) -> IJEPAEncoder:
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"target_encoder": _Tiny().state_dict()}, checkpoint)
    return IJEPAEncoder(checkpoint, model_factory=_Tiny)


def test_parent_train_call_cannot_unfreeze_the_encoder(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    parent = nn.Sequential(encoder)

    parent.train()

    assert encoder.training is False and encoder.model.training is False
    assert all(not p.requires_grad for p in encoder.parameters())


def test_requires_grad_cannot_be_enabled_and_tampering_is_caught(
    tmp_path: Path,
) -> None:
    encoder = _encoder(tmp_path)
    with pytest.raises(RuntimeError, match="frozen"):
        encoder.requires_grad_(True)

    next(encoder.model.parameters()).requires_grad_(True)  # direct tampering
    with pytest.raises(RuntimeError, match="trainable"):
        encoder.encode(torch.zeros(1, 3, 224, 224))


def test_encode_restores_eval_mode_and_emits_no_grad(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    encoder.model.train()  # switched directly, bypassing the wrapper

    latents = encoder.encode(torch.zeros(1, 3, 224, 224, dtype=torch.float64))

    assert encoder.model.training is False
    assert latents.dtype == torch.float32 and not latents.requires_grad


def _predictor() -> LatentTransformerPredictor:
    return LatentTransformerPredictor(
        PredictorConfig(
            embedding_dim=4,
            num_tokens=196,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
        )
    )


def test_predictor_rejects_mismatched_latent_dtype_with_clear_error() -> None:
    mask = torch.zeros(1, 196, dtype=torch.bool)
    mask[0, 0] = True

    with pytest.raises(TypeError, match="cast the latents"):
        _predictor()(torch.zeros(1, 196, 4, dtype=torch.float16), mask)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_scorer_casts_cached_half_precision_latents_to_predictor_dtype(
    dtype: torch.dtype,
) -> None:
    scorer = MaskedLatentScorer(_predictor().eval())

    result = scorer.score(torch.randn(1, 196, 4).to(dtype), mode="fixed")

    assert result.patch_scores.dtype == torch.float32
    assert torch.isfinite(result.patch_scores).all()


def test_loss_reduces_in_float32_for_half_precision_inputs() -> None:
    predicted = torch.randn(1, 3, 8).bfloat16()
    target = torch.randn(1, 3, 8).bfloat16()
    mask = torch.ones(1, 3, dtype=torch.bool)

    assert cosine_prediction_loss(predicted, target, mask).dtype == torch.float32
