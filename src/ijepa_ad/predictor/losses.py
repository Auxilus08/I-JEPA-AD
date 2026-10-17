"""Prediction losses for latent-space target reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor


def cosine_prediction_loss(
    predicted: Tensor, target: Tensor, target_mask: Tensor | None = None
) -> Tensor:
    """Mean ``1 - cosine_similarity`` over selected latent tokens."""
    pred, tgt = _selected(predicted, target, target_mask)
    numerator = (pred * tgt).sum(dim=-1)
    denominator = pred.norm(dim=-1).clamp_min(1e-12) * tgt.norm(dim=-1).clamp_min(1e-12)
    return (1.0 - numerator / denominator).mean()


def l2_prediction_loss(
    predicted: Tensor, target: Tensor, target_mask: Tensor | None = None
) -> Tensor:
    """Mean squared error over selected latent tokens."""
    pred, tgt = _selected(predicted, target, target_mask)
    return ((pred - tgt) ** 2).mean()


def prediction_loss(
    predicted: Tensor, target: Tensor, target_mask: Tensor, loss_type: str
) -> Tensor:
    """Dispatch a configured predictor loss."""
    if loss_type == "cosine":
        return cosine_prediction_loss(predicted, target, target_mask)
    if loss_type == "l2":
        return l2_prediction_loss(predicted, target, target_mask)
    raise ValueError(f"loss_type must be 'cosine' or 'l2', got {loss_type!r}")


def _selected(
    predicted: Tensor, target: Tensor, target_mask: Tensor | None
) -> tuple[Tensor, Tensor]:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must have the same shape, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.ndim != 3:
        raise ValueError(
            f"predicted and target must be B x tokens x dim, got {predicted.ndim}D"
        )
    # Reduce in at least float32 so half-precision tensors cannot lose the cosine.
    dtype = torch.promote_types(
        torch.promote_types(predicted.dtype, target.dtype), torch.float32
    )
    predicted, target = predicted.to(dtype), target.to(dtype)
    if target_mask is None:
        return predicted, target
    if target_mask.shape != predicted.shape[:2]:
        raise ValueError(
            f"target_mask must be B x tokens, got {tuple(target_mask.shape)} for "
            f"predictions {tuple(predicted.shape)}"
        )
    mask = target_mask.to(device=predicted.device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("target_mask must select at least one token")
    return predicted[mask], target[mask]
