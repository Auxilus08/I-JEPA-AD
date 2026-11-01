"""Opt-in scoring variants for masked latent prediction.

Everything here is a *scoring-only* change: the predictor and the masks are
untouched. ``MaskedLatentScorer`` keeps its original ``1 - cos`` behaviour unless
one of these discrepancies is passed explicitly.

Statistics are always fitted from nominal data handed in by the caller. Nothing in
this module reads a dataset, so the caller decides (and records) which split a
statistic came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


class TokenDiscrepancy(Protocol):
    """Map predicted and actual tokens ``[B, T, D]`` to per-token scores ``[B, T]``."""

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor: ...


def cosine_discrepancy(predicted: Tensor, actual: Tensor) -> Tensor:
    """``1 - cos`` clamped to [0, 2]; the scorer's built-in definition."""
    return (1 - F.cosine_similarity(predicted, actual, dim=-1, eps=1e-8)).clamp(0, 2)


def layer_norm_cosine_discrepancy(predicted: Tensor, actual: Tensor) -> Tensor:
    """``1 - cos`` after a parameter-free layer norm over channels.

    This is I-JEPA's target normalisation, ``F.layer_norm(x, (D,))``. The cosine of
    two layer-normed vectors is their Pearson correlation over channels.
    """
    dim = (predicted.shape[-1],)
    return cosine_discrepancy(F.layer_norm(predicted, dim), F.layer_norm(actual, dim))


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel mean and standard deviation of nominal latent tokens."""

    mean: Tensor  # [D]
    std: Tensor  # [D]


def fit_channel_stats(latents: Tensor, *, eps: float = 1e-6) -> ChannelStats:
    """Fit per-channel statistics from ``[N, T, D]`` (or ``[M, D]``) nominal tokens.

    ``eps`` floors the standard deviation so a near-constant channel cannot blow up.
    """
    if latents.ndim not in (2, 3):
        raise ValueError("latents must be [N, T, D] or [M, D]")
    tokens = latents.reshape(-1, latents.shape[-1]).to(torch.float64)
    if tokens.shape[0] < 2:
        raise ValueError("need at least two tokens to estimate a standard deviation")
    mean = tokens.mean(dim=0)
    std = tokens.std(dim=0).clamp_min(eps)
    return ChannelStats(mean=mean.float(), std=std.float())


@dataclass(frozen=True)
class ChannelStandardisedCosine:
    """``1 - cos`` after standardising both vectors with fixed per-channel stats."""

    stats: ChannelStats

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        mean = self.stats.mean.to(device=actual.device, dtype=actual.dtype)
        std = self.stats.std.to(device=actual.device, dtype=actual.dtype)
        return cosine_discrepancy((predicted - mean) / std, (actual - mean) / std)


@dataclass(frozen=True)
class MahalanobisResidual:
    """Squared Mahalanobis distance of ``actual - predicted`` in a PCA subspace.

    The score is ``mean_i (z_i ** 2)`` over the ``d`` retained components, where
    ``z`` is the centred residual projected on the principal axes and divided by
    ``sqrt(eigenvalue + ridge)``. Normal residuals therefore score about 1.
    Unlike cosine this is unbounded above.
    """

    mean: Tensor  # [D]
    components: Tensor  # [D, d], orthonormal columns
    inv_scale: Tensor  # [d]

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        mean = self.mean.to(device=actual.device, dtype=actual.dtype)
        components = self.components.to(device=actual.device, dtype=actual.dtype)
        inv_scale = self.inv_scale.to(device=actual.device, dtype=actual.dtype)
        projected = (actual - predicted - mean) @ components * inv_scale
        return projected.square().mean(dim=-1)


def fit_mahalanobis_residual(
    residuals: Tensor, *, n_components: int = 64, ridge: float = 1e-3
) -> MahalanobisResidual:
    """Fit a shared Gaussian to nominal residuals ``[M, D]``.

    ``ridge`` is relative: ``ridge * mean(retained eigenvalues)`` is added to every
    retained eigenvalue before inversion. The covariance is eigendecomposed in
    float64. ``n_components`` must be well below ``M`` so the fit is not singular.
    """
    if residuals.ndim != 2:
        raise ValueError("residuals must be [M, D]")
    count, dim = residuals.shape
    if not 1 <= n_components <= min(dim, count - 1):
        raise ValueError("n_components must be in [1, min(D, M - 1)]")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    data = residuals.to(torch.float64)
    mean = data.mean(dim=0)
    centred = data - mean
    covariance = centred.T @ centred / (count - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues[-n_components:].flip(0).clamp_min(0)
    components = eigenvectors[:, -n_components:].flip(1)
    regularised = eigenvalues + ridge * eigenvalues.mean()
    if not bool((regularised > 0).all()):
        raise ValueError("residual covariance is degenerate; reduce n_components")
    return MahalanobisResidual(
        mean=mean.float(),
        components=components.float(),
        inv_scale=regularised.rsqrt().float(),
    )


@dataclass(frozen=True)
class PositionNormaliser:
    """Per-token-position normalisation of ``[..., T]`` scores.

    ``apply`` returns ``(scores - mean) / std``. With ``divide=False`` at fit time
    ``std`` is all ones, so only the per-position mean is removed.
    """

    mean: Tensor  # [T]
    std: Tensor  # [T]

    @classmethod
    def fit(
        cls, scores: Tensor, *, divide: bool = True, std_floor_fraction: float = 0.5
    ) -> PositionNormaliser:
        """Fit from nominal ``[N, T]`` scores.

        The per-position standard deviation is floored at
        ``std_floor_fraction * median(std)`` because it is estimated from only a
        few dozen nominal images and a near-zero value would explode the z-score.
        """
        if scores.ndim != 2 or scores.shape[0] < 2:
            raise ValueError("scores must be [N, T] with N >= 2")
        if std_floor_fraction < 0:
            raise ValueError("std_floor_fraction must be non-negative")
        data = scores.to(torch.float64)
        mean = data.mean(dim=0)
        if divide:
            std = data.std(dim=0)
            floor = std_floor_fraction * std.median()
            std = std.clamp_min(max(float(floor), 1e-8))
        else:
            std = torch.ones_like(mean)
        return cls(mean=mean.float(), std=std.float())

    def apply(self, scores: Tensor) -> Tensor:
        mean = self.mean.to(device=scores.device, dtype=scores.dtype)
        std = self.std.to(device=scores.device, dtype=scores.dtype)
        return (scores - mean) / std
