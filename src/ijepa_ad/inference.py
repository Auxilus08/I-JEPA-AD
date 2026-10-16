"""Image-level masked latent prediction and adaptive anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from ijepa_ad.masking.coarse_masker import CoarseMasker, MaskWindow, PatchGrid
from ijepa_ad.masking.fine_masker import FineMasker, FineSelection
from ijepa_ad.masking.fusion import fuse_scores
from ijepa_ad.masking.score_maps import patch_scores_to_heatmap
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)

InferenceMode = Literal["adaptive", "dense", "fixed"]


class MaskingSettings(TypedDict):
    coarse_size: int
    fine_size: int
    threshold: float
    lam: float


def load_masking_settings(path: str | Path) -> MaskingSettings:
    """Read and validate the project masking configuration."""
    import yaml  # type: ignore[import-untyped]

    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid masking configuration: {config_path}")
    if config.get("grid_size") != 14:
        raise ValueError("real I-JEPA inference requires a 14x14 patch grid")
    return {
        "coarse_size": int(config["coarse_size"]),
        "fine_size": int(config["fine_size"]),
        "threshold": float(config["threshold"]),
        "lam": float(config["lambda"]),
    }


def load_predictor_checkpoint(
    checkpoint_path: str | Path,
    *,
    category: str,
    encoder_sha256: str,
    device: str | torch.device,
) -> LatentTransformerPredictor:
    """Load a category predictor only when its training provenance matches."""
    predictor, _ = load_predictor_checkpoint_with_provenance(
        checkpoint_path,
        category=category,
        encoder_sha256=encoder_sha256,
        device=device,
    )
    return predictor


def load_predictor_checkpoint_with_provenance(
    checkpoint_path: str | Path,
    *,
    category: str,
    encoder_sha256: str,
    device: str | torch.device,
) -> tuple[LatentTransformerPredictor, dict[str, Any]]:
    """Load a category predictor plus the training metadata stored with it.

    Validation is identical to ``load_predictor_checkpoint``. The returned
    metadata is copied verbatim from the checkpoint and is JSON-serializable:

    - ``training_config``: the saved ``TrainingConfig`` mapping.
    - ``reproducibility``: seeds, library versions and git state, or ``None``.
    - ``mask_strategy``: ``training_config["mask_strategy"]`` when it was recorded,
      else ``None``. ``None`` means "not recorded", never "random".
    - ``status``: ``"complete"`` (``reproducibility`` and ``mask_strategy`` both
      recorded), ``"legacy"`` (neither recorded) or ``"partial"``.

    Nothing is inferred for fields a checkpoint did not record.
    """
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"predictor checkpoint must contain a mapping: {path}")

    provenance = payload.get("cache_provenance")
    if not isinstance(provenance, dict) or provenance.get("category") != category:
        raise ValueError(f"predictor checkpoint category does not match {category!r}")
    if provenance.get("checkpoint_sha256") != encoder_sha256:
        raise ValueError("predictor checkpoint was trained with a different encoder")
    training_config = payload.get("training_config")
    if (
        not isinstance(training_config, dict)
        or training_config.get("loss_type") != "cosine"
    ):
        raise ValueError("expected a predictor trained with the cosine objective")
    reproducibility = payload.get("reproducibility")
    if reproducibility is not None and not isinstance(reproducibility, dict):
        raise ValueError(f"predictor checkpoint reproducibility is invalid: {path}")
    config_data = payload.get("model_config")
    state_dict = payload.get("model_state_dict")
    if not isinstance(config_data, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"predictor checkpoint is missing model configuration: {path}")

    config = PredictorConfig(**config_data)
    if (config.num_tokens, config.embedding_dim) != (196, 1408):
        raise ValueError(
            "predictor checkpoint must use 196 tokens and 1408-dimensional latents"
        )
    predictor = LatentTransformerPredictor(config)
    predictor.load_state_dict(state_dict, strict=True)
    predictor.to(device)
    predictor.eval()

    has_mask_strategy = "mask_strategy" in training_config
    if reproducibility is not None and has_mask_strategy:
        status = "complete"
    elif reproducibility is None and not has_mask_strategy:
        status = "legacy"
    else:
        status = "partial"
    metadata: dict[str, Any] = {
        "status": status,
        "mask_strategy": training_config.get("mask_strategy"),
        "training_config": dict(training_config),
        "reproducibility": None if reproducibility is None else dict(reproducibility),
    }
    return predictor, metadata


def image_score_upper_bound(mode: InferenceMode, lam: float) -> float:
    """Largest value ``max(patch_scores)`` can take for ``mode`` and ``lam``.

    Token scores are ``1 - cos`` clamped to [0, 2]. Fusion is
    ``coarse + lam * fine`` (``fuse_scores``), and each mode fills one or both terms:

    - ``fixed``: only coarse scores        -> ``2``
    - ``dense``: only fine scores          -> ``2 * lam``
    - ``adaptive``: coarse + refined fine  -> ``2 * (1 + lam)``
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if mode == "fixed":
        return 2.0
    if mode == "dense":
        return 2.0 * lam
    if mode == "adaptive":
        return 2.0 * (1.0 + lam)
    raise ValueError("mode must be adaptive, dense, or fixed")


@dataclass(frozen=True)
class InferenceResult:
    """Patch-level scores and mask-work accounting for one image."""

    coarse_scores: Tensor
    fine_scores: Tensor
    patch_scores: Tensor
    heatmap: Tensor
    evaluated_regions: int
    total_regions: int
    # Largest attainable ``max(patch_scores)`` for the producing mode and lambda
    # (see ``image_score_upper_bound``). The default is the unfused [0, 2] range.
    score_upper_bound: float = 2.0

    @property
    def raw_image_score(self) -> float:
        """Maximum fused patch score, in [0, ``score_upper_bound``]."""
        return float(self.patch_scores.max().item())

    @property
    def image_score(self) -> float:
        """``raw_image_score / score_upper_bound``, in [0, 1].

        A monotone rescaling of the maximum fused patch score, so image ranking
        (and AUROC) equals ranking by ``raw_image_score``. It does not saturate
        early: the bound is the true maximum of the fusion, so only a score at the
        theoretical maximum maps to 1. The divisor depends on the mode and lambda,
        so values are comparable only within one (mode, lambda) configuration; use
        ``raw_image_score`` across configurations.
        """
        if self.score_upper_bound <= 0:
            return 0.0  # e.g. dense mode with lambda == 0 scores nothing
        scaled = self.raw_image_score / self.score_upper_bound
        return min(max(scaled, 0.0), 1.0)  # guards float rounding only


class MaskedLatentScorer:
    """Score one encoded image with fixed, dense, or adaptive masked prediction."""

    def __init__(
        self,
        predictor: LatentTransformerPredictor,
        *,
        coarse_size: int = 4,
        fine_size: int = 1,
        threshold: float = 0.5,
        lam: float = 1.0,
        mask_batch_size: int = 16,
    ) -> None:
        if not torch.isfinite(torch.tensor(threshold)) or threshold < 0:
            raise ValueError("threshold must be a finite non-negative value")
        if lam < 0:
            raise ValueError("lam must be non-negative")
        if mask_batch_size < 1:
            raise ValueError("mask_batch_size must be positive")
        self.predictor = predictor
        self.grid = PatchGrid()
        self.coarse = CoarseMasker(self.grid, coarse_size)
        self.fine = FineMasker(self.grid, fine_size)
        self.threshold = threshold
        self.lam = lam
        self.mask_batch_size = mask_batch_size

    def score(
        self, latents: Tensor, *, mode: InferenceMode = "adaptive"
    ) -> InferenceResult:
        """Predict masked targets and return cosine-discrepancy patch scores."""
        if mode not in {"adaptive", "dense", "fixed"}:
            raise ValueError("mode must be adaptive, dense, or fixed")
        if latents.ndim != 3 or latents.shape[:2] != (1, self.grid.tokens):
            raise ValueError("latents must have shape [1, 196, embedding_dim]")
        if not torch.isfinite(latents).all():
            raise ValueError("latents must contain only finite values")
        parameter = next(self.predictor.parameters(), None)
        if parameter is not None and parameter.device != latents.device:
            raise ValueError(
                f"predictor is on {parameter.device}, "
                f"but latents are on {latents.device}"
            )
        if parameter is not None:
            latents = latents.to(dtype=parameter.dtype)

        self.predictor.eval()
        coarse_windows = self.coarse.windows()
        zero_scores = latents.new_zeros(self.grid.tokens)

        if mode == "dense":
            selection = FineSelection(self.fine.candidates(), self.threshold)
            fine_scores = self._score_windows(latents, selection.windows)
            return self._result(
                zero_scores,
                fine_scores,
                mode=mode,
                evaluated=len(selection.windows),
                total=len(self.fine.candidates()),
            )

        coarse_scores = self._score_windows(latents, coarse_windows)
        if mode == "fixed":
            return self._result(
                coarse_scores,
                zero_scores,
                mode=mode,
                evaluated=len(coarse_windows),
                total=len(coarse_windows),
            )

        region_scores = zero_scores.clone()
        for window in coarse_windows:
            indices = list(window.token_indices)
            region_scores[indices] = coarse_scores[indices].mean()
        selection = self.fine.select(
            region_scores.reshape(self.grid.size, self.grid.size).detach().cpu(),
            self.threshold,
        )
        fine_scores = self._score_windows(latents, selection.windows)
        return self._result(
            coarse_scores,
            fine_scores,
            mode=mode,
            evaluated=len(coarse_windows) + selection.count,
            total=len(coarse_windows) + len(self.fine.candidates()),
        )

    def _score_windows(
        self, latents: Tensor, windows: tuple[MaskWindow, ...]
    ) -> Tensor:
        scores = latents.new_zeros(self.grid.tokens)
        if not windows:
            return scores

        masks = torch.zeros(
            (len(windows), self.grid.tokens), dtype=torch.bool, device=latents.device
        )
        for row, window in enumerate(windows):
            masks[row, list(window.token_indices)] = True

        with torch.inference_mode():
            for start in range(0, len(windows), self.mask_batch_size):
                stop = min(start + self.mask_batch_size, len(windows))
                batch_masks = masks[start:stop]
                batch_latents = latents.expand(stop - start, -1, -1)
                predicted = self.predictor(batch_latents, batch_masks)
                discrepancies = (
                    1 - F.cosine_similarity(predicted, batch_latents, dim=-1, eps=1e-8)
                ).clamp_(0, 2)
                for row, window in enumerate(windows[start:stop]):
                    indices = list(window.token_indices)
                    scores[indices] = discrepancies[row, indices]
        return scores

    def _result(
        self,
        coarse_scores: Tensor,
        fine_scores: Tensor,
        *,
        mode: InferenceMode,
        evaluated: int,
        total: int,
    ) -> InferenceResult:
        patch_scores = fuse_scores(coarse_scores, fine_scores, self.lam)
        return InferenceResult(
            coarse_scores=coarse_scores,
            fine_scores=fine_scores,
            patch_scores=patch_scores,
            heatmap=patch_scores_to_heatmap(patch_scores, self.grid),
            evaluated_regions=evaluated,
            total_regions=total,
            score_upper_bound=image_score_upper_bound(mode, self.lam),
        )
