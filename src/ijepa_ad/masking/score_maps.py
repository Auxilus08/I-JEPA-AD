"""Conversion of patch scores to image-sized anomaly maps.

Two paths share one interpolation rule (bilinear, ``align_corners=False``):

* ``upsample_patch_scores`` returns raw, un-normalized scores. Evaluation (pixel
  AUROC, AUPRO) must use this so scores stay comparable across images.
* ``patch_scores_to_heatmap`` additionally min-max normalizes each map to [0, 1].
  That scaling is for display only; it discards absolute score magnitude.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .coarse_masker import PatchGrid


def upsample_patch_scores(
    patch_scores: Tensor,
    grid: PatchGrid | None = None,
    *,
    output_size: int | tuple[int, int] = 224,
) -> Tensor:
    """Bilinearly upsample `[196]` or `[N,196]` patch scores to `[N,1,H,W]` raw maps."""
    grid = grid or PatchGrid()
    if patch_scores.shape[-1] != grid.tokens:
        raise ValueError("patch_scores must end with the number of grid tokens")
    if patch_scores.ndim == 1:
        patch_scores = patch_scores.unsqueeze(0)
    if patch_scores.ndim != 2:
        raise ValueError("patch_scores must be one- or two-dimensional")
    maps = patch_scores.reshape(-1, 1, grid.size, grid.size).float()
    size = (output_size, output_size) if isinstance(output_size, int) else output_size
    return torch.nn.functional.interpolate(
        maps, size=size, mode="bilinear", align_corners=False
    )


def patch_scores_to_heatmap(
    patch_scores: Tensor,
    grid: PatchGrid | None = None,
    *,
    output_size: int | tuple[int, int] = 224,
    normalize: bool = True,
) -> Tensor:
    """Expand patch scores to image heatmaps; ``normalize`` is display-only scaling."""
    maps = upsample_patch_scores(patch_scores, grid, output_size=output_size)
    if not normalize:
        return maps
    minimum = maps.amin(dim=(-2, -1), keepdim=True)
    maximum = maps.amax(dim=(-2, -1), keepdim=True)
    return (maps - minimum) / (maximum - minimum).clamp_min(torch.finfo(maps.dtype).eps)


anomaly_heatmap = patch_scores_to_heatmap
build_heatmap = patch_scores_to_heatmap
