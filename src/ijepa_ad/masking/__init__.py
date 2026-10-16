"""Masking, fusion, and anomaly-map utilities."""

from .adaptive_engine import AdaptiveMaskingEngine, MaskingResult
from .coarse_masker import (
    CoarseMasker,
    MaskWindow,
    PatchGrid,
    coarse_anomaly_score,
    latent_anomaly_scores,
)
from .fine_masker import FineMasker, FineSelection, select_suspicious_regions
from .fusion import fuse_anomaly_scores, fuse_scores, score_fusion
from .score_maps import (
    anomaly_heatmap,
    build_heatmap,
    patch_scores_to_heatmap,
    upsample_patch_scores,
)

__all__ = [
    "AdaptiveMaskingEngine",
    "CoarseMasker",
    "FineMasker",
    "FineSelection",
    "MaskWindow",
    "MaskingResult",
    "PatchGrid",
    "anomaly_heatmap",
    "build_heatmap",
    "coarse_anomaly_score",
    "fuse_anomaly_scores",
    "fuse_scores",
    "patch_scores_to_heatmap",
    "score_fusion",
    "select_suspicious_regions",
    "latent_anomaly_scores",
    "upsample_patch_scores",
]
