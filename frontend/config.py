"""Frontend configuration loader conforming to project guidelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Default constants adhering to PDF specifications
DEFAULT_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

DEFAULT_METHODS = {
    "adaptive_ijepa": "Adaptive I-JEPA (Coarse-to-Fine)",
    "dense_masked_ijepa": "Dense-mask I-JEPA Baseline",
    "fixed_mask_ijepa": "Fixed-mask I-JEPA Baseline",
    "patchcore": "PatchCore Baseline",
    "mae_reconstruction": "MAE Reconstruction Baseline",
}

# Directories written by notebooks/01 (baseline) and notebooks/02 (augmented).
DEFAULT_PREDICTOR_DIRS = {
    "baseline": "checkpoints/predictors",
    "augmented": "checkpoints/predictors_augmented",
}
DEFAULT_PREDICTOR_CHECKPOINTS = "baseline"

DEFAULT_ANOMALY_THRESHOLD = 0.50
DEFAULT_FINE_THRESHOLD = 0.40
DEFAULT_LAMBDA_FUSION = 1.0
DEFAULT_COLORMAP = "jet"
DEFAULT_OVERLAY_ALPHA = 0.55


def load_frontend_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from configs/frontend/base.yaml with fallback."""
    if config_path is None:
        root = Path(__file__).resolve().parent.parent
        config_path = root / "configs" / "frontend" / "base.yaml"

    if config_path.is_file():
        try:
            import yaml

            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    return {
        "categories": DEFAULT_CATEGORIES,
        "methods": DEFAULT_METHODS,
        "predictor_checkpoint_dirs": DEFAULT_PREDICTOR_DIRS,
        "defaults": {
            "predictor_checkpoints": DEFAULT_PREDICTOR_CHECKPOINTS,
            "category": "bottle",
            "method": "adaptive_ijepa",
            "anomaly_threshold": DEFAULT_ANOMALY_THRESHOLD,
            "fine_threshold": DEFAULT_FINE_THRESHOLD,
            "lambda_fusion": DEFAULT_LAMBDA_FUSION,
            "colormap": DEFAULT_COLORMAP,
            "overlay_alpha": DEFAULT_OVERLAY_ALPHA,
        },
    }
