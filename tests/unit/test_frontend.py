"""Unit and contract tests for Person 4 Frontend and Visualization.

Verifies:
- Interface contract compliance (keys, value ranges, array shapes).
- Graceful rejection of invalid inputs.
- Mask count invariant: Adaptive evaluates fewer masks than Dense.
- Heatmap and overlay visualizer logic.
- Training and benchmark data loader services.
- Region detection consistency between nominal and defective samples.
"""

from __future__ import annotations

import numpy as np
import pytest
from frontend.components.visualizer import (
    create_heatmap_overlay,
    draw_detected_regions,
)
from frontend.config import load_frontend_config
from frontend.services.backend_interface import run_inference
from frontend.services.experiment_service import (
    load_benchmark_summary,
    load_category_auroc_breakdown,
    load_category_threshold,
    load_training_history,
)
from PIL import Image


def test_frontend_config_loads_defaults() -> None:
    """Verify frontend configuration contains required categories and methods."""
    cfg = load_frontend_config()
    assert "categories" in cfg
    assert "methods" in cfg
    assert len(cfg["categories"]) == 15
    assert "adaptive_ijepa" in cfg["methods"]
    assert "dense_masked_ijepa" in cfg["methods"]
    assert "fixed_mask_ijepa" in cfg["methods"]
    assert "patchcore" in cfg["methods"]
    assert "mae_reconstruction" in cfg["methods"]


def test_backend_contract_schema() -> None:
    """Verify run_inference satisfies the schema on Page 8 of the design spec."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    res = run_inference(
        image=img, method="adaptive_ijepa", category="bottle", use_mock=True
    )

    # Required contract keys
    assert "image_score" in res
    assert "pixel_heatmap" in res
    assert "detected_regions" in res
    assert "mask_count" in res
    assert "latency_ms" in res
    assert "peak_vram_mb" in res
    assert "method" in res
    assert "category" in res

    # Types and value invariants
    assert isinstance(res["image_score"], float)
    assert 0.0 <= res["image_score"] <= 1.0
    assert isinstance(res["pixel_heatmap"], np.ndarray)
    assert res["pixel_heatmap"].ndim == 2
    assert res["pixel_heatmap"].shape == (224, 224)
    assert isinstance(res["mask_count"], int)
    assert res["mask_count"] > 0
    assert isinstance(res["latency_ms"], float)
    assert res["latency_ms"] >= 0.0
    assert res["method"] == "adaptive_ijepa"
    assert res["category"] == "bottle"


def test_invalid_image_input_rejected_gracefully() -> None:
    """Verify non-image input raises a clear ValueError instead of crashing."""
    with pytest.raises(ValueError, match="Expected image to be a PIL.Image instance"):
        run_inference(image="invalid_image_path.xyz", method="adaptive_ijepa")


def test_adaptive_evaluates_fewer_masks_than_dense() -> None:
    """Verify the core invariant: Adaptive evaluates fewer regions than Dense."""
    img = Image.new("RGB", (224, 224))
    res_adaptive = run_inference(image=img, method="adaptive_ijepa", use_mock=True)
    res_dense = run_inference(image=img, method="dense_masked_ijepa", use_mock=True)

    assert res_adaptive["mask_count"] < res_dense["mask_count"]
    assert res_dense["mask_count"] == 196
    assert res_adaptive["mask_count"] <= 20


def test_nominal_samples_produce_no_detected_regions() -> None:
    """Verify consistency: when image score is below threshold, no boxes are drawn."""
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    # Threshold 0.99 ensures the sample is nominal
    res = run_inference(
        image=img,
        method="adaptive_ijepa",
        category="bottle",
        use_mock=True,
        anomaly_threshold=0.99,
    )
    assert res["image_score"] < 0.99
    assert res["detected_regions"] == []


def test_heatmap_overlay_generation() -> None:
    """Verify heatmap overlay blending preserves image dimensions and mode."""
    img = Image.new("RGB", (200, 150), color=(100, 150, 200))
    heat = np.random.uniform(0.0, 1.0, size=(150, 200)).astype(np.float32)

    overlay = create_heatmap_overlay(img, heat, alpha=0.5, colormap_name="jet")
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (200, 150)
    assert overlay.mode == "RGB"


def test_draw_detected_regions() -> None:
    """Verify bounding box rendering does not crash or distort image size."""
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    regions = [{"box": [20, 30, 80, 90], "confidence": 0.88, "label": "Defect"}]

    annotated = draw_detected_regions(img, regions)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (224, 224)


def test_training_and_benchmark_services() -> None:
    """Verify training curves and benchmark summaries return valid DataFrames."""
    df_train, is_synth_train = load_training_history(category="bottle", epochs=30)
    assert len(df_train) == (30 if is_synth_train else 100)
    assert "Train Loss" in df_train.columns
    assert "Val Loss" in df_train.columns
    assert isinstance(is_synth_train, bool)

    df_bench, is_synth_bench = load_benchmark_summary()
    assert len(df_bench) == (5 if is_synth_bench else 3)
    assert "Image AUROC (%)" in df_bench.columns
    assert isinstance(is_synth_bench, bool)

    df_cat, is_synth_cat = load_category_auroc_breakdown()
    assert len(df_cat) == (15 if is_synth_cat else 3)
    assert "Category" in df_cat.columns
    assert isinstance(is_synth_cat, bool)

    threshold, is_calibrated = load_category_threshold("bottle")
    assert 0.0 <= threshold <= 1.0
    assert isinstance(is_calibrated, bool)
