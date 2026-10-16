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

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from frontend.components.visualizer import (
    create_heatmap_overlay,
    draw_detected_regions,
)
from frontend.config import load_frontend_config
from frontend.services import backend_interface, experiment_service
from frontend.services.backend_interface import run_inference
from frontend.services.experiment_service import (
    benchmark_efficiency_highlights,
    load_benchmark_summary,
    load_category_auroc_breakdown,
    load_category_threshold,
    load_training_history,
)
from numpy.typing import NDArray
from PIL import Image

from ijepa_ad.caching import latent_cache


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


def test_mock_inference_remains_available_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_interface, "PYTORCH_AVAILABLE", False)
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))

    result = run_inference(img, use_mock=True)
    assert result["is_mock"] is True

    with pytest.raises(RuntimeError, match="requires both PyTorch"):
        run_inference(img, use_mock=False)


def test_disconnected_high_score_clusters_get_separate_boxes() -> None:
    heatmap: NDArray[np.float32] = np.zeros((20, 20), dtype=np.float32)
    heatmap[2:6, 3:7] = 0.9
    heatmap[12:16, 13:17] = 0.8

    regions = backend_interface.detected_regions_for_threshold(
        heatmap, (20, 20), 0.9, 0.5, "bottle"
    )

    assert [region["box"] for region in regions] == [
        [3, 2, 7, 6],
        [13, 12, 17, 16],
    ]
    assert (
        backend_interface.detected_regions_for_threshold(
            heatmap, (20, 20), 0.5, 0.5, "bottle"
        )
        == []
    )


def test_region_detection_does_not_include_background_when_cutoff_is_zero() -> None:
    heatmap: NDArray[np.float32] = np.zeros((20, 20), dtype=np.float32)
    heatmap[3, 4] = 0.9
    heatmap[15, 16] = 0.8

    regions = backend_interface.detected_regions_for_threshold(
        heatmap, (20, 20), 0.9, 0.5, "bottle"
    )

    assert [region["box"] for region in regions] == [
        [4, 3, 5, 4],
        [16, 15, 17, 16],
    ]


def test_efficiency_highlights_handle_partial_and_legacy_benchmarks() -> None:
    partial = pd.DataFrame({"Method": ["Adaptive I-JEPA"], "Image AUROC (%)": [92.5]})
    partial_highlights = benchmark_efficiency_highlights(partial)
    assert partial_highlights[0][1] == "—"
    assert partial_highlights[1][1] == "92.5 %"
    assert partial_highlights[2][1] == "—"

    legacy = pd.DataFrame(
        {
            "Method": ["Adaptive I-JEPA", "Dense-mask I-JEPA"],
            "Image AUROC (%)": [92.5, 93.0],
            "Mask Reduction vs Dense (%)": [80.0, 0.0],
        }
    )
    legacy_highlights = benchmark_efficiency_highlights(legacy)
    assert legacy_highlights[0][1] == "80.0 %"
    assert legacy_highlights[2][1] == "—"
    assert "Unavailable" in legacy_highlights[2][2]


def test_partial_pilot_results_build_an_incomplete_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment_service,
        "_load_pilot_results",
        lambda _root: {
            "adaptive": [
                {
                    "category": "bottle",
                    "num_images": 1,
                    "image_auroc": 0.9,
                    "mean_masks_per_image": 14.0,
                }
            ],
            "dense": [],
        },
    )

    benchmark, is_synthetic = load_benchmark_summary()
    highlights = benchmark_efficiency_highlights(benchmark)

    assert is_synthetic is False
    assert len(benchmark) == 2
    assert highlights[1][1] == "90.0 %"
    assert highlights[0][1] == "—"
    assert highlights[2][1] == "—"


def test_encoder_hash_cache_persists_with_checkpoint_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "IN22K-vit.g.16-600e.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[Path] = []

    def compute_hash(path: Path) -> str:
        calls.append(path)
        return "a" * 64

    monkeypatch.setattr(latent_cache, "compute_file_sha256", compute_hash)
    backend_interface._checkpoint_sha256.cache_clear()
    try:
        assert backend_interface._checkpoint_sha256(str(tmp_path)) == "a" * 64
        backend_interface._checkpoint_sha256.cache_clear()
        assert backend_interface._checkpoint_sha256(str(tmp_path)) == "a" * 64
        assert calls == [checkpoint]

        checkpoint.write_bytes(b"changed checkpoint size")
        backend_interface._checkpoint_sha256.cache_clear()
        assert backend_interface._checkpoint_sha256(str(tmp_path)) == "a" * 64
        assert calls == [checkpoint, checkpoint]
    finally:
        backend_interface._checkpoint_sha256.cache_clear()


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


def _write_pilot_file(root: Path, **row: object) -> None:
    path = root / "results" / "benchmarks" / "pilot_adaptive.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"category": "bottle", "mode": "adaptive", **row}
    path.write_text(json.dumps({"results": [record]}), encoding="utf-8")


def test_calibrated_threshold_requires_current_score_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        experiment_service,
        "__file__",
        str(tmp_path / "frontend" / "services" / "experiment_service.py"),
    )
    # Calibrated on the pre-normalization scale: must not be reused.
    _write_pilot_file(tmp_path, decision_threshold=0.4)
    assert load_category_threshold("bottle", default=0.5) == (0.5, False)

    _write_pilot_file(
        tmp_path,
        decision_threshold=0.4,
        image_score_definition={"upper_bound": 4.0},
    )
    assert load_category_threshold("bottle", default=0.5) == (0.4, True)


# --- predictor checkpoint directory selection (notebook 01 vs notebook 02 outputs) ---

_ENCODER_SHA = "b" * 64


def test_resolve_predictor_dir_default_relative_and_absolute(tmp_path: Path) -> None:
    resolve = backend_interface.resolve_predictor_dir
    assert resolve(tmp_path) == tmp_path / "checkpoints" / "predictors"
    assert (
        resolve(tmp_path, "checkpoints/predictors_augmented")
        == tmp_path / "checkpoints" / "predictors_augmented"
    )
    assert resolve(tmp_path, tmp_path / "elsewhere") == tmp_path / "elsewhere"


def test_frontend_config_lists_both_notebook_checkpoint_dirs() -> None:
    dirs = load_frontend_config()["predictor_checkpoint_dirs"]
    assert dirs["baseline"] == "checkpoints/predictors"
    assert dirs["augmented"] == "checkpoints/predictors_augmented"
    assert load_frontend_config()["defaults"]["predictor_checkpoints"] in dirs


def _write_predictor_checkpoint(
    path: Path, seed: int, category: str = "bottle"
) -> None:
    import torch

    from ijepa_ad.predictor.transformer_predictor import (
        LatentTransformerPredictor,
        PredictorConfig,
    )

    torch.manual_seed(seed)
    config = PredictorConfig(1408, 196, 16, 1, 2, 2.0, 0.0)
    model = LatentTransformerPredictor(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "training_config": {"loss_type": "cosine", "mask_strategy": "x"},
            "history": [{"epoch": 1, "train_loss": 0.5 + seed, "val_loss": 0.6}],
            "cache_provenance": {
                "category": category,
                "checkpoint_sha256": _ENCODER_SHA,
            },
        },
        path,
    )


@pytest.fixture  # type: ignore[misc]
def fake_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A project root with two real-format predictor sets and a stub encoder."""
    import torch

    masking = tmp_path / "configs" / "masking"
    masking.mkdir(parents=True)
    source = Path(backend_interface.__file__).resolve().parents[2]
    (masking / "coarse_to_fine.yaml").write_text(
        (source / "configs" / "masking" / "coarse_to_fine.yaml").read_text("utf-8"),
        encoding="utf-8",
    )
    _write_predictor_checkpoint(tmp_path / "checkpoints/predictors/bottle_cosine.pt", 1)
    _write_predictor_checkpoint(
        tmp_path / "checkpoints/predictors_augmented/bottle_cosine.pt", 2
    )

    class StubEncoder:
        def encode(self, batch: torch.Tensor) -> torch.Tensor:
            generator = torch.Generator().manual_seed(0)
            return torch.randn(batch.shape[0], 196, 1408, generator=generator)

    monkeypatch.setattr(backend_interface, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        backend_interface, "_checkpoint_sha256", lambda _r: _ENCODER_SHA
    )
    monkeypatch.setattr(backend_interface, "_load_encoder", lambda *_a: StubEncoder())
    backend_interface._load_predictor.cache_clear()
    yield tmp_path
    backend_interface._load_predictor.cache_clear()


def test_both_notebook_checkpoint_dirs_run_through_the_same_real_inference(
    fake_project: Path,
) -> None:
    image = Image.new("RGB", (64, 64), color=(120, 120, 120))

    default = run_inference(image, category="bottle")
    baseline = run_inference(
        image, category="bottle", predictor_dir="checkpoints/predictors"
    )
    augmented = run_inference(
        image, category="bottle", predictor_dir="checkpoints/predictors_augmented"
    )

    for result in (default, baseline, augmented):
        assert result["is_mock"] is False
        assert result["pixel_heatmap"].shape == (224, 224)
    # No directory means the baseline notebook's directory (previous behavior).
    assert default["image_score"] == baseline["image_score"]
    # Same stub latents, different weights: the selected directory was really loaded.
    assert augmented["image_score"] != baseline["image_score"]


def test_selected_directory_still_goes_through_provenance_checked_loader(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (64, 64), color=(120, 120, 120))
    # Checkpoint trained for another category must be refused, not loaded.
    _write_predictor_checkpoint(
        fake_project / "checkpoints/predictors_augmented/capsule_cosine.pt",
        3,
        category="bottle",
    )
    with pytest.raises(ValueError, match="category does not match"):
        run_inference(
            image,
            category="capsule",
            predictor_dir="checkpoints/predictors_augmented",
        )
    # A checkpoint trained with a different encoder must be refused too.
    monkeypatch.setattr(backend_interface, "_checkpoint_sha256", lambda _r: "c" * 64)
    backend_interface._load_predictor.cache_clear()
    with pytest.raises(ValueError, match="different encoder"):
        run_inference(
            image,
            category="bottle",
            predictor_dir="checkpoints/predictors_augmented",
        )


def test_missing_checkpoint_in_selected_directory_names_that_directory(
    fake_project: Path,
) -> None:
    image = Image.new("RGB", (64, 64), color=(120, 120, 120))
    with pytest.raises(FileNotFoundError, match="predictors_augmented"):
        run_inference(
            image,
            category="capsule",
            predictor_dir="checkpoints/predictors_augmented",
        )


def test_training_history_and_threshold_follow_the_selected_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        experiment_service,
        "__file__",
        str(tmp_path / "frontend" / "services" / "experiment_service.py"),
    )
    _write_predictor_checkpoint(tmp_path / "checkpoints/predictors/bottle_cosine.pt", 1)
    _write_predictor_checkpoint(
        tmp_path / "checkpoints/predictors_augmented/bottle_cosine.pt", 2
    )
    baseline, _ = load_training_history(
        "bottle", predictor_dir="checkpoints/predictors"
    )
    augmented, _ = load_training_history(
        "bottle", predictor_dir="checkpoints/predictors_augmented"
    )
    assert float(baseline["Train Loss"].iloc[0]) == 1.5
    assert float(augmented["Train Loss"].iloc[0]) == 2.5

    # A threshold calibrated for one checkpoint set is never reused for another.
    _write_pilot_file(
        tmp_path,
        decision_threshold=0.4,
        image_score_definition={"upper_bound": 4.0},
        predictor_checkpoint="checkpoints/predictors/bottle_cosine.pt",
    )
    baseline_dir, augmented_dir = (
        "checkpoints/predictors",
        "checkpoints/predictors_augmented",
    )
    assert load_category_threshold("bottle", predictor_dir=baseline_dir) == (0.4, True)
    assert load_category_threshold("bottle", predictor_dir=augmented_dir) == (
        0.5,
        False,
    )
    # Without a selection the original behavior is unchanged.
    assert load_category_threshold("bottle") == (0.4, True)
