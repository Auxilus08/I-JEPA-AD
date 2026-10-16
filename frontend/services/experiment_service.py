"""Experiment and training log service for Person 4 visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PILOT_MODE_LABELS = {
    "adaptive": "Adaptive I-JEPA",
    "dense": "Dense-mask I-JEPA",
    "fixed": "Fixed-mask I-JEPA",
}


def _load_pilot_results(root: Path) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for mode in PILOT_MODE_LABELS:
        path = root / "results" / "benchmarks" / f"pilot_{mode}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("results"), list
        ):
            raise ValueError(f"invalid pilot benchmark file: {path}")
        rows = payload["results"]
        if any(not isinstance(row, dict) or row.get("mode") != mode for row in rows):
            raise ValueError(f"pilot benchmark mode mismatch: {path}")
        results[mode] = rows
    return results


def load_category_threshold(
    category: str, *, mode: str = "adaptive", default: float = 0.5
) -> tuple[float, bool]:
    """Return a category/mode threshold calibrated on nominal validation scores."""
    if mode not in PILOT_MODE_LABELS:
        raise ValueError(f"unsupported calibration mode: {mode!r}")
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "results" / "benchmarks" / f"pilot_{mode}.json"
    if not path.is_file():
        return default, False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"invalid pilot benchmark file: {path}")
    for row in payload["results"]:
        if not isinstance(row, dict):
            raise ValueError(f"invalid pilot benchmark record: {path}")
        if row.get("category") == category and row.get("mode") == mode:
            threshold = row.get("decision_threshold")
            if threshold is None:
                return default, False
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int | float)
                or not np.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError(f"invalid calibrated threshold in {path}")
            return float(threshold), True
    return default, False


def load_training_history(
    category: str = "bottle", epochs: int = 50
) -> tuple[pd.DataFrame, bool]:
    """Load real training history from a checkpoint or generate illustrative curves.

    Inspects `results/predictor/{category}.pt` (the trainer's saved checkpoint).
    If found, extracts recorded `history` (epoch, train_loss, val_loss).

    Returns:
        tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)
    """
    root = Path(__file__).resolve().parent.parent.parent
    ckpt_path = root / "checkpoints" / "predictors" / f"{category}_cosine.pt"

    if ckpt_path.is_file():
        import torch

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"predictor checkpoint must contain a mapping: {ckpt_path}")
        history = checkpoint.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(
                f"predictor checkpoint has no training history: {ckpt_path}"
            )
        records: list[dict[str, Any]] = []
        for entry in history:
            if not isinstance(entry, dict):
                raise TypeError(f"invalid training-history entry in {ckpt_path}")
            records.append(
                {
                    "Epoch": entry["epoch"],
                    "Train Loss": entry["train_loss"],
                    "Val Loss": entry["val_loss"],
                }
            )
        df = pd.DataFrame(records)
        return df, False

    # Reproducible illustrative learning curves demonstrating expected convergence
    rng = np.random.default_rng(seed=42 + len(category))
    epoch_arr = np.arange(1, epochs + 1)

    train_loss = (
        0.42 * np.exp(-epoch_arr / 14.0) + 0.08 + rng.normal(0, 0.004, size=epochs)
    )
    val_loss = (
        0.45 * np.exp(-epoch_arr / 14.0) + 0.09 + rng.normal(0, 0.006, size=epochs)
    )

    df = pd.DataFrame(
        {
            "Epoch": epoch_arr,
            "Train Loss": np.clip(train_loss, 0.0, None),
            "Val Loss": np.clip(val_loss, 0.0, None),
        }
    )
    return df, True


def load_benchmark_summary() -> tuple[pd.DataFrame, bool]:
    """Return benchmark comparison table across MVTec categories and methods.

    Returns:
        tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)
    """
    root = Path(__file__).resolve().parent.parent.parent
    pilot_results = _load_pilot_results(root)
    if pilot_results:
        dense_by_category = {
            row["category"]: row["mean_masks_per_image"]
            for row in pilot_results.get("dense", [])
        }
        records: list[dict[str, Any]] = []
        for mode, rows in pilot_results.items():
            measured = [row for row in rows if row.get("num_images", 0) > 0]
            reductions = [
                100.0
                * (
                    1.0
                    - row["mean_masks_per_image"] / dense_by_category[row["category"]]
                )
                for row in measured
                if row["category"] in dense_by_category
                and dense_by_category[row["category"]] > 0
            ]
            records.append(
                {
                    "Method": PILOT_MODE_LABELS[mode],
                    "Image AUROC (%)": 100.0
                    * float(np.mean([row["image_auroc"] for row in measured])),
                    "Pixel AUROC (%)": 100.0
                    * float(np.mean([row["pixel_auroc"] for row in measured])),
                    "AUPRO (%)": 100.0
                    * float(np.mean([row["aupro"] for row in measured])),
                    "Avg Evaluated Masks": float(
                        np.mean([row["mean_masks_per_image"] for row in measured])
                    ),
                    "Mask Reduction vs Dense (%)": (
                        float(np.mean(reductions)) if reductions else 0.0
                    ),
                    "Cached-latent Latency (ms)": float(
                        np.mean([row["mean_predictor_latency_ms"] for row in measured])
                    ),
                    "Peak VRAM (MB)": _maximum_available_vram(measured),
                }
            )
        return pd.DataFrame(records), False

    bench_file = root / "results" / "benchmarks" / "summary.csv"

    if bench_file.is_file():
        return pd.read_csv(bench_file), False

    # Target / illustrative baseline comparisons from project proposal
    data = {
        "Method": [
            "Adaptive I-JEPA (Ours)",
            "Dense-mask I-JEPA",
            "Fixed-mask I-JEPA",
            "PatchCore",
            "MAE Reconstruction",
        ],
        "Image AUROC (%)": [98.4, 98.7, 91.5, 97.9, 92.6],
        "Pixel AUROC (%)": [96.8, 97.1, 87.8, 96.2, 88.3],
        "Avg Evaluated Masks": [14, 196, 16, 0, 1],
        "Mask Reduction vs Dense (%)": [92.9, 0.0, 91.8, 100.0, 99.5],
        "Inference Latency (ms)": [42.5, 218.4, 31.0, 115.8, 88.2],
    }
    return pd.DataFrame(data), True


def _maximum_available_vram(rows: list[dict[str, Any]]) -> float | None:
    values = [
        value
        for row in rows
        if (value := row.get("peak_vram_mb")) is not None
        and isinstance(value, int | float)
        and np.isfinite(value)
    ]
    return float(max(values)) if values else None


def benchmark_efficiency_highlights(
    bench_df: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    """Summarize adaptive-vs-dense efficiency when both measurements are present."""
    adaptive_rows = bench_df.loc[bench_df["Method"] == "Adaptive I-JEPA"]
    dense_rows = bench_df.loc[bench_df["Method"] == "Dense-mask I-JEPA"]
    if adaptive_rows.empty or dense_rows.empty:
        row = bench_df.iloc[0] if not bench_df.empty else None
        image_auroc = "—" if row is None else f"{row['Image AUROC (%)']:.1f} %"
        image_method = "No measured results" if row is None else str(row["Method"])
        return [
            ("Mask reduction", "—", "Requires adaptive and dense results"),
            ("Image AUROC", image_auroc, image_method),
            ("Cached latency gain", "—", "Requires adaptive and dense results"),
        ]

    adaptive = adaptive_rows.iloc[0]
    dense = dense_rows.iloc[0]
    adaptive_latency = float(adaptive["Cached-latent Latency (ms)"])
    latency_gain = (
        float(dense["Cached-latent Latency (ms)"]) / adaptive_latency
        if adaptive_latency > 0
        else float("nan")
    )
    return [
        (
            "Mask reduction",
            f"{adaptive['Mask Reduction vs Dense (%)']:.1f} %",
            "Adaptive vs Dense (pilot)",
        ),
        (
            "Image AUROC",
            f"{adaptive['Image AUROC (%)']:.1f} %",
            "Adaptive, macro average",
        ),
        (
            "Cached latency gain",
            f"{latency_gain:.1f}×" if np.isfinite(latency_gain) else "—",
            "Adaptive vs Dense",
        ),
    ]


def load_category_auroc_breakdown() -> tuple[pd.DataFrame, bool]:
    """Return category-wise AUROC breakdown for all 15 MVTec objects and textures.

    Returns:
        tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)
    """
    root = Path(__file__).resolve().parent.parent.parent
    pilot_results = _load_pilot_results(root)
    if pilot_results:
        by_category: dict[str, dict[str, Any]] = {}
        for mode, rows in pilot_results.items():
            for row in rows:
                category = row["category"]
                by_category.setdefault(category, {"Category": category})[
                    PILOT_MODE_LABELS[mode]
                ] = 100.0 * float(row["image_auroc"])
        return (
            pd.DataFrame([by_category[name] for name in sorted(by_category)]),
            False,
        )

    cat_file = root / "results" / "benchmarks" / "category_breakdown.csv"

    if cat_file.is_file():
        return pd.read_csv(cat_file), False

    categories = [
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

    # Target numbers per proposal
    adaptive_scores = [
        99.2,
        98.1,
        98.5,
        97.4,
        97.9,
        98.8,
        99.5,
        98.0,
        97.8,
        96.9,
        98.2,
        97.5,
        98.6,
        99.1,
        98.4,
    ]
    dense_scores = [
        99.4,
        98.3,
        98.7,
        97.6,
        98.1,
        99.0,
        99.6,
        98.2,
        98.0,
        97.1,
        98.4,
        97.7,
        98.8,
        99.3,
        98.6,
    ]
    patchcore_scores = [
        98.5,
        97.2,
        97.8,
        96.5,
        97.0,
        98.1,
        99.0,
        97.4,
        97.0,
        95.8,
        97.6,
        96.8,
        98.0,
        98.5,
        97.8,
    ]

    df = pd.DataFrame(
        {
            "Category": categories,
            "Adaptive I-JEPA": adaptive_scores,
            "Dense I-JEPA": dense_scores,
            "PatchCore": patchcore_scores,
        }
    )
    return df, True
