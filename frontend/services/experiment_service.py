"""Experiment and training log service for Person 4 visualization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frontend.services.backend_interface import resolve_predictor_dir

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


def _is_same_file(recorded: object, expected: Path, root: Path) -> bool:
    if not isinstance(recorded, str) or not recorded:
        return False
    path = Path(recorded)
    path = path if path.is_absolute() else root / path
    return os.path.normcase(os.path.abspath(path)) == os.path.normcase(
        os.path.abspath(expected)
    )


def load_category_threshold(
    category: str,
    *,
    mode: str = "adaptive",
    default: float = 0.5,
    predictor_dir: str | Path | None = None,
) -> tuple[float, bool]:
    """Return a category/mode threshold calibrated on nominal validation scores.

    With ``predictor_dir`` the threshold is used only if its benchmark row records
    that directory's ``<category>_cosine.pt`` as the evaluated predictor, so a
    threshold calibrated for one checkpoint set is never applied to another.
    """
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
            if predictor_dir is not None and not _is_same_file(
                row.get("predictor_checkpoint"),
                resolve_predictor_dir(root, predictor_dir) / f"{category}_cosine.pt",
                root,
            ):
                return default, False
            if "image_score_definition" not in row:
                # Calibrated under the pre-normalization score (max / 2, clamped);
                # not on the scale the live scorer reports now.
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
    category: str = "bottle",
    epochs: int = 50,
    predictor_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load real training history from a checkpoint or generate illustrative curves.

    Inspects `<predictor_dir>/{category}_cosine.pt` (default
    `checkpoints/predictors`), the trainer's saved checkpoint.
    If found, extracts recorded `history` (epoch, train_loss, val_loss).

    Returns:
        tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)
    """
    root = Path(__file__).resolve().parent.parent.parent
    ckpt_path = resolve_predictor_dir(root, predictor_dir) / f"{category}_cosine.pt"

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
            row["category"]: row.get("mean_masks_per_image")
            for row in pilot_results.get("dense", [])
            if isinstance(row.get("category"), str)
        }
        records: list[dict[str, Any]] = []
        for mode, rows in pilot_results.items():
            measured = [
                row
                for row in rows
                if isinstance(row.get("num_images"), int | float)
                and not isinstance(row.get("num_images"), bool)
                and row["num_images"] > 0
            ]
            reductions = [
                100.0 * (1.0 - float(adaptive_masks) / float(dense_masks))
                for row in measured
                if isinstance(row.get("category"), str)
                and (adaptive_masks := row.get("mean_masks_per_image")) is not None
                and (dense_masks := dense_by_category.get(row["category"])) is not None
                and isinstance(adaptive_masks, int | float)
                and not isinstance(adaptive_masks, bool)
                and np.isfinite(adaptive_masks)
                and isinstance(dense_masks, int | float)
                and not isinstance(dense_masks, bool)
                and np.isfinite(dense_masks)
                and dense_masks > 0
            ]

            records.append(
                {
                    "Method": PILOT_MODE_LABELS[mode],
                    "Image AUROC (%)": 100.0
                    * _mean_available_metric(measured, "image_auroc"),
                    "Pixel AUROC (%)": 100.0
                    * _mean_available_metric(measured, "pixel_auroc"),
                    "AUPRO (%)": 100.0 * _mean_available_metric(measured, "aupro"),
                    "Avg Evaluated Masks": _mean_available_metric(
                        measured, "mean_masks_per_image"
                    ),
                    "Mask Reduction vs Dense (%)": (
                        float(np.mean(reductions)) if reductions else float("nan")
                    ),
                    "Cached-latent Latency (ms)": _mean_available_metric(
                        measured, "mean_predictor_latency_ms"
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


def _mean_available_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        value
        for row in rows
        if (value := row.get(key)) is not None
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and np.isfinite(value)
    ]
    return float(np.mean(values)) if values else float("nan")


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
    if "Method" not in bench_df:
        return [
            ("Mask reduction", "—", "Unavailable: adaptive and dense results required"),
            ("Image AUROC", "—", "No measured results"),
            (
                "Cached latency gain",
                "—",
                "Unavailable: adaptive and dense results required",
            ),
        ]

    adaptive_rows = bench_df.loc[bench_df["Method"] == "Adaptive I-JEPA"]
    dense_rows = bench_df.loc[bench_df["Method"] == "Dense-mask I-JEPA"]
    adaptive = adaptive_rows.iloc[0] if not adaptive_rows.empty else None
    dense = dense_rows.iloc[0] if not dense_rows.empty else None

    def metric(row: pd.Series[Any] | None, column: str) -> float | None:
        if row is None:
            return None
        value = row.get(column)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, int | float | np.number)
            or not np.isfinite(value)
        ):
            return None
        return float(value)

    image_row = (
        adaptive
        if metric(adaptive, "Image AUROC (%)") is not None
        else next(
            (
                row
                for _, row in bench_df.iterrows()
                if metric(row, "Image AUROC (%)") is not None
            ),
            None,
        )
    )
    image_value = metric(image_row, "Image AUROC (%)")
    image_method = (
        str(image_row.get("Method", "Unknown")) if image_row is not None else ""
    )
    adaptive_masks = metric(adaptive, "Avg Evaluated Masks")
    dense_masks = metric(dense, "Avg Evaluated Masks")
    if adaptive is None or dense is None:
        reduction = None
    else:
        reduction = metric(adaptive, "Mask Reduction vs Dense (%)")
        if (
            reduction is None
            and adaptive_masks is not None
            and dense_masks is not None
            and dense_masks > 0
        ):
            reduction = 100.0 * (1 - adaptive_masks / dense_masks)
    adaptive_latency = metric(adaptive, "Cached-latent Latency (ms)")
    dense_latency = metric(dense, "Cached-latent Latency (ms)")
    latency_gain = (
        dense_latency / adaptive_latency
        if dense_latency is not None
        and adaptive_latency is not None
        and dense_latency >= 0
        and adaptive_latency > 0
        else None
    )
    return [
        (
            "Mask reduction",
            f"{reduction:.1f} %" if reduction is not None else "—",
            (
                "Adaptive vs Dense (pilot)"
                if reduction is not None
                else "Unavailable: adaptive and dense results required"
            ),
        ),
        (
            "Image AUROC",
            f"{image_value:.1f} %" if image_value is not None else "—",
            image_method if image_value is not None else "Unavailable",
        ),
        (
            "Cached latency gain",
            f"{latency_gain:.1f}×" if latency_gain is not None else "—",
            (
                "Adaptive vs Dense"
                if latency_gain is not None
                else "Unavailable: cached latency measurements required"
            ),
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
                category = row.get("category")
                if not isinstance(category, str):
                    continue
                by_category.setdefault(category, {"Category": category})[
                    PILOT_MODE_LABELS[mode]
                ] = (
                    100.0 * float(row["image_auroc"])
                    if isinstance(row.get("image_auroc"), int | float)
                    and not isinstance(row.get("image_auroc"), bool)
                    and np.isfinite(row["image_auroc"])
                    else float("nan")
                )
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
