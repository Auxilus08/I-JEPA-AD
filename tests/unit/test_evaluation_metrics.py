"""Tests for anomaly-detection metrics."""

import numpy as np
import pytest
from numpy.typing import NDArray

from ijepa_ad.evaluation.metrics import (
    aupro,
    binary_auroc,
    calibrate_image_threshold,
    image_auroc,
    pixel_auroc,
)
from ijepa_ad.evaluation.runner import evaluate_pilot


def test_image_auroc_handles_ordering_and_tied_scores() -> None:
    assert image_auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert image_auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_binary_auroc_vectorizes_large_tied_score_groups() -> None:
    labels = np.tile([0, 1], 50_000)
    scores = np.ones(labels.size, dtype=np.float32)

    assert binary_auroc(labels, scores) == pytest.approx(0.5)


def test_image_threshold_uses_only_nominal_validation_scores() -> None:
    threshold, observed_fpr = calibrate_image_threshold(
        [0.1, 0.2, 0.3, 0.4], target_fpr=0.25
    )

    assert threshold == pytest.approx(0.3)
    assert observed_fpr == pytest.approx(0.25)


@pytest.mark.parametrize(  # type: ignore[misc]
    "masking_threshold", [float("nan"), float("inf"), -0.1]
)
def test_pilot_rejects_invalid_masking_threshold(
    masking_threshold: float,
) -> None:
    with pytest.raises(ValueError, match="masking_threshold must be finite"):
        evaluate_pilot(
            categories=(),
            dataset_root="unused",
            cache_root="unused",
            checkpoint_dir="unused",
            encoder_checkpoint="unused",
            device="cpu",
            mode="adaptive",
            masking_config="unused",
            masking_threshold=masking_threshold,
        )


def test_pixel_auroc_uses_all_image_pixels() -> None:
    normal_scores: NDArray[np.float32] = np.zeros((2, 2), dtype=np.float32)
    normal_mask: NDArray[np.bool_] = np.zeros((2, 2), dtype=bool)
    defect_scores: NDArray[np.float32] = np.array(
        [[0.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    defect_mask: NDArray[np.bool_] = np.array([[False, False], [False, True]])

    assert pixel_auroc(
        [normal_scores, defect_scores], [normal_mask, defect_mask]
    ) == pytest.approx(1.0)


def test_aupro_is_one_for_perfectly_ranked_defect_region() -> None:
    scores: NDArray[np.float32] = np.zeros((4, 4), dtype=np.float32)
    scores[1:3, 1:3] = 1.0
    mask: NDArray[np.bool_] = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    assert aupro([scores], [mask], num_thresholds=50) == pytest.approx(1.0)


def test_aupro_matches_direct_threshold_scan() -> None:
    scores: NDArray[np.float32] = np.array(
        [[0.0, 0.1, 0.7], [0.2, 0.8, 0.4], [0.9, 0.3, 0.6]],
        dtype=np.float32,
    )
    mask: NDArray[np.bool_] = np.zeros((3, 3), dtype=bool)
    mask[1, 1:3] = True
    mask[0, 0] = True
    components = (np.array([0]), np.array([4, 5]))
    max_fpr = 0.3
    num_thresholds = 11
    epsilon = max(np.finfo(np.float64).eps, (scores.max() - scores.min()) * 1e-9)
    thresholds = np.linspace(
        scores.max() + epsilon, scores.min() - epsilon, num_thresholds
    )
    negative_pixels = int((~mask).sum())
    fpr = []
    pro = []
    for threshold in thresholds:
        predicted = scores >= threshold
        fpr.append(np.count_nonzero(predicted & ~mask) / negative_pixels)
        flat_prediction = predicted.reshape(-1)
        pro.append(
            float(
                np.mean(
                    [
                        np.count_nonzero(flat_prediction[component]) / component.size
                        for component in components
                    ]
                )
            )
        )
    last_inside = int(np.searchsorted(fpr, max_fpr, side="right"))
    if last_inside == 0:
        fpr_curve = np.array([0.0, max_fpr])
        pro_curve = np.array([pro[0], pro[0]])
    elif last_inside == num_thresholds:
        fpr_curve = np.asarray(fpr)
        pro_curve = np.asarray(pro)
    else:
        fpr_curve = np.asarray(fpr[:last_inside])
        pro_curve = np.asarray(pro[:last_inside])
        fraction = (max_fpr - fpr_curve[-1]) / (fpr[last_inside] - fpr_curve[-1])
        crossing_pro = pro_curve[-1] + fraction * (pro[last_inside] - pro_curve[-1])
        fpr_curve = np.append(fpr_curve, max_fpr)
        pro_curve = np.append(pro_curve, crossing_pro)
    expected = (
        np.sum((pro_curve[:-1] + pro_curve[1:]) * np.diff(fpr_curve) / 2) / max_fpr
    )

    assert aupro(
        [scores], [mask], max_fpr=max_fpr, num_thresholds=num_thresholds
    ) == pytest.approx(expected)


def test_aupro_requires_an_anomalous_region() -> None:
    with pytest.raises(ValueError, match="at least one anomalous"):
        aupro([np.zeros((2, 2))], [np.zeros((2, 2), dtype=bool)])


def test_pixel_metrics_reject_mismatched_maps() -> None:
    with pytest.raises(ValueError, match="share a 2D shape"):
        pixel_auroc(
            [np.zeros((2, 2)), np.zeros((3, 3))],
            [np.zeros((2, 2), dtype=bool), np.zeros((3, 2), dtype=bool)],
        )
