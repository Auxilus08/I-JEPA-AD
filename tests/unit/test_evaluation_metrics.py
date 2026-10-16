"""Tests for anomaly-detection metrics."""

import numpy as np
import pytest

from ijepa_ad.evaluation.metrics import (
    aupro,
    calibrate_image_threshold,
    image_auroc,
    pixel_auroc,
)


def test_image_auroc_handles_ordering_and_tied_scores() -> None:
    assert image_auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert image_auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_image_threshold_uses_only_nominal_validation_scores() -> None:
    threshold, observed_fpr = calibrate_image_threshold(
        [0.1, 0.2, 0.3, 0.4], target_fpr=0.25
    )

    assert threshold == pytest.approx(0.3)
    assert observed_fpr == pytest.approx(0.25)


def test_pixel_auroc_uses_all_image_pixels() -> None:
    normal_scores = np.zeros((2, 2), dtype=np.float32)
    normal_mask = np.zeros((2, 2), dtype=bool)
    defect_scores = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    defect_mask = np.array([[False, False], [False, True]])

    assert pixel_auroc(
        [normal_scores, defect_scores], [normal_mask, defect_mask]
    ) == pytest.approx(1.0)


def test_aupro_is_one_for_perfectly_ranked_defect_region() -> None:
    scores = np.zeros((4, 4), dtype=np.float32)
    scores[1:3, 1:3] = 1.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    assert aupro([scores], [mask], num_thresholds=50) == pytest.approx(1.0)


def test_aupro_requires_an_anomalous_region() -> None:
    with pytest.raises(ValueError, match="at least one anomalous"):
        aupro([np.zeros((2, 2))], [np.zeros((2, 2), dtype=bool)])


def test_pixel_metrics_reject_mismatched_maps() -> None:
    with pytest.raises(ValueError, match="share a 2D shape"):
        pixel_auroc(
            [np.zeros((2, 2)), np.zeros((3, 3))],
            [np.zeros((2, 2), dtype=bool), np.zeros((3, 2), dtype=bool)],
        )
