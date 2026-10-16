"""Image-level and pixel-level anomaly-detection metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def image_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute binary AUROC with average ranks for tied scores."""
    return binary_auroc(labels, scores)


def calibrate_image_threshold(
    nominal_scores: Sequence[float], *, target_fpr: float = 0.05
) -> tuple[float, float]:
    """Choose a score threshold from nominal validation scores only.

    The order statistic allows at most ``floor(target_fpr * n)`` validation
    exceedances. Ties can lower the observed rate.
    """
    scores = np.asarray(nominal_scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size:
        raise ValueError("nominal_scores must be a non-empty one-dimensional sequence")
    if not np.isfinite(scores).all():
        raise ValueError("nominal_scores must contain only finite values")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    allowed_exceedances = int(np.floor(target_fpr * scores.size))
    threshold = float(np.sort(scores)[-(allowed_exceedances + 1)])
    observed_fpr = float(np.mean(scores > threshold))
    return threshold, observed_fpr


def pixel_auroc(
    score_maps: Sequence[NDArray[np.floating]],
    masks: Sequence[NDArray[np.bool_]],
) -> float:
    """Compute pixel AUROC across all images without per-image score scaling."""
    score_arrays, mask_arrays = _validate_maps(score_maps, masks)
    labels = np.concatenate([mask.reshape(-1) for mask in mask_arrays])
    scores = np.concatenate([score.reshape(-1) for score in score_arrays])
    return binary_auroc(labels.astype(np.uint8), scores)


def aupro(
    score_maps: Sequence[NDArray[np.floating]],
    masks: Sequence[NDArray[np.bool_]],
    *,
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """Integrate per-component overlap over false-positive rates up to ``max_fpr``.

    Ground-truth regions use 4-connected components. PRO is averaged equally over
    components, while FPR is computed over all ground-truth-negative pixels.
    """
    score_arrays, mask_arrays = _validate_maps(score_maps, masks)
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be in (0, 1]")
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    components = [
        (image_index, component)
        for image_index, mask in enumerate(mask_arrays)
        for component in _connected_components(mask)
    ]
    if not components:
        raise ValueError("AUPRO requires at least one anomalous ground-truth region")

    negative_pixels = sum(int((~mask).sum()) for mask in mask_arrays)
    if negative_pixels == 0:
        raise ValueError("AUPRO requires at least one ground-truth-negative pixel")

    minimum = min(float(score.min()) for score in score_arrays)
    maximum = max(float(score.max()) for score in score_arrays)
    epsilon = max(float(np.finfo(np.float64).eps), (maximum - minimum) * 1e-9)
    thresholds = np.linspace(maximum + epsilon, minimum - epsilon, num_thresholds)
    false_positive_rates: NDArray[np.float64] = np.empty(
        num_thresholds, dtype=np.float64
    )
    region_overlaps: NDArray[np.float64] = np.empty(num_thresholds, dtype=np.float64)

    flat_scores = [score.reshape(-1) for score in score_arrays]
    flat_masks = [mask.reshape(-1) for mask in mask_arrays]
    for threshold_index, threshold in enumerate(thresholds):
        predicted = [score >= threshold for score in flat_scores]
        false_positives = sum(
            int(np.count_nonzero(prediction & ~mask))
            for prediction, mask in zip(predicted, flat_masks, strict=True)
        )
        false_positive_rates[threshold_index] = false_positives / negative_pixels
        region_overlaps[threshold_index] = np.mean(
            [
                np.count_nonzero(predicted[image_index][component]) / component.size
                for image_index, component in components
            ]
        )

    last_inside = int(np.searchsorted(false_positive_rates, max_fpr, side="right"))
    if last_inside == 0:
        fpr_curve = np.array([0.0, max_fpr])
        pro_curve = np.array([region_overlaps[0], region_overlaps[0]])
    elif last_inside == num_thresholds:
        fpr_curve = false_positive_rates
        pro_curve = region_overlaps
    else:
        fpr_curve = false_positive_rates[:last_inside]
        pro_curve = region_overlaps[:last_inside]
        lower_fpr = fpr_curve[-1]
        upper_fpr = false_positive_rates[last_inside]
        fraction = (max_fpr - lower_fpr) / (upper_fpr - lower_fpr)
        crossing_pro = pro_curve[-1] + fraction * (
            region_overlaps[last_inside] - pro_curve[-1]
        )
        fpr_curve = np.append(fpr_curve, max_fpr)
        pro_curve = np.append(pro_curve, crossing_pro)

    return float(np.trapezoid(pro_curve, fpr_curve) / max_fpr)


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC using the Mann–Whitney rank-sum identity."""
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if label_array.shape != score_array.shape or not label_array.size:
        raise ValueError("labels and scores must have equal non-zero lengths")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must contain only finite values")
    if not np.isin(label_array, (0, 1, False, True)).all():
        raise ValueError("labels must contain only 0 and 1")

    binary_labels = label_array.astype(bool)
    positives = int(binary_labels.sum())
    negatives = int(binary_labels.size - positives)
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")

    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(score_array.size, dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop

    rank_sum = float(ranks[binary_labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _validate_maps(
    score_maps: Sequence[NDArray[np.floating]],
    masks: Sequence[NDArray[np.bool_]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.bool_]]]:
    if not score_maps or len(score_maps) != len(masks):
        raise ValueError("score_maps and masks must have equal non-zero lengths")
    scores_out: list[NDArray[np.float64]] = []
    masks_out: list[NDArray[np.bool_]] = []
    for score, mask in zip(score_maps, masks, strict=True):
        score_array = np.asarray(score, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=bool)
        if score_array.ndim != 2 or score_array.shape != mask_array.shape:
            raise ValueError("each score map and mask must share a 2D shape")
        if not np.isfinite(score_array).all():
            raise ValueError("score maps must contain only finite values")
        scores_out.append(score_array)
        masks_out.append(mask_array)
    return scores_out, masks_out


def _connected_components(mask: NDArray[np.bool_]) -> list[NDArray[np.intp]]:
    foreground = np.flatnonzero(mask)
    visited = np.zeros(mask.size, dtype=bool)
    height, width = mask.shape
    components: list[NDArray[np.intp]] = []
    for start in foreground:
        if visited[start]:
            continue
        visited[start] = True
        stack = [int(start)]
        component: list[int] = []
        while stack:
            index = stack.pop()
            component.append(index)
            row, column = divmod(index, width)
            neighbors = []
            if row:
                neighbors.append(index - width)
            if row + 1 < height:
                neighbors.append(index + width)
            if column:
                neighbors.append(index - 1)
            if column + 1 < width:
                neighbors.append(index + 1)
            for neighbor in neighbors:
                if mask.flat[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.intp))
    return components
