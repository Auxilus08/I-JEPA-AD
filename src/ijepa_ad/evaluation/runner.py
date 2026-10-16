"""Evaluation of trained predictors against the official MVTec test split."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.latent_cache import compute_file_sha256
from ijepa_ad.data.mvtec_dataset import validate_category_splits
from ijepa_ad.data.splits import split_nominal_train_val
from ijepa_ad.data.transforms import PreprocessConfig
from ijepa_ad.evaluation.metrics import (
    aupro,
    calibrate_image_threshold,
    image_auroc,
    pixel_auroc,
)
from ijepa_ad.inference import (
    InferenceMode,
    MaskedLatentScorer,
    MaskingSettings,
    load_masking_settings,
    load_predictor_checkpoint,
)
from ijepa_ad.predictor.cache_dataset import load_cached_latent_dataset

ENCODER_NAME = "ijepa_vit_g16_in22k_ep44"
PATCH_SIZE = 16
NUM_TOKENS = 196
EMBEDDING_DIM = 1408


def evaluate_category(
    *,
    category: str,
    dataset_root: str | Path,
    cache_root: str | Path,
    predictor_path: str | Path,
    encoder_sha256: str,
    device: str,
    mode: InferenceMode,
    masking_settings: MaskingSettings,
    target_fpr: float = 0.05,
    num_thresholds: int = 200,
) -> dict[str, Any]:
    """Score official test samples and compute image/pixel metrics."""
    train_dataset, test_dataset = validate_category_splits(dataset_root, category)
    nominal_split = split_nominal_train_val(train_dataset, val_fraction=0.1, seed=0)
    val_expected = CacheExpectations(
        category=category,
        split="val",
        encoder_name=ENCODER_NAME,
        checkpoint_sha256=encoder_sha256,
        patch_size=PATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        preprocessing=PreprocessingSpec.from_config(PreprocessConfig()),
        sample_ids=tuple(sample.image_id for sample in nominal_split.val),
        num_tokens=NUM_TOKENS,
    )
    cached_val = load_cached_latent_dataset(
        cache_root, val_expected, samples=train_dataset.samples
    )
    test_expected = CacheExpectations(
        category=category,
        split="test",
        encoder_name=ENCODER_NAME,
        checkpoint_sha256=encoder_sha256,
        patch_size=PATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        preprocessing=PreprocessingSpec.from_config(PreprocessConfig()),
        sample_ids=tuple(sample.image_id for sample in test_dataset.samples),
        num_tokens=NUM_TOKENS,
    )
    cached_test = load_cached_latent_dataset(
        cache_root, test_expected, samples=test_dataset.samples
    )
    predictor = load_predictor_checkpoint(
        predictor_path,
        category=category,
        encoder_sha256=encoder_sha256,
        device=device,
    )
    scorer = MaskedLatentScorer(predictor, **masking_settings)
    validation_scores = [
        scorer.score(sample.latent.unsqueeze(0).to(device), mode=mode).image_score
        for sample in cached_val
    ]
    decision_threshold, validation_fpr = calibrate_image_threshold(
        validation_scores, target_fpr=target_fpr
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    labels: list[int] = []
    image_scores: list[float] = []
    score_maps: list[np.ndarray[Any, Any]] = []
    masks: list[np.ndarray[Any, Any]] = []
    latencies_ms: list[float] = []
    evaluated_masks: list[int] = []
    for sample in cached_test:
        annotation = sample.annotation
        if annotation is None:
            raise ValueError(f"missing test annotation for {sample.image_id}")
        if annotation.mask_path is None:
            with Image.open(annotation.image_path) as image:
                width, height = image.size
            ground_truth = np.zeros((height, width), dtype=bool)
        else:
            with Image.open(annotation.mask_path) as mask_image:
                ground_truth = np.asarray(mask_image.convert("L")) > 0

        latents = sample.latent.unsqueeze(0).to(device)
        start_time = time.perf_counter()
        result = scorer.score(latents, mode=mode)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        latencies_ms.append((time.perf_counter() - start_time) * 1000.0)
        score_map = F.interpolate(
            result.patch_scores.reshape(1, 1, 14, 14),
            size=ground_truth.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        labels.append(int(annotation.label != "good"))
        image_scores.append(result.image_score)
        score_maps.append(score_map.detach().cpu().numpy().astype(np.float32))
        masks.append(ground_truth)
        evaluated_masks.append(result.evaluated_regions)

    if device.startswith("cuda"):
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_vram_mb = None
    return {
        "category": category,
        "mode": mode,
        "objective": "cosine",
        "num_images": len(labels),
        "num_defects": sum(labels),
        "decision_threshold": decision_threshold,
        "validation_target_fpr": target_fpr,
        "validation_observed_fpr": validation_fpr,
        "image_auroc": image_auroc(labels, image_scores),
        "pixel_auroc": pixel_auroc(score_maps, masks),
        "aupro": aupro(score_maps, masks, num_thresholds=num_thresholds),
        "mean_masks_per_image": float(np.mean(evaluated_masks)),
        "mean_predictor_latency_ms": float(np.mean(latencies_ms)),
        "p95_predictor_latency_ms": float(np.percentile(latencies_ms, 95)),
        "peak_vram_mb": peak_vram_mb,
        "latency_scope": "cached-latent predictor scoring; excludes image encoding",
    }


def evaluate_pilot(
    *,
    categories: tuple[str, ...],
    dataset_root: str | Path,
    cache_root: str | Path,
    checkpoint_dir: str | Path,
    encoder_checkpoint: str | Path,
    device: str,
    mode: InferenceMode,
    masking_config: str | Path,
    threshold: float | None = None,
    target_fpr: float = 0.05,
    num_thresholds: int = 200,
) -> dict[str, Any]:
    """Evaluate selected categories with one shared scoring configuration."""
    encoder_sha256 = compute_file_sha256(encoder_checkpoint)
    settings = load_masking_settings(masking_config)
    if threshold is not None:
        settings["threshold"] = threshold
    results = []
    for category in categories:
        results.append(
            evaluate_category(
                category=category,
                dataset_root=dataset_root,
                cache_root=cache_root,
                predictor_path=Path(checkpoint_dir) / f"{category}_cosine.pt",
                encoder_sha256=encoder_sha256,
                device=device,
                mode=mode,
                masking_settings=settings,
                target_fpr=target_fpr,
                num_thresholds=num_thresholds,
            )
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "encoder_sha256": encoder_sha256,
        "encoder_name": ENCODER_NAME,
        "device": device,
        "mode": mode,
        "masking_settings": settings,
        "results": results,
    }
