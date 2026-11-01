"""Re-score the paper's example images with the validation-calibrated gate.

Reproduces the setup of notebook 04 (baseline predictors, cached test latents,
tau = q0.99 of nominal-validation region scores read from
results/calibration/calibration_summary.json, lambda = 1) and:

* checks the aggregate pixel metrics against results/calibration/pixel_metrics.csv,
* reports per-image numbers for the three example images in
  results/report_figures/heatmap_examples.json,
* saves the raw maps to --out (npz) for plotting.

Run from the repository root:
    .venv/Scripts/python.exe paper/calibrated_heatmaps.py --out <maps.npz>
Test labels are used only for metrics, never to choose tau.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec  # noqa: E402
from ijepa_ad.data.mvtec_dataset import validate_category_splits  # noqa: E402
from ijepa_ad.data.transforms import PreprocessConfig  # noqa: E402
from ijepa_ad.evaluation.metrics import aupro, pixel_auroc  # noqa: E402
from ijepa_ad.evaluation.runner import (  # noqa: E402
    EMBEDDING_DIM,
    ENCODER_NAME,
    NUM_TOKENS,
    PATCH_SIZE,
)
from ijepa_ad.inference import (  # noqa: E402
    MaskedLatentScorer,
    load_masking_settings,
    load_predictor_checkpoint_with_provenance,
)
from ijepa_ad.masking.score_maps import upsample_patch_scores  # noqa: E402
from ijepa_ad.predictor.cache_dataset import load_cached_latent_dataset  # noqa: E402

CATEGORIES = ("bottle", "capsule", "hazelnut")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ground_truth(annotation) -> np.ndarray:
    if annotation.mask_path is None:
        with Image.open(annotation.image_path) as image:
            width, height = image.size
        return np.zeros((height, width), dtype=bool)
    with Image.open(annotation.mask_path) as mask:
        return np.asarray(mask.convert("L")) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    summary = json.loads(
        (ROOT / "results/calibration/calibration_summary.json").read_text()
    )
    encoder_sha = summary["encoder_sha256"]
    settings = load_masking_settings(ROOT / "configs/masking/coarse_to_fine.yaml")
    data_cfg = yaml.safe_load((ROOT / "configs/data/mvtec.yaml").read_text())
    spec = PreprocessingSpec.from_config(
        PreprocessConfig(image_size=int(data_cfg["preprocessing"]["image_size"]))
    )
    examples = json.loads(
        (ROOT / "results/report_figures/heatmap_examples.json").read_text()
    )
    wanted = {Path(e["image"]).resolve(): e for e in examples}

    saved: dict[str, np.ndarray] = {}
    for category in CATEGORIES:
        root = next(
            r
            for r in (ROOT / data_cfg["root"], ROOT / "dataset")
            if (r / category / "train").is_dir()
        )
        _, test_ds = validate_category_splits(root, category)
        samples = test_ds.samples
        cache = load_cached_latent_dataset(
            ROOT / "data/cache/latents_test",
            CacheExpectations(
                category=category,
                split="test",
                encoder_name=ENCODER_NAME,
                checkpoint_sha256=encoder_sha,
                patch_size=PATCH_SIZE,
                embedding_dim=EMBEDDING_DIM,
                preprocessing=spec,
                sample_ids=tuple(s.image_id for s in samples),
                num_tokens=NUM_TOKENS,
            ),
            samples=samples,
        )
        predictor, _ = load_predictor_checkpoint_with_provenance(
            ROOT / f"checkpoints/predictors/{category}_cosine.pt",
            category=category,
            encoder_sha256=encoder_sha,
            device=DEVICE,
        )
        tau_cal = float(summary["calibrated_tau"][category]["0.99"])
        taus = {"default": settings["threshold"], "calibrated": tau_cal}
        print(f"\n== {category}: tau default={taus['default']}, calibrated={tau_cal:.4f}")

        patch = {k: [] for k in ("default", "calibrated")}
        work = {k: [] for k in ("default", "calibrated")}
        masks = []
        for sample in cache:
            masks.append(ground_truth(sample.annotation))
        for key, tau in taus.items():
            scorer = MaskedLatentScorer(predictor, **{**settings, "threshold": tau})
            for index, sample in enumerate(cache):
                result = scorer.score(
                    sample.latent.unsqueeze(0).to(DEVICE), mode="adaptive"
                )
                patch[key].append(result.patch_scores.detach().cpu())
                work[key].append(result.evaluated_regions)
                path = Path(sample.annotation.image_path).resolve()
                if path in wanted:
                    tag = f"{category}/{wanted[path]['role']}"
                    saved[f"{tag}/{key}/coarse"] = result.coarse_scores.cpu().numpy()
                    saved[f"{tag}/{key}/fine"] = result.fine_scores.cpu().numpy()
                    saved[f"{tag}/{key}/fused"] = result.patch_scores.cpu().numpy()
                    saved[f"{tag}/path"] = np.array(str(path))
                    saved[f"{tag}/mask"] = masks[index]

        if not args.skip_aggregate:
            for key in taus:
                maps = [
                    upsample_patch_scores(p, output_size=m.shape)[0, 0].numpy()
                    for p, m in zip(patch[key], masks)
                ]
                print(
                    f"  {key:10s} pixel AUROC={pixel_auroc(maps, masks):.4f} "
                    f"AUPRO={aupro(maps, masks, num_thresholds=200):.4f} "
                    f"masks/img={np.mean(work[key]):.1f}"
                )
        # per-example numbers
        for index, sample in enumerate(cache):
            path = Path(sample.annotation.image_path).resolve()
            if path not in wanted or not masks[index].any():
                continue
            row = [f"  {wanted[path]['role'][:22]:22s}"]
            for key in taus:
                amap = upsample_patch_scores(
                    patch[key][index], output_size=masks[index].shape
                )[0, 0].numpy()
                row.append(
                    f"{key}: pxAUROC={pixel_auroc([amap], [masks[index]]):.3f} "
                    f"masks={work[key][index]}"
                )
            print(" | ".join(row))

    np.savez(args.out, **saved)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
