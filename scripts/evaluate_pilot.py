"""Evaluate trained pilot predictors on official MVTec test splits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ijepa_ad.evaluation.runner import evaluate_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", nargs="+", default=["bottle", "capsule", "hazelnut"]
    )
    parser.add_argument(
        "--mode", choices=("adaptive", "dense", "fixed"), default="adaptive"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/latents"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/predictors")
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=Path("checkpoints/IN22K-vit.g.16-600e.pth.tar"),
    )
    parser.add_argument(
        "--masking-config",
        type=Path,
        default=Path("configs/masking/coarse_to_fine.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--masking-threshold",
        type=float,
        help=(
            "non-negative threshold that gates adaptive fine masking; "
            "not the validation-calibrated image decision threshold"
        ),
    )
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--num-thresholds", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.masking_threshold is not None and (
        not math.isfinite(args.masking_threshold) or args.masking_threshold < 0
    ):
        parser.error("--masking-threshold must be finite and non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is unavailable")
    summary = evaluate_pilot(
        categories=tuple(args.category),
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        checkpoint_dir=args.checkpoint_dir,
        encoder_checkpoint=args.encoder_checkpoint,
        device=args.device,
        mode=args.mode,
        masking_config=args.masking_config,
        masking_threshold=args.masking_threshold,
        target_fpr=args.target_fpr,
        num_thresholds=args.num_thresholds,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
