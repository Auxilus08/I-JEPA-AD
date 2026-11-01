#!/usr/bin/env python3
"""Generate the synthetic-anomaly validation set and its latent cache (plan item 12).

Base images are the nominal *validation* images (same split as training), textures
come from nominal *training* images; the official test split is never read.

    python scripts/build_synthetic_validation.py --category bottle capsule hazelnut \
        --device cuda

Reruns are idempotent: identical files are accepted, different ones refuse to be
replaced, and an existing cache is reused after its provenance is verified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ijepa_ad.caching.latent_cache import compute_file_sha256  # noqa: E402
from ijepa_ad.caching.synthetic_cache import (  # noqa: E402
    encode_synthetic_set,
    load_synthetic_cache,
    save_synthetic_cache,
    synthetic_cache_path,
    write_synthetic_set,
)
from ijepa_ad.data.mvtec_dataset import validate_category_splits  # noqa: E402
from ijepa_ad.data.splits import split_nominal_train_val  # noqa: E402
from ijepa_ad.data.synthetic_anomalies import SyntheticConfig  # noqa: E402
from ijepa_ad.data.transforms import PreprocessConfig  # noqa: E402

ENCODER_NAME = "ijepa_vit_g16_in22k_ep44"


class _OnDevice:
    """Moves each batch to the encoder's device; the encoder returns latents."""

    def __init__(self, encoder: Any, device: torch.device) -> None:
        self._encoder = encoder
        self._device = device

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        return self._encoder.encode(batch.to(self._device))  # type: ignore[no-any-return]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", dest="categories", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anomalies-per-image", type=int, default=4)
    parser.add_argument(
        "--image-root", type=Path, default=PROJECT_ROOT / "data/synthetic_val"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=PROJECT_ROOT / "data/cache/synthetic_val"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/IN22K-vit.g.16-600e.pth.tar",
    )
    parser.add_argument(
        "--official-repo", type=Path, default=PROJECT_ROOT / "external/ijepa"
    )
    args = parser.parse_args()

    data = yaml.safe_load(
        (PROJECT_ROOT / "configs/data/mvtec.yaml").read_text(encoding="utf-8")
    )
    split_cfg = data["nominal_split"]
    config = SyntheticConfig(
        seed=args.seed, anomalies_per_image=args.anomalies_per_image
    )
    preprocess = PreprocessConfig(image_size=int(data["preprocessing"]["image_size"]))
    print(f"synthetic fingerprint {config.fingerprint()}")

    encoder = None
    sha = None
    summary: dict[str, dict[str, object]] = {}
    for category in args.categories:
        root = next(
            r
            for r in (
                PROJECT_ROOT / data["root"],
                PROJECT_ROOT / "dataset",
                PROJECT_ROOT / "dataset" / category,
            )
            if (r / category / "train").is_dir()
        )
        train, _ = validate_category_splits(
            root, category
        )  # the test split is not used
        nominal = split_nominal_train_val(
            train.samples,
            val_fraction=float(split_cfg["val_fraction"]),
            seed=int(split_cfg["seed"]),
            min_train=int(split_cfg["min_train"]),
            min_val=int(split_cfg["min_val"]),
        )
        samples = write_synthetic_set(
            nominal.val,
            nominal.train,
            category=category,
            config=config,
            root=args.image_root,
        )
        cache_path = synthetic_cache_path(args.cache_root, config, category)
        if sha is None:
            sha = compute_file_sha256(args.checkpoint)
        if cache_path.exists():
            load_synthetic_cache(
                cache_path,
                samples,
                config=config,
                encoder_name=ENCODER_NAME,
                checkpoint_sha256=sha,
                preprocess=preprocess,
            )
            status = "reused (provenance verified)"
        else:
            if encoder is None:
                from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder

                device = torch.device(args.device)
                inner = IJEPAEncoder(
                    args.checkpoint,
                    official_repo_path=args.official_repo,
                    device=device,
                )

                encoder = _OnDevice(inner, device)
            latents = encode_synthetic_set(
                samples, encoder, preprocess=preprocess, batch_size=args.batch_size
            )
            save_synthetic_cache(
                cache_path,
                latents,
                samples,
                config=config,
                encoder_name=ENCODER_NAME,
                checkpoint_sha256=sha,
                preprocess=preprocess,
            )
            status = "encoded"
        summary[category] = {
            "base_validation_images": len(nominal.val),
            "synthetic_images": len(samples),
            "cache": str(cache_path),
            "status": status,
        }
        print(category, summary[category])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
