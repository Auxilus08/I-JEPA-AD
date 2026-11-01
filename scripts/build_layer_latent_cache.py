#!/usr/bin/env python3
"""Cache one encoder layer's tokens in the standard latent-cache format.

    python scripts/build_layer_latent_cache.py --layer 32 --category bottle capsule \
        hazelnut --device cuda

The result is an ordinary ``LatentCache`` (so ``ijepa_ad.predictor.training`` and
every loader work unchanged) with two differences that make it impossible to mix up
with the final-layer cache:

* ``encoder_name`` is ``<name>_layerNN``;
* ``checkpoint_sha256`` is a *derived identity*, the SHA-256 of the real checkpoint
  hash, the layer and the feature normalisation. A predictor trained on this cache
  records that identity, and ``load_predictor_checkpoint`` (which compares it with
  the hash the caller passes) therefore refuses to run it on final-layer latents.

Train/val go to ``data/cache/latents_layerNN`` and the official test split to
``data/cache/latents_layerNN_test``, mirroring the final-layer layout.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ijepa_ad.caching.builder import (  # noqa: E402
    build_latent_cache,
    build_nominal_latent_caches,
)
from ijepa_ad.caching.latent_cache import compute_file_sha256  # noqa: E402
from ijepa_ad.data.mvtec_dataset import validate_category_splits  # noqa: E402
from ijepa_ad.data.transforms import PreprocessConfig  # noqa: E402
from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder  # noqa: E402
from ijepa_ad.encoder.layer_features import LayerFeatureExtractor  # noqa: E402

ENCODER_NAME = "ijepa_vit_g16_in22k_ep44"
NORMALISATION = "token_layer_norm"


def derived_identity(checkpoint_sha256: str, layer: int) -> str:
    """SHA-256 identifying 'this checkpoint, this layer, this normalisation'."""
    text = f"{checkpoint_sha256}|layer={layer}|{NORMALISATION}|v1"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _LayerEncoder:
    """``LatentEncoder`` that returns one layer's normalised tokens."""

    def __init__(
        self, extractor: LayerFeatureExtractor, layer: int, device: str
    ) -> None:
        self._extractor = extractor
        self._layer = layer
        self._device = device

    def encode(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self._extractor.extract(image_batch.to(self._device))[self._layer]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--category", dest="categories", nargs="+", required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["nominal", "test"], choices=["nominal", "test"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
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
    preprocess = PreprocessConfig(image_size=int(data["preprocessing"]["image_size"]))
    real_sha = compute_file_sha256(args.checkpoint)
    identity = derived_identity(real_sha, args.layer)
    name = f"{ENCODER_NAME}_layer{args.layer:02d}"
    nominal_root = PROJECT_ROOT / f"data/cache/latents_layer{args.layer:02d}"
    test_root = PROJECT_ROOT / f"data/cache/latents_layer{args.layer:02d}_test"
    print(f"layer {args.layer}: encoder_name={name}, derived identity {identity}")

    inner = IJEPAEncoder(
        args.checkpoint,
        official_repo_path=args.official_repo,
        device=torch.device(args.device),
    )
    encoder = _LayerEncoder(
        LayerFeatureExtractor(inner, [args.layer]), args.layer, args.device
    )
    common = {
        "encoder_name": name,
        "checkpoint_sha256": identity,
        "patch_size": 16,
        "embedding_dim": 1408,
        "num_tokens": 196,
        "config": preprocess,
        "batch_size": args.batch_size,
    }
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
        train_ds, test_ds = validate_category_splits(root, category)
        if "nominal" in args.splits:
            reports = build_nominal_latent_caches(
                train_ds.samples,
                encoder,
                val_fraction=float(split_cfg["val_fraction"]),
                seed=int(split_cfg["seed"]),
                min_train=int(split_cfg["min_train"]),
                min_val=int(split_cfg["min_val"]),
                cache_root=nominal_root,
                category=category,
                **common,
            )
            print(category, {k: v.encoded for k, v in reports.items()})
        if "test" in args.splits:
            report = build_latent_cache(
                test_ds.samples,
                encoder,
                cache_root=test_root,
                category=category,
                split="test",
                **common,
            )
            print(category, "test", report.encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
