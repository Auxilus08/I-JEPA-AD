#!/usr/bin/env python3
"""Cache intermediate-layer patch features of the frozen I-JEPA target encoder.

    python scripts/build_layer_features.py --category bottle capsule hazelnut \
        --splits train val synthetic --device cuda

``train``/``val`` are the nominal split used everywhere else; ``synthetic`` is the
synthetic-anomaly validation set of plan item 12; ``test`` is the official test
split and is only encoded when asked for explicitly (plan item 8 chooses a layer on
validation data first and touches test once afterwards).

Features go to ``data/cache/layer_features/layerNN/<category>_<split>.pt`` (float16,
token layer norm, provenance-checked). Existing files are verified and reused.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ijepa_ad.caching.cache_schema import PreprocessingSpec  # noqa: E402
from ijepa_ad.caching.latent_cache import compute_file_sha256  # noqa: E402
from ijepa_ad.caching.synthetic_cache import load_synthetic_manifest  # noqa: E402
from ijepa_ad.data.mvtec_dataset import validate_category_splits  # noqa: E402
from ijepa_ad.data.splits import split_nominal_train_val  # noqa: E402
from ijepa_ad.data.synthetic_anomalies import SyntheticConfig  # noqa: E402
from ijepa_ad.data.transforms import PreprocessConfig, preprocess_image  # noqa: E402
from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder  # noqa: E402
from ijepa_ad.encoder.layer_features import (  # noqa: E402
    LayerFeatureExtractor,
    layer_feature_path,
    load_layer_features,
    save_layer_features,
)

DEFAULT_LAYERS = (8, 12, 16, 20, 24, 28, 32, 36, 40)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", dest="categories", nargs="+", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "synthetic"],
        choices=["train", "val", "synthetic", "test"],
    )
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cache-root", type=Path, default=PROJECT_ROOT / "data/cache/layer_features"
    )
    parser.add_argument(
        "--synthetic-root", type=Path, default=PROJECT_ROOT / "data/synthetic_val"
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
    preprocess = PreprocessConfig(image_size=int(data["preprocessing"]["image_size"]))
    spec = asdict(PreprocessingSpec.from_config(preprocess))
    sha = compute_file_sha256(args.checkpoint)
    synthetic_config = SyntheticConfig()

    extractor = None
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
        nominal = split_nominal_train_val(
            train_ds.samples,
            val_fraction=float(split_cfg["val_fraction"]),
            seed=int(split_cfg["seed"]),
            min_train=int(split_cfg["min_train"]),
            min_val=int(split_cfg["min_val"]),
        )
        records = {
            "train": nominal.train,
            "val": nominal.val,
            "synthetic": (
                load_synthetic_manifest(args.synthetic_root, synthetic_config, category)
                if "synthetic" in args.splits
                else ()
            ),
            "test": test_ds.samples if "test" in args.splits else (),
        }
        for split in args.splits:
            samples = records[split]
            ids = [s.image_id for s in samples]
            todo = [
                layer
                for layer in args.layers
                if not layer_feature_path(
                    args.cache_root, layer, category, f"{split}"
                ).exists()
            ]
            for layer in set(args.layers) - set(todo):  # verify what exists
                load_layer_features(
                    args.cache_root,
                    layer=layer,
                    category=category,
                    split=split,
                    sample_ids=ids,
                    checkpoint_sha256=sha,
                    preprocessing=spec,
                )
            if not todo:
                print(f"{category}/{split}: all layers present (verified)")
                continue
            if extractor is None:
                encoder = IJEPAEncoder(
                    args.checkpoint,
                    official_repo_path=args.official_repo,
                    device=torch.device(args.device),
                )
                extractor = LayerFeatureExtractor(encoder, args.layers)
            extractor = LayerFeatureExtractor(extractor.encoder, todo)
            started = time.perf_counter()
            chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in todo}
            for start in range(0, len(samples), args.batch_size):
                batch = samples[start : start + args.batch_size]
                images = torch.stack(
                    [preprocess_image(s.image_path, preprocess) for s in batch]
                )
                for layer, features in extractor.extract(
                    images.to(args.device)
                ).items():
                    chunks[layer].append(features.to(torch.float16))
            for layer in todo:
                save_layer_features(
                    args.cache_root,
                    layer=layer,
                    category=category,
                    split=split,
                    features=torch.cat(chunks[layer]),
                    sample_ids=ids,
                    checkpoint_sha256=sha,
                    preprocessing=spec,
                )
            print(
                f"{category}/{split}: {len(samples)} images, layers {todo}, "
                f"{time.perf_counter() - started:.0f}s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
