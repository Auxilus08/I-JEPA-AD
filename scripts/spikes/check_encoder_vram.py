#!/usr/bin/env python3
"""Throwaway ViT-g/16 CUDA memory spike; intentionally outside the package/CI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_IJEPA_ROOT = REPO_ROOT / "external" / "ijepa"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "IN22K-vit.g.16-600e.pth.tar"
IMAGE_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZES = (1, 4, 8, 16)


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is required to run this CUDA spike.", file=sys.stderr)
        return 2

    from ijepa_ad.encoder.ijepa_wrapper import _extract_target_encoder

    if not torch.cuda.is_available():
        print("CUDA is not available; no VRAM measurements were produced.")
        return 2
    if not OFFICIAL_IJEPA_ROOT.is_dir():
        print(
            "Missing external/ijepa. Clone "
            "https://github.com/facebookresearch/ijepa there first.",
            file=sys.stderr,
        )
        return 2
    if not CHECKPOINT_PATH.is_file():
        print(f"Missing checkpoint: {CHECKPOINT_PATH}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(OFFICIAL_IJEPA_ROOT))
    try:
        from src.models.vision_transformer import vit_giant
    finally:
        sys.path.remove(str(OFFICIAL_IJEPA_ROOT))

    checkpoint = _load_checkpoint(torch, CHECKPOINT_PATH)
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"checkpoint top-level keys: {sorted(checkpoint)}")
    if "target_encoder" not in checkpoint:
        print(
            "ERROR: exact checkpoint key 'target_encoder' is absent. "
            f"Actual top-level keys: {sorted(checkpoint)}",
            file=sys.stderr,
        )
        return 3

    model = vit_giant(patch_size=PATCH_SIZE, img_size=[IMAGE_SIZE])
    model.load_state_dict(_extract_target_encoder(checkpoint), strict=True)
    model = model.cuda().eval().half()
    print("loaded exact checkpoint key: target_encoder")
    print("forward peak memory is reported in GB using 2^30 bytes")

    for batch_size in BATCH_SIZES:
        try:
            peak_gb, output_shape = _measure_batch(torch, model, batch_size)
        except torch.cuda.OutOfMemoryError:
            print(f"batch_size={batch_size}: OOM")
            _recover_after_oom(torch)
            continue
        print(
            f"batch_size={batch_size}: peak_vram_gb={peak_gb:.3f}, "
            f"output_shape={output_shape}"
        )
    return 0


def _load_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 does not accept weights_only.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict, received {type(checkpoint)!r}")
    return checkpoint


def _measure_batch(
    torch: Any, model: Any, batch_size: int
) -> tuple[float, tuple[int, ...]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    images = torch.randn(
        batch_size,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device="cuda",
        dtype=torch.float16,
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(images)
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    output_shape = tuple(output.shape)
    del images, output
    return peak_gb, output_shape


def _recover_after_oom(torch: Any) -> None:
    """Release failed-attempt allocations so the next batch can be measured."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(main())
