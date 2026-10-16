# ViT-g/16 CUDA Reproduction Runbook

This is the reproducible procedure for CUDA-dependent encoder validation. The pilot
run was completed on Windows/Python 3.12.4 with PyTorch 2.2.2+cu121 on an RTX 4090;
recorded versions, checkpoint identity, and measured memory are in
[`GPU_VALIDATION_CHECKLIST.md`](GPU_VALIDATION_CHECKLIST.md). Re-run it when changing
the environment, checkpoint, or target GPU.

## 1. Create the environment

Use Python 3.11 on Linux for the most reproducible setup:

```bash
python3.11 -m venv .venv-vitg16
source .venv-vitg16/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch CUDA build selected by the official installer for the host's
driver. Do not use a CPU-only build. The selector is maintained at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

For example, if the selector recommends CUDA 12.8:

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

Record the actual versions before continuing:

```bash
nvidia-smi
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY
```

## 2. Restore the pinned Meta source

```bash
mkdir -p external
if [ ! -d external/ijepa/.git ]; then
  git clone https://github.com/facebookresearch/ijepa.git external/ijepa
fi
git -C external/ijepa checkout 52c1ae95d05f743e000e8f10a1f3a79b10cff048
test "$(git -C external/ijepa rev-parse HEAD)" = \
  52c1ae95d05f743e000e8f10a1f3a79b10cff048
```

## 3. Restore and verify the checkpoint

```bash
mkdir -p checkpoints
if [ ! -f checkpoints/IN22K-vit.g.16-600e.pth.tar ]; then
  curl -L \
    -o checkpoints/IN22K-vit.g.16-600e.pth.tar \
    https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar
fi
sha256sum checkpoints/IN22K-vit.g.16-600e.pth.tar
```

Confirm that the exact `target_encoder` key exists:

```bash
python - <<'PY'
from pathlib import Path
import torch

path = Path("checkpoints/IN22K-vit.g.16-600e.pth.tar")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
print("top-level keys:", sorted(checkpoint))
assert "target_encoder" in checkpoint
print("target_encoder parameters:", len(checkpoint["target_encoder"]))
PY
```

## 4. Run the non-skipped unit tests

These tests use a small injected model factory, so they do not allocate the full
ViT-g model. They must execute, not skip:

```bash
pytest -q tests/unit/test_ijepa_wrapper.py
```

Expected result:

```text
2 passed
```

If the output says `skipped`, stop and fix the PyTorch installation; do not record
the tests as passing.

## 5. Run the actual ViT-g/16 VRAM spike

```bash
python scripts/spikes/check_encoder_vram.py
```

The script loads `target_encoder`, constructs Meta's `vit_giant` architecture, and
measures FP16 forward-pass peak allocation at batch sizes 1, 4, 8, and 16.
Record output in this table:

| Batch size | Peak VRAM (GB) | Result |
|---:|---:|---|
| 1 | 1.914 | pass |
| 4 | 1.943 | pass |
| 8 | 1.980 | pass |
| 16 | 2.058 | pass |

If batch 4 or 8 exceeds approximately 6.5 GB, report that ViT-g/16 leaves
insufficient headroom for the predictor and Stage-2 fine masking. Do not proceed to
latent caching without deciding whether to fall back to ViT-H/14 or reduce the
runtime batch size.

## 6. Run the real wrapper smoke test

Only after the spike completes successfully:

```bash
python - <<'PY'
from pathlib import Path
import torch

from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder

encoder = IJEPAEncoder(
    Path("checkpoints/IN22K-vit.g.16-600e.pth.tar"),
    official_repo_path=Path("external/ijepa"),
    device="cuda",
)
images = torch.randn(1, 3, 224, 224, device="cuda")
latents = encoder.encode(images)
print("latents:", tuple(latents.shape))
assert tuple(latents.shape) == (1, 196, 1408)
assert not any(parameter.requires_grad for parameter in encoder.parameters())
PY
```

The expected contract is `(1, 196, 1408)`. Save the command output and the VRAM
table alongside the exact PyTorch/CUDA versions in the reproducibility record.
