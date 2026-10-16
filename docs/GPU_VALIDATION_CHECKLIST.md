# GPU validation checklist and pilot record

The pilot run below was completed on 2026-10-15. The unchecked boxes that follow are a
repeat-validation checklist; they are not evidence that the recorded pilot was skipped.
For procedure details, see [`ENCODER_VITG16_CUDA_RUNBOOK.md`](ENCODER_VITG16_CUDA_RUNBOOK.md).

Run from the repository root. Linux paths shown; on Windows use `.venv\Scripts\python.exe`.
Stop at the first failed step, record the exact error under §F, and report it; do not
patch code on the GPU host without recording the change.

## Verified pilot run

| Item | Recorded result |
| --- | --- |
| Python / PyTorch / CUDA runtime | Python 3.12.4 / 2.2.2+cu121 / CUDA available |
| GPU | NVIDIA RTX 4090, 24 GB |
| NumPy / PyArrow | 1.26.4 / 18.1.0; `pip check` clean |
| Meta I-JEPA source | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` |
| Official checkpoint | 16,534,564,553 bytes |
| Checkpoint SHA-256 | `a373c7b16165092e29bd5714c6ce79c574c4a52b01b64bf4c82dc81e6a82cf0b` |
| ViT-g/16 output | finite float32 `(N, 196, 1408)` |
| FP16 spike peak allocation | batch 1: 1.914 GB; 4: 1.943 GB; 8: 1.980 GB; 16: 2.058 GB |
| Real cache entries | 9 entries; 819 nominal train/validation images and 325 test images; zero failures |
| Predictor runs | 6 (3 categories × cosine/L2), 100 epochs each |
| Validation-only extensions | Bottle reached 300 cap; Capsule stopped at 215; Hazelnut stopped at 133/145; no extension test evaluation |
| Test evaluation | Adaptive/fixed/dense I-JEPA on Bottle, Capsule, and Hazelnut; 100 AUPRO thresholds |

See `REPRODUCIBILITY.md` for per-category test metrics, predictor validation results,
threshold calibration, cache integrity checks, and latency scope. The available GPU
does not establish compliance with the separate 8 GB target.

## A. Environment

- [ ] `git rev-parse HEAD` → record. `git status --short` is empty (no local edits).
- [ ] `python --version` (CI uses 3.11; local CPU runs used 3.12.10). Record.
- [ ] `pip install -e ".[dev]"` then
      `python -c "import torch, PIL; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), PIL.__version__)"`.
      `is_available()` must be `True`.
- [ ] `nvidia-smi` → record driver version, GPU name, total memory.
- [ ] CPU gate on this host: `pytest tests/unit tests/integration -q`, `ruff check .`,
      `black --check --workers 1 .`, `mypy src tests` all pass. Record counts. A failure
      here is an environment difference (e.g. Python 3.11 vs 3.12); investigate before
      going further.
- [ ] Paths exist: `data/raw/mvtec_ad`, `checkpoints/IN22K-vit.g.16-600e.pth.tar`,
      `external/ijepa` checked out at `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
      (`git -C external/ijepa rev-parse HEAD`).

## B. Checkpoint (do not run the model yet)

- [ ] Source: the URL in `REPRODUCIBILITY.md` § Official checkpoint
      (`https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar`).
- [ ] File exists; record size: `ls -l checkpoints/IN22K-vit.g.16-600e.pth.tar`.
- [ ] `sha256sum checkpoints/IN22K-vit.g.16-600e.pth.tar` → record. This is the value
      every cache entry will carry as `checkpoint_sha256`. No reference hash is recorded in
      the repository, so this first measurement becomes the project reference. Cross-check it equals
      `compute_file_sha256(...)` from `ijepa_ad.caching.latent_cache`.
- [ ] Key check (runbook §3): `target_encoder` is present. Record the number of
      parameters and whether keys start with `module.`.
- [ ] Confirm the official state dictionary uses a uniform `module.` prefix. The project
      loader strips it only when every key has it and rejects mixed-prefix dictionaries;
      the actual checkpoint loaded strictly with this normalization.

## C. Encoder smoke test

- [ ] `pytest -q tests/unit/test_ijepa_wrapper.py` → `2 passed` (not skipped).
- [ ] VRAM table per runbook §5 (`python scripts/spikes/check_encoder_vram.py`).
- [ ] Real wrapper, one tiny batch:

```python
import torch
from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder
from ijepa_ad.utils.profiling import profile_peak_vram

enc = IJEPAEncoder("checkpoints/IN22K-vit.g.16-600e.pth.tar",
                   official_repo_path="external/ijepa", device="cuda")
for n in (1, 4, 8):                      # add/remove sizes to taste
    x = torch.randn(n, 3, 224, 224, device="cuda")
    with profile_peak_vram(f"bs{n}") as peak:
        z = enc.encode(x)
    print(n, tuple(z.shape), z.dtype, bool(torch.isfinite(z).all()), f"{peak.allocated_mb:.0f} MB")
```

- [ ] Shape is `(N, 196, 1408)`, all values finite, for every N tried. Record dtype
      (the wrapper does not autocast; the model dtype decides).
- [ ] Record peak memory per batch size, including any OOM. Per
      `DECISIONS.md`, if batch 4 or 8 exceeds ~6.5 GB, report that ViT-g/16 leaves no
      headroom for the predictor.
- [ ] Same input twice gives the same output on this GPU? Record max abs difference
      (do not assume zero).

## D. Real dataset smoke test (CPU is fine)

- [ ] `python validate_mvtec.py data/raw/mvtec_ad` → record result per category.
- [ ] For the pilot categories (`bottle capsule hazelnut`, `configs/data/mvtec.yaml`):
      `validate_category_splits(root, category)` succeeds. A
      `DatasetValidationError` mentioning `duplicate image ID` means the category has
      byte-identical images in `train/good`; that is rejected by design, not deduplicated.
      Record which categories, how many ids, and report; do not work around it silently.
- [ ] Preprocess a few real images with `preprocess_image(path)`: shape
      `(3, 224, 224)`, float32, finite. Include one non-RGB image if any exist
      (grayscale categories such as `grid`, `screw`, `zipper` are expected to be `L`).
- [ ] Mock dry run on real data (no checkpoint, no GPU), which exercises dataset,
      split and cache writing: `python -m ijepa_ad.caching.cli --category bottle --mock-encoder`.
      Writes under `data/cache/latents/_mock`. Expect exit code 0.
- [ ] Split sanity for one category: train/val sizes are `n - round(n*0.1)` / `round(n*0.1)`;
      train ∩ val is empty; no `test/` path or non-`good` label appears in either; train
      and val ids are absent from the official test ids.

## E. Cache generation

Pilot first: one category, small batch.

- [ ] Pilot: `python -m ijepa_ad.caching.cli --category bottle --device cuda --batch-size 4 --summary-json results/pilot_nominal.json`.
      The summary says `encoder_kind: "real"`, `real_checkpoint_inference: true`, the
      checkpoint hash from §B, and `failed: 0`. Record totals.
- [ ] Reload both entries with the consumer API and check shape/dtype/metadata:

```python
from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.latent_cache import latent_cache_path, load_latent_cache
from ijepa_ad.data.transforms import PreprocessConfig

CKPT_SHA = "<from §B>"
for split in ("train", "val"):
    exp = CacheExpectations(
        category="bottle", split=split, encoder_name="ijepa_vit_g16_in22k_ep44",
        checkpoint_sha256=CKPT_SHA, patch_size=16, embedding_dim=1408,
        preprocessing=PreprocessingSpec.from_config(PreprocessConfig()))
    c = load_latent_cache(latent_cache_path("data/cache/latents", "bottle", split), exp)
    print(split, tuple(c.latents.shape), c.latents.dtype, c.metadata.latents_sha256[:12])
```

- [ ] Resume: rerun the identical command; summary shows `encoded_new: 0`,
      `skipped_compatible` = all samples.
- [ ] Interrupt and resume: `--rebuild`, start the command, kill it (Ctrl-C) part way,
      rerun without `--rebuild`. Record that finished shards were skipped
      (`skipped_compatible > 0`) and the final entries load. If the kill landed mid-write,
      record whether `recomputed > 0`.
- [ ] Batch-size agreement: build the same category with `--batch-size 1` and
      `--batch-size 8` into separate `--cache-root`s, load both and record
      `(a - b).abs().max()` per split. Do not assume a tolerance; record what you see
      (the CPU tests only show `atol=1e-4` for the mock).
- [ ] Float16 vs float32: there is no storage-dtype option in the CLI; latents are saved
      in the dtype the encoder returns. Using the float32 pilot cache, record
      `c.latents.half().float()` vs `c.latents` (max abs error, mean cosine similarity per
      token) and size (`numel * 2` vs `* 4` bytes; ~0.55 vs ~1.1 MB per sample). Choose
      the production dtype from these numbers and write the decision in `DECISIONS.md`.
      Implementing a dtype option is a code change; do not do it as part of validation.
- [ ] Only after the pilot passes: full run for the pilot categories, then `--purpose test`.
      Same checks (`failed: 0`, reload, resume). Host RAM during final assembly is
      about one whole (category, split) tensor.
- [ ] Confirm `git status --short` is still empty (`data/`, `checkpoints/`, `external/ijepa`
      and `results/*` are gitignored; no cache or checkpoint may be committed).

## F. Results to record (fill in; leave "not run" if skipped)

| Item | Value |
| --- | --- |
| Date, operator | |
| Machine, OS | |
| GPU, driver, total memory | |
| Repository commit, working tree clean? | |
| Python / PyTorch / CUDA / Pillow versions | |
| `external/ijepa` commit | |
| Checkpoint size and SHA-256 | |
| Checkpoint load worked as-is? (error text if not) | |
| Peak GPU memory per batch size (1 / 4 / 8 / 16) | |
| Output shape, dtype, finite? | |
| Categories validated; duplicate-image errors | |
| Pilot samples: discovered / encoded / skipped / recomputed / failed | |
| Batch-size 1 vs 8 max abs difference | |
| Float16 vs float32: max abs error, cosine, size | |
| Chosen storage dtype and total cache size | |
| CPU test counts on this host (pytest / ruff / black / mypy) | |
| Known limitations or surprises | |

Status: the three-category RTX 4090 pilot is recorded above. Re-run the unchecked
procedure when validating a different checkout, environment, GPU, or cache.
