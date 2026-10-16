# REPRODUCIBILITY.md

## Latest verified run

Run date: **2026-10-15**. This is the first real-model pilot record for the three
categories available in this checkout. The dataset tree contains Bottle, Capsule, and
Hazelnut only; the 15-category table from earlier snapshots is not evidence that the
other categories are staged here.

Environment: Windows, Python 3.12.4, PyTorch 2.2.2+cu121, CUDA available, NVIDIA RTX
4090 (24 GB), NumPy 1.26.4, and PyArrow 18.1.0. `pip check` passed. The frontend extra
constrains PyArrow to `<19` because the current PyTorch/NumPy 1.x environment cannot use
newer PyArrow wheels that require NumPy 2.x. The latest focused inference/evaluation/
frontend/encoder/seeding test command passed **30 tests**; Ruff and mypy passed for the
recently changed Python files. The full test suite has not been rerun for this run.

| Category | Train | Test | Nominal test | Defect test / masks |
| --- | ---: | ---: | ---: | ---: |
| bottle | 209 | 83 | 20 | 63 |
| capsule | 219 | 132 | 23 | 109 |
| hazelnut | 391 | 110 | 40 | 70 |

The real ViT-g/16 encoder returned finite float32 latents of shape `(N, 196, 1408)`.
The official I-JEPA checkout and checkpoint identity are recorded below. Nine real
latent caches were built (train/validation/test for each category); all entries were
finite, had the expected shape and sample order, and passed the recorded integrity/
split checks. Six predictors (three categories × cosine/L2 objectives) were trained
for 100 epochs, then extended using nominal validation loss only. The extension models
are stored separately under `results/predictor_extension_200/` and
`results/predictor_extension_300/`; the official-test metrics below still use the original
100-epoch checkpoints.

| Category | Objective | 100-epoch best loss | Extended best loss (epoch) |
| --- | --- | ---: | ---: |
| bottle | cosine | 0.051810 (99) | 0.033670 (290; reached 300-epoch cap) |
| bottle | L2 | 0.089058 (98) | 0.055142 (290; reached 300-epoch cap) |
| capsule | cosine | 0.105533 (100) | 0.095222 (190; early-stopped at 215) |
| capsule | L2 | 0.179605 (100) | 0.160209 (190; early-stopped at 215) |
| hazelnut | cosine | 0.182006 (100) | 0.180196 (108; early-stopped at 133) |
| hazelnut | L2 | 0.278700 (99) | 0.276434 (120; early-stopped at 145) |

The Bottle runs were still improving at epoch 290, so their convergence is not
established. Capsule early-stopped after a validation plateau; Hazelnut's additional
validation-loss reduction was below 1.1%. None of these extension checkpoints has been
evaluated on official test data.

The official test split was evaluated with adaptive, fixed-mask, and dense-mask I-JEPA.
Image AUROC / pixel AUROC / AUPRO:

| Category | Adaptive | Fixed | Dense |
| --- | --- | --- | --- |
| bottle | 0.9349 / 0.8900 / 0.6638 | 0.9349 / 0.8904 / 0.6656 | 0.8246 / 0.8843 / 0.6546 |
| capsule | 0.4986 / 0.6939 / 0.0959 | 0.4986 / 0.6939 / 0.0959 | 0.5325 / 0.7962 / 0.2372 |
| hazelnut | 0.6536 / 0.8774 / 0.4876 | 0.6179 / 0.8746 / 0.4880 | 0.5421 / 0.7703 / 0.3681 |

AUPRO used 100 thresholds. Adaptive decision thresholds were calibrated on nominal
validation scores, never test labels. Benchmark latency uses cached latents and excludes
image encoding; the separate frontend inference measurement includes the resident
encoder. PatchCore and MAE are not yet valid/implemented benchmarks. These figures
support pilot debugging only, not full MVTec claims.

## Official I-JEPA source

- Repository: https://github.com/facebookresearch/ijepa
- Local checkout: `external/ijepa`
- Exact commit: `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
- Commit date: 2023-06-13
- Working tree: clean, detached at the recorded commit

The repository is archived/read-only. The commit SHA above pins the exact source
used by `scripts/spikes/check_encoder_vram.py` and the production encoder wrapper.

## Image preprocessing

`src/ijepa_ad/data/transforms.py` matches `src/transforms.py` at the commit above:
`ToTensor()` scaling to `[0, 1]`, then `Normalize` with mean
`(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`, at 224px with bilinear
interpolation. The official random crop is replaced by a deterministic full-image
resize (see `DECISIONS.md`, 2026-10-10). Requires Pillow (`pillow==12.3.0` in
`requirements-lock.txt`).

## Official checkpoint

- URL: https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar
- Local path: `checkpoints/IN22K-vit.g.16-600e.pth.tar`
- Size: 16,534,564,553 bytes
- SHA-256: `a373c7b16165092e29bd5714c6ce79c574c4a52b01b64bf4c82dc81e6a82cf0b`
- Required state-dict key: `target_encoder`
- Architecture: ViT-g/16, 224px input, 196 patch tokens, 1408-dimensional output

The official checkpoint stores uniformly `module.`-prefixed target-encoder keys. The
project loader strips that prefix only when every key has it and rejects mixed-prefix
state dictionaries before strict architecture loading.

## VRAM spike

Run from the repository root on a CUDA host:

```bash
python scripts/spikes/check_encoder_vram.py
```

The script is intentionally a throwaway diagnostic and is not part of the tested
package or CI. It measures forward-pass `torch.cuda.max_memory_allocated()` at batch
sizes 1, 4, 8, and 16 using FP16, `torch.no_grad()`, and CUDA autocast.

For the complete RTX 4090 setup, installation, checkpoint verification, non-skipped
unit-test check, VRAM table, and real-wrapper smoke test, follow
[`docs/ENCODER_VITG16_CUDA_RUNBOOK.md`](docs/ENCODER_VITG16_CUDA_RUNBOOK.md).

## Latent cache generation

A cache entry is reusable only when schema version (2), category, split, ordered
sample ids (SHA-256 of image content), encoder name, patch size, embedding dim,
checkpoint SHA-256, and the full preprocessing spec (image size, mean, std, resize,
crop, `PREPROCESSING_VERSION`) all match. Change any of them and the entry is rebuilt;
nothing is reused silently. Absolute paths are never part of the identity.

Procedure (GPU host, after the runbook's checkpoint verification):

1. `ckpt_sha = compute_file_sha256("checkpoints/IN22K-vit.g.16-600e.pth.tar")`, once
   per run, and record it with the run. Also record the `external/ijepa` commit above.
2. `train, test = validate_category_splits(root, category)`.
3. `build_nominal_latent_caches(train.samples, IJEPAEncoder(...), val_fraction=0.1,
   seed=0, cache_root="data/cache/latents", category=category,
   encoder_name="ijepa_vit_g16_in22k_ep44", checkpoint_sha256=ckpt_sha, patch_size=16,
   embedding_dim=1408)`.
4. Interrupted? Rerun the same call; compatible per-sample shards in
   `<category>/.<split>.parts/` are skipped. Delete that directory to force a rebuild.
5. Consumers call `load_latent_cache(path, CacheExpectations(...))` and `.get(image_id)`.

Latents may differ in the last float bits when the batch size changes, so record the
batch size and dtype with a run. Preprocessing compatibility: change
`preprocess_image` output, and `PREPROCESSING_VERSION` must be bumped.

Validation status: the generation path is covered by synthetic tests and has also been
run with the official checkpoint on CUDA for the three pilot categories. Nine real
train/validation/test cache entries were verified: 819 nominal train/validation images
and 325 official test images, with finite float32 `(N, 196, 1408)` latents, expected
sample order and checkpoint/preprocessing metadata, and no cache failures.

### Cache-generation command (Phase 4)

Inputs: MVTec AD under `--dataset-root` (default `data/raw/mvtec_ad`), the checkpoint
(`--checkpoint`, default `checkpoints/IN22K-vit.g.16-600e.pth.tar`) and the pinned
`external/ijepa` checkout (`--official-repo`). Nothing machine-specific is hardcoded.

```bash
# CPU smoke test, no checkpoint (writes to data/cache/latents/_mock):
python -m ijepa_ad.caching.cli --category bottle --mock-encoder

# Real run on the GPU host: nominal train/val caches, then the evaluation cache.
python -m ijepa_ad.caching.cli --category bottle capsule hazelnut --device cuda \
    --batch-size 8 --summary-json results/cache_nominal.json
python -m ijepa_ad.caching.cli --category bottle capsule hazelnut --device cuda \
    --purpose test --summary-json results/cache_test.json
```

Record with the run: the summary JSON (it contains the checkpoint SHA-256, device,
batch size, and `encoder_kind`), the `external/ijepa` commit, `--val-fraction` (0.1) and
`--seed` (0), and the torch version. Resume: rerun the identical command; compatible
entries and shards are skipped, missing/corrupt/incompatible shards are recomputed
(`recomputed` in the summary). A corrupted final entry is reported as failed and left
untouched; `--rebuild` discards the selected entries first and is the way to replace it.
An encoder exception (e.g. out of memory) fails its batch only: those samples appear
under `failed_samples` and a rerun encodes just them. Exit
code 0 means every entry is complete, 1 that some failed (see `failed_samples`), 2 a
configuration error. Mock caches live under `_mock` and carry encoder name
`mock_encoder` and a checkpoint hash derived from `--seed`, so they never satisfy a real
run's compatibility check and a different seed never reuses another projection.

Validation status: the CLI and summary have synthetic CPU coverage and a completed
real-checkpoint CUDA run for Bottle, Capsule, and Hazelnut. The full 15-category dataset
has not been staged or evaluated.

## Historical CPU verification records

The following Phase 5 and Phase 6 sections preserve results from earlier
environments. They are evidence for the synthetic test coverage, not a claim that
the current Python environment can reproduce them; the latest audit is at the top of
this file. Statements below describing a missing GPU run or future real training refer
to those earlier handoffs and are superseded by the verified pilot record above.

### Phase 5 validation: reproducing the CPU test run

Environment used (Windows, CPU only, no network, no CUDA, no dataset, no checkpoint):
Python 3.12.10 (CI uses 3.11), torch 2.14.1+cpu, Pillow 12.3.0, ruff 0.8.6, black
25.1.0, mypy 1.14.1.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"      # Linux: .venv/bin/python
./.venv/Scripts/python.exe -m pytest tests/unit tests/integration        # 137 passed
./.venv/Scripts/python.exe -m pytest tests/integration/test_pipeline_end_to_end.py  # 18 passed
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/black.exe --check --workers 1 .
./.venv/Scripts/mypy.exe src tests
```

Obtained with synthetic data and the mock encoder only: every test result above, the
end-to-end pipeline, the cache failure/recovery tests, and the identical-latents check.

Deterministic guarantees (tested): split membership depends only on image content, the
seed, and `val_fraction` (not on filenames, enumeration or input order, and global RNG
state is left untouched); preprocessing is a pure function of the file bytes and
`PreprocessConfig`; cache metadata JSON is canonical (sorted, compact) and, apart from
`created_at_utc`, two runs give equal metadata; the mock encoder on CPU gives
bit-identical latents for the same inputs and batch size.

Not guaranteed or not measured: latents are only equal within float tolerance across
different batch sizes (tested at `atol=1e-4` with the mock); real-model inference on
GPU or CPU, other dtypes, and other torch/CUDA/cuDNN versions are unmeasured, as are
the `created_at_utc` and `torch_version` provenance fields, which vary by environment.
No bitwise-reproducibility claim is made for the real ViT-g/16.

Required for future real execution: MVTec AD under `data/raw/mvtec_ad`
(`python validate_mvtec.py data/raw/mvtec_ad` first), the checkpoint at
`checkpoints/IN22K-vit.g.16-600e.pth.tar`, the pinned `external/ijepa` checkout above,
and a CUDA host per `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.

Unverified on external GPU hardware: checkpoint load and hashing, the VRAM spike,
`IJEPAEncoder` inside the CLI (`_load_real_encoder`, device adapter), real
`[N, 196, 1408]` caches, float16 vs float32 storage, real-MVTec behaviour (including
whether any category has duplicate training images), and CI on Linux/Python 3.11.

### Phase 6 handoff: CPU commands and GPU validation (historical)

CPU commands (dev environment from `TRANSFER.md` §5):

```bash
./.venv/Scripts/python.exe -m pytest tests/integration/test_person2_cache_consumer.py -q  # 13 passed
./.venv/Scripts/python.exe -m pytest tests/unit tests/integration -q                      # 168 passed
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/black.exe --check --workers 1 .
./.venv/Scripts/mypy.exe src tests
```

What a downstream reader provides and receives: `FLOW.md`, "Downstream consumption
contract". It is tested with a stand-in consumer, not with Person 2's code.

The external-GPU procedure and the completed RTX 4090 pilot evidence are recorded in
[`docs/GPU_VALIDATION_CHECKLIST.md`](docs/GPU_VALIDATION_CHECKLIST.md). Float16-vs-
float32 storage comparisons, 8 GB target-device validation, and Linux/Python 3.11 CI
remain unverified.

## Person 2 predictor path: earlier CPU verification (historical)

Verified locally on Windows/Python 3.13.2 with synthetic cached latents only:

```bash
./.venv/Scripts/python.exe -m pytest tests/unit/test_predictor_losses.py tests/unit/test_transformer_predictor.py tests/integration/test_predictor_training.py -q  # 8 passed
./.venv/Scripts/python.exe -m pytest tests/unit tests/integration  # 165 passed
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/black.exe --check --workers 1 .
./.venv/Scripts/mypy.exe src tests
```

The following command documents the original interface; the real pilot was subsequently
trained with `scripts/train_predictor.py`, producing checkpoints under
`checkpoints/predictors/` and histories under `results/predictor/`:

```bash
./.venv/Scripts/python.exe scripts/train_predictor.py \
  --cache-root data/cache/latents \
  --category bottle \
  --checkpoint-sha256 <real_checkpoint_sha256_from_cache_summary> \
  --output results/predictor/bottle.pt \
  --device cuda
```

The pilot cosine and L2 runs used 100 epochs. Best validation epochs were near the end,
so longer training and validation-only ablations remain open work. Never select
hyperparameters from the official test labels.
