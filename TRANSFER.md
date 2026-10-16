# TRANSFER.md
_Last updated: 2026-10-17, by Claude Code, at commit a64a2a0 (HEAD) plus **uncommitted** working-tree changes_

## 1. One-paragraph project state

The pipeline code is complete for the three-category pilot (bottle, capsule, hazelnut): frozen I-JEPA wrapper, latent caches, Transformer predictor, coarse-to-fine inference, fusion, metrics and a Streamlit UI. An earlier session (RTX 4090 machine) recorded real-model pilot results (see §3), but **none of those artifacts exist in this checkout** (no `checkpoints/`, `external/ijepa`, `data/cache/`, or `results/*`), so they cannot be re-verified here. On 2026-10-17 an audit led to code changes (uncommitted): predictor training masks now match the inference masks, the UI heatmap is bilinear, augmentation outputs and caches can no longer go stale, the encoder wrapper cannot be unfrozen, and checkpoints record seeds and code provenance. **Because training masks changed, all previously trained predictors must be retrained and the pilot numbers in §3 are not comparable with future runs.** No real-model inference or GPU run was performed in the 2026-10-17 session.

## 2. Phase checklist

- **Pilot data and environment:** DONE (local copies at `dataset/` and `data/raw/mvtec_ad`; split validation re-run on 2026-10-17: 0 overlap train/val/test by content hash).
- **Encoder and caches:** code DONE; real encoder/caches IN PROGRESS locally (checkpoint and official repo not present here).
- **Predictor:** code DONE incl. inference-aligned masks; retraining NOT STARTED.
- **Inference and metrics:** code DONE; a new real evaluation NOT STARTED.
- **Augmentation workflow:** code DONE (fingerprinted, per-category); notebook 02 not executed with the real encoder.
- **Dashboard:** DONE for the pilot; heatmap now bilinear.
- **Full benchmark and comparisons:** NOT DONE. Only 3 of 15 categories; `baselines/` PatchCore and MAE are placeholders, and `DenseMaskedIJEPA`/`FixedMaskIJEPA` run on precomputed scores, not on the predictor.
- **Target hardware (RTX 4060, 8 GB):** NOT DONE. Nothing measured on it.

## 3. What is verified working right now

Backed by tests run on 2026-10-17 (`pytest -p no:cacheprovider -q tests`: **283 passed**, 3 consecutive runs; `ruff check` and `black --check` clean):
- Inference-aligned mask sampler: block rows equal the 16 coarse inference windows, single rows are isolated tokens, determinism, context always remains (`tests/unit/test_target_masks.py`).
- No target leakage: rewriting target latents leaves predictor output unchanged; loss equals mean `1 - cos` over target tokens only.
- Bilinear upsampling shared by evaluation and UI; normalisation is display-only (`tests/unit/test_score_maps.py`).
- Augmentation reproducibility, per-category capsule override, fingerprinted outputs, cache rebuilds on changed checkpoint/preprocessing/content (`tests/unit/test_augmentation.py`, `tests/integration/test_augmented_cache_invalidation.py`).
- Encoder stays frozen and in eval mode under a parent `.train()`; dtype guards (`tests/unit/test_dtype_and_freeze_safety.py`); non-finite loss stops training; checkpoint `reproducibility` block (`tests/integration/test_predictor_training_masks.py`).
- Wiring smoke on CPU with the **mock** encoder on real bottle images (outputs in the session scratchpad only): cache CLI, 2-epoch training via `scripts/train_predictor.py`, checkpoint load with `load_predictor_checkpoint`, adaptive scoring. This proves plumbing, not model quality.

Earlier-session pilot results (reported in the previous TRANSFER.md; **not reproducible from this checkout and produced with scattered-mask predictors**). Image AUROC / pixel AUROC / AUPRO, official test split:

| Category | Adaptive | Fixed | Dense |
| --- | --- | --- | --- |
| bottle | 0.9349 / 0.8900 / 0.6638 | 0.9349 / 0.8904 / 0.6656 | 0.8246 / 0.8843 / 0.6546 |
| capsule | 0.4986 / 0.6939 / 0.0959 | 0.4986 / 0.6939 / 0.0959 | 0.5325 / 0.7962 / 0.2372 |
| hazelnut | 0.6536 / 0.8774 / 0.4876 | 0.6179 / 0.8746 / 0.4880 | 0.5421 / 0.7703 / 0.3681 |

Adaptive averaged 16.39 / 16.00 / 17.60 masks per image (bottle/capsule/hazelnut) vs 196 dense, i.e. the gate (τ=0.5, cosine-distance units) almost never fired, so adaptive ≈ fixed. Do not use test results to tune thresholds.

## 4. What is NOT done yet (explicit, so it isn't assumed)

- Real-model verification after the 2026-10-17 changes: loading Meta ViT-g/16 on this machine, building real caches, retraining predictors with `inference_aligned` masks, re-evaluating. None done.
- Real behaviour of the train/inference mask alignment on real latents (effect size unknown); the synthetic experiment only showed a gap exists in principle.
- Possible context leakage via full-image encoding (context tokens are computed with attention over the masked region too); not measured.
- τ and λ selection on nominal validation data only; τ=0.5 is uncalibrated.
- `image_score` divides by 2 and clamps, but adaptive scores can reach 2(1+λ); not changed.
- Real baselines (PatchCore, MAE), 12 remaining MVTec categories, ablations, multi-seed statistics, RTX 4060/8 GB validation.
- No LR scheduler, gradient clipping, AMP or optimizer-state saving (deliberately not added; no evidence they are needed).
- `python -m ijepa_ad.predictor.training` silently does nothing (no `__main__` guard); use `scripts/train_predictor.py`.

## 5. Environment & reproduction

- Windows, Python 3.12, `.venv\Scripts\python.exe`, torch 2.2.2+cpu locally (CUDA not available on this machine). Install: `python -m pip install -e ".[dev,frontend]"`.
- Verify: `.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests`, `.\.venv\Scripts\ruff.exe check .`, `.\.venv\Scripts\black.exe --check .`, `.\.venv\Scripts\mypy.exe src tests` (one pre-existing error: unused `type: ignore` at `src/ijepa_ad/evaluation/metrics.py:10`).
- Local-only state a new environment lacks: Meta checkpoint `checkpoints/IN22K-vit.g.16-600e.pth.tar` (SHA-256 recorded earlier: `a373c7b16165092e29bd5714c6ce79c574c4a52b01b64bf4c82dc81e6a82cf0b`), pinned Meta repo at `external/ijepa` (commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`), latent caches under `data/cache/`, and predictors under `checkpoints/`. Rebuild caches with `scripts/build_latent_cache.py`; train with `scripts/train_predictor.py` or notebook 01/02.
- `dataset/` (untracked, not gitignored) duplicates `data/raw/mvtec_ad` (counts match; not hash-compared).

## 6. Known issues / open threads

- Old predictor checkpoints (scattered masks) still load and score; they are simply not aligned with inference.
- UI `heatmap` is min-max normalised per image (display only); never feed it to metrics.
- `baselines/` placeholders must not be reported as PatchCore/MAE results.
- Evaluation uses the validation set for both early stopping and the 5%-FPR threshold (slightly optimistic).
- `torch.load` of the Meta checkpoint is not `weights_only` (trusted file).

## 7. Where to look for more detail

- Rationale: `DECISIONS.md` entries dated 2026-10-17 (masks, upsampling, augmentation, capsule flip, freeze/dtype/provenance).
- Architecture and flows: `FLOW.md` §2–§4. Real-model pilot environment: `REPRODUCIBILITY.md`, `docs/GPU_VALIDATION_CHECKLIST.md`.

## 8. Immediate next step

On the GPU machine: load the real encoder, rebuild/verify the bottle caches, retrain bottle with `inference_aligned` masks (config default) next to the old scattered-mask run on the same seed, compare **validation** loss on inference-style masks, then pick τ and λ on validation only before any test evaluation.
