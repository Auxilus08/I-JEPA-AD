# TRANSFER.md
_Last updated: 2026-10-15_

## 1. Current project state

The three-category MVTec pilot has completed its first real-model pass: the official
I-JEPA ViT-g/16 checkpoint was loaded on CUDA, real train/validation/test latent caches
were built, six cosine/L2 predictors were trained, and adaptive/fixed/dense I-JEPA
inference was evaluated on the official test splits. The Streamlit dashboard now uses
real inference by default and reads the measured pilot results. These are pilot results,
not full-benchmark claims.

The environment used was Python 3.12.4, PyTorch 2.2.2+cu121, NumPy 1.26.4, and PyArrow
18.1.0 on an RTX 4090 with 24 GB VRAM. CUDA was available, `pip check` passed, and the
frontend constrains PyArrow to `<19` because newer wheels require NumPy 2.x, which is
incompatible with the pinned PyTorch 2.2 environment.

## 2. Phase checklist

- **Pilot data and environment:** DONE. Bottle, Capsule, and Hazelnut are staged at
  `data/raw/mvtec_ad`; originals were preserved. Category-level split validation passed.
- **Encoder and caches:** DONE for the pilot. Meta I-JEPA is pinned at
  `52c1ae95d05f743e000e8f10a1f3a79b10cff048`; the official checkpoint is
  16,534,564,553 bytes, SHA-256
  `a373c7b16165092e29bd5714c6ce79c574c4a52b01b64bf4c82dc81e6a82cf0b`.
  Real ViT-g/16 inference and all nine train/validation/test cache entries were verified.
- **Predictor:** DONE for the initial pilot. Six 100-epoch runs were extended using only
  nominal validation loss: all six to a 200-epoch cap, then Bottle and Capsule to 300.
  Bottle still improved through epoch 290; its cosine/L2 runs are not yet converged.
- **Inference and metrics:** DONE for the three I-JEPA schedules. Real masked-latent
  scoring, image/pixel AUROC, and AUPRO are implemented and exercised. The evaluation
  used 100 AUPRO thresholds and calibrated decision thresholds from nominal validation
  scores only.
- **Dashboard:** DONE for pilot integration. Real mode is the default; mock mode requires
  explicit selection. Measured benchmark tables render. A compatible frontend dependency
  range is recorded in `pyproject.toml`.
- **Full benchmark and comparisons:** NOT DONE. Only 3 of 15 categories are available;
  PatchCore and MAE are not implemented as valid comparable baselines. Ablations,
  multi-seed uncertainty/statistical analysis, and paper assets remain outstanding.
- **Target hardware:** NOT DONE. Only the RTX 4090 (24 GB) was tested; no RTX 4060/8 GB
  compliance claim is supported.

## 3. Pilot evidence

Dataset counts:

| Category | Train | Test | Nominal test | Defect test / masks |
| --- | ---: | ---: | ---: | ---: |
| bottle | 209 | 83 | 20 | 63 |
| capsule | 219 | 132 | 23 | 109 |
| hazelnut | 391 | 110 | 40 | 70 |

Image AUROC / pixel AUROC / AUPRO on the official test split:

| Category | Adaptive | Fixed | Dense |
| --- | --- | --- | --- |
| bottle | 0.9349 / 0.8900 / 0.6638 | 0.9349 / 0.8904 / 0.6656 | 0.8246 / 0.8843 / 0.6546 |
| capsule | 0.4986 / 0.6939 / 0.0959 | 0.4986 / 0.6939 / 0.0959 | 0.5325 / 0.7962 / 0.2372 |
| hazelnut | 0.6536 / 0.8774 / 0.4876 | 0.6179 / 0.8746 / 0.4880 | 0.5421 / 0.7703 / 0.3681 |

Adaptive evaluated an average of 16.39, 16.00, and 17.60 masks per image for Bottle,
Capsule, and Hazelnut, respectively, versus 196 for dense. The benchmark UI reports
macro-average measured pilot values. Its latency is cached-latent predictor scoring
only and excludes image encoding; end-to-end UI inference includes the encoder. Do not
use these test results to tune thresholds or hyperparameters.

Validation-calibrated adaptive decision thresholds (target 5% nominal validation FPR):
Bottle `0.31648`, Capsule `0.41187`, Hazelnut `0.44221`. The observed validation rates
were 1/21, 1/22, and 1/39. Threshold comparisons use strict `score > threshold`.

Validation-only training extensions used seed 0 and patience 25. The official-test
benchmarks above still use the original 100-epoch checkpoints; the extension checkpoints
are isolated under `results/predictor_extension_200/` and
`results/predictor_extension_300/` and were not evaluated on the test split.

| Category | Objective | Best at 100 | Best extended | Extended run |
| --- | --- | ---: | ---: | --- |
| bottle | cosine | 0.051810 (epoch 99) | 0.033670 (epoch 290) | 300 epochs |
| bottle | L2 | 0.089058 (epoch 98) | 0.055142 (epoch 290) | 300 epochs |
| capsule | cosine | 0.105533 (epoch 100) | 0.095222 (epoch 190) | stopped at 215 |
| capsule | L2 | 0.179605 (epoch 100) | 0.160209 (epoch 190) | stopped at 215 |
| hazelnut | cosine | 0.182006 (epoch 100) | 0.180196 (epoch 108) | stopped at 133 |
| hazelnut | L2 | 0.278700 (epoch 99) | 0.276434 (epoch 120) | stopped at 145 |

## 4. Reproduction and verification

Run commands from the `I-JEPA-AD` project root. On this Windows machine, use
`.venv\Scripts\python.exe`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_inference.py tests\unit\test_evaluation_metrics.py tests\unit\test_frontend.py tests\unit\test_ijepa_wrapper.py tests\unit\test_seeding.py -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src tests
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --help
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

The latest focused verification passed 30 tests; Ruff and mypy passed for the recently
changed inference, evaluation, and frontend files. The complete test suite has not yet
been rerun on this environment. Real encoder smoke test, cache counts, training losses,
metrics, and environment details are recorded in `REPRODUCIBILITY.md` and
`docs/GPU_VALIDATION_CHECKLIST.md`.

## 5. Remaining implementation sequence

1. Decide whether to extend Bottle training beyond 300 epochs; its best validation loss
   was still at epoch 290. Keep the test split isolated until the selection rule is fixed.
2. Obtain the other 12 MVTec categories; stage and validate them without altering source
   copies.
3. Implement reproducible PatchCore and MAE comparators with equivalent preprocessing,
   splits, and evaluation; otherwise keep comparison claims explicitly scoped to I-JEPA
   schedules.
4. Run the frozen protocol across all 15 categories, repeat selected experiments across
   seeds, and report uncertainty/statistical tests plus full encoder-inclusive latency
   and memory.
5. Validate the intended 8 GB device separately, generate paper assets from machine-
   readable results, then update the claims and protocol documentation.
