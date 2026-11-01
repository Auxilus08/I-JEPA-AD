# Metrics Improvement Plan

_Written 2026-11-01 at commit 7688586. Analysis only: nothing in this plan has been implemented yet._

## 1. Current state (read this first)

`TRANSFER.md` §1–§3 is stale: real artifacts now exist locally. Present in this checkout:

- Encoder checkpoint `checkpoints/IN22K-vit.g.16-600e.pth.tar` and Meta repo `external/ijepa/`
- Latent caches `data/cache/latents` (nominal train/val) and `data/cache/latents_test`
- Predictors `checkpoints/predictors/` (baseline), `checkpoints/predictors_augmented/`, `checkpoints/predictors_block2/capsule_cosine.pt` (2×2 coarse windows, **never evaluated**)
- Results `results/evaluation/` (notebook 03), `results/calibration/` (notebook 04), `results/benchmarks/` (RTX 4060 latency)

Measured on the official test split (baseline predictors, adaptive mode, default τ = 0.5):

| Category | Image AUROC (max / top-5) | Pixel AUROC | AUPRO | kNN reference, same latents (img / pix / AUPRO) |
|---|---|---|---|---|
| bottle   | 0.925 / 0.956 | 0.930 | 0.766 | 0.999 / 0.973 / 0.859 |
| capsule  | 0.522 / 0.528 | 0.684 | 0.098 | 0.805 / 0.916 / 0.513 |
| hazelnut | 0.802 / 0.930 | 0.907 | 0.577 | 0.988 / 0.979 / 0.868 |

What the numbers say:

1. **The frozen ViT-g latents carry the signal; the predictor does not extract it.** The nearest-neighbour reference on the same latents is far better. Work on the predictor and the scoring, not the encoder.
2. **Masking policy hardly matters.** Fixed, adaptive and dense give nearly identical pixel AUROC (`results/calibration/pixel_metrics.csv`). More τ/λ tuning will not help much.
3. **The max score is noisy.** Top-k already helps image AUROC (`results/calibration/image_auroc_grid.csv`).
4. **Augmented predictors are worse in every category** (`results/evaluation/predictor_comparison.csv`).
5. **Capsule is at chance.** Its defects are small, and 1000 px images resized to 224 px make each 16 px token cover about 70 original pixels.

## 2. Ground rules (do not break these)

- **Never select anything on the test split.** Choose settings on nominal validation data or on the synthetic-anomaly validation set (item 12). Fix the choice before scoring test.
- Report every change as a **paired-bootstrap 95% CI** on the delta against the current baseline row, in the same format as `results/calibration/bootstrap_deltas.csv`. A CI that contains 0 is "inconclusive", not "improved".
- Report masks per image and latency next to accuracy.
- Log every design decision in `DECISIONS.md` (project convention).
- Run `pytest tests` after any change under `src/`. Tests use the CPU and a mock encoder.
- Never overwrite existing result files. Write new results to a new directory, e.g. `results/improvements/<experiment>/`.

## 3. Recommendations, ranked by expected gain per unit of effort

### Tier 1: no retraining

These reuse the current predictors and cached latents. Extend notebook 04 or add a notebook 05.

1. **Per-position score normalisation.** For each of the 196 token positions, compute the mean and std of the discrepancy over nominal validation images (per category, per mode). Z-score each test patch score with them. Predictability depends on position: tokens in the centre of a 4×4 window are harder to predict than those at its edge, and background differs from object. For aligned categories (bottle, capsule) this bias probably swamps small defects. **Top cheap bet for capsule.**
2. **Per-channel standardisation before the cosine.** Standardise predicted and actual latents per channel with the mean and std of the nominal **training** latents, then compute `1 − cos`. Also try I-JEPA's target normalisation, `F.layer_norm(x, (x.size(-1),))` with no affine parameters. Big ViTs have a few outlier channels that dominate cosine similarity. These are scoring-only changes in `MaskedLatentScorer._score_windows` (`src/ijepa_ad/inference.py:283`). Make the behaviour a parameter, not a replacement, so the current results stay reproducible.
3. **Gaussian smoothing of the score map** before taking max or top-k (σ ≈ 4 px on the upsampled 224×224 map, as PatchCore and PaDiM do). Reuse the upsampling in `src/ijepa_ad/masking/score_maps.py`.
4. **Mahalanobis residual score.** Fit a Gaussian (per position, or one shared across positions) to the residual `actual − predicted` on nominal images. Score with Mahalanobis distance instead of `1 − cos`. Reduce the dimension first (e.g. PCA to 64–128 dims), since 1408 dims and about 200 images give a singular covariance.
5. **Evaluate `checkpoints/predictors_block2/capsule_cosine.pt`** with `coarse_size = 2` in the masking settings. Compare it with the baseline on validation first.

### Tier 2: fix the likely root cause, context leakage

6. **Mask at the input, not in latent space.** Today the whole image is encoded once, and latent tokens are replaced by a mask token afterwards (`src/ijepa_ad/predictor/transformer_predictor.py:94`, called from `inference.py:301`). Every context token has already attended to the defect through all 40 ViT-g layers, so the predictor can partly reconstruct the anomaly and the discrepancy shrinks. The paper lists this as an untested hypothesis.
   - **Diagnostic first:** on about 20 test images, compare discrepancies under two setups. (a) Current: full-image encoding, then latent masking. (b) Input masking: Meta's `VisionTransformer.forward(x, masks=...)` in `external/ijepa/src/models/vision_transformer.py:401` accepts *kept* token indices, so run the encoder on the visible tokens only. Expect anomalous tokens to score higher under (b).
   - **If confirmed:** cache context latents per (image, coarse window). That is about 16 × 200 × 196 × 1408 fp16 ≈ 1.7 GB per category. Retrain on them. Inference becomes 16 batched encoder passes per image, roughly 0.5–1.4 s on the RTX 4060 (the encoder at batch 1 takes ≈ 88 ms; see `results/benchmarks/e2e_latency_vram.json`).
7. **Use Meta's pretrained I-JEPA predictor (top bet for a large gain).** The 16.5 GB checkpoint very likely holds `encoder` (context), `predictor` and `target_encoder`. Verify the keys first. `IJEPAEncoder` (`src/ijepa_ad/encoder/ijepa_wrapper.py`) loads only `target_encoder`.
   - The pretrained predictor (`external/ijepa/src/models/vision_transformer.py:285`, `forward(x, masks_x, masks)`) was trained on ImageNet-22k for exactly this task: context encoder on visible patches → target-encoder latents.
   - Steps: (a) zero-shot, i.e. context encoder on visible tokens, pretrained predictor, discrepancy against the cached target latents, with the same windows and metrics; (b) if promising, fine-tune it on nominal images.
   - This replaces a 4-layer predictor trained from scratch on 188–352 images, and it removes leakage at the same time.
   - Watch host RAM when loading the checkpoint (15.8 GB machine). Use `torch.load(..., mmap=True)` if needed and log that change in `DECISIONS.md`.

### Tier 3: features and resolution

8. **Intermediate encoder layers.** Final-layer tokens are semantic and spatially diffuse; PatchCore uses mid-level features. First re-run the kNN reference (notebook 03 §9b) on blocks of about 20–35 (of 40) to find the layer with the most signal. This is cheap and needs no predictor. Then train the predictor on that layer, or on a concatenation of layers. This needs new caches with a different cache fingerprint.
9. **Higher resolution or a tight crop for capsule.** Options: 448 input with interpolated position embeddings (a 28×28 grid), or a fixed crop around the object band. This is expensive: 196/14 is hard-coded in `inference.py:40` and `inference.py:113`, in `PatchGrid`, and in the predictor config. Do it last.

### Tier 4: predictor training

10. **Give the predictor more training signal.**
    - Today each epoch sees one mask per image, about 12 batches. Sample several masks per image per step, as I-JEPA does with multiple targets.
    - Train longer with warmup and a cosine LR schedule.
    - Add **randomly offset (non-grid-aligned) blocks** to `build_inference_aligned_target_mask` (`src/ijepa_ad/predictor/target_masks.py:44`). Then score with **overlapping windows**: the 4 offsets of the 4×4 grid give 64 passes, still 3× fewer than dense. Average them to remove the window-boundary bias.
11. **Drop the current augmentations**, at least for aligned categories. They lowered image AUROC everywhere. If augmentation stays, use mild photometric changes only.

### Evaluation hygiene, needed before choosing settings from Tiers 1–4

12. **Synthetic-anomaly validation set.** Paste CutPaste- or DRAEM-style (Perlin-noise) defects onto the nominal validation images, generated with a fixed seed. Choose the layer, σ, k, λ, τ, block size and normalisation on it. Today only nominal validation data exists, so nothing can be chosen for detection quality without touching test. `src/ijepa_ad/data/augmentation.py` only augments nominal images; the defect generator is new code.
13. **Multiple seeds (≥ 3), keeping the paired bootstrap.** Most current deltas are within noise. Consider a 20% validation split for more stable τ and threshold calibration. That trades away training images, so record the choice.

### Hybrid, ablation only

14. A z-normalised sum of predictor discrepancy and kNN distance would almost certainly raise every metric, but the kNN term would do most of the work. Report it only as an ablation, never as the method, because it undercuts the paper's decoder-free and memory-free claim.

## 4. Execution order

1. Tier 1 (items 1–5) in a new notebook. Hours of work, no retraining.
2. Item 12 (synthetic-anomaly validation), so that later choices are legitimate.
3. Item 6 diagnostic, then item 7 zero-shot.
4. Depending on step 3: item 7 fine-tuning, or item 6 retraining, plus items 8 and 10.
5. Item 9 last.

After each step, update `paper/main.tex` only with numbers that come from saved result files, with their CIs.

## 5. Key file references

| Purpose | Path |
|---|---|
| Scoring (windows, cosine, fusion, image score) | `src/ijepa_ad/inference.py` |
| Predictor | `src/ijepa_ad/predictor/transformer_predictor.py` |
| Training loop / CLI | `src/ijepa_ad/predictor/training.py`, `scripts/train_predictor.py` |
| Training masks | `src/ijepa_ad/predictor/target_masks.py` |
| Encoder wrapper (target encoder only) | `src/ijepa_ad/encoder/ijepa_wrapper.py` |
| Meta ViT + predictor source | `external/ijepa/src/models/vision_transformer.py` |
| Metrics (AUROC, AUPRO, thresholds) | `src/ijepa_ad/evaluation/metrics.py` |
| Score-map upsampling | `src/ijepa_ad/masking/score_maps.py` |
| Calibration, top-k, bootstrap (template for new experiments) | `notebooks/04_calibrate_tau_and_image_score.ipynb` |
| kNN reference | `notebooks/03_evaluate_and_compare_predictors.ipynb` §9b |
| Predictor and masking configs | `configs/predictor/base.yaml`, `configs/masking/coarse_to_fine.yaml` |
