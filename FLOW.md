# FLOW.md
_Living HLD + LLD + runtime-flow reference for I-JEPA-AD._
_This file is rewritten in place as the system changes; it is NOT append-only. Update rules: see AGENT.md._
_Last reviewed against src/: 2026-10-11 - Person 2 cache-backed predictor dataset implemented._

## 1. High-Level Design
The data path is implemented through latent caching (generation, resume, lookup) and
validated end to end on synthetic data with a mock encoder (Phase 5). Person 2's
predictor-side work is implemented through CPU-verifiable training code: cached-latent
dataset/batches, masked-token Transformer predictor, cosine/L2 losses, training loop,
checkpoint saving, and a training CLI. Masking, score fusion, heatmaps, and comparison
baselines are implemented with synthetic CPU coverage. Evaluation metrics and real GPU
training remain open; the real encoder has never run on real weights or images.

```text
<root>/<category>/{train,test,ground_truth}
        │
        ▼
data.mvtec_dataset.MVTecADDataset ──> data.validators (labels, masks, leakage)
        │  tuple[MVTecSample]   (path metadata only, no pixels)
        ▼
data.splits.split_nominal_train_val ──> NominalSplit(train, val)   [train/good only]
        │  image paths
        ▼
data.transforms.preprocess_batch ──> Tensor[N, 3, 224, 224]  (I-JEPA normalized)
        │
        ▼
encoder.ijepa_wrapper.IJEPAEncoder.encode ──> Tensor[N, 196, 1408]  (frozen)
        │
        ▼
caching.cli (CLI → CacheRunConfig → run_cache_generation → RunSummary)
        │   for each category/purpose calls ↓
caching.builder.build_latent_cache  (batches; per-sample shards
        │   <split>.parts/<id>.pt; resumable; assembles only when complete)
        ▼                                 ──> <root>/<category>/<split>.pt
        │  load_latent_cache(path, CacheExpectations) ──> LatentCache(latents, metadata)
        │  .get(image_id) / .select(ids)   - no encoder re-run
        ▼
predictor.cache_dataset.CachedLatentDataset
        │
        ▼
predictor.training.train_predictor ──> predictor.transformer_predictor
        │                                      │
        │                                      ▼
        └────────────── checkpoint.pt <── losses on masked latent tokens
        │
        ▼
masking / baselines ──> fused heatmaps; evaluation metrics remain future work
```

The `utils` package now provides cross-cutting controls used by every future
training and evaluation entrypoint: `set_seed()` establishes reproducible random
state, `setup_logging()` emits structured JSONL records, and `profile_peak_vram()`
records peak CUDA allocation around a code block.

The `encoder` package wraps Meta's official I-JEPA ViT-g/16 ImageNet-22K
44-epoch checkpoint. It is configured for 224×224 RGB inputs, 16×16 patches,
196 patch tokens, and 1408-dimensional latents. The wrapper requires Meta's
official source checkout to construct the architecture and loads the checkpoint's
`target_encoder` state dict.

The `data` package owns MVTec AD ingestion and split invariants. It discovers only
`<root>/<category>/train/good/*` for training and all label directories under
`<root>/<category>/test/*` for testing. Training records must be nominal and
unmasked. Test `good` records are nominal and unmasked; every defect record must
have the corresponding official mask at
`<root>/<category>/ground_truth/<defect>/<image_stem>_mask.*`. Image identity is a
SHA-256 digest of file content, which catches copied images even when MVTec
filenames repeat across its official train and test directories. The split validator
rejects duplicate IDs within a split and any ID shared across train and test.
`MVTecADDataset` accepts only `split` `train` or `test` (anything else raises
`ValueError`).

The predictor's validation set is a nominal-only holdout carved from the official
training records by `data.splits`; the official test split is never read or
modified by splitting. `data.transforms` turns any split's image paths into
encoder-ready tensors with one deterministic transform for train, validation, and
test: RGB conversion, full-image bilinear resize to 224×224 (no crop), `[0, 1]`
scaling, and the official I-JEPA ImageNet normalization.

## 2. Low-Level Design
Foundational utility interfaces:

- [`seeding.py`](src/ijepa_ad/utils/seeding.py): `set_seed(seed: int) -> None`
  seeds Python, NumPy, PyTorch, and CUDA and enables deterministic PyTorch
  algorithms. `seeded_rng(seed: int) -> random.Random` returns an isolated RNG
  with the same seed validation, for library code that must be deterministic
  without touching global state.
- [`logging_setup.py`](src/ijepa_ad/utils/logging_setup.py):
  `setup_logging(level, log_file, logger_name) -> logging.Logger` configures
  idempotent stream/file JSON logging.
- [`profiling.py`](src/ijepa_ad/utils/profiling.py):
  `profile_peak_vram(label, logger, device) -> Iterator[PeakMemory]` resets and
  reads `torch.cuda.max_memory_allocated()` on CUDA hosts and safely reports zero
  on CPU-only hosts.

The predictor package contains the Person 2 path up to training execution. It can read
cached latents, join optional annotations, train a lightweight Transformer predictor on
masked latent-token reconstruction, validate on the nominal validation cache, and save a
checkpoint. Masking and comparison baselines are implemented with CPU-testable
tensor interfaces; evaluation metrics and full benchmark orchestration remain open.
Hydra/OmegaConf is not wired in: `configs/data/mvtec.yaml` records `root`,
`pilot_categories`, `preprocessing.{image_size, resize, mean, std}`, and
`nominal_split.{val_fraction, seed, min_train, min_val}`, `latent_cache.root`, but no code loads it yet.
Its values mirror `PreprocessConfig` defaults and the arguments callers must pass to
`split_nominal_train_val`.

Data interfaces:

- [`mvtec_dataset.py`](src/ijepa_ad/data/mvtec_dataset.py):
  `MVTecADDataset(root, category, split)` discovers immutable `MVTecSample` records
  and validates the selected official split. `validate_category_splits(root,
  category)` validates both boundaries together.
- [`validators.py`](src/ijepa_ad/data/validators.py):
  `validate_training_records()`, `validate_test_records()`, and
  `validate_no_split_leakage()` enforce nominal-only training, official mask/label
  pairing, and no cross-split image IDs.
- [`splits.py`](src/ijepa_ad/data/splits.py):
  `split_nominal_train_val(samples, *, val_fraction, seed, min_train=1, min_val=1)
  -> NominalSplit` runs `validate_training_records()`, orders records by SHA-256
  `image_id`, shuffles with `seeded_rng(seed)`, and holds out
  `round(n * val_fraction)` records as validation. It raises
  `InsufficientNominalSamplesError` (a `DatasetValidationError`) when either side
  would fall below its minimum. `NominalSplit(train, val, seed, val_fraction)` is
  frozen, with both tuples sorted by image path.
- [`transforms.py`](src/ijepa_ad/data/transforms.py):
  `PreprocessConfig(image_size=224, mean=IJEPA_MEAN, std=IJEPA_STD)`,
  `load_image(path)`, `to_rgb(image)`, `preprocess_image(path | PIL.Image, config)
  -> Tensor[3, S, S]`, and `preprocess_batch(images, config) -> Tensor[N, 3, S, S]`.
  Grayscale is replicated to three channels, images with transparency are
  alpha-composited onto black, and 16-bit or float modes raise
  `ImagePreprocessingError` instead of being clipped.

Encoder interface:

- [`ijepa_wrapper.py`](src/ijepa_ad/encoder/ijepa_wrapper.py):
  `IJEPAEncoder(checkpoint_path, official_repo_path, device)` loads the official
  ViT-g/16 target encoder, switches it to `eval()`, freezes every parameter, and
  exposes `encode(image_batch) -> Tensor[N, 196, 1408]`. Loading fails if the
  checkpoint is missing, lacks `target_encoder`, or does not exactly match the
  architecture.

Caching interfaces (`src/ijepa_ad/caching/`):

- [`cli.py`](src/ijepa_ad/caching/cli.py): `python -m ijepa_ad.caching.cli
  --category NAME [NAME ...] [--purpose nominal|test] [--dataset-root] [--cache-root]
  [--batch-size] [--device] [--val-fraction] [--seed] [--rebuild] [--mock-encoder]
  [--checkpoint] [--official-repo] [--encoder-name] [--summary-json]`;
  `CacheRunConfig`, `run_cache_generation(config, encoder=None) -> RunSummary`,
  `main(argv) -> int` (0 all complete, 1 some entries/samples failed, 2 bad
  configuration). `RunSummary.to_dict()` holds totals (`samples_discovered`,
  `skipped_compatible`, `encoded_new`, `recomputed`, `failed`), per-entry results,
  `failed_samples`, cache root, device, encoder name/hash, and `encoder_kind`
  (`mock`/`real`) with `real_checkpoint_inference`. Mock runs write to
  `<cache-root>/_mock`. `scripts/build_latent_cache.py` is a thin wrapper.

- [`builder.py`](src/ijepa_ad/caching/builder.py): `build_latent_cache(samples,
  encoder, *, cache_root, category, split, encoder_name, checkpoint_sha256,
  patch_size, embedding_dim, num_tokens=None, config=None, batch_size=8) ->
  BuildReport` (`encoded`, `reused`, `recomputed`, `failed`, `already_complete`);
  `build_nominal_latent_caches(train_samples, encoder, *, val_fraction, seed,
  ...)` (split, then cache `train` and `val`); `parts_dir(...)`. `encoder` is
  anything with `encode(Tensor[N,3,S,S]) -> Tensor[N,tokens,dim]` (`IJEPAEncoder`
  or a mock). Error: `CacheBuildError` (has `.failed` and the partial `.report`);
  encoder output with the wrong row count, token count, or dim raises it before saving.
  An `Exception` raised by `encoder.encode` fails that batch only: its samples are in
  `.failed`, nothing from it is saved, it is not retried, later batches still run, and
  the error is raised at the end. `KeyboardInterrupt`/`SystemExit` propagate.

- [`cache_schema.py`](src/ijepa_ad/caching/cache_schema.py): `SCHEMA_VERSION = 2`;
  frozen `CacheMetadata` (`to_json()`, `from_json()`, `from_dict()`),
  `PreprocessingSpec.from_config(PreprocessConfig)` (validated and normalized to float
  tuples on every construction path), and `CacheExpectations` (`mismatches(metadata)`;
  optional `sample_ids` and `num_tokens`). Errors: `CacheMetadataError`,
  `UnsupportedSchemaVersionError`.
- [`latent_cache.py`](src/ijepa_ad/caching/latent_cache.py):
  `latent_cache_path(root, category, split)`, `build_metadata(latents, *, category,
  split, sample_ids, encoder_name, checkpoint_sha256, patch_size, preprocessing)`,
  `save_latent_cache(path, latents, metadata)`, `load_latent_cache(path, expected)`,
  `try_load_latent_cache(path, expected)`, `compute_file_sha256(path)`, `cache_exists(path, expected)`. `LatentCache` offers
  `get(id)`, `select(ids)`, `index_of(id)`, and `in`. Errors:
  `CacheNotFoundError`, `CacheCorruptedError`, `CacheIncompatibleError` (all
  `CacheError`).

Predictor interfaces (`src/ijepa_ad/predictor/`):

- [`cache_dataset.py`](src/ijepa_ad/predictor/cache_dataset.py):
  `load_cached_latent_dataset(cache_root, expected, *, samples=None) ->
  CachedLatentDataset` loads `<cache_root>/<category>/<split>.pt` through
  `load_latent_cache` and optional `MVTecSample` annotations. `CachedLatentDataset`
  is indexable by integer/slice, exposes `sample_ids`, `latents`, `category`, `split`,
  and `select(image_ids) -> Tensor[N, tokens, dim]`. `CachedLatentSample` returns one
  sample (`latent` is `tokens x dim`, no batch axis). `annotations_by_image_id()`
  builds the optional label/mask join table, rejecting duplicate IDs.
  `collate_cached_latents(samples) -> CachedLatentBatch` stacks samples into
  `N x tokens x dim` and rejects empty or mixed category/split batches.
- [`losses.py`](src/ijepa_ad/predictor/losses.py):
  `cosine_prediction_loss(predicted, target, target_mask)`, `l2_prediction_loss(...)`,
  and `prediction_loss(..., loss_type)` compute losses only over selected target
  tokens. Inputs are `B x tokens x dim`; `target_mask` is `B x tokens`.
- [`transformer_predictor.py`](src/ijepa_ad/predictor/transformer_predictor.py):
  `PredictorConfig` validates architecture fields. `LatentTransformerPredictor`
  projects cached latents to a hidden dimension, replaces masked tokens with a learned
  mask token, adds learned positional embeddings, runs a Transformer encoder, and
  projects back to the latent dimension.
- [`training.py`](src/ijepa_ad/predictor/training.py):
  `TrainingConfig` (incl. `mask_strategy`, `block_size`, `block_probability`,
  `single_token_count`), `train_predictor(model, train_dataset, val_dataset, config,
  checkpoint_path=None) -> TrainingResult`, `build_random_target_mask(...)`,
  `build_target_mask(config, ...)` (strategy dispatch), and
  `main(argv)` for `scripts/train_predictor.py`. Training uses AdamW,
  random or inference-aligned target masks, cosine or L2 loss, validation loss tracking, optional early
  stopping, and checkpoint saving. The CLI seeds before predictor initialization;
  training requires distinct train/val cache splits, batches lazily, casts cached
  latents to the model dtype, and keeps validation masks fixed across epochs.
  Validation masks are per sample (`build_validation_target_mask`): each sample's
  generator is seeded by `sample_mask_seed(seed + 10000, image_id)` (SHA-256, so
  independent of batch size, order and `PYTHONHASHSEED`); training masks still use
  one shared generator. Validation losses from different mask strategies are not
  directly comparable.
  Checkpoints include cache provenance (encoder/checkpoint/preprocessing, latent
  dtype, and train/val sample IDs) and a `reproducibility` block (seeds, mask
  strategy, torch/python versions, git commit/dirty flag). A non-finite loss raises.
  `scripts/train_predictor.py` is a thin wrapper.
- [`target_masks.py`](src/ijepa_ad/predictor/target_masks.py):
  `build_inference_aligned_target_mask(batch, tokens, generator, block_size=4,
  block_probability=0.5, single_token_count=8)` draws, per row, either one of the
  `CoarseMasker` windows (exactly the coarse inference masks) or isolated
  non-adjacent single tokens (stand-ins for 1x1 fine masks). Selection only; the
  predictor overwrites target latents with its mask token, so no target leaks.
  `sample_mask_seed(base_seed, sample_id, epoch=None)` derives the per-sample
  validation seed: first 8 bytes of `SHA-256("ijepa-ad/validation-mask/v1|base_seed|
  epoch-or-'-'|sample_id")`, top bit cleared.
- [`inference.py`](src/ijepa_ad/inference.py):
  `load_predictor_checkpoint` (predictor only) wraps
  `load_predictor_checkpoint_with_provenance` -> `(predictor, {status, mask_strategy,
  training_config, reproducibility})`; fields a checkpoint lacks stay `None`
  (`status` = `complete` / `partial` / `legacy`). `image_score_upper_bound(mode, lam)`
  gives the maximum fused patch score (fixed `2`, dense `2*lam`, adaptive
  `2*(1+lam)`); `InferenceResult.image_score = raw_image_score / score_upper_bound`
  in [0, 1] (comparable only within one mode and lambda), where `raw_image_score` is
  `max(patch_scores)`. `evaluation/runner.py` writes `predictor_provenance`,
  `predictor_checkpoint` and `image_score_definition` into each category result.
- [`data/augmentation.py`](src/ijepa_ad/data/augmentation.py):
  `AugmentationConfig` (+ `fingerprint()`), `load_augmentation_config(path, category)`
  (defaults + `category_overrides`), `augment_image`, `materialize_augmented_records`
  (writes `<root>/<category>/train/good/<fingerprint>/*.png`, ids = file SHA-256).
- [`masking/score_maps.py`](src/ijepa_ad/masking/score_maps.py):
  `upsample_patch_scores` (raw bilinear, `align_corners=False`; used by evaluation)
  and `patch_scores_to_heatmap(normalize=True)` (same interpolation + display-only
  min-max; used by the UI).

Latent cache schema (version 2; v1 lacked `preprocessing.version`). Layout: `<cache_root>/<category>/<split>.pt`, where
split is `train`, `val` or `test`; default root `data/cache/latents` (gitignored via
`/data/`). The file holds exactly `latents` (`N×tokens×dim` tensor) and
`metadata_json` (canonical JSON of):

| field | meaning |
| --- | --- |
| `schema_version` | integer, must equal 2 |
| `category`, `split` | MVTec category and split name |
| `sample_ids` | ordered, unique SHA-256 `image_id`s, one per latent row |
| `encoder_name`, `patch_size` | encoder identifier (`configs/encoder/vitg16.yaml`) and patch size |
| `checkpoint_sha256` | SHA-256 of the checkpoint file |
| `input_shape` | `[3, S, S]` fed to the encoder |
| `preprocessing` | `image_size`, `mean`, `std`, `resize`, `crop`, `version` (`PREPROCESSING_VERSION`) |
| `latent_shape`, `latent_dtype`, `num_tokens`, `embedding_dim` | tensor description (`float16`/`bfloat16`/`float32`) |
| `latents_sha256` | SHA-256 of the latent bytes (integrity) |
| `created_at_utc`, `torch_version` | provenance |

Invalidation: an entry is reusable only if schema version, category, split, encoder
name, checkpoint hash, patch size, embedding dim, and the full preprocessing spec
(plus `sample_ids` and `num_tokens`, when the caller passes them) equal the caller's
`CacheExpectations`. Otherwise `CacheIncompatibleError`; `try_load_latent_cache`
returns `None` so the caller rebuilds. A checksum, shape, dtype, metadata, or
non-finite-value failure is `CacheCorruptedError` and is never swallowed.

### Downstream consumption contract (Person 2)

Status: **implemented through CPU-tested training mechanics**. The original stand-in
contract test still exists (`tests/integration/test_person2_cache_consumer.py`), real
Person 2 adapter code lives in `predictor.cache_dataset`, and predictor/loss/training
code lives under `predictor/`. Real GPU training on real caches has not been run.

What a consumer provides: the cache root, a category, a split (`train`, `val` or
`test`), and a `CacheExpectations` describing the run that produced the entry
(`encoder_name`, `checkpoint_sha256`, `patch_size`, `embedding_dim`, `preprocessing`
via `PreprocessingSpec.from_config(PreprocessConfig())`, optionally the ordered
`sample_ids`). For the real encoder these are `ijepa_vit_g16_in22k_ep44`, the
checkpoint hash recorded in the run's summary JSON, 16, 1408, and the default
`PreprocessConfig`. Pass `sample_ids` when exact membership and order matter.

```python
path = latent_cache_path(cache_root, "bottle", "train")      # <root>/<cat>/<split>.pt
cache = load_latent_cache(path, expected)                    # -> LatentCache (CPU)
cache.latents          # Tensor[N, 196, 1408]   the whole entry (a batch of N)
cache.metadata         # CacheMetadata: category, split, sample_ids, dtype, hashes
cache.get(image_id)    # Tensor[196, 1408]      ONE sample, no batch dimension
cache.select(ids)      # Tensor[len(ids), 196, 1408]   a batch, in the order asked
cache.index_of(id)     # row number;  `id in cache` -> bool

from ijepa_ad.predictor import load_cached_latent_dataset, collate_cached_latents

dataset = load_cached_latent_dataset(cache_root, expected, samples=test.samples)
dataset[0].latent      # Tensor[196, 1408]
dataset[0].annotation  # optional SampleAnnotation(label, mask_path, image_path)
batch = collate_cached_latents([dataset[2], dataset[0]])
batch.latents          # Tensor[2, 196, 1408], in requested order
```

- Sample identity: `image_id`, the SHA-256 of the image bytes (the same value as
  `MVTecSample.image_id`). Row `i` of `latents` belongs to `metadata.sample_ids[i]`.
- Category and split live in the metadata and the path, and `expected` is compared with
  the metadata, so a file renamed into another category/split is
  `CacheIncompatibleError`. `train`/`val` come from the nominal split of `train/good`;
  `test` is the official test set from a separate run (`--purpose test`).
- Labels and masks are **not** in the cache. For `test`, pass the official
  `MVTecSample` records to `load_cached_latent_dataset(..., samples=test.samples)` to
  join `image_id` to `label`/`mask_path`.
- Dtype and device: the dtype the encoder returned at build time
  (`metadata.latent_dtype`; production dtype undecided), loaded on CPU. Move or cast in
  the consumer. `select` returns a copy; `get` and `.latents` share memory with the
  loaded object, so treat them as read-only or `.clone()`. In-memory edits never reach
  the file.
- Outcomes: missing file → `CacheNotFoundError` (also a `FileNotFoundError`) from
  `load_latent_cache`, `None` from `try_load_latent_cache`, `False` from `cache_exists`.
  Built for another configuration or schema → `CacheIncompatibleError` (`None`/`False`
  from the soft variants). Truncated, tampered, or inconsistent → `CacheCorruptedError`,
  which the soft variants also raise. Unknown sample id → `KeyError`. Unsafe
  category/split name → `ValueError`. `load_latent_cache(path)` without `expected` still
  validates the file but cannot know it is the right one; pass `expected` whenever
  latents are reused.
- Consumers never need the `.<split>.parts/` shards (removed once an entry is complete),
  the preprocessing code, or the encoder.

## 3. Runtime Flow

### Notebook predictor workflows
1. `notebooks/01_train_predictor.ipynb` loads predictor settings, validates each MVTec category, and reuses or builds nominal `train`/`val` latent caches through `run_cache_generation`.
2. It constructs `LatentTransformerPredictor` from cache dimensions and config architecture fields, then calls `train_predictor` with early stopping and writes the best model to `checkpoints/predictors/<category>_<loss>.pt`.
3. `notebooks/02_augment_and_train_predictor.ipynb` splits original nominal training images first (using `nominal_split` from `configs/data/mvtec.yaml`, independent of the augmentation seed), augments only the training side in image space via `materialize_augmented_records` (fingerprinted directory, content-hash ids, capsule never mirrored), encodes the resulting records into `data/cache/latents_augmented`, and writes augmented-run checkpoints under `checkpoints/predictors_augmented`.
4. Both checkpoint payloads retain model configuration, training configuration, a `reproducibility` block, loss history, selected epoch, and train/validation cache provenance; evaluation copies the training and reproducibility metadata into its result JSON. The official test split is evaluation-only.
Before training or inference, an entrypoint calls `set_seed(seed)` once. Runtime
blocks can be wrapped as `with profile_peak_vram("stage"):` and all run output is
sent through the logger returned by `setup_logging()`.

Implemented data flow for one category (verified on synthetic data in
`tests/integration/test_data_pipeline.py` and `test_latent_cache_pipeline.py`
with mock encoders):

1. `train, test = validate_category_splits(root, category)`
2. `split = split_nominal_train_val(train, val_fraction=0.1, seed=0)`
3. `batch = preprocess_batch([sample.image_path for sample in split.train])`
4. `latents = IJEPAEncoder(...).encode(batch)` → `[N, 196, 1408]`
5. `meta = build_metadata(latents, category=..., split=..., sample_ids=[s.image_id
   ...], encoder_name=..., checkpoint_sha256=compute_file_sha256(ckpt),
   patch_size=16, preprocessing=PreprocessingSpec.from_config(config))`, then
   `save_latent_cache(latent_cache_path(root, category, split), latents, meta)`
6. Later runs: `try_load_latent_cache(path, CacheExpectations(...))`; `None` means
   rebuild from step 3.

Steps 3–6 are wrapped by `build_nominal_latent_caches(train.samples, encoder,
val_fraction=0.1, seed=0, cache_root=..., category=..., encoder_name=...,
checkpoint_sha256=..., patch_size=16, embedding_dim=1408)`. Per batch it
preprocesses each sample, encodes, and writes one shard per sample; a rerun skips
compatible shards; the final `<split>.pt` is written only when every sample has a
shard. An unreadable image raises `CacheBuildError` (good shards kept, no final
entry). Only mock encoders have been run; no real-encoder cache exists.

The CLI performs the same flow per category: `validate_category_splits`, then for
`--purpose nominal` `split_nominal_train_val` and caches `train` and `val`; for
`--purpose test` only the official test records, as a separate `test` entry. Samples
that cannot be read, and entries that are unusable (e.g. a corrupted final file), are
recorded in the summary and the run moves on; a corrupted final entry is left
untouched until `--rebuild`. An exception from the encoder fails only its batch (its
samples are listed as failed, finished shards are kept, rerun to resume). A
`val_fraction` outside (0, 1) is a configuration error (exit code 2). Only the mock
encoder has been run.

Every entry point re-validates records before looking at the cache: `build_latent_cache`
calls `validate_training_records` / `validate_test_records` first, and the CLI calls
`validate_category_splits` first, so an existing cache never bypasses dataset or leakage
checks. Reuse then requires an exact match of the metadata (see Invalidation); a cache
file copied to another category or split, or a shard copied under another sample's
name, is incompatible and is rebuilt. Each encoder output batch must have the right
rows/tokens/dim and be finite before any of its shards are written (`CacheBuildError`).
Row order in an entry is the order of the records given (split output is path-sorted).

Person 2 training flow after real `train` and `val` caches exist:

1. Build `CacheExpectations` for the category/split/checkpoint/preprocessing.
2. Optionally discover records (`train, test = validate_category_splits(...)`) when
   labels or masks must be joined.
3. `dataset = load_cached_latent_dataset(cache_root, expected, samples=...)`.
4. Construct `LatentTransformerPredictor(PredictorConfig(...))`.
5. Run `train_predictor(model, train_dataset, val_dataset, TrainingConfig(...),
   checkpoint_path=...)`, or use `scripts/train_predictor.py`.
6. Training samples target masks via `build_target_mask` (`inference_aligned` in
   `configs/predictor/base.yaml`: one coarse window or isolated tokens; the legacy
   `random` strategy is the dataclass default), predicts all tokens, computes loss only on
   masked tokens, validates on `val`, restores the best state, and writes a checkpoint
   if requested. Predictor code does not call the encoder or parse cache `.pt` payloads
   directly.

Test defects remain evaluation-only and never participate in model selection.

For training-time caching, nominal images flow through `IJEPAEncoder.encode()` once
and produce `[N, 196, 1408]` latent tensors. The first ViT-g/16 run must use batch
size 1, inference mode, and `profile_peak_vram()` to establish whether the RTX 4060
8 GB target is viable.

## 4. Divergences from the original PRD
The initial scaffold uses `requirements-lock.txt` as the permitted lockfile variant
from §1 and keeps modeling dependencies empty. Foundational utilities use lazy
PyTorch loading so CPU-only tooling remains importable. See `DECISIONS.md`,
2026-08-20.

The official I-JEPA repository has only a training transform (`RandomResizedCrop`).
Preprocessing here uses a deterministic full-image resize instead, so no border
region is cropped away. See `DECISIONS.md`, 2026-10-10 "Deterministic I-JEPA
preprocessing".

Predictor training masks follow the inference masks (coarse 4x4 windows and single
tokens) instead of scattered 40% masks; the PRD does not specify training masks. See
`DECISIONS.md`, 2026-10-17 "Inference-aligned predictor training masks". The display
heatmap is bilinear like the PRD; only its per-image min-max scaling is display-only
(`DECISIONS.md`, 2026-10-17 "One bilinear upsampling rule").

## 5. Frontend and Visualization Architecture (Person 4)

Person 4 delivers the user-facing web dashboard and visualization layer (`frontend/`).

```text
User Image / MVTec Sample
        │
        ▼
frontend.app (Streamlit Dashboard)
        │
        ├──> sidebar: Category, Method (Adaptive/Dense/Fixed/PatchCore/MAE), Threshold, Opacity
        │
        ▼
frontend.services.backend_interface.run_inference(...)
        │
        ├──> Calls src.ijepa_ad.masking.adaptive_engine / baselines
        │    (or contract-compliant synthetic mock when checkpoints are unmounted)
        │
        ▼
Returns Result Schema:
{
    "image_score": float,   # in [0, 1]; max patch score / mode-and-lambda maximum
    "pixel_heatmap": ndarray [H, W],
    "overlay": PIL.Image | ndarray,
    "detected_regions": list[dict],
    "mask_count": int,
    "latency_ms": float,
    "peak_vram_mb": float | None,
    "method": str,
    "category": str | None,
}
        │
        ▼
frontend.components.visualizer
        ├──> create_heatmap_overlay (continuous colormap alpha blending)
        └──> draw_detected_regions (bounding box localization)
        │
        ▼
frontend.services.experiment_service
        ├──> load_training_history (Cosine and L2 loss curves)
        └──> load_benchmark_summary (AUROC, mask reduction, latency comparisons)
```
