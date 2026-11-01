# DECISIONS.md
_Append-only log of design/implementation decisions for I-JEPA-AD._
_Entry format and rules for what belongs here: see AGENT.md § Mandatory living documentation._
_Do not edit or remove past entries — if a decision is later reversed, add a new entry that supersedes it and links back._

---

## [2026-08-20] Establish initial repository toolchain
**Context:** §1 and §4.4 of the implementation plan require a reproducible Python
package layout, automated quality checks, and CPU-only tests before modeling work.
**Options considered:** Keep dependencies implicit; use a requirements-only layout;
or define package and development tooling in `pyproject.toml` with a pinned tool
lockfile.
**Decision:** Use a `src/` package layout, `pyproject.toml` for setuptools plus
Ruff, Black, mypy, and pytest configuration, `requirements-lock.txt` for exact
development-tool versions, and GitHub Actions/pre-commit for the same checks.
**Rationale:** This keeps local and CI commands aligned while leaving runtime and
modeling dependencies empty until their implementation phases.
**Baseline / prior state:** Documentation-only repository with no Python package or
automated checks.
**Outcome after change:** Empty package scaffold and smoke test are present; local
tooling verification is recorded in `TRANSFER.md` after the checks run.

## [2026-08-20] Centralize deterministic seeding and observability
**Context:** Weeks 1–2 require reproducibility controls and structured resource
logging before data and model modules are added.
**Options considered:** Let each future script seed its own libraries; configure
logging ad hoc per script; or expose shared utility entrypoints for seeding,
structured logs, and CUDA memory measurement.
**Decision:** Use one `set_seed(seed)` entrypoint for Python, NumPy, PyTorch, and
CUDA determinism; `setup_logging()` for idempotent JSONL logging; and
`profile_peak_vram()` around measurable code blocks using
`torch.cuda.max_memory_allocated()`.
**Rationale:** Centralized controls prevent inconsistent experiment state and make
peak VRAM and run events observable without coupling future modules together. Lazy
PyTorch loading keeps CPU-only configuration and test environments importable.
**Baseline / prior state:** Utility package contained only an empty initializer.
**Outcome after change:** All three utilities have unit tests; CPU test environments
report zero CUDA memory safely, while CUDA environments use PyTorch's peak allocator
counter. Further integration with training and inference is pending Phase 1+.

## [2026-08-20] Keep official MVTec test data evaluation-only
**Context:** MVTec AD ingestion needs strict train/test boundaries and a future
nominal validation strategy without leaking defect labels into model selection.
**Options considered:** Randomly split all images including test defects; use the
official test split for validation; or hold out validation samples from nominal
training images and reserve the official test split for final evaluation.
**Decision:** Discover training only from `train/good`, discover mixed labels from
the official `test` tree, require each defect image's matching
`ground_truth/<label>/<stem>_mask.*`, and reserve future validation as a nominal-only
holdout from training. Detect cross-split leakage using SHA-256 content IDs.
**Rationale:** This follows the MVTec AD protocol, keeps defect labels out of
hyperparameter selection, and catches copied images even when filenames are reused
between official directories.
**Baseline / prior state:** No data discovery or split validation existed.
**Outcome after change:** Synthetic fixtures verify nominal-only training, 2:2
nominal/defect test counts, mask pairing, non-nominal directory rejection, and
cross-split duplicate-content rejection. Future validation split implementation is
pending the predictor phase.

## [2026-08-20] Adopt Meta official I-JEPA ViT-g/16 checkpoint
**Context:** The PRD requested an official ViT-B/16 checkpoint, but Meta's official
I-JEPA release provides no pretrained ViT-B/16 weights. The user authorized the
official ViT-g/16 alternative.
**Options considered:** Train or source a non-official ViT-B/16 checkpoint; use
Meta's official ViT-g/16 checkpoint; or stop until an official ViT-B/16 release
exists.
**Decision:** Use Meta's official ImageNet-22K ViT-g/16 44-epoch checkpoint at
`https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar`, with the
official `in22k_vitg16_ep44.yaml` architecture settings: 224px input, 16px patch,
196 tokens, and embedding dimension 1408.
**Rationale:** This is the only available official Meta checkpoint that matches the
project's 224px/16px/196-token geometry. The tradeoff is substantially higher VRAM
usage and a changed latent dimension; the first run must profile memory on the
RTX 4060 before pipeline commitment.
**Baseline / prior state:** Encoder implementation was blocked on a nonexistent
official ViT-B/16 checkpoint.
**Outcome after change:** The project configuration and wrapper contract now target
ViT-g/16; production construction remains pending a local checkout of Meta's
archived source repository and a memory smoke test.

## [2026-10-10] Deterministic I-JEPA preprocessing
**Context:** Phase 1 needs one preprocessing path that turns MVTec images into
`N×3×224×224` tensors for the frozen ViT-g/16 encoder.
**Options considered:** Reuse Meta's training transform (`RandomResizedCrop`,
scale 0.3–1.0); `Resize(256)` + `CenterCrop(224)` as in many ImageNet eval and
PatchCore setups; or a direct full-image resize to 224×224.
**Decision:** Decode with Pillow, convert to RGB, bilinear-resize the full image to
224×224 with no crop, scale to `[0, 1]`, then normalize with mean
`(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. Use the same transform for
train, validation, and test. Grayscale is replicated to 3 channels; images with
transparency are alpha-composited onto black; 16-bit and float modes are rejected.
Implemented in `src/ijepa_ad/data/transforms.py`.
**Rationale:** Normalization, `[0, 1]` scaling, 224px size, and bilinear interpolation
are verified against Meta's official `facebookresearch/ijepa` `src/transforms.py`
(`make_transforms` defaults) and `configs/in22k_vitg16_ep44.yaml` (flip, color
distortion, and blur all disabled). The official repo ships no deterministic eval
transform, so the resize policy is a project choice: random crops would make
anomaly scores non-deterministic, and center-cropping would discard border regions
where MVTec defects can occur.
**Baseline / prior state:** No preprocessing existed; `MVTecADDataset` returns path
metadata only.
**Outcome after change:** 16 unit tests cover shapes, dtype, exact normalized values,
grayscale, RGBA, palette transparency, no-crop borders, determinism, and error paths.
Not numerically cross-checked against torchvision (not installed). Pillow added as a
runtime dependency because torch alone cannot decode PNG/JPEG.

## [2026-10-10] Nominal-only train/validation split
**Context:** Predictor training needs a validation set for early stopping that never
contains defect images (IMPLEMENTATION_PLAN §3 Weeks 5–6, §5).
**Options considered:** Use `set_seed()` plus global `random.shuffle`; shuffle in
filesystem order with a local RNG; or sort by content hash and shuffle with an
isolated seeded RNG.
**Decision:** `split_nominal_train_val(samples, *, val_fraction, seed, min_train=1,
min_val=1)` in `src/ijepa_ad/data/splits.py`. It accepts only records that pass
`validate_training_records`, holds out `round(n * val_fraction)` samples (ties to
even), orders by SHA-256 `image_id` before shuffling with
`ijepa_ad.utils.seeding.seeded_rng(seed)`, and raises
`InsufficientNominalSamplesError` when either side would fall below its minimum.
`seeded_rng` is a new public helper sharing `set_seed`'s seed validation.
**Rationale:** Calling `set_seed()` inside a data function would reset the caller's
global RNG and torch determinism flags as a side effect. Ordering by content hash
makes the split depend only on image content and seed, not on paths or directory
listing order. Reusing the training validator guarantees no defect, masked, or
test-split record can enter train or validation.
**Baseline / prior state:** No validation split existed; the 2026-08-20 entry
reserved it as future nominal-only holdout work.
**Outcome after change:** 12 unit tests cover nominal-only output, disjointness,
completeness, seed reproducibility, input-order independence, global-state
isolation, rejection of defect/masked/test records, insufficient-sample errors, and
invalid arguments. Default `val_fraction: 0.1`, `seed: 0` recorded in
`configs/data/mvtec.yaml`.

## [2026-10-10] Latent cache format and schema
**Context:** Phase 2 needs a way to store frozen-encoder outputs (`N×196×1408`) so the
predictor never re-runs ViT-g/16, and to know when a stored cache is stale. Real
cache generation happens later on an external GPU machine; this phase is
CPU/synthetic only.
**Options considered:** `torch.save` of a plain dict with a pickled metadata object;
`safetensors` (new dependency); NumPy `.npz`; or `torch.save` of tensors and strings
only, with metadata as a canonical JSON string.
**Decision:** One `.pt` file per (category, split) at `<root>/<category>/<split>.pt`,
containing exactly `{"latents": Tensor, "metadata_json": str}`, loaded with
`torch.load(weights_only=True)`. Metadata is the frozen dataclass `CacheMetadata`
(`schema_version=1`) in `src/ijepa_ad/caching/cache_schema.py`, serialized as sorted,
compact JSON and fully re-validated on load (unknown or missing keys rejected, bool
is not an int, shape/dtype/token-count cross-checked, unsupported version raises
`UnsupportedSchemaVersionError`). It records category, split, ordered SHA-256 sample
ids, encoder name, checkpoint SHA-256, patch size, input shape, a `PreprocessingSpec`
(image size, mean, std, resize, crop), latent shape/dtype, token count, embedding
dim, a SHA-256 of the latent bytes, UTC creation time, and torch version. Reuse goes
through `CacheExpectations`; any difference raises `CacheIncompatibleError`.
`try_load_latent_cache` maps missing/stale to `None` but lets corruption raise.
Writes use a same-directory temp file, `fsync`, then `os.replace`. Category and split
must be plain names so they cannot escape the cache root.
**Rationale:** Tensors plus strings is the whole payload, so `weights_only=True`
works and no arbitrary object is ever unpickled; no dependency is added. The content
checksum catches truncation and bit rot that `torch.load` might not. Keeping the
metadata in the same file as the tensor means they cannot drift apart, and atomic
replace means a crash leaves either the old entry or none. One file per
category/split matches how the predictor consumes data (MVTec categories are a few
hundred images). Hostname, user, and absolute paths are deliberately not recorded.
**Baseline / prior state:** none — new (`caching/` held only an empty `__init__.py`).
**Outcome after change:** 37 unit tests in `tests/unit/test_latent_cache.py` pass on
CPU with small synthetic tensors (round trip, dtypes, bad/missing fields, corruption,
tampering, each invalidation rule, atomic-write failure). Not verified: real
checkpoint hashing, real ViT-g/16 outputs, real MVTec images, GPU. Latents are
stored in the dtype the caller passes (float32/float16/bfloat16); choosing the
production dtype is pending the VRAM spike.

## [2026-10-10] Resumable cache generation and schema version 2
**Context:** Phase 3 connects the data pipeline to the cache. A split-level entry held
whole in memory cannot be resumed after a crash, a failed sample must never shift
latents onto the wrong sample, and the schema did not record a preprocessing version.
Supersedes the schema-version-1 details of the 2026-10-10 "Latent cache format and
schema" entry; the file layout, payload keys, safe loading, and atomic write are
unchanged.
**Options considered:** Shard files per batch (fragile id association); one per-sample
file as the only layout (changes the decided layout); or per-sample *shards* that are
themselves valid one-sample cache entries, assembled into the unchanged
`<category>/<split>.pt`.
**Decision:** (1) `caching/builder.py::build_latent_cache` preprocesses each sample
individually, encodes in batches, and writes every sample's latents to
`<root>/<category>/.<split>.parts/<image_id>.pt` (a one-sample entry saved with
`save_latent_cache`, so it carries and is checked against full metadata). Compatible
shards are skipped on resume; missing, incompatible, or corrupted shards are
recomputed (shards are derived data). When every sample has a shard they are
assembled in sample order into the final entry, then the parts directory is removed.
A compatible final entry short-circuits everything; a corrupted final entry raises
and is never overwritten silently. (2) Any sample that cannot be preprocessed is
recorded and raises `CacheBuildError` after the remaining samples are processed; good
shards are kept and no final entry is written from partial results. An encoder that
returns the wrong number of rows raises before anything is saved. (3) `train`/`val`
entries accept only records passing `validate_training_records` (`train/good`,
unmasked, `split == "train"`); `build_nominal_latent_caches` goes through
`split_nominal_train_val`. (4) `PreprocessingSpec` gains `version`
(`transforms.PREPROCESSING_VERSION = 1`); `SCHEMA_VERSION` is now 2. Bump the
preprocessing version whenever `preprocess_image` changes output for the same config.
(5) `LatentCache` gains `get/select/index_of/in` for lookup by `image_id`, and
`cache_exists(path, expected)` fully validates before answering. Cache identity is
category + split + ordered sample ids (content hashes) + encoder name + checkpoint
SHA-256 + patch size + embedding dim + preprocessing spec + schema version; no
absolute path is part of it.
**Rationale:** Reusing `save_latent_cache` and `CacheExpectations` for shards avoids a
second format and gives resume the same checks as reuse. Writing shards per batch
bounds memory during encoding. Failing loudly on a bad sample keeps split-level
entries aligned with the dataset instead of silently shorter.
**Baseline / prior state:** Schema v1 (no preprocessing version); nothing called
`save_latent_cache` outside tests; no lookup by id; no resume.
**Outcome after change:** New unit tests (lookup, `cache_exists`, preprocessing version)
and 11 integration tests (mock encoder, synthetic MVTec tree, CPU) cover association,
resume, corrupt shards, invalidation, failed samples, and nominal-only enforcement. No
v1 caches existed, so nothing is migrated: v1 files are rejected as incompatible. Known
limits: assembly still builds the whole split tensor in host RAM (about 0.55 MB per
sample in float16); latents from different batch compositions can differ by float
rounding (not bit-identical); real checkpoint and GPU runs are unverified.

## [2026-10-10] Cache-generation CLI, run summary, and failure policy
**Context:** Phase 4 asks for a configurable, resumable generation pipeline with a
structured run summary, usable on an external GPU host. Phase 3's `caching/builder.py`
already is the orchestrator (batching, per-sample shards, resume, invalidation), so
only the gaps below were filled; no second orchestrator was created.
**Options considered:** A new `cache_builder.py` duplicating the builder; Hydra/
OmegaConf for configuration; per-sample retry when the encoder raises; or a thin CLI
and summary layer over the existing builder with argparse.
**Decision:** (1) `BuildReport` gains `recomputed` (shards that existed but were
corrupt or incompatible) and `failed`; `CacheBuildError` carries the partial `report`.
`build_latent_cache` gains `num_tokens`, checked with `embedding_dim` on every encoder
output before anything is written; a wrong shape raises `CacheBuildError`. (2) New
`caching/cli.py`: `CacheRunConfig`, `run_cache_generation(config, encoder=None) ->
RunSummary`, `main(argv)` (`python -m ijepa_ad.caching.cli`, thin wrapper
`scripts/build_latent_cache.py`). Plain argparse; Hydra stays unwired (no new
dependency). Defaults mirror `configs/data/mvtec.yaml` and `configs/encoder/vitg16.yaml`.
(3) Purposes: `nominal` caches `train`+`val` from `train/good` via
`split_nominal_train_val`; `test` caches only the official test split, as a separate
entry. Both go through `validate_category_splits`. (4) Failure policy: unreadable
samples and unusable final entries are recorded in the summary and the run continues
with the next entry; exit code 1 if any failed, 2 for bad configuration. (Superseded
2026-10-12: an exception raised by the encoder now fails only its batch, see "PR review
fixes".) (5) The
mock encoder (`--mock-encoder`) is CPU-only, writes under `<cache-root>/_mock`, uses
the identity `mock_encoder` with a seed-derived checkpoint hash (2026-10-12; it was
the fixed `0*64`), and the summary reports
`encoder_kind: "mock"`, `real_checkpoint_inference: false`. (6) `--rebuild` deletes the
selected entries and shards before building; resume is otherwise automatic. (7) The
real encoder's input batches are moved to `--device` by a small adapter; the checkpoint
is hashed once per run.
**Rationale:** The mock cache directory cannot shadow or overwrite an expensive real
cache. Retrying per sample would hide systematic failures and double GPU time, while
resume already gives safe recovery. Continuing past a bad sample or entry gives one
complete report per run instead of one error at a time.
**Baseline / prior state:** Phase 3 builder and tests only; no entry point, no
recomputed/failed accounting, no token-count check, no device handling.
**Outcome after change:** 118 tests pass on CPU (102 before): 5 more builder tests and
11 CLI/summary tests (summary totals, resume counts, recomputed shards, nominal-only
and test-only purposes, `--rebuild`, config errors, exit codes). A mock CLI run on a
synthetic tree produced `[5, 196, 1408]`. Not verified: real checkpoint, GPU, real MVTec.

## [2026-10-10] Phase 5 end-to-end validation and two integration fixes
**Context:** Phase 5 asks whether dataset → validation → nominal split → preprocessing
→ encoder interface → cache → reload works as one system, and whether leakage,
stale-cache reuse, or misassociation is possible. Verified against the code, not the
earlier docs. No existing decision is reversed; the split policy, preprocessing, cache
schema (v2), and failure policy are unchanged.
**Options considered:** Only add tests; fix defects found by the tests with a regression
test written first; or also redesign the failure policy (per-sample retry, catching all
exceptions in the CLI).
**Decision:** (1) `MVTecADDataset` raises `ValueError` for any split other than `train`
or `test`. Before, an unknown split such as `"val"` fell through to the test-tree
branch, and with `validate=False` returned official test images labelled `val`; a
validation set must never be readable from `test/`. (2) `build_latent_cache` checks
that every encoder output batch is finite and raises `CacheBuildError` before any shard
of that batch is written. Before, `save_latent_cache` raised a bare `CacheMetadataError`
partway through the batch, which the CLI did not catch (traceback, no summary). It now
follows the existing wrong-shape policy: the CLI records the entry as failed and moves
on; shards from earlier batches stay and are valid. (3) Confirmed, unchanged: duplicate
image content inside `train/good` is rejected, not deduplicated (`duplicate image ID`),
so one bad category stops that category's run. (4) Confirmed, unchanged: cache rows
follow the order of the records passed in (split output is path-sorted), so the same
images under different filenames give a different row order and are therefore rebuilt,
never misread. (5) Reuse always re-runs record validation first; an existing cache never
bypasses dataset or leakage checks.
**Rationale:** Both defects were found by probing, then pinned by a failing regression
test before the fix. The fixes are minimal and keep existing behaviour for valid input.
**Baseline / prior state:** 118 tests passed; the two behaviours above were untested.
**Outcome after change:** 137 tests pass on CPU: 18 in
`tests/integration/test_pipeline_end_to_end.py` and 1 in `test_mvtec_dataset.py`. Covered:
full pipeline with association checks, order/filename-independent split membership,
duplicate and leaked images, reuse after dataset invalidation, global RNG and source
files untouched, cache files moved across category/split, shards copied under another
sample's name, failed atomic replace, truncated and tampered entries, reordered ids,
changed dim/seed, repeat-run identity, batch-size tolerance. Synthetic data and a mock
encoder only; no real MVTec, checkpoint, or GPU.

## [2026-10-10] Phase 6 handoff: document and test the existing cache API, no new format
**Context:** Person 2 needs to read cached embeddings without knowing the data
pipeline. Checked against the code: `load_latent_cache`, `CacheExpectations` and
`LatentCache` already give one validated consumption path, and `predictor/` is empty, so
Person 2's actual interface is not in the repository.
**Options considered:** Add a facade or re-exports in `caching/__init__.py`; add a
consumer-specific loader (dataset-style, batching, label join); or document the existing
API and pin it with a stand-in-consumer test.
**Decision:** Change no source. Document the contract in `FLOW.md` (what the consumer
provides, one sample `[196, 1408]` via `get` versus a batch `[N, 196, 1408]` via
`select` or `.latents`, the error or status for each failure, labels not cached) and add
`tests/integration/test_person2_cache_consumer.py` (13 tests). Add
`docs/GPU_VALIDATION_CHECKLIST.md`, an unexecuted procedure for the GPU host. These are
**proposals needing Person 2's agreement, not decisions**: re-export the consumer names
from `ijepa_ad.caching`; a helper that builds `CacheExpectations` from a run's summary
JSON so consumers need not repeat encoder name and hash; a storage-dtype option for the
CLI once float16 versus float32 has been measured.
**Rationale:** The API already satisfies the contract, and a second loader would
duplicate validation that must not be bypassed. Re-exports and helpers are cheap to add
once the consumer's needs are known and awkward to remove once relied on.
**Baseline / prior state:** 137 tests; the contract was implicit in `FLOW.md` §2/§3 and
in test code; single versus batch return shapes and the absence of labels in the cache
were not stated.
**Outcome after change:** 150 tests pass on CPU. Verified with synthetic data, the mock
encoder and a stand-in consumer: shape/dtype/metadata round trip, id-to-latent
association against an independent re-encode, train/val/test separation, ten kinds of
incompatibility rejected, missing and corrupted entries, unsafe names, and the
`[N, 196, 1408]` geometry (mock latents). No defect was found in the cache or split
code. Not verified: Person 2's code, the real checkpoint, GPU, real MVTec. Noted for the
GPU host: `IJEPAEncoder._load_checkpoint` uses torch's default `weights_only` (the
runbook and spike pass `False`); not changed because it cannot be tested without the
checkpoint.

## [2026-10-11] Person 2 cache-backed predictor dataset
**Context:** The handoff said Person 2's real code did not exist; only a stand-in
consumer test documented how cached latents should be read. The next role needs a
stable predictor-facing API before the Transformer predictor and training loop are
implemented.
**Options considered:** Keep using `LatentCache` directly in future predictor code;
add re-exports only; or add a small dataset-style adapter in `predictor/` that loads
through the public cache API and joins optional labels/masks by image id.
**Decision:** Add `src/ijepa_ad/predictor/cache_dataset.py` with
`load_cached_latent_dataset`, `CachedLatentDataset`, per-sample and batch dataclasses,
`annotations_by_image_id`, and `collate_cached_latents`. The adapter keeps cache
validation in `load_latent_cache`, preserves cache row order, returns one latent as
`tokens x dim`, stacks batches as `N x tokens x dim`, and treats labels/masks as an
optional join from `MVTecSample` records rather than adding them to cache files.
**Rationale:** This gives Person 2 real code without changing the cache schema or
duplicating integrity checks. Keeping annotations outside the cache preserves the
existing producer/consumer boundary and supports train/val nominal-only use as well as
test-time label/mask joins.
**Baseline / prior state:** `predictor/` only contained an empty `__init__.py`; Person 2
was represented by `tests/integration/test_person2_cache_consumer.py` and prose in
`FLOW.md`.
**Outcome after change:** 157 tests pass on CPU. Added 7 unit tests for the new adapter:
load/order/shape, id-ordered selection, optional annotation join, missing annotation
failure, duplicate annotation failure, collate ordering, and empty/mixed-split collate
failure. Existing 13 Person 2 contract tests still pass. Not implemented yet:
Transformer predictor, losses, training loop, masking, baselines, metrics, real MVTec,
GPU, or real checkpoint inference.

## [2026-10-11] Complete Person 2 predictor training path without GPU execution
**Context:** Person 2 needed to be finished as far as possible before GPU access is
available, so the later GPU session only has to generate real caches and run training.
The cache-backed dataset existed, but the predictor model, losses, training loop, CLI,
and overfit sanity test were still missing.
**Options considered:** Wait for real GPU caches before writing training code; build a
notebook-only prototype; or implement the reusable library path and verify it on tiny
synthetic cached latents on CPU.
**Decision:** Add `losses.py`, `transformer_predictor.py`, `training.py`, and
`scripts/train_predictor.py`. The predictor is a lightweight Transformer encoder over
cached latent tokens with learned mask and positional embeddings. Training samples
random target masks, optimizes cosine or L2 loss only on masked tokens, validates on
the nominal validation cache, tracks best validation loss, and optionally saves a
checkpoint containing model config, training config, history, and best epoch. The CLI
loads `train`/`val` cache entries using `CacheExpectations` and writes the checkpoint
to a caller-provided path.
**Rationale:** CPU synthetic cache tests can verify the training mechanics now without
pretending real MVTec/I-JEPA numbers exist. Keeping this as source modules plus a thin
script makes the later GPU run a command execution task instead of new engineering.
The training loop uses a small internal batch iterator instead of `torch.utils.data`
because, in this Windows/Python 3.13 environment, mypy followed that import path into
NumPy stubs that are incompatible with the repo's configured Python 3.11 target.
**Baseline / prior state:** Person 2 could load and batch cached latents, but there was
no model, loss, optimizer loop, checkpoint output, or overfit test.
**Outcome after change:** 165 tests pass on CPU. Added 8 predictor tests: hand-computed
cosine/L2 loss checks, predictor shape/input validation, deterministic random mask
checks, and a tiny synthetic cached-latent training test that substantially reduces
validation loss and saves a checkpoint. `ruff check .`, `black --check --workers 1 .`,
and `mypy src tests` pass. Not executed: real MVTec cache generation, real GPU
training, real predictor quality metrics, masking/inference, baselines, or evaluation.

## 2026-10-12 - PR review fixes (Qodo findings on the data pipeline and caching PR)
**Context:** Eight findings were verified against the code before any change; seven
were reproduced by a failing test first, one (corrupted-entry recovery) was a
documentation inconsistency, not a defect.
**Decision:** (1) *Encoder exceptions.* `build_latent_cache` catches `Exception` from
`encoder.encode` per batch, records every sample of that batch as failed
(`<file>: encoder failed: <Type>: <message>`), saves nothing from it, never retries or
splits it, continues with later batches, and raises `CacheBuildError` (with `.failed`
and the partial `.report`) at the end. The CLI therefore records it in the summary and
moves on to the next entry or category. `KeyboardInterrupt`/`SystemExit` still
propagate. This replaces the 2026-10-10 rule that an encoder exception aborts the run.
(2) *Corrupted final entry.* Unchanged on purpose: it is reported, left byte-for-byte
untouched, and never auto-replaced, because an entry that failed its checksum may be
the only copy of an expensive GPU run and silent replacement hides disk damage.
Recovery is explicit: `--rebuild`. Only the CLI help text and REPRODUCIBILITY.md
overstated this ("corrupted ones are recomputed" holds for shards only); they were
corrected. (3) *Mock identity.* The mock checkpoint hash is
`sha256("mock_encoder:random-projection:seed=<seed>")` (`mock_checkpoint_sha256`), so a
different seed (a different projection) never reuses a cache; the name stays
`mock_encoder` and caches stay under `_mock`. (4) *Token count.* `CacheExpectations`
gained optional `num_tokens`, which `build_latent_cache` always passes, so a completed
entry is judged by the same rule as a new encoding; a mismatch is not reused, the
existing entry is left in place, and the encode path then raises `CacheBuildError`.
(5) *`val_fraction`.* `_check_config` rejects values outside (0, 1), NaN, infinity and
booleans with `CacheRunConfigError` (exit code 2) for `--purpose nominal`, the only
purpose that uses it; nothing is clamped. (6) *`PreprocessingSpec`.* Validation and
normalization moved into `__post_init__`, so direct construction, `replace`,
`from_config` and `from_dict` share one rule set: positive integer sizes/version, string
resize/crop, exactly three finite numeric mean and std values, std > 0 (the old
`<= 0` test let a NaN std through). (7) *List/tuple.* `mean`/`std` are normalized to
float tuples there, so a YAML-style list and the tuple read back from disk compare
equal; a genuinely different value still mismatches. `transforms.PreprocessConfig` is
unchanged. (8) *Loading.* The finite-value check now lives in
`_check_tensor_matches_metadata`, shared by save and load; a non-finite file raises
`CacheCorruptedError` on load.
**Rationale:** Findings 1 and 4 to 8 were real gaps between the documented invariants
and the code paths; 3 would have let a test run silently reuse another projection.
For 2, replacing a corrupt entry automatically would contradict two existing tests
and the earlier decision, and `--rebuild` already provides the recovery.
**Test changes:** Tests that simulated a crashed run by raising `RuntimeError` from the
encoder (`test_cache_cli.py` x2, `test_latent_cache_pipeline.py` x4 incl. the shared
mock, `test_pipeline_end_to_end.py` x1) now raise `KeyboardInterrupt`, the interruption
that still aborts a run; their assertions are unchanged. Two files use
`mock_checkpoint_sha256(0)` instead of the removed `MOCK_CHECKPOINT_SHA256`.
**Outcome after change:** 168 tests pass on CPU (150 before). ruff, black and mypy are
clean. Not verified: real encoder, GPU, real MVTec; a CUDA out-of-memory error is only
simulated with a `RuntimeError`.

## 2026-10-14 - Implement Person 4 Frontend and Visualization Suite
**Context:** The four-person division of the project designates Person 4 to build
the user-facing demo and visualization dashboard. Requirements from the specification
mandate an interactive interface for MVTec category/image selection, anomaly detection
method switching (Adaptive, Dense, Fixed, PatchCore, MAE), heatmaps, overlays,
suspicious region localization, mask counts, latency/VRAM metrics, loss curves, and
benchmark summaries.
**Options considered:**
1. React + FastAPI: High overhead, slower development, multiple runtime dependencies.
2. Gradio: Simple, but less flexible for multi-tab research dashboards with loss curves
   and custom overlay parameter controls.
3. Streamlit: Explicitly recommended by the specification, rapid development, rich native
   data/image widgets, and supports immediate integration with Python backend modules.
**Decision:** Selected Streamlit for the frontend application under `frontend/app.py`,
supported by a modular architecture (`frontend/config.py`, `frontend/services/`,
`frontend/components/`), adhering to the backend integration contract defined on
Page 5 & 8 of the design specification.
**Rationale:** Streamlit allows end-to-end interactivity (image upload, slider-controlled
alpha blending, colormap selection, and loss curve plotting) while decoupling the UI
from backend internals via the stable `run_inference(image, method, category)` application
interface.
**Baseline / prior state:** No frontend, visualization, or interactive demo existed in
the repository.
**Outcome after change:** Complete frontend application implemented and smoke-tested.
Added `tests/unit/test_frontend.py` verifying contract schema compliance, input validation,
adaptive vs dense mask reduction invariants, heatmap overlay blending, and training/benchmark
services (all 7 unit tests pass).

## 2026-10-14 - Person 4 PR Review Fixes (Qodo findings on PR #7)
**Context:** Automated code review on PR #7 flagged 9 issues covering dependency
declarations, package installation, real-vs-mock pipeline honesty, baseline handling,
training checkpoint discovery, and session state retention.
**Options considered:**
1. Keep mock indicators implicit and leave dependencies outside setuptools.
2. Address all findings comprehensively: declare frontend dependencies in pyproject.toml,
   add CLI entrypoint `ijepa-frontend`, support trainer `.pt` checkpoints directly,
   persist Streamlit session state across widget reruns, and enforce consistency between
   decision thresholds and detected bounding boxes.
**Decision:** Selected option 2. Added `frontend` optional dependencies to `pyproject.toml`,
updated GitHub Actions CI workflow to install `.[dev,frontend]`, implemented `ijepa-frontend`
CLI launcher, ensured `is_mock=True` is explicitly returned when checkpoints are unmounted,
added `.pt` checkpoint history discovery in `experiment_service.py`, persisted inspection
results in `st.session_state`, and eliminated false-positive defect boxes on nominal samples.
**Rationale:** Preserves research reproducibility, guarantees clean environment
installs in CI, and prevents UI desynchronization on user interaction.
**Baseline / prior state:** Frontend lacked session state retention, CI failed on missing
dependencies, and checkpoint loader searched for non-existent JSONL files.
**Outcome after change:** All quality checks pass (`ruff check .`, `black --check .`,
`mypy src tests frontend` 0 errors across 63 files, 8/8 frontend unit tests pass).
CI workflow updated.
## [2026-10-16] Notebook predictor and augmentation workflow
**Context:** The project needed a proper predictor-training notebook with checkpoint persistence and a separate data-augmentation workflow.
**Options considered:** Keep the existing demo notebook; augment cached latent tensors; augment nominal images before encoding and use a separate cache/checkpoint root.
**Decision:** Make the training notebook use the existing validated cache and training APIs and save one best checkpoint per category. Add a separate notebook that applies deterministic image-space augmentation only to the nominal training subset, keeps original validation images, and writes augmented caches/checkpoints separately.
**Rationale:** Image-space augmentation produces valid encoder inputs, avoids corrupting latent geometry, prevents test leakage, and preserves a directly comparable non-augmented baseline.
**Baseline / prior state:** An untracked demo notebook with hardcoded pilot settings and mock mode enabled by default.
**Outcome after change:** Two reproducible notebooks and an augmentation config are present; execution remains pending on the local encoder/checkpoint setup.

## [2026-10-17] Inference-aligned predictor training masks
**Context:** An audit found training used ~40% scattered random target masks, while inference hides one contiguous 4x4 coarse window (16/8/4 tokens on the 14x14 grid) and then single tokens. Measured on the actual samplers: training masks averaged 25 connected components with 60% of each masked token's 4-neighbours visible, and a given 4x4 block is fully masked with probability 0.4^16 (~4e-7); coarse windows leave 23% visible. A synthetic-latent experiment (not real data) showed a predictor trained on scattered masks scoring worse on blocks than one trained on blocks. Validation and early stopping also used the scattered masks.
**Options considered:** (a) keep scattered masks and rely on generalisation; (b) block-only masks; (c) a per-row mixture of exact coarse windows and isolated single tokens; (d) multi-scale random rectangles.
**Decision:** Add `mask_strategy: inference_aligned` (c): each row targets either one of the grid-aligned `CoarseMasker` windows (the exact inference masks) with probability `block_probability`, or `single_token_count` mutually non-adjacent tokens standing in for 1x1 masks. Sampler in `predictor/target_masks.py`; validation uses the same sampler with its fixed seed. The `TrainingConfig` default stays `random` so existing callers, tests and checkpoints keep their behaviour; `configs/predictor/base.yaml` and the CLI select `inference_aligned`.
**Rationale:** Exact coarse windows remove the dominant shift with no new hyperparameter (`block_size` must equal `coarse_size`; a test enforces it). Several isolated singles per row give ~8x more loss terms per forward pass than one 1x1 target, at the cost of a slightly sparser context than a true 1x1 mask (documented trade-off). Leakage prevention is unchanged: targets are overwritten by the mask token inside the predictor, and a test shows rewriting target latents leaves outputs bit-identical.
**Baseline / prior state:** `build_random_target_mask` only (kept, used by `random`).
**Outcome after change:** Unit and integration tests pass. The effect on real metrics is pending a real-data pilot; predictors trained with scattered masks must be retrained and earlier pilot numbers are not comparable.

## [2026-10-17] One bilinear upsampling rule for evaluation and UI heatmaps
**Context:** `patch_scores_to_heatmap` used nearest-neighbour plus per-image min-max, while the evaluation runner interpolated raw scores bilinearly (`align_corners=False`). The PRD specifies bilinear.
**Options considered:** leave the paths separate; make the UI bilinear only; share one upsampling function with normalisation as a separate display-only step.
**Decision:** `masking/score_maps.py` now has `upsample_patch_scores` (raw, bilinear, `align_corners=False`) used by `evaluation/runner.py`, and `patch_scores_to_heatmap(..., normalize=True)` which min-max scales that same interpolation for display only.
**Rationale:** One interpolation rule; evaluation scores are never rescaled per image. The runner's maths is identical to before (same `F.interpolate` call), so pixel AUROC/AUPRO are not changed by this edit (`evaluation/metrics.py` untouched).
**Baseline / prior state:** Nearest upsampling in the UI path; inline interpolation in the runner.
**Outcome after change:** `tests/unit/test_score_maps.py` pins equivalence with `F.interpolate`, score-scale preservation, and row-major token mapping.

## [2026-10-17] Fingerprinted augmentation outputs and content-hash sample ids
**Context:** Notebook 02 skipped an augmented image if a file of the same name existed, named samples `<id16>_augNN` (not content), and reused the augmentation seed as the nominal-split seed, so changing augmentation settings could silently reuse old images/latents or reshuffle the validation set.
**Options considered:** delete outputs on every run; add the settings to the cache schema; fingerprint the settings into the output directory and use file content hashes as sample ids.
**Decision:** `data/augmentation.py`: `AugmentationConfig.fingerprint()` (settings + `AUGMENTATION_VERSION`) names the output directory; augmented sample ids are the SHA-256 of the written PNG; writes are atomic. The split now reads `nominal_split` from `configs/data/mvtec.yaml`, independent of `augmentation.seed`; the evaluation runner takes `val_fraction`/`split_seed` explicitly (defaults unchanged).
**Rationale:** Cache validity is already keyed by `sample_ids`, checkpoint SHA and preprocessing spec; making the ids content hashes closes the only gap without a schema change, and old fingerprint directories are preserved rather than deleted.
**Baseline / prior state:** Flat output directory, name-based ids, split seed = augmentation seed (both 0 today, so existing splits are unchanged).
**Outcome after change:** `tests/unit/test_augmentation.py` and `tests/integration/test_augmented_cache_invalidation.py`.

## [2026-10-17] Capsule horizontal flip disabled; rotation fills with border colour
**Context:** Capsule images have a black cap on the left and an orange body with a "500" imprint and the "actavis" logo on the right (inspected visually); `faulty_imprint` is a test defect class. Rotation filled corners with black, which paints dark wedges onto the bright MVTec backgrounds.
**Options considered:** keep the flip everywhere; remove it for all categories; per-category override.
**Decision:** `category_overrides.capsule.horizontal_flip_probability: 0.0` in `configs/augmentation/base.yaml`; bottle and hazelnut keep 0.5. Rotation corners use the image's border median colour.
**Rationale:** A mirrored capsule (cap on the right, reversed print) never occurs in nominal or test data, so it would widen the "normal" manifold toward reversed print. Bottle and hazelnut are not chirally marked. The fill-colour change affects all categories and is part of `AUGMENTATION_VERSION` 2, hence new fingerprints.
**Baseline / prior state:** Flip p=0.5 everywhere, black fill.
**Outcome after change:** Tests cover per-category resolution, flip on/off pixel equality, and bright corners; no augmented artifacts existed to invalidate.

## [2026-10-17] Frozen-encoder lock, dtype guards and training provenance
**Context:** `IJEPAEncoder.train()` flipped the wrapper to train mode; `_assert_frozen` used `assert`; the predictor gave an opaque error on a latent/parameter dtype mismatch; checkpoints lacked seeds and code provenance.
**Options considered:** document the hazards; auto-cast inside the predictor; fail fast.
**Decision:** `IJEPAEncoder.train()` always stays in eval, `requires_grad_(True)` raises, `encode()` re-asserts eval and frozen parameters (a `RuntimeError`, not `assert`) and casts input to the model's device/dtype. `LatentTransformerPredictor.forward` raises a clear `TypeError` on dtype mismatch (callers already cast); losses reduce in at least float32; training raises `FloatingPointError` on a non-finite loss; checkpoints gain a `reproducibility` block (seeds, mask strategy, torch/python versions, git commit and dirty flag).
**Rationale:** Cheap invariants that fail loudly. Gradient clipping, an LR scheduler, AMP and optimizer-state saving were considered and not added: the 14M-parameter predictor fits comfortably in 8 GB in float32, the cosine loss is bounded, and there is no evidence of instability, so default hyperparameters stay untouched.
**Baseline / prior state:** See context.
**Outcome after change:** `tests/unit/test_dtype_and_freeze_safety.py`, `tests/integration/test_predictor_training_masks.py`. An existing flaky test (`test_predictor_overfits_tiny_synthetic_cache_and_saves_checkpoint` failed for one of 12 global seeds because model init was unseeded) now seeds before construction.

## [2026-10-17] Predictor training provenance recorded in evaluation output
**Context:** `load_predictor_checkpoint` validated category/encoder/objective and then dropped `training_config` and `reproducibility`, so an evaluation JSON could not say whether the predictor was trained with random or `inference_aligned` masks.
**Options considered:** infer the strategy from the checkpoint filename; change `load_predictor_checkpoint` to return a tuple; add a second loader that returns metadata and keep the old one as a wrapper.
**Decision:** `inference.load_predictor_checkpoint_with_provenance` returns `(predictor, metadata)`; `load_predictor_checkpoint` calls it and returns only the predictor (frontend and existing callers unchanged). `metadata` = `{status, mask_strategy, training_config, reproducibility}` copied verbatim; `status` is `complete` / `partial` / `legacy`. Missing fields stay `None` (a legacy checkpoint reports `mask_strategy: null`, which means "not recorded", not "random"). `evaluate_category` writes `predictor_checkpoint` and `predictor_provenance` into each result. A non-mapping `reproducibility` raises.
**Rationale:** Never fabricate provenance; filenames are not evidence. Predictor inference behaviour is untouched.
**Baseline / prior state:** Metadata discarded at load time.
**Outcome after change:** `tests/unit/test_inference.py` (current, legacy, partial, integrity checks) and `tests/unit/test_evaluation_provenance.py` (end-to-end result record with a real checkpoint file).

## [2026-10-17] Per-sample, batch-size-independent validation masks
**Context:** Validation masks were drawn from one generator consumed in batch order, so the mask given to an image depended on the validation batch size and the images before it. Early stopping compares validation loss across epochs, and a batch-size change silently changed the validation task.
**Options considered:** keep the shared generator and require a fixed validation batch size; precompute one mask tensor for the whole validation set; derive a generator per sample from a stable hash.
**Decision:** `target_masks.sample_mask_seed(base_seed, sample_id, epoch=None)` = first 8 bytes of `SHA-256("ijepa-ad/validation-mask/v1|base_seed|epoch-or-'-'|sample_id")` with the top bit cleared. `training.build_validation_target_mask` seeds a fresh CPU `torch.Generator` per sample from it and calls the same `build_target_mask` with batch size 1. `base_seed` is `seed + VALIDATION_MASK_SEED_OFFSET` (unchanged). `epoch=None` in training, so each sample keeps one fixed mask across epochs (comparable early-stopping signal, as before); passing an epoch gives per-epoch masks. Training masks still use the shared generator and are unchanged. Checkpoints gain `reproducibility.validation_mask_protocol`.
**Rationale:** Per-sample generators make a mask a pure function of (config, seed, sample id, epoch): independent of batch size, order, neighbours and global RNG. SHA-256 is stable across processes, unlike the salted built-in `hash()`.
**Baseline / prior state:** Shared validation generator re-seeded each epoch; masks depended on batching.
**Outcome after change:** `tests/unit/test_validation_masks.py` (batch sizes 1/4/16 with partial final batches, shuffled/reversed/subset orders, golden seeds, differing `PYTHONHASHSEED` subprocesses, no global-RNG use). Validation loss values change slightly versus earlier runs (different masks, same distribution), so validation losses are not comparable with checkpoints trained before this change. They are also not comparable between mask strategies (different target geometry), so loss alone must not be used to rank a `random` run against an `inference_aligned` run; compare on test-time scoring instead.

## [2026-10-17] Image-level score normalised by the mode/lambda maximum
**Context:** `image_score` was `clamp(max(patch_scores) / 2, 0, 1)`. Token scores lie in [0, 2], but fusion is `coarse + lambda * fine`, so adaptive maxima reach `2(1 + lambda)` (4 at lambda = 1) and dense maxima `2 * lambda`. Everything above 2 saturated at 1.0 and tied, which can lower image AUROC and distorts the calibrated threshold.
**Options considered:** remove the division (scores in [0, 4], breaks the UI/threshold [0, 1] contract); divide by a universal `2(1 + lambda)` for every mode (compresses fixed scores); divide by the mode-specific maximum (chosen); rank-normalise (loses calibration meaning).
**Decision:** `inference.image_score_upper_bound(mode, lam)`: fixed `2`, dense `2 * lam`, adaptive `2 * (1 + lam)`. `InferenceResult.score_upper_bound` carries it (default 2.0 keeps externally constructed results as before), `raw_image_score` is the unscaled maximum, and `image_score = raw / bound` (0 when the bound is 0, i.e. dense with lambda = 0). Evaluation records `image_score_definition` (formula, bound, lambda, mode) per category. `frontend/services/experiment_service.load_category_threshold` ignores calibrated thresholds whose result row lacks `image_score_definition` (old scale) and falls back to the default. Patch scores, pixel AUROC and AUPRO are untouched.
**Rationale:** It is a monotone rescale, so ranking is preserved exactly wherever the old clamp did not bind, and it removes the ties where it did. Calibration is computed on the same definition (validation and test use one scorer). The UI/API contract stays [0, 1], so no frontend validation changed.
**Baseline / prior state:** `max / 2` clamped; saturation above raw score 2.
**Outcome after change:** `tests/unit/test_inference.py` (bounds per mode/lambda, extremes map to 0 and 1, raw 2 vs 3 no longer tie, ranking preserved, lambda = 0 dense) and `tests/unit/test_frontend.py`. **Previously reported adaptive/dense numbers are not directly reproducible** (image AUROC may change through tie resolution; `decision_threshold` is on a new scale). `image_score` values are comparable only within one (mode, lambda); use `raw_image_score` or the recorded bound across configurations. Fixed mode is unchanged (bound 2).

## [2026-11-01] Opt-in scoring variants for Tier 1 of the metrics plan (no retraining)
**Context:** `docs/METRICS_IMPROVEMENT_PLAN.md` Tier 1 asks for five scoring-only changes evaluated against the current baseline (baseline predictors, adaptive mode, default tau 0.5, max score) with a paired-bootstrap CI. Items 2 (channel standardisation / layer norm) and 4 (Mahalanobis residual) change the per-token discrepancy inside `MaskedLatentScorer._score_windows`; item 3 needs a Gaussian blur of the upsampled map; item 1 is a per-position z-score.
**Options considered:** replace `1 - cos` in place; subclass the scorer in the notebook; add an opt-in discrepancy parameter. For the fine-pass gate: let each variant gate on its own score scale (tau 0.5 would mean something different per variant and change masks per image); or always gate on the baseline `1 - cos` region scores. For item 1: inside the scorer, or post hoc on raw coarse/fine scores. For the Mahalanobis covariance: per position (infeasible with 21-39 validation images in 1408 dimensions) or one shared Gaussian after PCA.
**Decision:** `MaskedLatentScorer(..., discrepancy=None)` takes an optional `TokenDiscrepancy` callable `(predicted, actual) -> [B, T]`; `None` keeps the original code path and results. The fine-pass gate is always evaluated on the baseline `1 - cos` region scores, so every variant evaluates exactly the baseline's windows (`masks/image` unchanged; a paired comparison isolates the scoring change). With a custom discrepancy `InferenceResult.score_upper_bound` is `inf` and `image_score` raises (use `raw_image_score`); Mahalanobis scores are unbounded. New `MaskedLatentScorer.window_predictions` exposes the masked predictions for fitting residual statistics. New module `ijepa_ad/scoring.py`: `ChannelStandardisedCosine` (+ `fit_channel_stats`), `layer_norm_cosine_discrepancy`, `MahalanobisResidual` (+ `fit_mahalanobis_residual`, PCA to `d` components, relative ridge 1e-3, one shared Gaussian), `PositionNormaliser` (z-score with std floored at 0.5 x the median std, or centring only). `masking.score_maps.smooth_score_maps` is a separable Gaussian blur with reflected borders. Item 1 is applied post hoc to the baseline raw scores (coarse stats from `fixed`, fine stats from `dense` validation passes; fused as `z_coarse + lambda * fired * z_fine`). Statistics: items 1 and 4 from nominal validation images only, item 2 from nominal training latents only. Settings declared before test scoring: top-k 5, sigma 4 px (sensitivity 8 and 16, plus a sigma-0 control through the same two-stage resampling), Mahalanobis d = 64 (sensitivity 32 and 128), position std floor 0.5. For smoothed maps the image score is read at the 196 token centres. Notebook `notebooks/05_tier1_scoring_improvements.ipynb`, results in `results/improvements/tier1/` (nothing existing overwritten; the notebook refuses to start if outputs exist).
**Rationale:** An opt-in parameter keeps every earlier result reproducible; a regression test compares the default path bit for bit against a verbatim copy of the original algorithm. Gating on the baseline score removes a confound (compute and window set differ per variant otherwise). The Mahalanobis `d` is a genuinely sensitive setting (see below), which is why it was fixed in advance and the other values are labelled sensitivity.
**Baseline / prior state:** `_score_windows` hard-coded `1 - cos`; no smoothing, per-position or Mahalanobis scoring.
**Outcome after change:** The baseline row reproduces `results/calibration` (image AUROC, pixel AUROC, AUPRO, masks/image to 1e-4) and the bootstrap procedure reproduces notebook 04's "top-5 only" interval exactly. Versus the baseline max score, using 95% paired CIs on image AUROC: item 1 (z-score) improves bottle (+0.068, [+0.013, +0.145]) and is inconclusive for capsule (+0.101, [-0.110, +0.318]) and hazelnut max (+0.062, [-0.037, +0.168]; top-5 improves), while pixel AUROC / AUPRO rise strongly (capsule 0.684 -> 0.901 / 0.098 -> 0.508; hazelnut 0.907 -> 0.969 / 0.577 -> 0.812; bottle 0.930 -> 0.952 / 0.766 -> 0.831; no CI). Item 2: per-channel standardisation worsens max image AUROC for bottle and hazelnut; layer norm changes nothing because cached tokens are already zero-mean over channels (mean/std about 0.02). Item 3 (sigma 4 px): improves bottle and hazelnut max against the baseline, but against the sigma-0 control only hazelnut max stays significant, so part of the gain is the resampling; capsule is inconclusive. Item 4 (d = 64): improves bottle, inconclusive for capsule and hazelnut; sensitivity rows move a lot with d (hazelnut max 0.59 / 0.74 / 0.93 for d = 32 / 64 / 128), so d = 64 stays as declared and no value is selected from test numbers. Item 5: the 2x2-block capsule predictor with 2x2 windows (49 masks/image) is not better than the baseline (max 0.418 vs 0.522, inconclusive; pixel AUROC 0.685 vs 0.684). Capsule image AUROC stays near chance for every variant. Tests: `tests/unit/test_scoring_variants.py` (20 tests), full suite passes.

## [2026-11-01] Synthetic-anomaly validation set for choosing settings (plan item 12)
**Context:** Only nominal validation images exist, so no setting (Mahalanobis dimension, smoothing sigma, top-k, lambda, gate tau, score family) could be chosen for detection quality without looking at the test split. The cache machinery (`latent_cache`, `builder`) accepts only official nominal-train/val and test records, and that restriction is what keeps the test split evaluation-only.
**Options considered:** extend the cache schema with a synthetic split (loosens the leakage checks); paste defects onto test images (uses test data); generate defects from nominal validation images with a separate, equally strict cache; choose on the test split (forbidden). For fitting and evaluating on the same validation images: ignore the overlap, or cross-fit.
**Decision:** `ijepa_ad/data/synthetic_anomalies.py` generates CutPaste scars and DRAEM-style Perlin texture blends as a pure function of `(SyntheticConfig, image_id, variant)` (SHA-256 seeded, fingerprinted; changing any setting or `SYNTHETIC_VERSION` changes the fingerprint). Base images are nominal validation images, texture sources nominal training images; the test split is never read (the build script calls `validate_category_splits` but only uses `train`). Defects stay on the object (foreground heuristic), cover 0.3% to 4% of the image (log-uniform), alternate kind per variant (4 per image) and change only pixels inside the returned mask. `ijepa_ad/caching/synthetic_cache.py` writes images, masks and a manifest under `data/synthetic_val/<fingerprint>/`, verifies file hashes on load, and stores latents under `data/cache/synthetic_val/<fingerprint>/` only if generator fingerprint, encoder checkpoint SHA-256, preprocessing and ordered ids match; it never overwrites. `scripts/build_synthetic_validation.py` builds and encodes (reusable and idempotent). `notebooks/06_synthetic_validation_settings.ipynb` selects settings with 3-fold cross-fitting over the clean validation images (z-score, Mahalanobis and gate-quantile statistics are fitted on the other folds only), then a greedy sequential selection in the declared order family, sigma, k, lambda, tau, with one criterion (mean over categories of mean(image AUROC, pixel AUROC, AUPRO)), one global choice, and ties within 0.005 broken toward the earlier (simpler) option. Pixel metrics on the proxy are computed at 224x224. The saved selection is written before the proxy is compared with notebook 05's test numbers, so the comparison cannot influence it.
**Rationale:** A separate cache keeps the official-split guarantees intact. Cross-fitting avoids fitting statistics and evaluating on the same clean images. Declaring the order, criterion and tie rule up front makes the selection reproducible and not a search over many knobs. The dense fine pass is independent of the gate, so every tau and lambda is evaluated post hoc from one set of predictions (verified equal to `MaskedLatentScorer` output).
**Baseline / prior state:** No labelled data for choosing settings other than the test split.
**Outcome after change:** 84, 88 and 156 synthetic images (bottle, capsule, hazelnut; fingerprint `46c947cb99e1faf4`). The baseline is neither saturated nor at chance on it (image AUROC with the max score 0.56 to 0.79). Selected (global): per-position z-score, sigma 0, k = 3, lambda = 1, default tau (`results/improvements/synthetic_val/selected_settings.json`). Several differences are inside the tie margin: Mahalanobis d = 128 tied with the z-score on the criterion (bottle preferred it), sigma 4 scored 0.0043 above sigma 0, k >= 3 are all within the margin, lambda has almost no effect, and calibrated quantile gates were worse than the default. After the selection was saved, the proxy's ranking of the eight Tier 1 families agreed with the test split from notebook 05 (mean Spearman 0.74 for image AUROC with the max score, 0.86 top-5, 0.84 pixel AUROC, 0.94 AUPRO) but is weak for capsule max-score image AUROC (0.42). The 2x2-block capsule predictor is not better than the baseline on the proxy either (top-5 0.709 vs 0.710). Encoder layer and mask block size are not covered (they need new caches or predictors). The selected settings have not been applied to the test split. Tests: `tests/unit/test_synthetic_anomalies.py` (13 tests); full suite passes.

## [2026-11-01] Selected settings applied to the test split once (notebook 07)
**Context:** Notebook 06 chose a scoring configuration on the synthetic-anomaly validation set. Choosing is only half the protocol: the choice has to be scored on the test split, once, against the baseline, without changing anything afterwards.
**Options considered:** re-run the whole grid on test (tuning on test); evaluate only the winner; evaluate the winner and the baseline at the same `k` so a top-k effect is not credited to the scoring change.
**Decision:** `notebooks/07_confirm_selected_settings_on_test.ipynb` reads `results/improvements/synthetic_val/selected_settings.json`, refuses to run if it says the test split influenced the choice, refits the statistics on all nominal validation images (position z-score) and scores test once for the baseline and the selected configuration. Intervals are paired bootstrap (1,000 resamples, seed 0) against the baseline max score, against the baseline top-k, and for the max score. When the selected configuration is a notebook 05 variant its pixel AUROC / AUPRO / max-score AUROC must equal that saved row (it does, to 1e-4). Outputs: `results/improvements/confirmation/`.
**Rationale:** One pre-declared configuration, one evaluation, interval reported either way.
**Baseline / prior state:** Baseline = baseline predictors, adaptive, default tau, max score.
**Outcome after change:** Selected (`pos_zscore`, sigma 0, k = 3, lambda = 1, default tau) versus the baseline max score, image AUROC: bottle +0.073 [+0.019, +0.152] improves; capsule +0.150 [-0.045, +0.356] inconclusive; hazelnut +0.137 [+0.054, +0.226] improves. Against the baseline at the same k: bottle +0.046 [+0.012, +0.097] improves, capsule +0.160 [-0.027, +0.357] and hazelnut +0.042 [-0.014, +0.104] inconclusive. Pixel AUROC / AUPRO (point estimates): bottle 0.930 -> 0.952 / 0.766 -> 0.831, capsule 0.684 -> 0.901 / 0.098 -> 0.508, hazelnut 0.907 -> 0.969 / 0.577 -> 0.812. Masks per image unchanged (16.0 to 16.2). So the synthetic-validation choice transferred: the gains it showed on the proxy appear on the real test split, but capsule image-level detection is still not distinguishable from the baseline.

## [2026-11-01] Meta's pretrained predictor with input masking, zero-shot (plan items 7a and 6)
**Context:** Plan item 6 hypothesises that our predictor sees latents of the full image, so defects leak into the visible tokens; item 7 proposes Meta's pretrained context encoder + predictor, which also removes the leakage because the encoder only sees visible patches. The 16.5 GB checkpoint was feared to be too large for a 15.8 GB host.
**Options considered:** load the whole checkpoint (the optimizer state alone is most of the 16.5 GB); `torch.load(..., mmap=True, weights_only=True)`; run the pretrained path only for the item 6 diagnostic on about 20 images; run it on the whole validation and test sets.
**Decision:** `ijepa_ad/encoder/ijepa_pretrained_predictor.py` (`IJEPAContextPredictor`) loads the `encoder` (context, vit_giant) and `predictor` (16 blocks, width 384, 16 heads, read off the saved shapes) states with `mmap=True` (the path is passed as `str`, which this torch version requires for mmap), strict key matching, frozen, eval, float32 by default. For each coarse window it runs the context encoder on the visible tokens only (Meta's `masks=` argument), then the predictor for the window's token indices; windows of equal size are batched because Meta's gather needs rectangular index tensors. Evaluation (`notebooks/08_pretrained_predictor_zero_shot.ipynb`): coarse 4x4 windows only (the dense pass would need 196 context-encoder passes per image; in the baseline the fine pass fires on at most 1.2% of test images), discrepancy `1 - cos` against the cached target latents (not recomputed), compared with our trained predictor in coarse-only mode (reproduces the saved `fixed (coarse only)` row), raw and with the per-position z-score selected in notebook 06 (statistics from nominal validation images of the same model). Predictions are cached under `data/cache/pretrained_zero_shot/` with the checkpoint hash and ordered ids. This is logged as the memory-mapping change the plan asked to record.
**Rationale:** mmap keeps host RAM under 2 GB (measured RSS 1.9 GB) and GPU memory about 4 GB, so the whole check fits on this machine. Zero-shot first, as the plan says, because fine-tuning is only worth its cost if zero-shot is promising. Evaluating the whole splits (not 20 images) costs about 1 s per image, 12 minutes in total, and gives intervals.
**Baseline / prior state:** Only the target encoder was ever loaded; the context encoder and the pretrained predictor were unused.
**Outcome after change:** The pipeline is sound (nominal 1 - cos 0.15 to 0.20, not near 1; unit tests with fake networks check that no masked token reaches the context). It does not beat our predictor. Image AUROC, pretrained versus trained, raw max score: bottle 0.661 vs 0.925 (worsens), capsule 0.556 vs 0.522 (inconclusive), hazelnut 0.458 vs 0.799 (worsens); raw pixel AUROC is below chance for capsule (0.398) and hazelnut (0.462). With the selected z-score on both: bottle 0.986 vs 0.998 (-0.012 [-0.034, +0.000], inconclusive), capsule 0.647 vs 0.672 (-0.026 [-0.108, +0.063], inconclusive), hazelnut 0.809 vs 0.939 (-0.130 [-0.212, -0.055], worsens). It costs about 980 ms per image against 23 ms. Conclusion: zero-shot transfer from ImageNet-22k does not help, so fine-tuning (item 7b) is not justified by this result, and the leakage hypothesis (item 6) is not supported as the main limit: removing leakage and adding a much larger predictor did not improve separation (confounded, since size and training data also differ; a decisive test would retrain on input-masked context latents).

## [2026-11-01] Encoder layer chosen on synthetic validation data, then a predictor trained on it (plan item 8)
**Context:** Every result so far scores final-layer tokens. A nearest-neighbour reference on those tokens is far better than the predictor (notebook 03), and PatchCore uses mid-level features, so a different layer might hold more usable signal. The plan suggests scanning blocks 20 to 35 with the kNN reference, but selecting a layer on the test split would be tuning on test.
**Options considered:** scan on the test split (forbidden); scan on nominal validation images only (no defects, nothing to rank); scan on the synthetic-anomaly validation set (item 12). For the cache: change `LatentCache` to carry a layer field; or build ordinary latent caches whose `encoder_name` and `checkpoint_sha256` identify the layer. For raw residual-stream features: use them as is, or apply a per-token layer norm.
**Decision:** `ijepa_ad/encoder/layer_features.py` (`LayerFeatureExtractor`, forward hooks on the frozen target encoder's blocks, no weights touched) returns the output of block `l` before the final LayerNorm with a parameter-free per-token layer norm, so scales are comparable across depth and float16 storage is safe; `scripts/build_layer_features.py` caches float16 features (provenance: checkpoint hash, preprocessing, layer, ordered ids; never overwritten) for layers 8, 12, ..., 40 on train, val and synthetic only. `notebooks/09_encoder_layer_scan.ipynb` scores the kNN reference (bank = all nominal training tokens) on clean validation plus synthetic images with a criterion declared in advance (mean over categories of mean of image AUROC top-3, pixel AUROC, AUPRO), two feature treatments (layer norm; plus train-statistics channel standardisation), ties broken toward the final cached latents and then later layers. The test split was not read. Layer 40 (pre-norm) matched the final cached latents (J 0.966 vs 0.969), which checks the pipeline. Mid layers were best (J: layer 28 0.987, 32 0.986, 24 0.987, final 0.969); the declared rule selected layer 32 with layer norm. `notebooks/10_confirm_layer_on_test.ipynb` then scored layer 32 on test once (features built only for that layer). For the predictor, `scripts/build_layer_latent_cache.py` writes ordinary latent caches (`data/cache/latents_layer32`, `..._test`) whose `encoder_name` is `<name>_layer32` and whose `checkpoint_sha256` is a derived identity `SHA-256(checkpoint_sha256|layer=32|token_layer_norm|v1)`. A predictor trained on it records that identity, so `load_predictor_checkpoint` raises if it is used with the real checkpoint hash (and the other way round); no cache or loader code changed. Predictors were trained with the unchanged CLI defaults (`scripts/train_predictor.py`; note that `python -m ijepa_ad.predictor.training` does nothing because the module has no `__main__` guard) into `checkpoints/predictors_layer32/`. `notebooks/11_layer32_predictor.ipynb` scores them with the notebook 06 settings carried over unchanged (not re-selected for layer 32).
**Rationale:** Choosing on the synthetic set keeps the test split out of selection; the derived identity turns a silent feature mix-up (same shape, 1408 dimensions) into an error; reusing the training CLI and the cache format keeps the recipe identical to the baseline so only the features differ.
**Baseline / prior state:** Final-layer latents only.
**Outcome after change:** kNN reference, test, layer 32 vs final (notebook 10, paired bootstrap): capsule image AUROC +0.105 [+0.022, +0.187] improves (max) and +0.074 [+0.017, +0.135] (top-3); bottle and hazelnut inconclusive at ceiling (>= 0.99); pixel AUROC / AUPRO rise everywhere (capsule AUPRO 0.51 -> 0.77, bottle 0.86 -> 0.94, hazelnut 0.87 -> 0.94). Predictor on layer 32 (notebook 11; same recipe, one run per category): with the notebook 06 settings, layer 32 vs final layer, image AUROC top-3: bottle 0.999 vs 0.998 inconclusive, capsule 0.767 vs 0.672 +0.095 [+0.038, +0.150] improves, hazelnut 0.993 vs 0.939 +0.054 [+0.017, +0.101] improves. Against the original baseline (final layer, max score): bottle +0.075 [+0.021, +0.153], capsule +0.246 [+0.049, +0.457], hazelnut +0.191 [+0.107, +0.284], all improve. Pixel AUROC / AUPRO (point estimates), original baseline -> layer 32 with selected settings: bottle 0.930 / 0.766 -> 0.966 / 0.871, capsule 0.684 / 0.098 -> 0.931 / 0.666, hazelnut 0.907 / 0.577 -> 0.989 / 0.901. The z-score slightly lowers bottle pixel metrics at layer 32 (0.979 / 0.904 with the baseline score), which is the cost of carrying the settings over instead of re-selecting. Caveats: one training run per category, scan measured kNN signal not predictor performance, test split scored once per row.

## [2026-11-01] Scoring settings re-selected for the layer-32 predictors on synthetic validation data only (notebook 12)
**Context:** Notebook 11 scored the layer-32 predictors with the notebook 06 settings carried over from the final-layer predictor (per-position z-score, k = 3, ...), and noted that they were not chosen for layer 32. The task was to re-select them, using only the synthetic-anomaly validation data, and to stop before any test scoring.
**Options considered:** keep the carried-over settings; re-run the notebook 06 selection with the layer-32 predictors and caches; widen or change the grid or criterion for layer 32 (rejected: that would make the two selections incomparable and invite tuning). For test-split isolation: reuse notebook 06's `validate_category_splits` (it also lists and hashes the official test images) or build only the train dataset. For storage precision: mix float32 clean tokens with float16 synthetic tokens, or put both through the float16 store.
**Decision:** `notebooks/12_layer32_synthetic_settings.ipynb` repeats the notebook 06 protocol unchanged (3-fold cross-fitting, greedy order family -> sigma -> k -> lambda -> tau, criterion J = mean over categories of mean(image AUROC at k, pixel AUROC, AUPRO), one global choice, ties within 0.005 go to the earliest declared option). The declared grid is asserted equal to `declared_before_results` in `results/improvements/synthetic_val/selected_settings.json`. Inputs: `checkpoints/predictors_layer32`, `data/cache/latents_layer32` (train, for channel standardisation), and `data/cache/layer_features/layer32` (val and synthetic; the val tokens are bit-identical to `latents_layer32` val after float16 rounding, so clean and synthetic tokens share one store; effect on `1 - cos` at most 3.6e-7). Checks that stop the notebook if they fail: derived layer-32 identity equals the one recorded in each predictor, a layer-32 predictor is rejected for the real checkpoint hash, predictor train/val ids equal the nominal split, synthetic PNG hashes equal the manifest and the feature keys, synthetic bases equal the validation images, post-hoc adaptive scores equal `MaskedLatentScorer`. Test isolation: only `MVTecADDataset(..., 'train')` is built (the notebook 06/11 call to `validate_category_splits` is avoided), no `*_test` cache is opened (asserted), no test result is consulted, so notebook 06's proxy-validity section and the 2x2-block capsule check are not repeated. No library or script code changed; the notebook was executed with a small `jupyter_client` runner because `nbformat`/`nbclient` are not installed. Outputs in `results/improvements/layer32_synthetic_val/` (`selected_settings.json`, `settings_grid.csv`, `provenance_checks.json`, `selected_vs_start.csv`, `carried_over_comparison.csv`, `gate_firing_diagnostic.csv`, `synthetic_set_summary.json`, figures); nothing existing was overwritten (the notebook refuses to start if outputs exist).
**Rationale:** Same protocol and same synthetic images make the layer-32 choice comparable with notebook 06's, and a changed grid would be a new selection, not a re-selection. Avoiding `validate_category_splits` removes the only place the earlier selection notebooks touched the test directory.
**Baseline / prior state:** Layer-32 predictors used the carried-over final-layer settings (notebook 11); no selection had been made for layer 32.
**Outcome after change:** Selected (global): `maha_d128`, sigma 0, k = 1, lambda 1, default tau. J on the layer-32 synthetic set (optimistic: chosen on it): start (baseline score, k = 5) 0.9322, carried-over settings 0.9509, selected 0.9610 (+0.0101 over carried over, above the 0.005 margin). Stage 1 is not a tie: maha_d128 0.9615 vs pos_zscore 0.9523 (bottle prefers maha_d128; capsule and hazelnut prefer pos_zscore); J rises steeply with d (d32 0.867, d64 0.922, d128 0.9615) and d128 is the largest value in the grid. Later stages are ties settled by the rule: sigma 0, 4 (0.9613), 8 within the margin; k = 3 scored best (0.9626) but k = 1 (0.9610) is earliest within the margin; lambda is irrelevant (0.9608 to 0.9612). Default tau (0.9610) beats every calibrated quantile gate (0.935 to 0.945). The default gate (tau 0.5) almost never fires on layer-32 predictions (0 to 0.1% of tokens; never for capsule and hazelnut), so the selected pipeline is in practice coarse-only. **The selection has not been applied to the test split**; previously reported layer-32 numbers (notebook 11, carried-over settings) are unchanged. For the next step: Mahalanobis scores are unbounded, so use `raw_image_score`; the covariance is fitted on coarse-window residuals of the nominal validation images and, as in notebook 06, applied to coarse and dense scores. Caveats: J is near its ceiling (0.93 to 0.97), so the 0.005 margin is comparable to the differences; no layer-32 check of proxy-versus-test agreement exists (it would need test numbers); one training run per category.

## [2026-11-01] Layer-32 selected settings applied to the test split once (notebook 13)
**Context:** Notebook 12 selected the layer-32 scoring settings on synthetic validation data (`maha_d128`, sigma 0, k = 1, lambda 1, default tau; `results/improvements/layer32_synthetic_val/selected_settings.json`). The protocol requires scoring that fixed choice on the official test split once, against matched references, and changing nothing afterwards.
**Options considered:** compare only with the original baseline (mixes layer, predictor and scoring); compare only with notebook 11's configuration (differs in score family and in k, so a gain or loss could not be attributed); add a like-for-like reference on the same layer-32 predictor. Whether to run a throwaway dry run on the test split (rejected: it would score test twice).
**Decision:** `notebooks/13_confirm_layer32_settings_on_test.ipynb`, outputs in `results/improvements/layer32_confirmation/` (new; the notebook refuses to start if outputs exist). It reads the saved record as authoritative: it stops if the record says test influenced the selection or was already applied, if the settings are not exactly `maha_d128 / 0 / 1 / 1.0 / default`, if the encoder or layer-32 identity differs, or if a layer-32 predictor file differs from the sha256 stored in the record; the record's hash is checked before and after. Mahalanobis statistics (d = 128, ridge 1e-3, coarse-window residuals) are fitted on all nominal validation images only, as notebooks 05 to 07. Image AUROC uses `InferenceResult.raw_image_score` (asserted equal to the max fused patch score; the bounded `image_score` is undefined for a custom discrepancy). Four rows, each passed once through the test set (counted and asserted): `original` (final-layer baseline, max), `l32 cosine` (layer-32 predictor, `1 - cos`, max, same windows, gate, tau, lambda and metrics as the selection: the like-for-like reference), `selected`, and `l32 notebook 11` (per-position z-score, top-3; derived post hoc from the `l32 cosine` adaptive pass exactly as in notebook 11, so no second test pass). Paired bootstrap as before (1,000 resamples, seed 0, good and anomalous resampled separately, same resamples for every comparison in a category). Comparisons: primary `selected` vs `l32 cosine` (max); vs notebook 11 as reported (family and k differ, labelled), plus two like-for-like diagnostics (both top-3, both max; the top-3 of `selected` is a diagnostic, k stays 1); historical `selected` vs `original`; context `l32 cosine` vs `original` to show how much of the historical gain is the layer. Reference rows must reproduce notebook 07's baseline and notebook 11's rows to 1e-4 (they did, difference 0.0). No library or script code changed, so no new tests; FLOW.md is unaffected (existing modules reused, no flow change).
**Rationale:** The like-for-like row is the only comparison that isolates the scoring change; the labelled notebook 11 comparison answers whether the re-selection improved on what was already reported; the historical comparison is kept separate because three things change at once. Hard guards make the one-time evaluation auditable.
**Baseline / prior state:** Layer-32 predictors scored with the notebook 06 settings carried over (notebook 11); the selection for layer 32 had not been applied to test.
**Outcome after change:** Test split, image AUROC (max score; 95% paired CI of the delta). Primary, `selected` vs layer-32 `1 - cos`: bottle 0.9913 vs 0.9587, +0.033 [+0.004, +0.078] improves; capsule 0.600 vs 0.543, +0.057 [-0.123, +0.259] inconclusive; hazelnut 0.960 vs 0.946, +0.013 [-0.031, +0.068] inconclusive. Pixel AUROC / AUPRO point estimates, `1 - cos` -> selected: bottle 0.979 / 0.904 -> 0.983 / 0.928, capsule 0.836 / 0.309 -> 0.956 / 0.720, hazelnut 0.986 / 0.877 -> 0.987 / 0.894. **Against notebook 11's reported configuration (z-score, top-3) the re-selection does not help on test**: as reported (family and k differ) bottle -0.008 [-0.029, +0.002] inconclusive, capsule -0.167 [-0.282, -0.057] worsens, hazelnut -0.034 [-0.067, -0.007] worsens; at the same top-3 score capsule -0.121 [-0.229, -0.016] and hazelnut -0.024 [-0.054, -0.001] still worsen; at the same max score capsule -0.121 [-0.245, -0.001] worsens, bottle and hazelnut inconclusive. Pixel AUROC / AUPRO versus notebook 11: bottle +0.017 / +0.056, capsule +0.026 / +0.054, hazelnut -0.001 / -0.008. Historical, vs the original baseline (final layer, max): bottle +0.067 [+0.012, +0.144] improves, capsule +0.079 [-0.134, +0.303] inconclusive, hazelnut +0.158 [+0.076, +0.254] improves; the layer-32 `1 - cos` row alone already gives +0.034 / +0.021 / +0.144 (hazelnut improves), so most of the historical gain is the layer and retrained predictor, not the scoring. Masks per image: 16.19 / 16.00 / 16.58 for layer 32 (original 16.19 / 16.00 / 16.15), identical between `selected` and `l32 cosine` (the default gate fires rarely). Scorer latency about 32 to 34 ms per image for every row (Mahalanobis adds 0.5 to 2.8 ms); not comparable with the 23 ms reported earlier (a different measurement run). The synthetic-validation choice therefore transferred only partly: it beats raw cosine but not the z-score configuration already reported, so the carried-over notebook 06 settings remain the better-supported layer-32 configuration on test; per notebook 12 the synthetic criterion preferred the z-score for capsule and hazelnut already, and `d = 128` is the largest value of a strongly d-sensitive grid. The settings were not changed. Caveats: one training run per category, capsule intervals are wide (132 test images), top-3 for the selected row is a diagnostic only.

## [2026-11-01] Paper updated with the validation-selected results; headline fixed before the layer-32 re-selection
**Context:** `paper/main.tex` (compiled 11:13) predated notebooks 05 to 13, with a deadline the same day, so the later results had to be added without new experiments.
**Options considered:** report the best test row (the layer-32 Mahalanobis row has the highest mean pixel AUROC and AUPRO) as the headline; report the configuration fixed before the layer-32 re-selection; leave the paper unchanged. For numbers: type them by hand, or generate the tables from the saved CSVs.
**Decision:** New section "Validation-Selected Improvements" (`sec:improve`) plus abstract, discussion, threats, future work, conclusion and reproducibility edits. The headline is layer 32 + position z-score (top-3), the configuration fixed before the notebook 12/13 re-selection; the Mahalanobis result is reported as a negative finding (worse at image level for capsule and hazelnut, better in pixel metrics). Tables are generated by `paper/make_improvement_tables.py` from `results/improvements/{layer_predictor,confirmation,layer_confirmation,layer32_confirmation}/`; mean rows average the rounded per-category values, as the existing Table I does. The same-score comparisons are included next to the top-3-versus-max ones so the two are not conflated.
**Rationale:** Choosing the headline after seeing test numbers would reintroduce test tuning; generating tables from saved files keeps every number traceable.
**Baseline / prior state:** Paper reported only the original pipeline, the calibrated gate and the kNN reference.
**Outcome after change:** `main.pdf` compiles (9 pages, no undefined references or overfull boxes in the log). Table rendering was not inspected visually in this session. No experiment was run for the paper update; the 24/28-layer, block-size, augmentation, multi-seed, hybrid and resolution items were not started.

## [2026-11-01] B.Tech project report written in LaTeX from the RBU template (Report/)
**Context:** The school's report format (`Report/2.Instructions for project report.docx`) requires a chapter-structured report in LaTeX (A4, Times New Roman 12 pt, 1.5 spacing, margins 1.5/1/1/1 in, 3000 to 6000 words), and `Report/main.tex` held the RBU sample template with placeholder content.
**Options considered:** write the report fresh; keep the template and replace its content; reuse the IEEE paper text unchanged (it is a different, two-column format with different sections).
**Decision:** Kept the template preamble and structure unchanged (so formatting rules stay as supplied) and replaced the content: cover (logo converted from the sample's EMF), declaration, approval sheet, acknowledgement, abstract, lists, six chapters, references, annexures A to C. Tables are extracted from the paper sources by `Report/make_report_tables.py` (which come from saved result files) and wrapped in `resizebox`; figures are copied from the paper and `results/improvements/`. Compile with XeLaTeX (`fontspec`). Fixed one template bug: `\chapterseparator` printed the previous chapter's number (the page before Chapter 1 said "0"), so it now prints the upcoming chapter number. The text follows the paper, including the negative results and limitations; no experiment was run.
**Rationale:** Following the supplied template avoids format rejections; generating tables from saved files keeps the report numbers identical to the paper's.
**Baseline / prior state:** Template with a "Smart Campus" placeholder project.
**Outcome after change:** Builds to 43 pages with no undefined references; about 4,400 words of chapter prose (limit 3000 to 6000). Roll numbers (A-35, A-11, A-30, A-02) and the guide's title (Dr. Sunita G. Rawat) were added afterwards, and the paper's affiliation was changed to Ramdeobaba University to match the report. Still blank for the students: the Student Contribution Statement and the publication status.
