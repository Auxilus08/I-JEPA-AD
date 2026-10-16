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
