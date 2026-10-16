"""Training loop for the cached-latent Transformer predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.data.transforms import PreprocessConfig
from ijepa_ad.predictor.cache_dataset import (
    CachedLatentBatch,
    CachedLatentDataset,
    collate_cached_latents,
    load_cached_latent_dataset,
)
from ijepa_ad.predictor.losses import prediction_loss
from ijepa_ad.predictor.target_masks import build_inference_aligned_target_mask
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)
from ijepa_ad.utils.seeding import set_seed

MASK_STRATEGIES = ("random", "inference_aligned")
VALIDATION_MASK_SEED_OFFSET = 10_000


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for predictor optimization."""

    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    mask_ratio: float
    loss_type: str
    patience: int
    min_delta: float
    seed: int
    device: str
    # ``random`` keeps the original scattered ``mask_ratio`` masks (and the default,
    # so existing callers/checkpoints behave as before). ``inference_aligned`` draws
    # the coarse-window / isolated-token targets used at inference; see
    # ``ijepa_ad.predictor.target_masks``. The sampler fields are ignored by
    # ``random``; ``mask_ratio`` is ignored by ``inference_aligned``.
    mask_strategy: str = "random"
    block_size: int = 4
    block_probability: float = 0.5
    single_token_count: int = 8

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 < self.mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0, 1)")
        if self.loss_type not in {"cosine", "l2"}:
            raise ValueError("loss_type must be 'cosine' or 'l2'")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.mask_strategy not in MASK_STRATEGIES:
            raise ValueError(f"mask_strategy must be one of {MASK_STRATEGIES}")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if not 0 <= self.block_probability <= 1:
            raise ValueError("block_probability must be in [0, 1]")
        if self.single_token_count < 1:
            raise ValueError("single_token_count must be positive")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class EpochMetrics:
    """Losses recorded for one training epoch."""

    epoch: int
    train_loss: float
    val_loss: float


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned by ``train_predictor``."""

    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path | None


def train_predictor(
    model: LatentTransformerPredictor,
    train_dataset: CachedLatentDataset,
    val_dataset: CachedLatentDataset,
    config: TrainingConfig,
    *,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train a predictor on cached train latents and validate on cached val latents."""
    _validate_training_inputs(model, train_dataset, val_dataset)

    set_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[EpochMetrics] = []
    best_state: dict[str, Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    mask_generator = torch.Generator(device="cpu").manual_seed(config.seed)

    for epoch in range(1, config.max_epochs + 1):
        shuffle_generator = torch.Generator(device="cpu").manual_seed(
            config.seed + epoch
        )
        train_loss = _run_epoch(
            model,
            _iter_batches(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=shuffle_generator,
            ),
            config,
            device,
            mask_generator,
            optimizer=optimizer,
        )
        val_loss = _run_epoch(
            model,
            _iter_batches(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                generator=shuffle_generator,
            ),
            config,
            device,
            torch.Generator(device="cpu").manual_seed(
                config.seed + VALIDATION_MASK_SEED_OFFSET
            ),
        )
        history.append(EpochMetrics(epoch, train_loss, val_loss))
        if val_loss < best_val - config.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if config.patience and epochs_without_improvement >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    saved_path = None
    if checkpoint_path is not None:
        saved_path = Path(checkpoint_path)
        _save_checkpoint(
            saved_path,
            model,
            config,
            history,
            best_epoch,
            best_val,
            train_dataset,
            val_dataset,
        )

    return TrainingResult(tuple(history), best_epoch, best_val, saved_path)


def build_random_target_mask(
    batch_size: int, num_tokens: int, mask_ratio: float, generator: torch.Generator
) -> Tensor:
    """Sample a boolean target mask with at least one target and one context token."""
    if batch_size <= 0 or num_tokens <= 1:
        raise ValueError("batch_size must be positive and num_tokens must exceed 1")
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be in (0, 1)")
    target_count = round(num_tokens * mask_ratio)
    target_count = min(max(target_count, 1), num_tokens - 1)
    scores = torch.rand(batch_size, num_tokens, generator=generator)
    indices = scores.argsort(dim=1)[:, :target_count]
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def build_target_mask(
    config: TrainingConfig,
    batch_size: int,
    num_tokens: int,
    generator: torch.Generator,
) -> Tensor:
    """Sample the target mask for ``config.mask_strategy`` (CPU, boolean)."""
    if config.mask_strategy == "inference_aligned":
        return build_inference_aligned_target_mask(
            batch_size,
            num_tokens,
            generator,
            block_size=config.block_size,
            block_probability=config.block_probability,
            single_token_count=config.single_token_count,
        )
    return build_random_target_mask(
        batch_size, num_tokens, config.mask_ratio, generator
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for later GPU training once real latent caches exist."""
    parser = _parser()
    args = parser.parse_args(argv)
    preprocessing = PreprocessingSpec.from_config(
        PreprocessConfig(image_size=args.image_size)
    )
    train_expected = _expectations(args, "train", preprocessing)
    val_expected = _expectations(args, "val", preprocessing)
    train_dataset = load_cached_latent_dataset(args.cache_root, train_expected)
    val_dataset = load_cached_latent_dataset(args.cache_root, val_expected)
    model_config = PredictorConfig(
        embedding_dim=args.embedding_dim,
        num_tokens=train_dataset.cache.metadata.num_tokens,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    set_seed(args.seed)
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        mask_strategy=args.mask_strategy,
        block_size=args.block_size,
        block_probability=args.block_probability,
        single_token_count=args.single_token_count,
        loss_type=args.loss_type,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device=args.device,
    )
    result = train_predictor(
        LatentTransformerPredictor(model_config),
        train_dataset,
        val_dataset,
        training_config,
        checkpoint_path=args.output,
    )
    print(json.dumps(_result_to_dict(result), sort_keys=True))
    return 0


def _run_epoch(
    model: LatentTransformerPredictor,
    batches: Iterator[CachedLatentBatch],
    config: TrainingConfig,
    device: torch.device,
    mask_generator: torch.Generator,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    dtype = next(model.parameters()).dtype
    for batch in batches:
        latents = batch.latents.to(device=device, dtype=dtype)
        mask = build_target_mask(
            config, latents.shape[0], latents.shape[1], mask_generator
        ).to(device)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predicted = model(latents, mask)
            loss = prediction_loss(predicted, latents, mask, config.loss_type)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError(
                f"non-finite {config.loss_type} loss; refusing to continue training"
            )
        if training:
            assert optimizer is not None
            loss.backward()
            optimizer.step()
        batch_size = int(latents.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("cannot train or validate on an empty dataset")
    return total_loss / total_samples


def _iter_batches(
    dataset: CachedLatentDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
) -> Iterator[CachedLatentBatch]:
    indices = torch.arange(len(dataset))
    if shuffle:
        indices = indices[torch.randperm(len(dataset), generator=generator)]
    for start in range(0, len(dataset), batch_size):
        batch_indices = indices[start : start + batch_size].tolist()
        yield collate_cached_latents([dataset[int(i)] for i in batch_indices])


def _save_checkpoint(
    path: Path,
    model: LatentTransformerPredictor,
    config: TrainingConfig,
    history: Sequence[EpochMetrics],
    best_epoch: int,
    best_val_loss: float,
    train_dataset: CachedLatentDataset,
    val_dataset: CachedLatentDataset,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": config.to_dict(),
        "history": [asdict(item) for item in history],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "cache_provenance": _cache_provenance(train_dataset, val_dataset),
        "reproducibility": _reproducibility_info(config),
    }
    torch.save(payload, path)


def _validate_training_inputs(
    model: LatentTransformerPredictor,
    train_dataset: CachedLatentDataset,
    val_dataset: CachedLatentDataset,
) -> None:
    if train_dataset.category != val_dataset.category:
        raise ValueError("train and val datasets must use the same category")
    if train_dataset.split != "train":
        raise ValueError(
            f"train_dataset split must be 'train', got {train_dataset.split!r}"
        )
    if val_dataset.split != "val":
        raise ValueError(f"val_dataset split must be 'val', got {val_dataset.split!r}")
    overlap = set(train_dataset.sample_ids) & set(val_dataset.sample_ids)
    if overlap:
        sample = sorted(overlap)[0]
        raise ValueError(f"train and val datasets overlap at sample id {sample!r}")
    if train_dataset.cache.metadata.embedding_dim != model.config.embedding_dim:
        raise ValueError("dataset embedding_dim does not match model")
    if train_dataset.cache.metadata.num_tokens != model.config.num_tokens:
        raise ValueError("dataset num_tokens does not match model")
    if val_dataset.cache.metadata.embedding_dim != model.config.embedding_dim:
        raise ValueError("val embedding_dim does not match model")
    if val_dataset.cache.metadata.num_tokens != model.config.num_tokens:
        raise ValueError("val num_tokens does not match model")


def _reproducibility_info(config: TrainingConfig) -> dict[str, Any]:
    """Seeds and software state needed to rerun a training job."""
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain", "--untracked-files=no")
    return {
        "seed": config.seed,
        "train_mask_seed": config.seed,
        "shuffle_seed": "seed + epoch",
        "validation_mask_seed": config.seed + VALIDATION_MASK_SEED_OFFSET,
        "mask_strategy": config.mask_strategy,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def _git_output(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _cache_provenance(
    train_dataset: CachedLatentDataset, val_dataset: CachedLatentDataset
) -> dict[str, Any]:
    train = train_dataset.cache.metadata
    val = val_dataset.cache.metadata
    return {
        "category": train.category,
        "encoder_name": train.encoder_name,
        "checkpoint_sha256": train.checkpoint_sha256,
        "preprocessing": train.preprocessing.to_dict(),
        "train": {
            "split": train.split,
            "latent_dtype": train.latent_dtype,
            "sample_ids_sha256": _sample_ids_sha256(train.sample_ids),
            "sample_ids": list(train.sample_ids),
        },
        "val": {
            "split": val.split,
            "latent_dtype": val.latent_dtype,
            "sample_ids_sha256": _sample_ids_sha256(val.sample_ids),
            "sample_ids": list(val.sample_ids),
        },
    }


def _sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    text = "\n".join(sample_ids)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expectations(
    args: argparse.Namespace, split: str, preprocessing: PreprocessingSpec
) -> CacheExpectations:
    return CacheExpectations(
        category=args.category,
        split=split,
        encoder_name=args.encoder_name,
        checkpoint_sha256=args.checkpoint_sha256,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        preprocessing=preprocessing,
    )


def _result_to_dict(result: TrainingResult) -> dict[str, Any]:
    return {
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "checkpoint_path": (
            None if result.checkpoint_path is None else str(result.checkpoint_path)
        ),
        "history": [asdict(item) for item in result.history],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/latents"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-name", default="ijepa_vit_g16_in22k_ep44")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=1408)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument(
        "--mask-strategy", choices=MASK_STRATEGIES, default="inference_aligned"
    )
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--block-probability", type=float, default=0.5)
    parser.add_argument("--single-token-count", type=int, default=8)
    parser.add_argument("--loss-type", choices=("cosine", "l2"), default="cosine")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser
