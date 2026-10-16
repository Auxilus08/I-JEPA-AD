"""Training with inference-aligned masks, non-finite guard and checkpoint provenance."""

from pathlib import Path

import pytest
import torch

from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.latent_cache import (
    build_metadata,
    latent_cache_path,
    save_latent_cache,
)
from ijepa_ad.data.transforms import PreprocessConfig
from ijepa_ad.predictor import training as training_module
from ijepa_ad.predictor.cache_dataset import (
    CachedLatentDataset,
    load_cached_latent_dataset,
)
from ijepa_ad.predictor.target_masks import build_inference_aligned_target_mask
from ijepa_ad.predictor.training import TrainingConfig, train_predictor
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)

Datasets = tuple[CachedLatentDataset, CachedLatentDataset]
CATEGORY = "bottle"
ENCODER = "synthetic_encoder"
CHECKPOINT = "b" * 64
GRID = 4  # 16 tokens
DIM = 4
PREPROCESS = PreprocessConfig(image_size=64)


def _expected(split: str) -> CacheExpectations:
    return CacheExpectations(
        category=CATEGORY,
        split=split,
        encoder_name=ENCODER,
        checkpoint_sha256=CHECKPOINT,
        patch_size=16,
        embedding_dim=DIM,
        preprocessing=PreprocessingSpec.from_config(PREPROCESS),
    )


def _write_cache(root: Path, split: str, count: int) -> None:
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn(count, GRID * GRID, DIM, generator=generator)
    metadata = build_metadata(
        latents,
        category=CATEGORY,
        split=split,
        sample_ids=tuple(f"{split}-{index}" for index in range(count)),
        encoder_name=ENCODER,
        checkpoint_sha256=CHECKPOINT,
        patch_size=16,
        preprocessing=PreprocessingSpec.from_config(PREPROCESS),
    )
    save_latent_cache(latent_cache_path(root, CATEGORY, split), latents, metadata)


def _model() -> LatentTransformerPredictor:
    return LatentTransformerPredictor(
        PredictorConfig(
            embedding_dim=DIM,
            num_tokens=GRID * GRID,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
        )
    )


def _config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "batch_size": 4,
        "max_epochs": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "mask_ratio": 0.5,
        "loss_type": "cosine",
        "patience": 0,
        "min_delta": 0.0,
        "seed": 3,
        "device": "cpu",
        "mask_strategy": "inference_aligned",
        "block_size": 2,
        "single_token_count": 3,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


@pytest.fixture  # type: ignore[misc]
def datasets(tmp_path: Path) -> Datasets:
    _write_cache(tmp_path, "train", 4)
    _write_cache(tmp_path, "val", 2)
    return (
        load_cached_latent_dataset(tmp_path, _expected("train")),
        load_cached_latent_dataset(tmp_path, _expected("val")),
    )


def test_training_and_validation_use_aligned_masks_and_validation_repeats(
    datasets: Datasets, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, val = datasets
    seen: list[torch.Tensor] = []
    original = build_inference_aligned_target_mask

    def record(*args: object, **kwargs: object) -> torch.Tensor:
        mask = original(*args, **kwargs)  # type: ignore[arg-type]
        seen.append(mask.clone())
        return mask

    monkeypatch.setattr(
        "ijepa_ad.predictor.training.build_inference_aligned_target_mask", record
    )

    train_predictor(_model(), train, val, _config())

    assert len(seen) == 4  # (train, val) x 2 epochs
    assert torch.equal(seen[1], seen[3])  # validation masks fixed across epochs
    assert not torch.equal(seen[0], seen[2])  # training masks keep varying
    for mask in seen:
        assert (mask.sum(dim=1) >= 1).all() and (mask.sum(dim=1) < GRID * GRID).all()


def test_training_is_reproducible_for_a_seed(datasets: Datasets) -> None:
    train, val = datasets
    torch.manual_seed(0)
    first = _model()
    second = _model()
    second.load_state_dict(first.state_dict())

    one = train_predictor(first, train, val, _config())
    two = train_predictor(second, train, val, _config())

    assert [m.val_loss for m in one.history] == [m.val_loss for m in two.history]
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            first.state_dict().values(), second.state_dict().values(), strict=False
        )
    )


def test_checkpoint_records_mask_strategy_seeds_and_environment(
    datasets: Datasets, tmp_path: Path
) -> None:
    train, val = datasets
    path = tmp_path / "predictor.pt"

    train_predictor(_model(), train, val, _config(), checkpoint_path=path)

    payload = torch.load(path, weights_only=True)
    assert payload["training_config"]["mask_strategy"] == "inference_aligned"
    assert payload["training_config"]["block_size"] == 2
    info = payload["reproducibility"]
    assert info["seed"] == 3 and info["validation_mask_seed"] == 3 + 10_000
    assert info["mask_strategy"] == "inference_aligned"
    assert info["torch_version"] == str(torch.__version__)
    assert "git_commit" in info and "git_dirty" in info
    assert payload["cache_provenance"]["checkpoint_sha256"] == CHECKPOINT


def test_non_finite_loss_stops_training_instead_of_saving_nans(
    datasets: Datasets, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, val = datasets
    monkeypatch.setattr(
        training_module,
        "prediction_loss",
        lambda predicted, *_args: predicted.sum() * float("nan"),
    )
    path = tmp_path / "predictor.pt"

    with pytest.raises(FloatingPointError, match="non-finite"):
        train_predictor(_model(), train, val, _config(), checkpoint_path=path)

    assert not path.exists()
