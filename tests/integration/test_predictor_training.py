from collections.abc import Iterator
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
from ijepa_ad.predictor.cache_dataset import load_cached_latent_dataset
from ijepa_ad.predictor.training import (
    TrainingConfig,
    TrainingResult,
    build_random_target_mask,
    train_predictor,
)
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)

CATEGORY = "bottle"
ENCODER = "synthetic_encoder"
CHECKPOINT = "b" * 64
PATCH_SIZE = 16
TOKENS = 4
DIM = 4
PREPROCESS = PreprocessConfig(image_size=32)


def _expected(split: str) -> CacheExpectations:
    return CacheExpectations(
        category=CATEGORY,
        split=split,
        encoder_name=ENCODER,
        checkpoint_sha256=CHECKPOINT,
        patch_size=PATCH_SIZE,
        embedding_dim=DIM,
        preprocessing=PreprocessingSpec.from_config(PREPROCESS),
    )


def _write_cache(
    root: Path,
    split: str,
    count: int,
    *,
    dtype: torch.dtype = torch.float32,
    sample_ids: tuple[str, ...] | None = None,
) -> None:
    if sample_ids is None:
        sample_ids = tuple(f"{split}-{index}" for index in range(count))
    latents = _synthetic_latents(count).to(dtype=dtype)
    metadata = build_metadata(
        latents,
        category=CATEGORY,
        split=split,
        sample_ids=sample_ids,
        encoder_name=ENCODER,
        checkpoint_sha256=CHECKPOINT,
        patch_size=PATCH_SIZE,
        preprocessing=PreprocessingSpec.from_config(PREPROCESS),
    )
    save_latent_cache(latent_cache_path(root, CATEGORY, split), latents, metadata)


def _synthetic_latents(count: int) -> torch.Tensor:
    rows = []
    for sample_index in range(count):
        base = float(sample_index + 1)
        tokens = [[base, base + 1.0, base - 1.0, base * 0.5] for _ in range(TOKENS)]
        rows.append(tokens)
    return torch.tensor(rows, dtype=torch.float32)


def _training_config(*, seed: int = 0, max_epochs: int = 1) -> TrainingConfig:
    return TrainingConfig(
        batch_size=4,
        max_epochs=max_epochs,
        learning_rate=0.001,
        weight_decay=0.0,
        mask_ratio=0.5,
        loss_type="l2",
        patience=0,
        min_delta=0.0,
        seed=seed,
        device="cpu",
    )


def _model() -> LatentTransformerPredictor:
    return LatentTransformerPredictor(
        PredictorConfig(
            embedding_dim=DIM,
            num_tokens=TOKENS,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
        )
    )


def test_random_target_mask_keeps_context_and_target_tokens() -> None:
    mask = build_random_target_mask(
        batch_size=5,
        num_tokens=8,
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(0),
    )

    assert mask.shape == (5, 8)
    assert mask.dtype == torch.bool
    assert torch.equal(mask.sum(dim=1), torch.full((5,), 4))


def test_predictor_overfits_tiny_synthetic_cache_and_saves_checkpoint(
    tmp_path: Path,
) -> None:
    _write_cache(tmp_path, "train", 4)
    _write_cache(tmp_path, "val", 4)
    train = load_cached_latent_dataset(tmp_path, _expected("train"))
    val = load_cached_latent_dataset(tmp_path, _expected("val"))
    torch.manual_seed(0)  # model init is outside train_predictor's own seeding
    model = LatentTransformerPredictor(
        PredictorConfig(
            embedding_dim=DIM,
            num_tokens=TOKENS,
            hidden_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        )
    )
    checkpoint = tmp_path / "predictor.pt"

    result = train_predictor(
        model,
        train,
        val,
        TrainingConfig(
            batch_size=4,
            max_epochs=120,
            learning_rate=0.02,
            weight_decay=0.0,
            mask_ratio=0.5,
            loss_type="l2",
            patience=0,
            min_delta=0.0,
            seed=0,
            device="cpu",
        ),
        checkpoint_path=checkpoint,
    )

    assert result.history[0].val_loss > result.best_val_loss
    assert result.best_val_loss < result.history[0].val_loss * 0.25
    assert result.best_val_loss < 1.0
    assert result.checkpoint_path == checkpoint
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["best_epoch"] == result.best_epoch
    assert payload["model_config"]["embedding_dim"] == DIM
    provenance = payload["cache_provenance"]
    assert provenance["category"] == CATEGORY
    assert provenance["encoder_name"] == ENCODER
    assert provenance["checkpoint_sha256"] == CHECKPOINT
    assert (
        provenance["preprocessing"]
        == PreprocessingSpec.from_config(PREPROCESS).to_dict()
    )
    assert provenance["train"]["latent_dtype"] == "float32"
    assert provenance["val"]["latent_dtype"] == "float32"
    assert provenance["train"]["sample_ids"] == [f"train-{index}" for index in range(4)]
    assert provenance["val"]["sample_ids"] == [f"val-{index}" for index in range(4)]


def test_training_rejects_same_cache_for_train_and_validation(
    tmp_path: Path,
) -> None:
    _write_cache(tmp_path, "train", 2)
    train = load_cached_latent_dataset(tmp_path, _expected("train"))

    with pytest.raises(ValueError, match="val_dataset split must be 'val'"):
        train_predictor(_model(), train, train, _training_config())


def test_training_rejects_overlapping_sample_ids_before_optimization(
    tmp_path: Path,
) -> None:
    _write_cache(tmp_path, "train", 2, sample_ids=("shared", "train-only"))
    _write_cache(tmp_path, "val", 2, sample_ids=("shared", "val-only"))
    train = load_cached_latent_dataset(tmp_path, _expected("train"))
    val = load_cached_latent_dataset(tmp_path, _expected("val"))
    model = _model()
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(ValueError, match="overlap at sample id 'shared'"):
        train_predictor(model, train, val, _training_config())

    assert all(
        torch.equal(value, initial_state[name])
        for name, value in model.state_dict().items()
    )


def _assert_training_accepts_dtype(tmp_path: Path, dtype: torch.dtype) -> None:
    _write_cache(tmp_path, "train", 2, dtype=dtype)
    _write_cache(tmp_path, "val", 2, dtype=dtype)
    train = load_cached_latent_dataset(tmp_path, _expected("train"))
    val = load_cached_latent_dataset(tmp_path, _expected("val"))

    result = train_predictor(_model(), train, val, _training_config())

    assert len(result.history) == 1
    assert torch.isfinite(torch.tensor(result.history[0].train_loss))
    assert torch.isfinite(torch.tensor(result.history[0].val_loss))


def test_training_accepts_float16_cached_latents(tmp_path: Path) -> None:
    _assert_training_accepts_dtype(tmp_path, torch.float16)


def test_training_accepts_bfloat16_cached_latents(tmp_path: Path) -> None:
    _assert_training_accepts_dtype(tmp_path, torch.bfloat16)


def test_validation_masks_repeat_each_epoch_independently_of_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(tmp_path, "train", 2)
    _write_cache(tmp_path, "val", 2)
    train = load_cached_latent_dataset(tmp_path, _expected("train"))
    val = load_cached_latent_dataset(tmp_path, _expected("val"))
    sampled_masks: list[torch.Tensor] = []
    original = training_module.build_random_target_mask

    def record_masks(
        batch_size: int,
        num_tokens: int,
        mask_ratio: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        mask = original(batch_size, num_tokens, mask_ratio, generator)
        sampled_masks.append(mask.clone())
        return mask

    monkeypatch.setattr(training_module, "build_random_target_mask", record_masks)

    train_predictor(_model(), train, val, _training_config(max_epochs=2))

    # Per epoch: training batch call(s) followed by one call per validation sample.
    per_epoch = len(sampled_masks) // 2
    assert per_epoch * 2 == len(sampled_masks) and per_epoch > len(val)
    first, second = sampled_masks[:per_epoch], sampled_masks[per_epoch:]
    assert not torch.equal(first[0], second[0])  # training masks keep varying
    for before, after in zip(first[-len(val) :], second[-len(val) :], strict=True):
        assert torch.equal(before, after)  # validation masks repeat each epoch


def test_batch_iterator_yields_batches_lazily(tmp_path: Path) -> None:
    _write_cache(tmp_path, "train", 4)
    train = load_cached_latent_dataset(tmp_path, _expected("train"))

    batches = training_module._iter_batches(
        train,
        batch_size=2,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )

    assert isinstance(batches, Iterator)
    assert next(batches).image_ids == ("train-0", "train-1")
    assert next(batches).image_ids == ("train-2", "train-3")


def test_cli_seed_reproduces_predictor_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_cache(tmp_path, "train", 2)
    _write_cache(tmp_path, "val", 2)
    initial_states: list[dict[str, torch.Tensor]] = []

    def capture_model(
        model: LatentTransformerPredictor,
        *_args: object,
        **_kwargs: object,
    ) -> TrainingResult:
        initial_states.append(
            {name: value.detach().clone() for name, value in model.state_dict().items()}
        )
        return TrainingResult((), 0, 0.0, None)

    monkeypatch.setattr(training_module, "train_predictor", capture_model)
    args = [
        "--cache-root",
        str(tmp_path),
        "--category",
        CATEGORY,
        "--checkpoint-sha256",
        CHECKPOINT,
        "--encoder-name",
        ENCODER,
        "--patch-size",
        str(PATCH_SIZE),
        "--embedding-dim",
        str(DIM),
        "--image-size",
        "32",
        "--hidden-dim",
        "16",
        "--depth",
        "1",
        "--num-heads",
        "2",
        "--mlp-ratio",
        "2",
        "--seed",
        "731",
        "--output",
        str(tmp_path / "predictor.pt"),
    ]

    assert training_module.main(args) == 0
    capsys.readouterr()
    assert training_module.main(args) == 0
    capsys.readouterr()

    assert len(initial_states) == 2
    assert all(
        torch.equal(value, initial_states[0][name])
        for name, value in initial_states[1].items()
    )
