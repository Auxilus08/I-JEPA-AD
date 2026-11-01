"""Tests for the pretrained context-encoder + predictor wrapper (fake networks)."""

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ijepa_ad.encoder.ijepa_pretrained_predictor import (
    EMBEDDING_DIM,
    IJEPAContextPredictor,
    PretrainedPredictorError,
)


class _FakeEncoder(nn.Module):
    """Returns each kept token's *index* in channel 0, so tests can see what it saw."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.seen: list[Tensor] = []

    def forward(self, x: Tensor, masks: list[Tensor]) -> Tensor:
        keep = masks[0]
        self.seen.append(keep.clone())
        out = torch.zeros(keep.shape[0], keep.shape[1], EMBEDDING_DIM)
        out[..., 0] = keep.float() * self.scale
        return out


class _FakePredictor(nn.Module):
    """Predicts, per target token: its index, #context tokens, context-overlap flag."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x: Tensor, masks_x: list[Tensor], masks: list[Tensor]) -> Tensor:
        targets = masks[0]
        context_index = x[..., 0].long()
        overlap = (context_index.unsqueeze(-1) == targets.unsqueeze(1)).any(dim=(1, 2))
        out = torch.zeros(targets.shape[0], targets.shape[1], EMBEDDING_DIM)
        out[..., 0] = targets.float() * self.scale
        out[..., 1] = float(x.shape[1])
        out[..., 2] = overlap.float().unsqueeze(-1)
        return out


def _checkpoint(tmp_path: Path, *, prefix: str = "module.") -> Path:
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "encoder": {prefix + k: v for k, v in _FakeEncoder().state_dict().items()},
            "predictor": {
                prefix + k: v for k, v in _FakePredictor().state_dict().items()
            },
            "target_encoder": {"unused": torch.zeros(1)},
        },
        path,
    )
    return path


def _model(tmp_path: Path) -> IJEPAContextPredictor:
    return IJEPAContextPredictor(
        _checkpoint(tmp_path),
        encoder_factory=_FakeEncoder,
        predictor_factory=_FakePredictor,
    )


IMAGE = torch.zeros(1, 3, 224, 224)
SQUARE = [r * 14 + c for r in range(4) for c in range(4)]


def test_context_never_contains_the_masked_window(tmp_path: Path) -> None:
    model = _model(tmp_path)
    (predicted,) = model.predict_windows(IMAGE, [SQUARE])

    assert predicted.shape == (16, EMBEDDING_DIM)
    assert torch.equal(predicted[:, 0], torch.tensor(SQUARE, dtype=torch.float32))
    assert predicted[0, 1] == 196 - 16  # 180 visible tokens
    assert predicted[0, 2] == 0  # no masked token leaked into the context
    seen = model.encoder.seen[0][0]
    assert not set(seen.tolist()) & set(SQUARE)


def test_unequal_windows_are_batched_by_size_and_keep_their_order(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path)
    windows = [SQUARE, [12, 13, 26, 27], [0], [1, 2], [195], [5, 6, 7, 8, 9, 10]]

    results = model.predict_windows(IMAGE, windows, batch_size=2)

    assert [r.shape[0] for r in results] == [len(w) for w in windows]
    for window, result in zip(windows, results, strict=False):
        assert torch.equal(
            result[:, 0], torch.tensor(sorted(window), dtype=torch.float32)
        )
        assert result[0, 1] == 196 - len(window)
        assert result[0, 2] == 0


def test_window_predictions_place_tokens_and_leave_the_rest_zero(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path)
    out = model.window_predictions(IMAGE, [[3, 4], [100]])

    assert out.shape == (196, EMBEDDING_DIM)
    assert out[3, 0] == 3 and out[4, 0] == 4 and out[100, 0] == 100
    assert out[[0, 1, 2, 5, 99]].abs().sum() == 0


def test_batch_size_does_not_change_results(tmp_path: Path) -> None:
    model = _model(tmp_path)
    windows = [[i, i + 1] for i in range(0, 20, 2)]
    one = model.predict_windows(IMAGE, windows, batch_size=1)
    many = model.predict_windows(IMAGE, windows, batch_size=7)

    assert all(torch.equal(a, b) for a, b in zip(one, many, strict=False))


def test_invalid_windows_and_images_are_rejected(tmp_path: Path) -> None:
    model = _model(tmp_path)
    for bad in ([], [-1], [196], [3, 3], list(range(196))):
        with pytest.raises(PretrainedPredictorError, match="invalid window"):
            model.predict_windows(IMAGE, [bad])
    with pytest.raises(PretrainedPredictorError, match="image must be"):
        model.predict_windows(torch.zeros(2, 3, 224, 224), [[1]])
    with pytest.raises(ValueError, match="batch_size"):
        model.predict_windows(IMAGE, [[1]], batch_size=0)


def test_model_is_frozen_eval_and_cannot_be_unfrozen(tmp_path: Path) -> None:
    model = _model(tmp_path)

    assert not model.training and not model.encoder.training
    assert not any(p.requires_grad for p in model.parameters())
    model.train()
    assert not model.training
    with pytest.raises(RuntimeError, match="frozen"):
        model.requires_grad_(True)


def test_checkpoint_must_match_exactly_and_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        IJEPAContextPredictor(
            tmp_path / "missing.pt",
            encoder_factory=_FakeEncoder,
            predictor_factory=_FakePredictor,
        )
    broken = tmp_path / "broken.pt"
    torch.save({"encoder": {"module.wrong": torch.zeros(1)}, "predictor": {}}, broken)
    with pytest.raises((PretrainedPredictorError, RuntimeError)):
        IJEPAContextPredictor(
            broken, encoder_factory=_FakeEncoder, predictor_factory=_FakePredictor
        )
    no_predictor = tmp_path / "nopred.pt"
    torch.save({"encoder": _FakeEncoder().state_dict()}, no_predictor)
    with pytest.raises(PretrainedPredictorError, match="predictor"):
        IJEPAContextPredictor(
            no_predictor, encoder_factory=_FakeEncoder, predictor_factory=_FakePredictor
        )


def test_plain_unprefixed_checkpoints_load_too(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, prefix="")
    model = IJEPAContextPredictor(
        path, encoder_factory=_FakeEncoder, predictor_factory=_FakePredictor
    )

    assert model.predict_windows(IMAGE, [[7]])[0].shape == (1, EMBEDDING_DIM)
