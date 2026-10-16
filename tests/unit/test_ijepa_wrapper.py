"""Tests for the frozen I-JEPA ViT-g/16 wrapper contract."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor, nn  # noqa: E402

from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder  # noqa: E402


class _TinyPatchEncoder(nn.Module):
    """Small patch encoder with the official ViT-g/16 output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3 * 16 * 16, 1408)

    def forward(self, image_batch: Tensor) -> Tensor:
        patches = image_batch.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(3)
        return self.projection(patches.flatten(0, 2)).view(
            image_batch.shape[0], 196, 1408
        )


def _make_encoder(tmp_path: Path) -> IJEPAEncoder:
    model = _TinyPatchEncoder()
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"target_encoder": model.state_dict()}, checkpoint)
    return IJEPAEncoder(
        checkpoint,
        model_factory=_TinyPatchEncoder,
    )


def test_encoder_is_frozen_at_load_time(tmp_path: Path) -> None:
    encoder = _make_encoder(tmp_path)

    assert encoder.training is False
    assert encoder.model.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_encoder_returns_196_vitg16_patch_tokens(tmp_path: Path) -> None:
    encoder = _make_encoder(tmp_path)

    latents = encoder.encode(torch.zeros(2, 3, 224, 224))

    assert latents.shape == (2, 196, 1408)


def test_encoder_loads_distributed_checkpoint_keys(tmp_path: Path) -> None:
    model = _TinyPatchEncoder()
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "target_encoder": {
                f"module.{name}": value for name, value in model.state_dict().items()
            }
        },
        checkpoint,
    )

    encoder = IJEPAEncoder(checkpoint, model_factory=_TinyPatchEncoder)

    assert encoder.encode(torch.zeros(1, 3, 224, 224)).shape == (1, 196, 1408)


def test_encoder_rejects_mixed_distributed_checkpoint_keys(tmp_path: Path) -> None:
    model = _TinyPatchEncoder()
    state_dict = model.state_dict()
    first_name, first_value = next(iter(state_dict.items()))
    state_dict.pop(first_name)
    state_dict[f"module.{first_name}"] = first_value
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"target_encoder": state_dict}, checkpoint)

    with pytest.raises(ValueError, match="mixes distributed and unprefixed keys"):
        IJEPAEncoder(checkpoint, model_factory=_TinyPatchEncoder)
