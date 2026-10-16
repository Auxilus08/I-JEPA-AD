"""Frozen wrapper for Meta's official image-based I-JEPA ViT-g/16 encoder."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

META_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar"
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
EMBEDDING_DIM = 1408


class IJEPAEncoder(nn.Module):
    """Load and expose Meta's frozen I-JEPA ViT-g/16 target encoder.

    ``official_repo_path`` must point to a checkout of Meta's archived ``ijepa``
    repository. ``model_factory`` is a test seam that avoids constructing the
    billion-parameter official model in unit tests; production callers should use
    the default factory.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        official_repo_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        model_factory: Callable[[], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"I-JEPA checkpoint not found: {checkpoint_path}")

        self.model = (
            model_factory()
            if model_factory is not None
            else _build_meta_vitg16(official_repo_path)
        )
        checkpoint = _load_checkpoint(self.checkpoint_path)
        state_dict = _extract_target_encoder(checkpoint)
        incompatible = self.model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "I-JEPA checkpoint does not exactly match the ViT-g/16 encoder: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        self.model.to(device)
        self.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._assert_frozen()

    def encode(self, image_batch: Tensor) -> Tensor:
        """Encode ``N×3×224×224`` images as ``N×196×1408`` patch latents."""
        if image_batch.ndim != 4:
            raise ValueError(
                f"image_batch must be N×C×H×W, received {tuple(image_batch.shape)}"
            )
        if image_batch.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                "I-JEPA ViT-g/16 expects N×3×224×224 input, received "
                f"{tuple(image_batch.shape)}"
            )

        with torch.inference_mode():
            latents = self.model(image_batch)
        if latents.shape != (image_batch.shape[0], NUM_PATCHES, EMBEDDING_DIM):
            raise RuntimeError(
                "I-JEPA encoder returned an unexpected shape: "
                f"{tuple(latents.shape)}; expected "
                f"({image_batch.shape[0]}, {NUM_PATCHES}, {EMBEDDING_DIM})"
            )
        return latents

    def _assert_frozen(self) -> None:
        trainable = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        assert not trainable, f"I-JEPA encoder parameters are trainable: {trainable}"


def _build_meta_vitg16(official_repo_path: str | Path | None) -> nn.Module:
    """Construct ``vit_giant(patch_size=16)`` from Meta's official source tree."""
    if official_repo_path is None:
        raise ValueError(
            "official_repo_path is required to load Meta's archived I-JEPA source"
        )
    repo_path = Path(official_repo_path).resolve()
    if not (repo_path / "src" / "models" / "vision_transformer.py").is_file():
        raise FileNotFoundError(
            "official_repo_path must point to the Meta ijepa checkout containing "
            "src/models/vision_transformer.py"
        )

    sys.path.insert(0, str(repo_path))
    try:
        module = importlib.import_module("src.models.vision_transformer")
        return module.vit_giant(patch_size=PATCH_SIZE, img_size=[IMAGE_SIZE])
    finally:
        sys.path.remove(str(repo_path))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"I-JEPA checkpoint must contain a mapping: {path}")
    return checkpoint


def _extract_target_encoder(checkpoint: dict[str, Any]) -> dict[str, Tensor]:
    state_dict = checkpoint.get("target_encoder")
    if not isinstance(state_dict, dict):
        raise KeyError(
            "Meta I-JEPA checkpoint is missing its 'target_encoder' state dict"
        )

    keys = tuple(state_dict)
    if not all(isinstance(key, str) for key in keys):
        raise TypeError("I-JEPA target_encoder state-dict keys must be strings")
    has_module_prefix = tuple(key.startswith("module.") for key in keys)
    if any(has_module_prefix) and not all(has_module_prefix):
        raise ValueError(
            "I-JEPA target_encoder state dict mixes distributed and unprefixed keys"
        )

    strip_module_prefix = bool(keys) and all(has_module_prefix)
    normalized: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"I-JEPA target_encoder value for {key!r} is not a tensor")
        normalized_key = key.removeprefix("module.") if strip_module_prefix else key
        if normalized_key in normalized:
            raise ValueError(
                f"I-JEPA target_encoder key collision after prefix normalization: "
                f"{normalized_key}"
            )
        normalized[normalized_key] = value
    return normalized
