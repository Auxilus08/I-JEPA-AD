"""Meta's pretrained I-JEPA context encoder and predictor, run with input masking.

The checkpoint ``IN22K-vit.g.16-600e.pth.tar`` holds three networks: ``encoder``
(the context encoder), ``predictor`` and ``target_encoder`` (the one
``IJEPAEncoder`` loads and whose latents are cached). Meta trained the predictor to
predict the target encoder's latents of masked blocks from the context encoder's
latents of the *visible* patches only.

This wrapper reproduces that inference path for one masked window at a time:

1. the context encoder sees only the tokens outside the window (input masking, so
   no visible token has attended to the masked region),
2. the pretrained predictor predicts the window's tokens from those latents,
3. the caller compares them with the cached target-encoder latents.

Nothing is trained or fine-tuned here and no weights are modified. The checkpoint is
memory-mapped (``mmap=True``), so loading needs a few hundred MB of host RAM instead
of the 16.5 GB file size (most of which is optimizer state).
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
EMBEDDING_DIM = 1408
# Hyper-parameters Meta used for this checkpoint (read off the saved shapes:
# 16 blocks, width 384, MLP 1536 = 4 x 384; heads follow the encoder, 16).
PREDICTOR_EMBED_DIM = 384
PREDICTOR_DEPTH = 16
PREDICTOR_HEADS = 16


class PretrainedPredictorError(ValueError):
    """The checkpoint or the requested windows cannot be used."""


class IJEPAContextPredictor(nn.Module):
    """Context encoder plus predictor of the official I-JEPA ViT-g/16 checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        official_repo_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        encoder_factory: Callable[[], nn.Module] | None = None,
        predictor_factory: Callable[[], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"I-JEPA checkpoint not found: {checkpoint_path}")
        if encoder_factory is None or predictor_factory is None:
            module = _meta_module(official_repo_path)
            encoder_factory = encoder_factory or (
                lambda: module.vit_giant(patch_size=PATCH_SIZE, img_size=[IMAGE_SIZE])
            )
            predictor_factory = predictor_factory or (
                lambda: module.vit_predictor(
                    num_patches=NUM_PATCHES,
                    embed_dim=EMBEDDING_DIM,
                    predictor_embed_dim=PREDICTOR_EMBED_DIM,
                    depth=PREDICTOR_DEPTH,
                    num_heads=PREDICTOR_HEADS,
                )
            )
        checkpoint = torch.load(
            str(path), map_location="cpu", mmap=True, weights_only=True
        )
        self.encoder = encoder_factory()
        self.predictor = predictor_factory()
        _load_exact(self.encoder, checkpoint, "encoder")
        _load_exact(self.predictor, checkpoint, "predictor")
        del checkpoint
        self.to(device=device, dtype=dtype)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> IJEPAContextPredictor:
        """Stay in eval mode: the pretrained weights are frozen."""
        super().train(False)
        return self

    def requires_grad_(self, requires_grad: bool = True) -> IJEPAContextPredictor:
        if requires_grad:
            raise RuntimeError("the pretrained I-JEPA networks are frozen")
        super().requires_grad_(False)
        return self

    def predict_windows(
        self,
        image: Tensor,
        windows: Sequence[Sequence[int]],
        *,
        batch_size: int = 8,
    ) -> list[Tensor]:
        """Predict each window's target latents from the tokens outside it.

        ``image`` is one normalised ``1x3x224x224`` tensor. ``windows`` are lists of
        flat token indices (row-major over the 14 x 14 grid). Returns one
        ``[len(window), 1408]`` float32 tensor per window, in the same order.
        Windows with the same number of tokens share a batch (the gather in Meta's
        code needs rectangular index tensors); results are identical to running
        them one by one up to floating-point batching noise.
        """
        if image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise PretrainedPredictorError(
                f"image must be 1x3x{IMAGE_SIZE}x{IMAGE_SIZE}, got {tuple(image.shape)}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        parameter = next(self.encoder.parameters())
        image = image.to(device=parameter.device, dtype=parameter.dtype)
        all_tokens = torch.arange(NUM_PATCHES, device=image.device)
        checked: list[tuple[int, ...]] = []
        for window in windows:
            tokens = tuple(sorted(window))
            if not tokens or tokens[0] < 0 or tokens[-1] >= NUM_PATCHES:
                raise PretrainedPredictorError(f"invalid window {list(window)}")
            if len(set(tokens)) != len(tokens) or len(tokens) >= NUM_PATCHES:
                raise PretrainedPredictorError(f"invalid window {list(window)}")
            checked.append(tokens)

        results: list[Tensor | None] = [None] * len(checked)
        by_size: dict[int, list[int]] = {}
        for position, tokens in enumerate(checked):
            by_size.setdefault(len(tokens), []).append(position)
        with torch.inference_mode():
            for positions in by_size.values():
                for start in range(0, len(positions), batch_size):
                    chunk = positions[start : start + batch_size]
                    targets = torch.tensor(
                        [checked[p] for p in chunk], device=image.device
                    )
                    keep = torch.stack(
                        [
                            all_tokens[~torch.isin(all_tokens, targets[row])]
                            for row in range(len(chunk))
                        ]
                    )
                    context = self.encoder(image.expand(len(chunk), -1, -1, -1), [keep])
                    predicted = self.predictor(context, [keep], [targets])
                    for row, position in enumerate(chunk):
                        results[position] = predicted[row].float().cpu()
        return [result for result in results if result is not None]

    def window_predictions(
        self,
        image: Tensor,
        windows: Sequence[Sequence[int]],
        *,
        batch_size: int = 8,
    ) -> Tensor:
        """``[196, 1408]`` predictions placed at each window's own token positions.

        The counterpart of ``MaskedLatentScorer.window_predictions``; tokens outside
        every window are zero.
        """
        out = torch.zeros(NUM_PATCHES, EMBEDDING_DIM)
        for window, predicted in zip(
            windows,
            self.predict_windows(image, windows, batch_size=batch_size),
            strict=False,
        ):
            out[sorted(window)] = predicted
        return out


def _meta_module(official_repo_path: str | Path | None) -> Any:
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
        return importlib.import_module("src.models.vision_transformer")
    finally:
        sys.path.remove(str(repo_path))


def _load_exact(model: nn.Module, checkpoint: dict[str, Any], key: str) -> None:
    state = checkpoint.get(key)
    if not isinstance(state, dict) or not state:
        raise PretrainedPredictorError(f"checkpoint has no {key!r} state dict")
    prefixed = [name.startswith("module.") for name in state]
    if any(prefixed) and not all(prefixed):
        raise PretrainedPredictorError(f"{key!r} mixes prefixed and plain keys")
    cleaned = {
        (name.removeprefix("module.") if all(prefixed) else name): value
        for name, value in state.items()
    }
    result = model.load_state_dict(cleaned, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise PretrainedPredictorError(f"{key!r} does not match the model")
