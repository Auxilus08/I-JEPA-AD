"""Intermediate-layer patch features of the frozen I-JEPA target encoder.

The cached latents are the encoder's *final* output (after its last LayerNorm).
Mid-level features are often more useful for anomaly detection (PatchCore uses
them), so this module reads the output of any transformer block with forward hooks
on the same frozen target encoder ``IJEPAEncoder`` already loads. No weights change.

Layer ``l`` (1-based) is the output of block ``l`` *before* the encoder's final
LayerNorm; layer ``depth`` is therefore the final block without that norm. Stored
features get a parameter-free per-token layer norm (``token_layer_norm``): it fixes
the scale across layers (the residual stream grows with depth) and keeps values
inside float16 range, so the features can be cached compactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder

LAYER_FEATURE_VERSION = 1


class LayerFeatureError(ValueError):
    """Layer features do not match what the caller expects."""


def token_layer_norm(features: Tensor) -> Tensor:
    """Parameter-free LayerNorm over the channel axis of ``[..., D]`` features."""
    return F.layer_norm(features.float(), (features.shape[-1],))


class LayerFeatureExtractor:
    """Capture the output of chosen blocks of a frozen ``IJEPAEncoder``."""

    def __init__(self, encoder: IJEPAEncoder, layers: Sequence[int]) -> None:
        blocks = getattr(encoder.model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise LayerFeatureError("encoder.model must expose a ModuleList 'blocks'")
        self.depth = len(blocks)
        self.layers = tuple(sorted(set(layers)))
        if not self.layers or self.layers[0] < 1 or self.layers[-1] > self.depth:
            raise LayerFeatureError(f"layers must be within 1..{self.depth}")
        self.encoder = encoder
        self._blocks = blocks

    def extract(self, image_batch: Tensor) -> dict[int, Tensor]:
        """Return ``{layer: [N, tokens, D] float32 (CPU, token-layer-normed)}``."""
        captured: dict[int, Tensor] = {}
        handles = []

        def make_hook(layer: int) -> Any:
            def hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
                captured[layer] = token_layer_norm(output.detach()).cpu()

            return hook

        try:
            for layer in self.layers:
                handles.append(
                    self._blocks[layer - 1].register_forward_hook(make_hook(layer))
                )
            self.encoder.encode(image_batch)
        finally:
            for handle in handles:
                handle.remove()
        missing = set(self.layers) - set(captured)
        if missing:
            raise LayerFeatureError(f"blocks {sorted(missing)} produced no output")
        return captured


@dataclass(frozen=True)
class LayerFeatureSet:
    """Features of one (category, split, layer) plus the provenance they carry."""

    features: Tensor  # [N, tokens, D] float16
    sample_ids: tuple[str, ...]
    layer: int


def layer_feature_path(root: str | Path, layer: int, category: str, split: str) -> Path:
    return Path(root) / f"layer{layer:02d}" / f"{category}_{split}.pt"


def _metadata(
    layer: int,
    checkpoint_sha256: str,
    preprocessing: dict[str, Any],
    ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "version": LAYER_FEATURE_VERSION,
        "layer": layer,
        "normalisation": "parameter-free token layer norm",
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing": preprocessing,
        "sample_ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
        "num_samples": len(ids),
    }


def save_layer_features(
    root: str | Path,
    *,
    layer: int,
    category: str,
    split: str,
    features: Tensor,
    sample_ids: Sequence[str],
    checkpoint_sha256: str,
    preprocessing: dict[str, Any],
) -> Path:
    """Write float16 features; refuses to overwrite."""
    if features.ndim != 3 or features.shape[0] != len(sample_ids):
        raise LayerFeatureError("features must be [N, tokens, D] with one row per id")
    if not bool(torch.isfinite(features).all()):
        raise LayerFeatureError("features contain non-finite values")
    path = layer_feature_path(root, layer, category, split)
    if path.exists():
        raise LayerFeatureError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": features.to(torch.float16).cpu(),
        "metadata": _metadata(layer, checkpoint_sha256, preprocessing, sample_ids),
        "sample_ids": list(sample_ids),
    }
    torch.save(payload, path)
    return path


def load_layer_features(
    root: str | Path,
    *,
    layer: int,
    category: str,
    split: str,
    sample_ids: Sequence[str],
    checkpoint_sha256: str,
    preprocessing: dict[str, Any],
) -> LayerFeatureSet:
    """Load features, rejecting any mismatch in layer, encoder, preprocessing or ids."""
    path = layer_feature_path(root, layer, category, split)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = _metadata(layer, checkpoint_sha256, preprocessing, sample_ids)
    if payload["metadata"] != expected:
        diff = sorted(k for k in expected if payload["metadata"].get(k) != expected[k])
        raise LayerFeatureError(f"{path} does not match the request in: {diff}")
    if tuple(payload["sample_ids"]) != tuple(sample_ids):
        raise LayerFeatureError(f"{path} sample ids differ in order")
    return LayerFeatureSet(payload["features"], tuple(sample_ids), layer)
