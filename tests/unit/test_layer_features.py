"""Tests for intermediate-layer feature extraction and its cache."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from ijepa_ad.encoder.layer_features import (
    LayerFeatureError,
    LayerFeatureExtractor,
    load_layer_features,
    save_layer_features,
    token_layer_norm,
)


class _Block(nn.Module):
    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        return x * 2 + self.shift


class _FakeModel(nn.Module):
    def __init__(self, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(float(i)) for i in range(depth))
        self.norm = nn.LayerNorm(6)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class _FakeEncoder:
    def __init__(self) -> None:
        self.model = _FakeModel()

    def encode(self, batch: Tensor) -> Tensor:
        return self.model(batch)


def _batch() -> Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(3, 5, 6, generator=generator)


def _extractor(layers: list[int]) -> LayerFeatureExtractor:
    return LayerFeatureExtractor(_FakeEncoder(), layers)  # type: ignore[arg-type]


def test_extractor_returns_requested_blocks_normed_and_on_cpu() -> None:
    encoder = _FakeEncoder()
    x = _batch()
    out = LayerFeatureExtractor(encoder, [2, 4]).extract(x)  # type: ignore[arg-type]

    assert sorted(out) == [2, 4]
    expected2 = token_layer_norm(encoder.model.blocks[1](encoder.model.blocks[0](x)))
    assert torch.allclose(out[2], expected2, atol=1e-5)
    assert out[2].device.type == "cpu" and out[2].dtype == torch.float32
    assert torch.allclose(out[4].mean(-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(out[4].std(-1, unbiased=False), torch.ones(3, 5), atol=1e-3)


def test_hooks_are_removed_after_extraction() -> None:
    encoder = _FakeEncoder()
    extractor = LayerFeatureExtractor(encoder, [1, 3])  # type: ignore[arg-type]
    extractor.extract(_batch())
    extractor.extract(_batch())

    assert all(len(block._forward_hooks) == 0 for block in encoder.model.blocks)


def test_invalid_layers_and_models_are_rejected() -> None:
    for layers in ([0], [5], []):
        with pytest.raises(LayerFeatureError, match="layers"):
            _extractor(layers)
    with pytest.raises(LayerFeatureError, match="blocks"):
        LayerFeatureExtractor(SimpleNamespace(model=nn.Linear(1, 1)), [1])  # type: ignore[arg-type]


PRE: dict[str, Any] = {"image_size": 224}


def test_cache_round_trip_and_every_mismatch_is_rejected(tmp_path: Path) -> None:
    features = torch.randn(2, 5, 6)
    ids = ["a", "b"]
    sha = "x" * 64
    save_layer_features(
        tmp_path,
        layer=12,
        category="bottle",
        split="train",
        features=features,
        sample_ids=ids,
        checkpoint_sha256=sha,
        preprocessing=PRE,
    )

    def load(**override: Any) -> Any:
        kwargs: dict[str, Any] = {
            "layer": 12,
            "category": "bottle",
            "split": "train",
            "sample_ids": ids,
            "checkpoint_sha256": sha,
            "preprocessing": PRE,
        }
        return load_layer_features(tmp_path, **{**kwargs, **override})

    loaded = load()
    assert loaded.features.dtype == torch.float16
    assert torch.allclose(loaded.features.float(), features, atol=2e-3)
    for override, key in (
        ({"checkpoint_sha256": "y" * 64}, "checkpoint_sha256"),
        ({"sample_ids": ["b", "a"]}, "sample_ids_sha256"),
        ({"preprocessing": {"image_size": 448}}, "preprocessing"),
    ):
        with pytest.raises(LayerFeatureError, match=key):
            load(**override)
    with pytest.raises(LayerFeatureError, match="overwrite"):
        save_layer_features(
            tmp_path,
            layer=12,
            category="bottle",
            split="train",
            features=features,
            sample_ids=ids,
            checkpoint_sha256=sha,
            preprocessing=PRE,
        )
    with pytest.raises(LayerFeatureError, match="one row per id"):
        save_layer_features(
            tmp_path,
            layer=13,
            category="bottle",
            split="train",
            features=features,
            sample_ids=["a"],
            checkpoint_sha256=sha,
            preprocessing=PRE,
        )
