"""Tests for deterministic random state setup."""

import os
import random
from importlib import import_module
from typing import Any

import pytest

from ijepa_ad.utils.seeding import seeded_rng, set_seed


def test_set_seed_repeats_python_and_numpy_sequences() -> None:
    np = import_module("numpy")
    set_seed(123)
    first = (random.random(), np.random.random())
    set_seed(123)
    second = (random.random(), np.random.random())

    assert first == second


def test_set_seed_rejects_invalid_values() -> None:
    with pytest.raises(TypeError):
        set_seed(True)
    with pytest.raises(ValueError):
        set_seed(-1)


def test_set_seed_configures_torch_when_installed() -> None:
    torch = pytest.importorskip("torch")

    set_seed(123)
    first = torch.rand(3)
    set_seed(123)
    second = torch.rand(3)

    assert torch.equal(first, second)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_calls_torch_and_cuda_controls(monkeypatch: Any) -> None:
    class FakeCudnn:
        deterministic = False
        benchmark = True

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def manual_seed(self, seed: int) -> None:
            calls.append(("cuda.manual_seed", seed))

        def manual_seed_all(self, seed: int) -> None:
            calls.append(("cuda.manual_seed_all", seed))

    class FakeTorch:
        cuda = FakeCuda()
        backends = type("Backends", (), {"cudnn": FakeCudnn()})()

        @staticmethod
        def manual_seed(seed: int) -> None:
            calls.append(("manual_seed", seed))

        @staticmethod
        def use_deterministic_algorithms(enabled: bool) -> None:
            calls.append(("deterministic", enabled))

    calls: list[tuple[str, int | bool]] = []
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr("ijepa_ad.utils.seeding._load_torch", lambda: FakeTorch)

    set_seed(7)

    assert calls == [
        ("manual_seed", 7),
        ("deterministic", True),
        ("cuda.manual_seed", 7),
        ("cuda.manual_seed_all", 7),
    ]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert FakeTorch.backends.cudnn.deterministic is True
    assert FakeTorch.backends.cudnn.benchmark is False


def test_set_seed_preserves_custom_cublas_workspace_config(monkeypatch: Any) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr("ijepa_ad.utils.seeding._load_torch", lambda: None)

    set_seed(7)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_seeded_rng_is_reproducible_and_isolated() -> None:
    random.seed(5)
    state = random.getstate()

    first = [seeded_rng(11).random() for _ in range(3)]
    second = [seeded_rng(11).random() for _ in range(3)]

    assert first == second
    assert random.getstate() == state


def test_seeded_rng_rejects_invalid_values() -> None:
    with pytest.raises(TypeError):
        seeded_rng(True)
    with pytest.raises(ValueError):
        seeded_rng(-1)
