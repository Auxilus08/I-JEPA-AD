"""Tests for peak VRAM profiling."""

import logging
from typing import Any

from ijepa_ad.utils.profiling import profile_peak_vram


def test_profile_peak_vram_is_safe_without_cuda(monkeypatch: Any, caplog: Any) -> None:
    monkeypatch.setattr("ijepa_ad.utils.profiling._load_torch", lambda: None)
    logger = logging.getLogger("profiling-test")
    logger.setLevel(logging.INFO)
    with profile_peak_vram("cpu-block", logger=logger) as result:
        assert result.label == "cpu-block"
        assert result.allocated_bytes == 0

    assert result.allocated_mb == 0
    assert len(caplog.records) == 1
    assert caplog.records[0].message.startswith("peak CUDA memory: 0 bytes")


def test_profile_peak_vram_reads_cuda_peak(monkeypatch: Any, caplog: Any) -> None:
    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def reset_peak_memory_stats(self, device: Any) -> None:
            calls.append(("reset", device))

        def max_memory_allocated(self, device: Any) -> int:
            calls.append(("max", device))
            return 2_000_000

    class FakeTorch:
        cuda = FakeCuda()

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr("ijepa_ad.utils.profiling._load_torch", lambda: FakeTorch)
    logger = logging.getLogger("cuda-profiling-test")
    logger.setLevel(logging.INFO)

    with profile_peak_vram("cuda-block", logger=logger, device=3) as result:
        assert result.allocated_bytes == 0

    assert result.allocated_bytes == 2_000_000
    assert result.allocated_mb == 2
    assert calls == [("reset", 3), ("max", 3)]
    assert caplog.records[0].message.startswith("peak CUDA memory: 2000000 bytes")
