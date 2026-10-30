from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from neuroflex.runtime import (
    ComputeDevice,
    HybridAngleEngine,
    LatestItemWorker,
    compute_status_text,
    resolve_compute_device,
)


def test_cpu_device_is_explicit() -> None:
    device = resolve_compute_device("cpu")
    assert device.selected == "cpu"
    assert device.requested == "cpu"


def test_latest_item_worker_processes_or_accounts_for_every_submission(caplog) -> None:
    def slow_double(value: int) -> int:
        time.sleep(0.002)
        return value * 2

    with caplog.at_level(logging.WARNING, logger="neuroflex.runtime"):
        worker = LatestItemWorker(slow_double, max_pending=1)
        worker.start()
        for value in range(120):
            worker.submit(value)
        worker.stop()
    stats = worker.stats
    assert stats.submitted == 120
    assert stats.processed + stats.dropped + stats.failed == stats.submitted
    assert stats.dropped > 0
    assert len(caplog.records) <= 3


def test_latest_item_worker_returns_only_the_newest_completed_result() -> None:
    worker = LatestItemWorker(lambda value: value * 2, max_pending=3)
    worker.start()
    for value in range(3):
        worker.submit(value)
    worker.stop()
    assert worker.get_latest_result() == 4
    assert worker.get_latest_result() is None


def test_hybrid_angle_engine_has_a_correct_cpu_fallback() -> None:
    engine = HybridAngleEngine("cpu")
    landmarks = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    angles = engine.calculate(landmarks, [(0, 1, 2), (0, 1, 3)])
    assert np.allclose(angles, [90.0, 90.0])


def test_hybrid_angle_engine_rejects_degenerate_limbs() -> None:
    engine = HybridAngleEngine("cpu")
    landmarks = np.zeros((3, 3))
    with pytest.raises(ValueError, match="zero-length limb"):
        engine.calculate(landmarks, [(0, 1, 2)])


def test_runtime_rejects_unknown_device_preferences() -> None:
    with pytest.raises(ValueError, match="Unsupported device preference"):
        resolve_compute_device("tpu")  # type: ignore[arg-type]


def test_compute_status_never_claims_inactive_cuda() -> None:
    cpu = ComputeDevice("auto", "cpu", False, False, "CPU", "CUDA unavailable")
    cuda = ComputeDevice("auto", "cuda", True, True, "Test GPU")
    assert compute_status_text(cpu).startswith("Compute: CPU active")
    assert "CUDA acceleration active" not in compute_status_text(cpu)
    assert compute_status_text(cuda) == "Compute: CUDA acceleration active • Test GPU"
