from __future__ import annotations

import importlib.util
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np

from neuroflex.domain.kinematics import joint_angle

T = TypeVar("T")
R = TypeVar("R")
DevicePreference = Literal["auto", "cpu", "cuda"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComputeDevice:
    requested: DevicePreference
    selected: Literal["cpu", "cuda"]
    torch_available: bool
    cuda_available: bool
    name: str
    fallback_reason: str | None = None


def resolve_compute_device(preference: DevicePreference = "auto") -> ComputeDevice:
    """Inspect the actual host and select a compute device without assuming CUDA."""
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device preference: {preference}")

    torch_available = importlib.util.find_spec("torch") is not None
    cuda_available = False
    cuda_name: str | None = None
    if torch_available:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_name = str(torch.cuda.get_device_name(0))

    if preference != "cpu" and cuda_available:
        return ComputeDevice(preference, "cuda", True, True, cuda_name or "CUDA device")

    reason = None
    if preference == "cuda":
        reason = "CUDA requested but unavailable; using CPU"
    elif preference == "auto" and not cuda_available:
        reason = "CUDA unavailable; using CPU"
    return ComputeDevice(
        preference,
        "cpu",
        torch_available,
        cuda_available,
        "CPU",
        reason,
    )


class HybridAngleEngine:
    """Compute angle batches on CUDA when present, otherwise use NumPy on CPU."""

    def __init__(self, preference: DevicePreference = "auto") -> None:
        self.device = resolve_compute_device(preference)

    def calculate(
        self, landmarks: np.ndarray, triplets: list[tuple[int, int, int]]
    ) -> np.ndarray:
        data = np.asarray(landmarks, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("Landmarks must have shape (joints, 3)")
        selected = data[np.asarray(triplets)]
        if self.device.selected == "cpu":
            return np.asarray(joint_angle(selected), dtype=np.float64)

        import torch

        points = torch.as_tensor(selected, dtype=torch.float32, device="cuda")
        left = points[:, 0] - points[:, 1]
        right = points[:, 2] - points[:, 1]
        denominator = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(
            right, dim=1
        )
        if bool(torch.any(denominator <= 1e-9).item()):
            raise ValueError("Cannot compute an angle from a zero-length limb")
        cosine = torch.sum(left * right, dim=1) / denominator
        angles = torch.rad2deg(torch.acos(torch.clamp(cosine, -1.0, 1.0)))
        return angles.detach().cpu().numpy().astype(np.float64)


def compute_status_text(device: ComputeDevice) -> str:
    if device.selected == "cuda":
        return f"Compute: CUDA acceleration active • {device.name}"
    if device.torch_available:
        return "Compute: CPU active • CUDA runtime/device not detected"
    return "Compute: CPU active • install the optional CUDA package for GPU acceleration"


@dataclass(frozen=True)
class PipelineStats:
    submitted: int
    processed: int
    dropped: int
    failed: int
    pending: int


class LatestItemWorker(Generic[T, R]):
    """Bounded background worker that drops the oldest stale item when saturated."""

    def __init__(self, process: Callable[[T], R], *, max_pending: int = 2) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be at least one")
        self._process = process
        self._items: queue.Queue[T | object] = queue.Queue(maxsize=max_pending)
        self._results: queue.Queue[R] = queue.Queue()
        self._sentinel = object()
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._failed = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="neuroflex-runtime", daemon=True)
        self._thread.start()

    def submit(self, item: T) -> None:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Worker must be started before submitting items")
        with self._lock:
            self._submitted += 1
        try:
            self._items.put_nowait(item)
        except queue.Full:
            try:
                self._items.get_nowait()
                self._items.task_done()
            except queue.Empty:
                pass
            else:
                self._record_drop("Dropped stale runtime item")
            self._items.put_nowait(item)

    def get_result(self, *, timeout: float | None = None) -> R:
        return self._results.get(timeout=timeout)

    def get_latest_result(self) -> R | None:
        """Return the newest completed result without blocking and discard older results."""
        latest: R | None = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                return latest
            else:
                self._results.task_done()

    def stop(self, *, timeout: float = 5.0) -> None:
        thread = self._thread
        if thread is None:
            return
        while True:
            try:
                self._items.put_nowait(self._sentinel)
                break
            except queue.Full:
                try:
                    self._items.get_nowait()
                    self._items.task_done()
                except queue.Empty:
                    continue
                self._record_drop("Dropped queued runtime item during shutdown")
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("Runtime worker did not stop in time")

    @property
    def stats(self) -> PipelineStats:
        with self._lock:
            return PipelineStats(
                submitted=self._submitted,
                processed=self._processed,
                dropped=self._dropped,
                failed=self._failed,
                pending=self._items.qsize(),
            )

    def _run(self) -> None:
        while True:
            item = self._items.get()
            try:
                if item is self._sentinel:
                    return
                result = self._process(item)  # type: ignore[arg-type]
                self._results.put(result)
                with self._lock:
                    self._processed += 1
            except Exception:  # noqa: BLE001 - worker failures are counted, not allowed to kill the thread
                with self._lock:
                    self._failed += 1
            finally:
                self._items.task_done()

    def _record_drop(self, message: str) -> None:
        with self._lock:
            self._dropped += 1
            dropped = self._dropped
        if dropped == 1 or dropped % 50 == 0:
            LOGGER.warning("%s; total_dropped=%d", message, dropped)
