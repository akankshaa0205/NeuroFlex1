from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from neuroflex.domain.kinematics import joint_angle
from neuroflex.domain.scoring import exact_dtw
from neuroflex.runtime import LatestItemWorker, resolve_compute_device


def timed_ms(function, iterations: int) -> list[float]:
    values = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    return values


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/runtime_benchmark.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rng = np.random.default_rng(config["seed"])
    device = resolve_compute_device(config["device"])

    points = rng.normal(size=(config["landmark_frames"], 3, 3))
    left = rng.normal(size=config["dtw_points"])
    right = rng.normal(size=config["dtw_points"])

    def kinematics_work() -> None:
        joint_angle(points)

    def dtw_work() -> None:
        exact_dtw(left, right)

    for _ in range(config["warmup_iterations"]):
        kinematics_work()
        dtw_work()

    kin_times = timed_ms(kinematics_work, config["timed_iterations"])
    dtw_times = timed_ms(dtw_work, config["timed_iterations"])

    worker = LatestItemWorker(lambda value: exact_dtw(value, right), max_pending=config["queue_capacity"])
    worker.start()
    queue_started = time.perf_counter_ns()
    for index in range(config["queue_items"]):
        worker.submit(left + index * 1e-9)
    worker.stop(timeout=30.0)
    queue_elapsed_ms = (time.perf_counter_ns() - queue_started) / 1_000_000

    payload = {
        "benchmark_version": 1,
        "config": config,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "device": device.__dict__,
        "kinematics": summary(kin_times),
        "exact_dtw": summary(dtw_times),
        "bounded_queue": {**worker.stats.__dict__, "elapsed_ms": queue_elapsed_ms},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
