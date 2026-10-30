from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from neuroflex.domain.kimore import KimoreQualityEstimator, pose_to_kimore_angles
from neuroflex.pose import synthetic_pose


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate packaged KIMORE runtime integration.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    training = json.loads(args.training_metrics.read_text(encoding="utf-8"))
    estimator = KimoreQualityEstimator(args.model)
    sequence = np.stack(
        [pose_to_kimore_angles(synthetic_pose(index / 10)) for index in range(120)]
    )
    timings = []
    predictions = []
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        result = estimator.predict(sequence, "Es1")
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
        predictions.append(result.quality_percent)
    ordered = sorted(timings)
    p95 = ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)]
    payload = {
        "status": "pass",
        "model_version": result.model_version,
        "training_records": training["records"],
        "training_subjects": training["subjects"],
        "held_out_metrics": training["test_metrics"],
        "runtime_input_frames": len(sequence),
        "runtime_quality_percent": predictions[-1],
        "runtime_inference_mean_ms": statistics.fmean(timings),
        "runtime_inference_p95_ms": p95,
        "deterministic": max(predictions) - min(predictions) <= 1e-9,
        "bounded": all(0 <= value <= 100 for value in predictions),
        "cross_sensor_status": result.status,
    }
    if not payload["deterministic"] or not payload["bounded"]:
        payload["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
