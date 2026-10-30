from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

PATTERN = re.compile(r"m(\d+)_s(\d+)_e(\d+)_angles")


def resample(x: np.ndarray, n: int = 100) -> np.ndarray:
    x = np.atleast_2d(x)
    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, n)
    return np.column_stack([np.interp(new, old, x[:, i]) for i in range(x.shape[1])])


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    costs = cdist(a, b)
    prev = np.full(len(b) + 1, np.inf)
    prev[0] = 0
    for row in costs:
        cur = np.full(len(b) + 1, np.inf)
        for j, cost in enumerate(row, 1):
            cur[j] = cost + min(cur[j - 1], prev[j], prev[j - 1])
        prev = cur
    return float(prev[-1] / max(len(a), len(b)))


def auc(y, score):
    y = np.asarray(y)
    ranks = rankdata(-np.asarray(score))
    p = y == 1
    np_, nn = p.sum(), (~p).sum()
    return float((ranks[p].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def threshold(y, score):
    y = np.asarray(y)
    score = np.asarray(score)
    values = np.unique(score)
    candidates = np.r_[values[0], (values[:-1] + values[1:]) / 2, values[-1]]

    def ba(t):
        pred = score <= t
        return 0.5 * ((pred[y == 1]).mean() + (~pred[y == 0]).mean())

    return float(max(candidates, key=lambda t: (ba(t), -t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    output = {"personal_baseline_repetitions": 3, "sensors": {}}
    for sensor in ("Kinect", "Vicon"):
        folders = [
            (a.root / "Segmented Movements/Segmented Movements" / sensor / "Angles", True),
            (
                a.root
                / "Incorrect Segmented Movements/Incorrect Segmented Movements"
                / sensor
                / "Angles",
                False,
            ),
        ]
        records = []
        for folder, correct in folders:
            for p in sorted(folder.glob("*.txt")):
                m = PATTERN.match(p.stem)
                if m:
                    records.append(
                        {
                            "m": int(m[1]),
                            "s": int(m[2]),
                            "e": int(m[3]),
                            "correct": correct,
                            "x": resample(np.loadtxt(p, delimiter=",")),
                        }
                    )
        sensor_out = {}
        for movement in range(1, 11):
            rs = [r for r in records if r["m"] == movement]
            protocols = {}
            for protocol in ("global", "personal"):
                scored = []
                for subject in range(1, 11):
                    if protocol == "global":
                        base = np.mean(
                            np.stack([r["x"] for r in rs if r["correct"] and r["s"] != subject]),
                            axis=0,
                        )
                        evals = [r for r in rs if r["s"] == subject]
                    else:
                        baseline = [
                            r["x"] for r in rs if r["correct"] and r["s"] == subject and r["e"] <= 3
                        ]
                        if not baseline:
                            continue
                        base = np.mean(np.stack(baseline), axis=0)
                        evals = [
                            r for r in rs if r["s"] == subject and (not r["correct"] or r["e"] > 3)
                        ]
                    for r in evals:
                        scored.append(
                            {"s": subject, "y": int(r["correct"]), "d": dtw(r["x"], base)}
                        )
                predictions = []
                for subject in sorted({r["s"] for r in scored}):
                    train = [r for r in scored if r["s"] != subject]
                    test = [r for r in scored if r["s"] == subject]
                    t = threshold([r["y"] for r in train], [r["d"] for r in train])
                    predictions.extend((r["y"], int(r["d"] <= t)) for r in test)
                y = np.array([r["y"] for r in scored])
                d = np.array([r["d"] for r in scored])
                pred = np.array(predictions)
                correct = d[y == 1]
                incorrect = d[y == 0]
                protocols[protocol] = {
                    "auc": auc(y, d),
                    "balanced_accuracy": float(
                        0.5
                        * (
                            (pred[pred[:, 0] == 1, 1] == 1).mean()
                            + (pred[pred[:, 0] == 0, 1] == 0).mean()
                        )
                    ),
                    "correct_median": float(np.median(correct)),
                    "correct_q1": float(np.quantile(correct, 0.25)),
                    "correct_q3": float(np.quantile(correct, 0.75)),
                    "incorrect_median": float(np.median(incorrect)),
                    "incorrect_q1": float(np.quantile(incorrect, 0.25)),
                    "incorrect_q3": float(np.quantile(incorrect, 0.75)),
                    "n": len(scored),
                }
            sensor_out[f"m{movement:02d}"] = protocols
        output["sensors"][sensor] = sensor_out
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
