from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuroflex.domain.kimore import ANGLE_NAMES, aggregate_angle_sequence, feature_names


def load_sessions(
    frame_file: Path, score_file: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sequences: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    with frame_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequences[(row["subject_id"], row["exercise_folder"])].append(
                [float(row[name]) for name in ANGLE_NAMES]
            )
    scores: dict[tuple[str, str], float] = {}
    with score_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["score_status"] == "ok" and row["cTS"]:
                scores[(row["subject_id"], row["exercise_folder"])] = float(row["cTS"])
    keys = sorted(set(sequences) & set(scores))
    x = np.stack([aggregate_angle_sequence(np.asarray(sequences[key]), key[1]) for key in keys])
    y = np.asarray([scores[key] for key in keys], dtype=np.float64)
    groups = np.asarray([key[0] for key in keys])
    return x, y, groups, [f"{subject}/{exercise}" for subject, exercise in keys]


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rho = float(spearmanr(y_true, prediction).statistic)
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
        "spearman_rho": rho,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train subject-grouped KIMORE quality regression.")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    x, y, groups, keys = load_sessions(args.frames, args.scores)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_index, test_index = next(split.split(x, y, groups))
    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, random_state=args.seed, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=150, min_samples_leaf=3, random_state=args.seed, n_jobs=-1
        ),
    }
    folds = GroupKFold(n_splits=5)
    cv_mae = {
        name: float(
            -cross_val_score(
                model,
                x[train_index],
                y[train_index],
                groups=groups[train_index],
                cv=folds,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            ).mean()
        )
        for name, model in candidates.items()
    }
    selected_name = min(cv_mae, key=cv_mae.get)
    selected = candidates[selected_name]
    selected.fit(x[train_index], y[train_index])
    prediction = selected.predict(x[test_index])
    test_metrics = metrics(y[test_index], prediction)
    artifact = {
        "model": selected,
        "model_version": "kimore-quality-1.0",
        "feature_names": feature_names(),
        "target": "cTS_0_50",
        "training_source": "KIMORE standardized derivative",
        "cross_sensor_runtime_status": "experimental",
    }
    payload = {
        "seed": args.seed,
        "records": len(x),
        "subjects": len(set(groups)),
        "train_records": len(train_index),
        "test_records": len(test_index),
        "train_subjects": sorted(set(groups[train_index])),
        "test_subjects": sorted(set(groups[test_index])),
        "candidate_cv_mae": cv_mae,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "target_mean": float(np.mean(y[test_index])),
        "baseline_mean_mae": float(
            mean_absolute_error(y[test_index], np.full(len(test_index), np.mean(y[train_index])))
        ),
        "test_predictions": [
            {"record": keys[index], "actual": float(y[index]), "predicted": float(value)}
            for index, value in zip(test_index, prediction, strict=True)
        ],
        "limitations": [
            "Clinical scores apply to complete recordings, not individual repetitions.",
            "Runtime MediaPipe inference is cross-sensor and remains experimental.",
            "Split is subject-disjoint to prevent identity leakage.",
        ],
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model)
    args.metrics.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "test_predictions"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
