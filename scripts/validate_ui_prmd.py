from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from neuroflex.domain.kinematics import joint_angle

FILE_PATTERN = re.compile(r"m(?P<movement>\d+)_s(?P<subject>\d+)_e(?P<episode>\d+)_positions")
ANGLE_TRIPLETS = {
    "trunk_lower": (0, 1, 2),
    "trunk_upper": (1, 2, 3),
    "left_shoulder": (6, 7, 8),
    "left_elbow": (7, 8, 9),
    "right_shoulder": (10, 11, 12),
    "right_elbow": (11, 12, 13),
    "left_knee": (14, 15, 16),
    "right_knee": (18, 19, 20),
}


@dataclass(frozen=True)
class Record:
    path: Path
    movement: int
    subject: int
    episode: int
    correct: bool


def discover(root: Path) -> list[Record]:
    correct_root = root / "Segmented Movements/Segmented Movements/Kinect/Positions"
    incorrect_root = (
        root / "Incorrect Segmented Movements/Incorrect Segmented Movements/Kinect/Positions"
    )
    records = []
    for folder, correct in ((correct_root, True), (incorrect_root, False)):
        for path in sorted(folder.glob("*.txt")):
            match = FILE_PATTERN.match(path.stem)
            if match is None:
                raise ValueError(f"Unexpected UI-PRMD file name: {path.name}")
            records.append(
                Record(
                    path,
                    int(match["movement"]),
                    int(match["subject"]),
                    int(match["episode"]),
                    correct,
                )
            )
    return records


def extract_angles(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] != 66:
        raise ValueError(f"Expected 66 Kinect position columns in {path}; got {raw.shape}")
    joints = raw.reshape(raw.shape[0], 22, 3)
    return np.column_stack(
        [joint_angle(joints[:, np.asarray(indices), :]) for indices in ANGLE_TRIPLETS.values()]
    )


def resample(sequence: np.ndarray, frames: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, sequence.shape[0])
    target = np.linspace(0.0, 1.0, frames)
    return np.column_stack([np.interp(target, source, sequence[:, index]) for index in range(sequence.shape[1])])


def multivariate_dtw(left: np.ndarray, right: np.ndarray) -> float:
    previous = np.full(right.shape[0] + 1, np.inf)
    previous[0] = 0.0
    for left_frame in left:
        current = np.full(right.shape[0] + 1, np.inf)
        costs = np.linalg.norm(right - left_frame, axis=1)
        for index, cost in enumerate(costs, start=1):
            current[index] = cost + min(current[index - 1], previous[index], previous[index - 1])
        previous = current
    return float(previous[-1] / max(left.shape[0], right.shape[0]))


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    return float(
        0.5
        * (
            np.mean(predictions[positive] == 1)
            + np.mean(predictions[negative] == 0)
        )
    )


def select_threshold(distances: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    unique = np.unique(distances)
    candidates = np.concatenate(
        ([np.nextafter(unique[0], -np.inf)], (unique[:-1] + unique[1:]) / 2, [unique[-1]])
    )
    scored = [(balanced_accuracy(labels, (distances <= value).astype(int)), value) for value in candidates]
    score, threshold = max(scored, key=lambda item: (item[0], -item[1]))
    return float(threshold), float(score)


def roc_auc(distances: np.ndarray, labels: np.ndarray) -> float:
    scores = -distances
    ranks = rankdata(scores)
    positive = labels == 1
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(~positive))
    return float((np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ui_prmd_validation.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = discover(args.data_root)
    split_by_subject = {}
    for split in ("train", "validation", "test"):
        for subject in config[f"{split}_subjects"]:
            if subject in split_by_subject:
                raise ValueError(f"Subject {subject} occurs in multiple splits")
            split_by_subject[subject] = split

    sequences = {}
    frame_counts = []
    all_angle_min = np.inf
    all_angle_max = -np.inf
    for index, record in enumerate(records):
        angles = extract_angles(record.path)
        frame_counts.append(angles.shape[0])
        all_angle_min = min(all_angle_min, float(np.min(angles)))
        all_angle_max = max(all_angle_max, float(np.max(angles)))
        sequences[index] = resample(angles, config["resampled_frames"])

    results = []
    movement_metrics = {}
    for movement in range(1, 11):
        train_ids = [
            i
            for i, record in enumerate(records)
            if record.movement == movement
            and record.correct
            and split_by_subject[record.subject] == "train"
        ]
        stack = np.stack([sequences[i] for i in train_ids])
        template = np.mean(stack, axis=0)

        movement_rows = []
        for index, record in enumerate(records):
            split = split_by_subject[record.subject]
            if record.movement != movement or split == "train":
                continue
            distance = multivariate_dtw(sequences[index], template)
            row = {
                "file": record.path.name,
                "movement": movement,
                "subject": record.subject,
                "episode": record.episode,
                "split": split,
                "correct": record.correct,
                "dtw_distance": distance,
            }
            results.append(row)
            movement_rows.append(row)

        validation = [row for row in movement_rows if row["split"] == "validation"]
        val_distances = np.asarray([row["dtw_distance"] for row in validation])
        val_labels = np.asarray([int(row["correct"]) for row in validation])
        threshold, val_balanced = select_threshold(val_distances, val_labels)

        test = [row for row in movement_rows if row["split"] == "test"]
        test_distances = np.asarray([row["dtw_distance"] for row in test])
        test_labels = np.asarray([int(row["correct"]) for row in test])
        predictions = (test_distances <= threshold).astype(int)
        movement_metrics[f"m{movement:02d}"] = {
            "threshold_selected_on_validation": threshold,
            "validation_balanced_accuracy": val_balanced,
            "test_balanced_accuracy": balanced_accuracy(test_labels, predictions),
            "test_roc_auc": roc_auc(test_distances, test_labels),
            "test_correct_median_dtw": float(np.median(test_distances[test_labels == 1])),
            "test_incorrect_median_dtw": float(np.median(test_distances[test_labels == 0])),
            "test_records": len(test),
        }

    test_rows = [row for row in results if row["split"] == "test"]
    overall_labels = np.asarray([int(row["correct"]) for row in test_rows])
    overall_predictions = []
    for row in test_rows:
        threshold = movement_metrics[f"m{row['movement']:02d}"]["threshold_selected_on_validation"]
        overall_predictions.append(int(row["dtw_distance"] <= threshold))

    payload = {
        "validation_version": 1,
        "config": config,
        "data_integrity": {
            "records": len(records),
            "correct_records": sum(record.correct for record in records),
            "incorrect_records": sum(not record.correct for record in records),
            "subjects": sorted({record.subject for record in records}),
            "movements": sorted({record.movement for record in records}),
            "records_by_split": dict(Counter(split_by_subject[r.subject] for r in records)),
            "minimum_frames": min(frame_counts),
            "maximum_frames": max(frame_counts),
            "minimum_computed_angle_deg": all_angle_min,
            "maximum_computed_angle_deg": all_angle_max,
            "kinematic_features": list(ANGLE_TRIPLETS),
        },
        "movement_metrics": movement_metrics,
        "overall_test_balanced_accuracy": balanced_accuracy(
            overall_labels, np.asarray(overall_predictions)
        ),
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
