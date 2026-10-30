from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261025
TARGETS = ["like", "palm", "peace", "fist", "one"]


def split_for(user_id: str) -> str:
    value = int(hashlib.sha256(f"{SEED}:{user_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "validation" if value < 85 else "test"


def features(landmarks: np.ndarray) -> np.ndarray:
    centered = landmarks - landmarks[:, :1, :]
    scale = np.max(np.linalg.norm(centered[:, :, :2], axis=2), axis=1)
    scale[scale < 1e-6] = 1.0
    return (centered / scale[:, None, None]).reshape(len(landmarks), -1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    classes = sorted(path.stem for path in args.cache.glob("*.npy"))
    positive_counts = []
    by_class = {}
    for label in classes:
        array = np.load(args.cache / f"{label}.npy")
        metadata = json.loads((args.cache / f"{label}.json").read_text(encoding="utf-8"))
        by_class[label] = (array, metadata)
        if label in TARGETS:
            positive_counts.append(len(array))
    negative_limit = int(np.median(positive_counts))
    negative_candidates = []
    for label in classes:
        array, metadata = by_class[label]
        for index, meta in enumerate(metadata):
            row = (label, index, meta, array[index])
            if label in TARGETS:
                rows.append(row)
            else:
                key = hashlib.sha256(f"{SEED}:{label}:{meta['image']}".encode()).hexdigest()
                negative_candidates.append((key, row))
    rows.extend(row for _, row in sorted(negative_candidates)[:negative_limit])

    split_data = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if split_for(str(row[2]["user_id"])) == split]
        detected = [row for row in selected if bool(row[2]["detected"])]
        split_data[split] = {
            "all": selected,
            "x": features(np.stack([row[3] for row in detected])),
            "y": np.asarray([row[0] if row[0] in TARGETS else "no_gesture" for row in detected]),
        }

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    random_state=SEED,
                    max_iter=300,
                    early_stopping=False,
                ),
            ),
        ]
    )
    model.fit(split_data["train"]["x"], split_data["train"]["y"])
    val_prob = model.predict_proba(split_data["validation"]["x"])
    class_names = model.classes_

    def threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
        indices = np.argmax(probabilities, axis=1)
        predictions = class_names[indices].copy()
        predictions[np.max(probabilities, axis=1) < threshold] = "no_gesture"
        return predictions

    candidates = np.unique(np.concatenate(([0.0], np.max(val_prob, axis=1), [1.0])))
    scored = [
        (accuracy_score(split_data["validation"]["y"], threshold_predictions(val_prob, value)), value)
        for value in candidates
    ]
    validation_accuracy, threshold = max(scored, key=lambda item: (item[0], item[1]))
    test_prob = model.predict_proba(split_data["test"]["x"])
    test_pred = threshold_predictions(test_prob, threshold)
    detected_accuracy = accuracy_score(split_data["test"]["y"], test_pred)
    test_all = split_data["test"]["all"]
    end_to_end_correct = int(np.sum(test_pred == split_data["test"]["y"]))
    labels = TARGETS + ["no_gesture"]
    payload = {
        "seed": SEED,
        "targets": TARGETS,
        "negative_source_classes": [label for label in classes if label not in TARGETS],
        "negative_images_selected": negative_limit,
        "split_method": "SHA-256 of seed:user_id; 70/15/15 subject-exclusive",
        "split_images": {split: len(data["all"]) for split, data in split_data.items()},
        "split_detected": {split: len(data["y"]) for split, data in split_data.items()},
        "threshold": float(threshold),
        "validation_detected_accuracy": float(validation_accuracy),
        "test_detected_accuracy": float(detected_accuracy),
        "test_end_to_end_accuracy_failures_are_errors": end_to_end_correct / len(test_all),
        "test_detection_failures": len(test_all) - len(split_data["test"]["y"]),
        "labels": labels,
        "test_confusion_matrix_detected": confusion_matrix(
            split_data["test"]["y"], test_pred, labels=labels
        ).tolist(),
        "iterations": int(model.named_steps["mlp"].n_iter_),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "labels": labels}, args.output)
    args.metrics.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
