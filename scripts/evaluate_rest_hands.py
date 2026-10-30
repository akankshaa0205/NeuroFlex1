from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure command false triggers on captured idle hands.")
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    landmarks = np.load(args.landmarks, allow_pickle=False)
    artifact = joblib.load(args.model)
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise ValueError("Expected landmarks with shape (samples, 21, 3)")

    centered = landmarks - landmarks[:, :1, :]
    scale = np.linalg.norm(centered, axis=2).max(axis=1)
    valid = scale > 1e-8
    features = centered[valid] / scale[valid, None, None]
    probabilities = model.predict_proba(features.reshape(len(features), -1))
    best_indices = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    labels = np.asarray(model.classes_, dtype=str)[best_indices]
    triggered = (labels != "no_gesture") & (confidences >= threshold)

    payload = {
        "samples": len(landmarks),
        "valid_samples": int(valid.sum()),
        "model_threshold": threshold,
        "false_triggers": int(triggered.sum()),
        "false_trigger_rate": float(triggered.mean()) if len(triggered) else None,
        "max_command_confidence": float(confidences[labels != "no_gesture"].max())
        if np.any(labels != "no_gesture")
        else None,
        "triggered_label_counts": {
            label: int(np.sum(labels[triggered] == label)) for label in sorted(set(labels[triggered]))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
