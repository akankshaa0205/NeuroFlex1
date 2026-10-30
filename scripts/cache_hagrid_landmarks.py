from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def make_landmarker(model: Path):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def target_center(record: dict, label: str) -> np.ndarray | None:
    for bbox, box_label in zip(record.get("bboxes", []), record.get("labels", [])):
        if box_label == label:
            x, y, width, height = bbox
            return np.asarray([x + width / 2, y + height / 2])
    return None


def select_hand(result, center: np.ndarray | None) -> np.ndarray | None:
    if not result.hand_landmarks:
        return None
    candidates = [np.asarray([[point.x, point.y, point.z] for point in hand]) for hand in result.hand_landmarks]
    if center is None or len(candidates) == 1:
        return candidates[0]
    distances = [np.linalg.norm(np.mean(hand[:, :2], axis=0) - center) for hand in candidates]
    return candidates[int(np.argmin(distances))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache MediaPipe landmarks for every HaGRID image.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_root = args.dataset_root / "hagrid_30k"
    annotation_root = args.dataset_root / "ann_train_val"

    with make_landmarker(args.hand_model) as landmarker:
        for directory in sorted(image_root.glob("train_val_*")):
            label = directory.name.removeprefix("train_val_")
            output = args.output / f"{label}.npy"
            metadata_output = args.output / f"{label}.json"
            if output.exists() and metadata_output.exists():
                print(f"class={label} status=cached", flush=True)
                continue
            paths = sorted(directory.glob("*.jpg"))
            annotations = json.loads((annotation_root / f"{label}.json").read_text(encoding="utf-8"))
            landmarks = np.full((len(paths), 21, 3), np.nan, dtype=np.float32)
            rows = []
            for index, path in enumerate(paths):
                image = cv2.imread(str(path))
                detected = False
                if image is not None:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    record = annotations[path.stem]
                    selected = select_hand(result, target_center(record, label))
                    if selected is not None:
                        landmarks[index] = selected
                        detected = True
                record = annotations.get(path.stem, {})
                rows.append(
                    {
                        "image": path.name,
                        "user_id": record.get("user_id"),
                        "detected": detected,
                    }
                )
            np.save(output, landmarks, allow_pickle=False)
            metadata_output.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
            failures = int(np.sum(np.isnan(landmarks[:, 0, 0])))
            print(f"class={label} images={len(paths)} failures={failures}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
