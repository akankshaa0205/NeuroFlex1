from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp

SEED = 20261025
SAMPLES_PER_CLASS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a local HaGRID image sample.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_landmarker(model: Path):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def annotate(image, annotation: dict, target: str):
    height, width = image.shape[:2]
    for bbox, label in zip(annotation.get("bboxes", []), annotation.get("labels", [])):
        x, y, box_width, box_height = bbox
        p1 = round(x * width), round(y * height)
        p2 = round((x + box_width) * width), round((y + box_height) * height)
        color = (40, 210, 40) if label == target else (0, 160, 255)
        cv2.rectangle(image, p1, p2, color, 3)
        cv2.putText(
            image,
            label,
            (p1[0], max(18, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def contact_sheet(images: list, label: str):
    tiles = []
    for index, image in enumerate(images, start=1):
        tile = cv2.resize(image, (240, 160), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            f"{label} {index:02d}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    rows = [cv2.hconcat(tiles[index : index + 5]) for index in range(0, 20, 5)]
    return cv2.vconcat(rows)


def main() -> None:
    args = parse_args()
    image_root = args.dataset_root / "hagrid_30k"
    annotation_root = args.dataset_root / "ann_train_val"
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    classes = sorted(path.name.removeprefix("train_val_") for path in image_root.iterdir())
    class_results = {}
    user_ids: set[str] = set()
    resolution_counts: Counter[str] = Counter()
    corrupt_images = 0
    with make_landmarker(args.hand_model) as landmarker:
        for label in classes:
            directory = image_root / f"train_val_{label}"
            image_paths = sorted(directory.glob("*.jpg"))
            annotations = json.loads((annotation_root / f"{label}.json").read_text())
            for image_path in image_paths:
                record = annotations.get(image_path.stem)
                if record and record.get("user_id"):
                    user_ids.add(str(record["user_id"]))
            selected = rng.sample(image_paths, min(SAMPLES_PER_CLASS, len(image_paths)))
            sheet_images = []
            failures = 0
            sample_rows = []
            for image_path in selected:
                image = cv2.imread(str(image_path))
                if image is None:
                    corrupt_images += 1
                    failures += 1
                    sample_rows.append({"image": image_path.name, "readable": False})
                    continue
                height, width = image.shape[:2]
                resolution_counts[f"{width}x{height}"] += 1
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                detected = bool(result.hand_landmarks)
                failures += int(not detected)
                annotation = annotations[image_path.stem]
                sheet_images.append(annotate(image.copy(), annotation, label))
                sample_rows.append(
                    {
                        "image": image_path.name,
                        "readable": True,
                        "resolution": [width, height],
                        "hand_detected": detected,
                        "annotation_labels": annotation.get("labels", []),
                        "user_id": annotation.get("user_id"),
                    }
                )
            if len(sheet_images) == SAMPLES_PER_CLASS:
                cv2.imwrite(str(args.output / f"contact_{label}.jpg"), contact_sheet(sheet_images, label))
            class_results[label] = {
                "image_count": len(image_paths),
                "sample_count": len(selected),
                "mediapipe_failures": failures,
                "mediapipe_failure_pct": 100.0 * failures / len(selected),
                "samples": sample_rows,
            }
            print(
                f"class={label} images={len(image_paths)} sample={len(selected)} "
                f"failures={failures}",
                flush=True,
            )
    payload = {
        "seed": SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "class_names": classes,
        "classes": class_results,
        "unique_user_ids": len(user_ids),
        "user_id_present": bool(user_ids),
        "sample_resolution_counts": dict(sorted(resolution_counts.items())),
        "corrupt_sample_images": corrupt_images,
    }
    (args.output / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
