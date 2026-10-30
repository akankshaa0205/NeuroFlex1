from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Record idle/random hand landmarks without video.")
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    samples: list[np.ndarray] = []
    timestamps: list[int] = []
    frame_count = 0
    started = time.monotonic()
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while time.monotonic() - started < args.seconds:
                ok, frame = capture.read()
                if not ok:
                    continue
                timestamp_ms = int((time.monotonic() - started) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                frame_count += 1
                for hand in result.hand_landmarks:
                    samples.append(
                        np.asarray([[point.x, point.y, point.z] for point in hand], dtype=np.float32)
                    )
                    timestamps.append(timestamp_ms)
                if not args.no_preview:
                    cv2.putText(
                        frame,
                        "Show idle, resting, and random non-command poses. Q to stop.",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("NeuroFlex rest-hand collection (video is not saved)", frame)
                    if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                        break
    finally:
        capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    landmarks = np.stack(samples) if samples else np.empty((0, 21, 3), dtype=np.float32)
    np.save(args.output, landmarks, allow_pickle=False)
    duration = time.monotonic() - started
    metadata = {
        "landmark_file": args.output.name,
        "video_saved": False,
        "camera_frames": frame_count,
        "detected_hands": len(samples),
        "duration_seconds": duration,
        "timestamps_ms": timestamps,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "timestamps_ms"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
