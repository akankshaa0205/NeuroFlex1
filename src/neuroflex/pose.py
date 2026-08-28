from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class PoseEstimator(Protocol):
    def estimate(self, frame: np.ndarray) -> tuple[np.ndarray, float] | None: ...


class LivePoseCamera:
    """Local webcam + MediaPipe pose adapter. Frames are never written to disk."""

    def __init__(self, model_path: Path, camera_index: int = 0) -> None:
        import mediapipe as mp

        self._mp = mp
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(camera_index)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.last_visibility = np.zeros(33, dtype=np.float64)
        self.last_world_landmarks: np.ndarray | None = None
        self._last_timestamp_ms = 0
        self._smoothed_coordinates: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return self.capture.isOpened()

    def read(self) -> tuple[np.ndarray, np.ndarray | None, float]:
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("The camera did not return a frame")
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(self._last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
        self._last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb),
            timestamp_ms,
        )
        if not result.pose_landmarks:
            self.last_world_landmarks = None
            self._smoothed_coordinates = None
            return rgb, None, 0.0
        landmarks = result.pose_landmarks[0]
        coordinates = np.asarray(
            [[point.x, point.y, point.z] for point in landmarks], dtype=np.float64
        )
        if self._smoothed_coordinates is None:
            self._smoothed_coordinates = coordinates
        else:
            # Temporal tracking removes the distracting frame-to-frame skeleton jitter while
            # retaining enough responsiveness for deliberately slow rehabilitation movement.
            self._smoothed_coordinates = 0.65 * self._smoothed_coordinates + 0.35 * coordinates
        visibility = [point.visibility or 0.0 for point in landmarks]
        self.last_visibility = np.asarray(visibility, dtype=np.float64)
        if result.pose_world_landmarks:
            world = result.pose_world_landmarks[0]
            self.last_world_landmarks = np.asarray(
                [[point.x, point.y, point.z] for point in world], dtype=np.float64
            )
        return rgb, self._smoothed_coordinates.copy(), float(np.mean(visibility))

    def close(self) -> None:
        self.capture.release()
        self.landmarker.close()


def synthetic_pose(phase: float) -> np.ndarray:
    """Generate 33 normalized landmarks for a deterministic display/demo."""
    points = np.full((33, 3), 0.5, dtype=np.float64)
    points[:, 2] = 0.0
    points[0, :2] = (0.50, 0.16)
    points[11, :2] = (0.40, 0.32)
    points[12, :2] = (0.60, 0.32)
    points[23, :2] = (0.44, 0.62)
    points[24, :2] = (0.56, 0.62)
    points[25, :2] = (0.44, 0.80)
    points[26, :2] = (0.56, 0.80)
    points[27, :2] = (0.43, 0.96)
    points[28, :2] = (0.57, 0.96)
    swing = 0.15 + 0.16 * (1 + math.sin(phase)) / 2
    points[13, :2] = (0.34, 0.47)
    points[15, :2] = (0.34 + swing, 0.46 - swing)
    points[14, :2] = (0.66, 0.47)
    points[16, :2] = (0.71, 0.62)
    return points


SKELETON_EDGES = (
    (0, 7), (0, 8), (7, 11), (8, 12),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
)
