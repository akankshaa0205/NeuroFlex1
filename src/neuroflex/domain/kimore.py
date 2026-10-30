from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from neuroflex.domain.kinematics import joint_angle

ANGLE_NAMES = (
    "LeftElbow_Flexion",
    "RightElbow_Flexion",
    "LeftShoulder_Flexion",
    "RightShoulder_Flexion",
    "LeftHip_Flexion",
    "RightHip_Flexion",
    "LeftKnee_Flexion",
    "RightKnee_Flexion",
    "Trunk_Lean",
)
STAT_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "p10",
    "p50",
    "p90",
    "range",
    "vel_mean",
    "vel_std",
    "vel_p90",
)
EXERCISE_FOLDERS = ("Es1", "Es2", "Es3", "Es4", "Es5")


def pose_to_kimore_angles(landmarks: np.ndarray) -> np.ndarray:
    """Map a MediaPipe pose to the KIMORE-compatible nine-angle subset."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (33, 3):
        raise ValueError("Expected 33 MediaPipe pose landmarks")
    triplets = np.asarray(
        [
            (11, 13, 15),
            (12, 14, 16),
            (23, 11, 13),
            (24, 12, 14),
            (11, 23, 25),
            (12, 24, 26),
            (23, 25, 27),
            (24, 26, 28),
        ]
    )
    angles = np.asarray(joint_angle(points[triplets]), dtype=np.float64)
    shoulder_midpoint = points[[11, 12]].mean(axis=0)
    hip_midpoint = points[[23, 24]].mean(axis=0)
    vertical_reference = hip_midpoint + np.asarray([0.0, -1.0, 0.0])
    trunk = float(
        joint_angle(np.asarray([[shoulder_midpoint, hip_midpoint, vertical_reference]]))[0]
    )
    return np.concatenate([angles, [trunk]])


def aggregate_angle_sequence(sequence: np.ndarray, exercise_folder: str) -> np.ndarray:
    data = np.asarray(sequence, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(ANGLE_NAMES):
        raise ValueError(f"Expected (frames, {len(ANGLE_NAMES)}) angle sequence")
    if len(data) < 2:
        raise ValueError("At least two frames are required for quality estimation")
    features: list[float] = []
    for column in data.T:
        velocity = np.abs(np.diff(column))
        features.extend(
            [
                float(np.mean(column)),
                float(np.std(column)),
                float(np.min(column)),
                float(np.max(column)),
                float(np.percentile(column, 10)),
                float(np.median(column)),
                float(np.percentile(column, 90)),
                float(np.ptp(column)),
                float(np.mean(velocity)),
                float(np.std(velocity)),
                float(np.percentile(velocity, 90)),
            ]
        )
    features.extend(float(exercise_folder == candidate) for candidate in EXERCISE_FOLDERS)
    return np.asarray(features, dtype=np.float64)


def feature_names() -> list[str]:
    return [f"{angle}__{stat}" for angle in ANGLE_NAMES for stat in STAT_NAMES] + [
        f"exercise__{folder}" for folder in EXERCISE_FOLDERS
    ]


@dataclass(frozen=True, slots=True)
class KimorePrediction:
    clinical_score_0_50: float
    quality_percent: float
    model_version: str
    status: str = "experimental_cross_sensor_estimate"


class KimoreQualityEstimator:
    def __init__(self, artifact_path: Path) -> None:
        artifact = joblib.load(artifact_path)
        if artifact.get("feature_names") != feature_names():
            raise ValueError("KIMORE model feature schema is incompatible")
        self.model = artifact["model"]
        self.version = str(artifact["model_version"])

    def predict(self, sequence: np.ndarray, exercise_folder: str) -> KimorePrediction:
        features = aggregate_angle_sequence(sequence, exercise_folder)
        raw = float(self.model.predict(features.reshape(1, -1))[0])
        score = float(np.clip(raw, 0.0, 50.0))
        return KimorePrediction(score, score * 2.0, self.version)
