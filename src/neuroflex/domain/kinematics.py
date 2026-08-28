from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import savgol_filter


def joint_angle(points: ArrayLike, *, eps: float = 1e-9) -> np.ndarray:
    """Return ABC angle(s) in degrees for arrays shaped (..., 3, 3)."""
    data = np.asarray(points, dtype=np.float64)
    if data.shape[-2:] != (3, 3):
        raise ValueError("Expected shape (..., 3, 3)")
    left = data[..., 0, :] - data[..., 1, :]
    right = data[..., 2, :] - data[..., 1, :]
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    if np.any(denominator <= eps):
        raise ValueError("Cannot compute an angle from a zero-length limb")
    cosine = np.sum(left * right, axis=-1) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def extract_joint_angle(landmarks: ArrayLike, triplet: tuple[int, int, int]) -> float:
    data = np.asarray(landmarks, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Landmarks must have shape (joints, 3)")
    return float(joint_angle(data[np.asarray(triplet)]))


def smooth_angles(values: ArrayLike, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Smooth a completed/rolling trajectory; short inputs pass through unchanged."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("Angle trajectory must be one-dimensional")
    if data.size < 3:
        return data.copy()
    effective = min(window, data.size if data.size % 2 else data.size - 1)
    if effective <= polyorder:
        return data.copy()
    return np.asarray(savgol_filter(data, effective, polyorder), dtype=np.float64)


def torso_lean_degrees(landmarks: ArrayLike) -> float:
    """Estimate frontal-plane torso lean from shoulder/hip midpoints."""
    data = np.asarray(landmarks, dtype=np.float64)
    if data.shape != (33, 3):
        raise ValueError("Expected 33 pose landmarks")
    shoulder = np.mean(data[[11, 12], :2], axis=0)
    hip = np.mean(data[[23, 24], :2], axis=0)
    torso = shoulder - hip
    if np.linalg.norm(torso) <= 1e-9:
        raise ValueError("Shoulders and hips overlap")
    return float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1]))))
