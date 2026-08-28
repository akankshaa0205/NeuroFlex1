from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(slots=True)
class GestureDebouncer:
    dwell_frames: int = 4
    cooldown_frames: int = 12
    confidence_threshold: float = 0.85
    _candidate: str | None = None
    _count: int = 0
    _cooldown: int = 0
    _must_release: bool = False

    @property
    def progress(self) -> float:
        return min(1.0, self._count / self.dwell_frames)

    @property
    def candidate(self) -> str | None:
        return self._candidate

    def reset(self) -> None:
        self._candidate, self._count = None, 0

    def clear(self) -> None:
        self._candidate, self._count, self._cooldown, self._must_release = None, 0, 0, False

    def update(self, label: str | None, confidence: float) -> str | None:
        if self._must_release:
            if label is None:
                self._must_release = False
            return None
        if self._cooldown:
            self._cooldown -= 1
            return None
        if label is None or confidence < self.confidence_threshold:
            self._candidate, self._count = None, 0
            return None
        if label != self._candidate:
            self._candidate, self._count = label, 1
            return None
        self._count += 1
        if self._count >= self.dwell_frames:
            fired = self._candidate
            self._candidate, self._count, self._cooldown = None, 0, self.cooldown_frames
            self._must_release = True
            return fired
        return None


def hands_together_at_chest(
    landmarks: ArrayLike, visibility: ArrayLike, *, min_visibility: float = 0.35
) -> tuple[bool, float]:
    """Recognize a deliberate command pose using pose joints visible at exercise distance."""
    points = np.asarray(landmarks, dtype=np.float64)
    quality = np.asarray(visibility, dtype=np.float64)
    if points.shape != (33, 3) or quality.shape != (33,):
        return False, 0.0
    required = np.asarray([11, 12, 15, 16, 23, 24])
    confidence = float(np.min(quality[required]))
    if confidence < min_visibility:
        return False, confidence
    left_shoulder, right_shoulder = points[11, :2], points[12, :2]
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    if shoulder_width <= 1e-6:
        return False, confidence
    wrists = points[[15, 16], :2]
    wrist_gap = float(np.linalg.norm(wrists[0] - wrists[1])) / shoulder_width
    shoulder_mid = np.mean(points[[11, 12], :2], axis=0)
    hip_mid = np.mean(points[[23, 24], :2], axis=0)
    chest = shoulder_mid + 0.32 * (hip_mid - shoulder_mid)
    chest_distance = float(np.linalg.norm(np.mean(wrists, axis=0) - chest)) / shoulder_width
    detected = wrist_gap <= 0.55 and chest_distance <= 0.75
    geometry_score = min(1.0, max(0.0, 1.0 - wrist_gap / 0.8))
    return detected, min(confidence, geometry_score)
