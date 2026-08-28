from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SessionState(str, Enum):
    READY = "ready"
    CALIBRATING = "calibrating"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class Exercise:
    id: str
    name: str
    joint_triplet: tuple[int, int, int]
    target_rom_deg: float
    theta_threshold_deg: float
    repetitions: int = 8
    cue_low: str = "Move a little farther"
    cue_high: str = "Reduce the range"
    body_region: str = "Upper body"
    movement_mode: str = "angle"


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    target_rom_deg: float
    dtw_max: float
    theta_threshold_deg: float

    def __post_init__(self) -> None:
        if min(self.target_rom_deg, self.dtw_max, self.theta_threshold_deg) <= 0:
            raise ValueError("Scoring denominators must be positive")


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    rom: float
    alignment: float
    accuracy: float
    penalty: float
    composite: float


@dataclass(frozen=True, slots=True)
class PoseFrame:
    frame_id: int
    timestamp_ns: int
    landmarks: FloatArray
    confidence: float


@dataclass(frozen=True, slots=True)
class PatientIntake:
    age_years: int
    body_area: str
    affected_side: str
    mobility_level: str
    goal: str
    discomfort: int

    def __post_init__(self) -> None:
        if not 5 <= self.age_years <= 110:
            raise ValueError("Age must be between 5 and 110")
        if self.affected_side not in {"Left", "Right", "Either"}:
            raise ValueError("Affected side must be Left, Right, or Either")
        if self.mobility_level not in {"Limited", "Moderate", "Good"}:
            raise ValueError("Unknown mobility level")
        if not 0 <= self.discomfort <= 10:
            raise ValueError("Discomfort must be between 0 and 10")


@dataclass(frozen=True, slots=True)
class PersonalBaseline:
    exercise_id: str
    side: str
    comfortable_rom_deg: float
    target_rom_deg: float
    rest_value_deg: float
    posture_reference_deg: float
    calibrated_at: str
