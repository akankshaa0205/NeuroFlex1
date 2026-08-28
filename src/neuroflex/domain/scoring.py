from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from neuroflex.domain.models import ScoreComponents, ScoreConfig


def exact_dtw(left: ArrayLike, right: ArrayLike) -> float:
    """Deterministic exact DTW reference used for short rehabilitation repetitions."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if not a.size or not b.size:
        raise ValueError("DTW trajectories cannot be empty")
    previous = np.full(b.size + 1, np.inf)
    previous[0] = 0.0
    for value in a:
        current = np.full(b.size + 1, np.inf)
        for index, target in enumerate(b, start=1):
            current[index] = abs(value - target) + min(
                current[index - 1], previous[index], previous[index - 1]
            )
        previous = current
    return float(previous[-1] / max(a.size, b.size))


def movement_score(
    *,
    achieved_rom_deg: float,
    dtw_distance: float,
    theta_error_deg: float,
    config: ScoreConfig,
    confidence: float = 1.0,
) -> ScoreComponents:
    """Provisional, bounded interpretation of the PRD composite score."""
    rom = float(np.clip(achieved_rom_deg / config.target_rom_deg, 0.0, 1.0))
    alignment = float(np.clip(1.0 - dtw_distance / config.dtw_max, 0.0, 1.0))
    accuracy = float(np.clip(1.0 - theta_error_deg / config.theta_threshold_deg, 0.0, 1.0))
    penalty = float(np.clip(confidence, 0.0, 1.0))
    composite = float(np.clip(100.0 * np.mean([rom, alignment, accuracy]) * penalty, 0, 100))
    return ScoreComponents(rom, alignment, accuracy, penalty, composite)
