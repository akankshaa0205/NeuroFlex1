from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LiveAssessment:
    score: float
    current_ratio: float
    achieved_ratio: float
    target_reached: bool
    returning: bool
    verdict: str


def assess_live_movement(
    *,
    current_progress_deg: float,
    achieved_rom_deg: float,
    target_rom_deg: float,
    confidence: float,
    continue_cue: str,
) -> LiveAssessment:
    if target_rom_deg <= 0:
        raise ValueError("Target ROM must be positive")
    current_ratio = float(np.clip(current_progress_deg / target_rom_deg, 0, 1))
    achieved_ratio = float(np.clip(achieved_rom_deg / target_rom_deg, 0, 1))
    confidence_factor = float(np.clip(confidence, 0, 1))
    score = float(100 * (0.65 * achieved_ratio + 0.35 * current_ratio) * confidence_factor)
    target_reached = current_ratio >= 0.9
    returning = achieved_ratio >= 0.8 and current_ratio < achieved_ratio * 0.65
    if target_reached:
        verdict = "Excellent! Target reached — return slowly with control"
    elif returning:
        verdict = "Great repetition — return smoothly to the start"
    elif current_ratio >= 0.7:
        verdict = "Almost there — you can do it, keep the movement controlled"
    elif current_ratio >= 0.4:
        verdict = "Strong progress — halfway there, keep going"
    elif current_ratio >= 0.15:
        verdict = f"Good start — {continue_cue.lower()} if comfortable"
    else:
        verdict = "You’ve got this — begin gently when ready"
    return LiveAssessment(
        score, current_ratio, achieved_ratio, target_reached, returning, verdict
    )


def choose_active_side(progress_history: dict[str, list[float]]) -> str:
    if not progress_history:
        raise ValueError("At least one side is required")
    peaks = {side: max(values) if values else 0.0 for side, values in progress_history.items()}
    return max(peaks, key=peaks.get)


def motivational_message(current_ratio: float, repetitions: int) -> str:
    if repetitions >= 8:
        return "Session complete — your consistency today builds tomorrow’s progress!"
    if current_ratio >= 0.9:
        return "Wonderful range! Stay smooth and breathe normally."
    if current_ratio >= 0.6:
        return "You’re doing well — controlled movement matters more than speed."
    if repetitions:
        return f"{repetitions} strong repetition{'s' if repetitions != 1 else ''} — keep it up!"
    return "Every careful movement counts. You can do this."
