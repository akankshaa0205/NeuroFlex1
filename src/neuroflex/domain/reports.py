from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SessionReport:
    completed_repetitions: int
    prescribed_repetitions: int
    achieved_rom_deg: float
    target_rom_deg: float
    comfortable_rom_deg: float
    target_attainment_pct: float
    baseline_attainment_pct: float
    posture_adherence_pct: float
    tracking_coverage_pct: float
    rom_change_from_day_one_deg: float | None
    guidance: tuple[str, ...]
    recalibration_recommended: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["guidance"] = list(self.guidance)
        return result


def build_session_report(
    payload: dict[str, Any], historical_sessions: list[dict[str, Any]]
) -> SessionReport:
    personalization = payload.get("personalization", {})
    if not isinstance(personalization, dict):
        personalization = {}
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        frames = []
    achieved = float(payload.get("achieved_rom_deg", 0))
    target = float(personalization.get("target_rom_deg") or 0)
    comfortable = float(personalization.get("comfortable_rom_deg") or 0)
    posture_values = [bool(row["posture_ok"]) for row in frames if "posture_ok" in row]
    tracked_values = [
        float(row.get("tracked_confidence", row.get("confidence", 0))) for row in frames
    ]
    posture_pct = 100 * sum(posture_values) / len(posture_values) if posture_values else 0.0
    tracking_pct = (
        100 * sum(value >= 0.25 for value in tracked_values) / len(tracked_values)
        if tracked_values
        else 0.0
    )
    comparable = _comparable_sessions(payload, historical_sessions)
    first_rom = float(comparable[0].get("achieved_rom_deg", 0)) if comparable else achieved
    rom_change = achieved - first_rom if comparable else None
    guidance: list[str] = []
    target_pct = 100 * achieved / target if target > 0 else 0.0
    if target_pct >= 95:
        guidance.append("You reached your personalized range target with controlled effort.")
    else:
        guidance.append("Continue within comfort; consistent controlled range matters more than force.")
    if posture_pct < 80:
        guidance.append("Next time, slow down and keep the calibrated torso position steady.")
    else:
        guidance.append("Your posture stayed consistent for most measured movement frames.")
    if tracking_pct < 85:
        guidance.append("Improve camera framing so the highlighted joints stay visible throughout.")
    eligible = recalibration_is_recommended(payload, historical_sessions)
    if eligible:
        guidance.append("Your recent sessions suggest a fresh comfort calibration may be useful.")
    return SessionReport(
        int(payload.get("repetitions", 0)),
        int(payload.get("prescribed_repetitions", 0)),
        achieved,
        target,
        comfortable,
        target_pct,
        100 * achieved / comfortable if comfortable > 0 else 0.0,
        posture_pct,
        tracking_pct,
        rom_change,
        tuple(guidance),
        eligible,
    )


def recalibration_is_recommended(
    payload: dict[str, Any], historical_sessions: list[dict[str, Any]]
) -> bool:
    rows = [*_comparable_sessions(payload, historical_sessions), payload]
    if len(rows) < 3:
        return False
    recent = rows[-3:]
    personalization = payload.get("personalization", {})
    comfortable = float(personalization.get("comfortable_rom_deg") or 0) if isinstance(personalization, dict) else 0
    if comfortable <= 0:
        return False
    for row in recent:
        if int(row.get("repetitions", 0)) < int(row.get("prescribed_repetitions", 0)):
            return False
        if float(row.get("posture_adherence_pct", 0)) < 85:
            return False
        if float(row.get("tracking_coverage_pct", 0)) < 85:
            return False
    roms = [float(row.get("achieved_rom_deg", 0)) for row in recent]
    return bool(np.median(roms) >= comfortable * 1.05 and np.median(roms) >= comfortable + 5 and roms[-1] >= roms[0])


def _comparable_sessions(
    payload: dict[str, Any], sessions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    current_personalization = payload.get("personalization", {})
    current_side = current_personalization.get("affected_side") if isinstance(current_personalization, dict) else None
    matching = []
    for row in sessions:
        row_personalization = row.get("personalization", {})
        row_side = row_personalization.get("affected_side") if isinstance(row_personalization, dict) else None
        if (
            row.get("patient_id") == payload.get("patient_id")
            and row.get("exercise_id") == payload.get("exercise_id")
            and row.get("algorithm_version") == payload.get("algorithm_version")
            and row_side == current_side
        ):
            matching.append(row)
    return sorted(matching, key=lambda row: str(row.get("started_at", "")))
