from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProgressSummary:
    sessions: int
    first_score: float
    latest_score: float
    score_change: float
    first_rom: float
    latest_rom: float
    rom_change: float


def summarize_progress(sessions: list[dict[str, Any]], exercise_id: str) -> ProgressSummary | None:
    matching = sorted(
        (row for row in sessions if row.get("exercise_id") == exercise_id),
        key=lambda row: str(row.get("started_at", "")),
    )
    if not matching:
        return None
    first, latest = matching[0], matching[-1]
    first_score, latest_score = float(first.get("score", 0)), float(latest.get("score", 0))
    first_rom = float(first.get("achieved_rom_deg", 0))
    latest_rom = float(latest.get("achieved_rom_deg", 0))
    return ProgressSummary(
        len(matching), first_score, latest_score, latest_score - first_score,
        first_rom, latest_rom, latest_rom - first_rom,
    )
