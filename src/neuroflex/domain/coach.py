from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoachPhase(str, Enum):
    IDLE = "idle"
    PREPARE = "prepare"
    MOVE_OUT = "move_out"
    HOLD = "hold"
    RETURN = "return"
    REST = "rest"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class CoachUpdate:
    phase: CoachPhase
    instruction: str
    repetitions: int
    target_repetitions: int
    countdown_seconds: int | None = None


class SessionCoach:
    """A slow, deterministic rehabilitation session state machine."""

    def __init__(
        self,
        target_repetitions: int,
        *,
        preparation_seconds: float = 5.0,
        target_dwell_seconds: float = 0.8,
        hold_seconds: float = 1.2,
        return_dwell_seconds: float = 0.6,
        rest_seconds: float = 2.0,
    ) -> None:
        self.target_repetitions = target_repetitions
        self.preparation_seconds = preparation_seconds
        self.target_dwell_seconds = target_dwell_seconds
        self.hold_seconds = hold_seconds
        self.return_dwell_seconds = return_dwell_seconds
        self.rest_seconds = rest_seconds
        self.phase = CoachPhase.IDLE
        self.repetitions = 0
        self._phase_started = 0.0
        self._threshold_since: float | None = None
        self._tracking_lost_since: float | None = None

    def start(self, now: float) -> CoachUpdate:
        self.phase = CoachPhase.PREPARE
        self.repetitions = 0
        self._phase_started = now
        self._threshold_since = None
        self._tracking_lost_since = None
        return self._result("Get into the starting position. We begin in 5 seconds.", 5)

    def update(
        self, *, progress_ratio: float, visible: bool, posture_ok: bool, now: float
    ) -> CoachUpdate:
        if self.phase in {CoachPhase.IDLE, CoachPhase.COMPLETE}:
            return self._current(now)
        if not visible:
            if self._tracking_lost_since is None:
                self._tracking_lost_since = now
            self._threshold_since = None
            return self._result(
                "Session paused. Step back until all highlighted joints are visible."
            )
        if self._tracking_lost_since is not None:
            pause_duration = now - self._tracking_lost_since
            self._phase_started += pause_duration
            self._tracking_lost_since = None
        if not posture_ok:
            if self._tracking_lost_since is None:
                self._tracking_lost_since = now
            self._threshold_since = None
            return self._result(
                "Pause here. Return to a tall, steady posture before continuing."
            )

        if self.phase == CoachPhase.PREPARE:
            remaining = max(0, int(self.preparation_seconds - (now - self._phase_started) + 0.999))
            if remaining == 0:
                self._transition(CoachPhase.MOVE_OUT, now)
                return self._result(
                    "Slowly perform the movement toward your comfortable target. You can do it."
                )
            return self._result(
                f"Settle into the starting position. Beginning in {remaining}…", remaining
            )

        if self.phase == CoachPhase.MOVE_OUT:
            if progress_ratio >= 0.85:
                if self._threshold_since is None:
                    self._threshold_since = now
                elif now - self._threshold_since >= self.target_dwell_seconds:
                    self._transition(CoachPhase.HOLD, now)
                    return self._result("Excellent range. Hold gently—no straining.")
            else:
                self._threshold_since = None
            return self._result(
                "Move slowly toward the target. Keep breathing and stay controlled."
            )

        if self.phase == CoachPhase.HOLD:
            if now - self._phase_started >= self.hold_seconds:
                self._transition(CoachPhase.RETURN, now)
                return self._result("Now return slowly to the starting position.")
            return self._result("Hold this comfortable position. Keep breathing.")

        if self.phase == CoachPhase.RETURN:
            if progress_ratio <= 0.25:
                if self._threshold_since is None:
                    self._threshold_since = now
                elif now - self._threshold_since >= self.return_dwell_seconds:
                    self.repetitions += 1
                    if self.repetitions >= self.target_repetitions:
                        self._transition(CoachPhase.COMPLETE, now)
                        return self._result(
                            "Session complete. Wonderful work—your controlled effort matters."
                        )
                    self._transition(CoachPhase.REST, now)
                    return self._result(
                        f"Repetition {self.repetitions} complete. Relax and rest briefly."
                    )
            else:
                self._threshold_since = None
            return self._result("Return slowly and fully. Control the movement back to rest.")

        if self.phase == CoachPhase.REST:
            remaining = max(0, int(self.rest_seconds - (now - self._phase_started) + 0.999))
            if remaining == 0:
                self._transition(CoachPhase.MOVE_OUT, now)
                return self._result("Begin the next slow repetition when ready.")
            return self._result(f"Rest and breathe. Next repetition in {remaining}…", remaining)

        return self._current(now)

    def pause(self, now: float) -> None:
        if self._tracking_lost_since is None:
            self._tracking_lost_since = now

    def resume(self, now: float) -> None:
        if self._tracking_lost_since is not None:
            self._phase_started += now - self._tracking_lost_since
            self._tracking_lost_since = None

    def _transition(self, phase: CoachPhase, now: float) -> None:
        self.phase = phase
        self._phase_started = now
        self._threshold_since = None

    def _current(self, now: float) -> CoachUpdate:
        if self.phase == CoachPhase.COMPLETE:
            return self._result("Session complete. Save your summary when you are ready.")
        return self._result("Choose an exercise and start when ready.")

    def _result(self, instruction: str, countdown: int | None = None) -> CoachUpdate:
        return CoachUpdate(
            self.phase, instruction, self.repetitions, self.target_repetitions, countdown
        )
