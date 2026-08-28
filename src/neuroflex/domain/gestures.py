from dataclasses import dataclass


@dataclass(slots=True)
class GestureDebouncer:
    dwell_frames: int = 4
    cooldown_frames: int = 12
    confidence_threshold: float = 0.85
    _candidate: str | None = None
    _count: int = 0
    _cooldown: int = 0

    def update(self, label: str | None, confidence: float) -> str | None:
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
            return fired
        return None
