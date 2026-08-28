from dataclasses import dataclass


@dataclass(slots=True)
class RepetitionCounter:
    low_deg: float
    high_deg: float
    repetitions: int = 0
    _raised: bool = False

    def update(self, angle_deg: float) -> bool:
        if not self._raised and angle_deg >= self.high_deg:
            self._raised = True
        elif self._raised and angle_deg <= self.low_deg:
            self._raised = False
            self.repetitions += 1
            return True
        return False
