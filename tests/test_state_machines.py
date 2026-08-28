from neuroflex.domain.gestures import GestureDebouncer
from neuroflex.domain.repetitions import RepetitionCounter


def test_rep_requires_high_then_low() -> None:
    counter = RepetitionCounter(50, 100)
    assert not counter.update(40)
    assert not counter.update(110)
    assert counter.update(45)
    assert counter.repetitions == 1


def test_gesture_requires_confident_dwell_and_cooldown() -> None:
    detector = GestureDebouncer(dwell_frames=3, cooldown_frames=2)
    assert detector.update("thumbs_up", 0.7) is None
    assert detector.update("thumbs_up", 0.9) is None
    assert detector.update("thumbs_up", 0.9) is None
    assert detector.update("thumbs_up", 0.9) == "thumbs_up"
    assert detector.update("thumbs_up", 0.9) is None
