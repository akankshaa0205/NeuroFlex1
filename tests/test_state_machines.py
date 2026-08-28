import numpy as np

from neuroflex.domain.gestures import GestureDebouncer, hands_together_at_chest
from neuroflex.domain.repetitions import RepetitionCounter
from neuroflex.pose import synthetic_pose


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


def test_gesture_progress_and_reset_are_visible_to_ui() -> None:
    detector = GestureDebouncer(dwell_frames=4, cooldown_frames=2)
    detector.update("Open_Palm", 0.9)
    detector.update("Open_Palm", 0.9)
    assert detector.candidate == "Open_Palm"
    assert detector.progress == 0.5
    detector.reset()
    assert detector.candidate is None
    assert detector.progress == 0.0


def test_hands_together_command_pose_is_distinct_from_normal_reach() -> None:
    pose = synthetic_pose(0)
    visibility = np.ones(33)
    shoulder_mid = np.mean(pose[[11, 12], :2], axis=0)
    hip_mid = np.mean(pose[[23, 24], :2], axis=0)
    chest = shoulder_mid + 0.32 * (hip_mid - shoulder_mid)
    pose[15, :2], pose[16, :2] = chest + (-0.01, 0), chest + (0.01, 0)
    detected, confidence = hands_together_at_chest(pose, visibility)
    assert detected
    assert confidence > 0.35
    pose[16, :2] = (0.9, 0.2)
    assert not hands_together_at_chest(pose, visibility)[0]
