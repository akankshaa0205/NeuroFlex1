import pytest

from neuroflex.domain.exercises import EXERCISES, movement_signal


def test_library_covers_major_body_regions() -> None:
    regions = {exercise.body_region for exercise in EXERCISES}
    assert len(EXERCISES) >= 10
    assert {"Shoulder", "Elbow", "Hip", "Knee", "Ankle", "Trunk", "Full body"} <= regions


def test_flexion_signal_increases_as_joint_angle_closes() -> None:
    assert movement_signal(60, "flexion") > movement_signal(150, "flexion")


def test_unknown_signal_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        movement_signal(90, "unknown")
