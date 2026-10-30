import pytest

from neuroflex.domain.exercises import CORE_EXERCISE_IDS, EXERCISES, movement_signal


def test_library_covers_major_body_regions() -> None:
    regions = {exercise.body_region for exercise in EXERCISES}
    assert len(EXERCISES) >= 10
    assert {"Shoulder", "Elbow", "Hip", "Knee", "Ankle", "Trunk", "Full body"} <= regions


def test_flexion_signal_increases_as_joint_angle_closes() -> None:
    assert movement_signal(60, "flexion") > movement_signal(150, "flexion")


def test_unknown_signal_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        movement_signal(90, "unknown")


def test_five_core_exercises_are_presented_before_experimental_options() -> None:
    assert len(CORE_EXERCISE_IDS) == 5
    assert [exercise.id for exercise in EXERCISES[:5]] == [
        "shoulder-flexion",
        "shoulder-abduction",
        "elbow-flexion",
        "knee-extension",
        "sit-to-stand",
    ]
    assert all(exercise.id in CORE_EXERCISE_IDS for exercise in EXERCISES[:5])
    assert all(exercise.id not in CORE_EXERCISE_IDS for exercise in EXERCISES[5:])
