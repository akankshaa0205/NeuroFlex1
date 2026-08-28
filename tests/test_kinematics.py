import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuroflex.domain.kinematics import joint_angle, smooth_angles, torso_lean_degrees


def test_known_right_angle() -> None:
    points = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=float)
    assert joint_angle(points) == pytest.approx(90)


def test_zero_length_limb_rejected() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        joint_angle(np.zeros((3, 3)))


@given(st.floats(min_value=0.01, max_value=100, allow_nan=False))
def test_angle_is_scale_invariant(scale: float) -> None:
    points = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=float)
    assert joint_angle(points * scale) == pytest.approx(joint_angle(points))


def test_smoothing_preserves_short_input() -> None:
    assert np.array_equal(smooth_angles([10, 20]), [10, 20])


def test_upright_torso_has_near_zero_frontal_lean() -> None:
    landmarks = np.zeros((33, 3))
    landmarks[11, :2] = (0.4, 0.3)
    landmarks[12, :2] = (0.6, 0.3)
    landmarks[23, :2] = (0.4, 0.7)
    landmarks[24, :2] = (0.6, 0.7)
    assert torso_lean_degrees(landmarks) == pytest.approx(0)
