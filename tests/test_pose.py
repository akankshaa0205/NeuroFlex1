import numpy as np

from neuroflex.pose import synthetic_pose


def test_synthetic_pose_has_mediapipe_landmark_contract() -> None:
    pose = synthetic_pose(0.5)
    assert pose.shape == (33, 3)
    assert pose.dtype == np.float64
    assert np.isfinite(pose).all()
