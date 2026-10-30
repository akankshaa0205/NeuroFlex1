from pathlib import Path

import joblib
import numpy as np

from neuroflex.domain.kimore import (
    KimoreQualityEstimator,
    aggregate_angle_sequence,
    feature_names,
    pose_to_kimore_angles,
)
from neuroflex.pose import synthetic_pose


class ConstantModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), 37.5)


def test_pose_maps_to_nine_finite_kimore_angles() -> None:
    angles = pose_to_kimore_angles(synthetic_pose(0))
    assert angles.shape == (9,)
    assert np.isfinite(angles).all()


def test_aggregate_schema_is_stable() -> None:
    sequence = np.tile(np.arange(9, dtype=float), (20, 1))
    values = aggregate_angle_sequence(sequence, "Es1")
    assert values.shape == (len(feature_names()),)
    assert values[-5:].tolist() == [1, 0, 0, 0, 0]


def test_estimator_clips_and_exposes_experimental_status(tmp_path: Path) -> None:
    artifact = tmp_path / "kimore.joblib"
    joblib.dump(
        {
            "model": ConstantModel(),
            "model_version": "test",
            "feature_names": feature_names(),
        },
        artifact,
    )
    result = KimoreQualityEstimator(artifact).predict(np.ones((20, 9)), "Es5")
    assert result.clinical_score_0_50 == 37.5
    assert result.quality_percent == 75.0
    assert result.status == "experimental_cross_sensor_estimate"
