from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import neuroflex.app as app_module
from neuroflex.domain.exercises import EXERCISES
from neuroflex.domain.models import PatientIntake, SessionState
from neuroflex.pose import synthetic_pose


class FakeCamera:
    def __init__(self, model_path: Path) -> None:
        self.last_visibility = np.ones(33)

    @property
    def available(self) -> bool:
        return True

    def read(self) -> tuple[np.ndarray, np.ndarray, float]:
        return np.zeros((480, 640, 3), dtype=np.uint8), synthetic_pose(0), 1.0

    def close(self) -> None:
        pass


def test_intake_calibration_then_personalized_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app_module, "LivePoseCamera", FakeCamera)
    application = QApplication.instance() or QApplication([])
    window = app_module.MainWindow()
    window.intake = PatientIntake(
        58, "Shoulder", "Right", "Moderate", "Reach overhead", 2
    )
    window._primary_action()
    assert window.calibration_armed
    window._primary_action()
    assert window.state == SessionState.CALIBRATING
    exercise = EXERCISES[window.exercise.currentIndex()]
    window._finish_calibration(exercise, 100, 12)
    assert window.baselines[exercise.id].target_rom_deg == pytest.approx(77.9)
    window._primary_action()
    assert window.state == SessionState.ACTIVE
    window.close()
    application.processEvents()
