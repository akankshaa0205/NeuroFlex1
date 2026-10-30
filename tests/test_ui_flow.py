import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import neuroflex.app as app_module
from neuroflex.domain.exercises import EXERCISES
from neuroflex.domain.gestures import GestureDebouncer
from neuroflex.domain.models import PatientIntake, SessionState
from neuroflex.pose import synthetic_pose


class FakeCamera:
    def __init__(
        self,
        model_path: Path,
        gesture_model_path: Path | None = None,
        hand_model_path: Path | None = None,
    ) -> None:
        self.last_visibility = np.ones(33)
        self.last_world_landmarks = None
        self.last_gesture: tuple[str | None, float] = (None, 0.0)
        self.gesture_recognizer = None
        self.gesture_error = None
        self.gesture_threshold = 0.75

    @property
    def available(self) -> bool:
        return True

    def read(self) -> tuple[np.ndarray, np.ndarray, float]:
        return np.zeros((480, 640, 3), dtype=np.uint8), synthetic_pose(0), 1.0

    def close(self) -> None:
        pass


def test_background_camera_pipeline_delivers_a_frame(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app_module, "LivePoseCamera", FakeCamera)
    application = QApplication.instance() or QApplication([])
    window = app_module.MainWindow()
    deadline = time.monotonic() + 1.0
    while window.canvas.frame is None and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert window.canvas.frame is not None
    assert np.allclose(window.pose_visibility, 1.0)
    assert window.camera_worker is not None
    window.close()
    application.processEvents()


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
    window._finish_calibration(exercise, 100, 12, 4)
    assert window.baselines[exercise.id].target_rom_deg == pytest.approx(77.9)
    window._primary_action()
    assert window.state == SessionState.ACTIVE
    window.gesture_debouncer = GestureDebouncer(dwell_frames=2, cooldown_frames=2)
    window.gesture_enabled.setChecked(True)
    command_pose = synthetic_pose(0)
    chest = np.mean(command_pose[[11, 12], :2], axis=0) + 0.32 * (
        np.mean(command_pose[[23, 24], :2], axis=0)
        - np.mean(command_pose[[11, 12], :2], axis=0)
    )
    command_pose[15, :2] = chest + (-0.01, 0)
    command_pose[16, :2] = chest + (0.01, 0)
    window._process_gesture(command_pose)
    window._process_gesture(command_pose)
    assert window.state == SessionState.PAUSED

    # The trained-model path uses the artifact threshold, the same consecutive-frame
    # debouncer, and state-safe command mappings.
    window._primary_action()
    window.gesture_debouncer.clear()
    window.camera.last_gesture = ("palm", 0.90)
    neutral_pose = synthetic_pose(0)
    window._process_gesture(neutral_pose)
    assert window.state == SessionState.ACTIVE
    window._process_gesture(neutral_pose)
    assert window.state == SessionState.PAUSED

    window._primary_action()
    window.gesture_debouncer.clear()
    window.camera.last_gesture = ("like", 0.50)
    window._process_gesture(neutral_pose)
    window._process_gesture(neutral_pose)
    assert window.state == SessionState.ACTIVE
    window.close()
    application.processEvents()
