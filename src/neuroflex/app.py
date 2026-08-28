from __future__ import annotations

import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from neuroflex.domain.coach import CoachPhase, CoachUpdate, SessionCoach
from neuroflex.domain.exercises import EXERCISES, movement_signal
from neuroflex.domain.feedback import assess_live_movement, choose_active_side
from neuroflex.domain.guidance import guidance_for
from neuroflex.domain.kinematics import extract_joint_angle, smooth_angles, torso_lean_degrees
from neuroflex.domain.models import PatientIntake, PersonalBaseline, SessionState
from neuroflex.domain.personalization import personalized_target, readiness_message
from neuroflex.domain.progress import summarize_progress
from neuroflex.persistence import SCHEMA_VERSION, SessionRepository, export_session
from neuroflex.pose import SKELETON_EDGES, LivePoseCamera

DATA_DIR = Path.home() / ".neuroflex"
MODEL_PATH = Path(__file__).parents[2] / "models" / "pose_landmarker_full.task"


class PoseCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.landmarks: np.ndarray | None = None
        self.visibility: np.ndarray | None = None
        self.frame: QImage | None = None
        self.feedback = "Stand where your full body is visible"
        self.phase_text = "READY"
        self.rep_text = "0 / 8"
        self.focus_indices: tuple[int, int, int] | None = None
        self.good = True
        self.setMinimumSize(620, 500)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#08111f"))
        if self.frame is not None:
            scaled = self.frame.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            frame_x = (self.width() - scaled.width()) / 2
            frame_y = (self.height() - scaled.height()) / 2
            painter.drawImage(round(frame_x), round(frame_y), scaled)
            painter.fillRect(self.rect(), QColor(2, 10, 22, 48))
        else:
            painter.fillRect(QRectF(18, 18, self.width() - 36, self.height() - 36), QColor("#0d2037"))
        if self.landmarks is not None:
            if self.frame is not None:
                scaled = self.frame.scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
                frame_x = (self.width() - scaled.width()) / 2
                frame_y = (self.height() - scaled.height()) / 2
                points = [
                    QPointF(frame_x + p[0] * scaled.width(), frame_y + p[1] * scaled.height())
                    for p in self.landmarks
                ]
            else:
                points = [QPointF(p[0] * self.width(), p[1] * self.height()) for p in self.landmarks]
            visible = self.visibility if self.visibility is not None else np.ones(len(points))
            painter.setPen(QPen(QColor(0, 0, 0, 180), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for left, right in SKELETON_EDGES:
                if min(visible[left], visible[right]) >= 0.25:
                    painter.drawLine(points[left], points[right])
            painter.setPen(QPen(QColor("#34d399" if self.good else "#fb7185"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for left, right in SKELETON_EDGES:
                if min(visible[left], visible[right]) >= 0.25:
                    painter.drawLine(points[left], points[right])
            if self.focus_indices is not None:
                first, middle, last = self.focus_indices
                painter.setPen(QPen(QColor("#fbbf24"), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(points[first], points[middle])
                painter.drawLine(points[middle], points[last])
            painter.setBrush(QColor("#f8fafc"))
            painter.setPen(Qt.PenStyle.NoPen)
            for index in range(len(points)):
                if visible[index] < 0.25:
                    continue
                radius = 8 if self.focus_indices and index in self.focus_indices else 5
                painter.drawEllipse(points[index], radius, radius)
        painter.setPen(QColor("#e2e8f0"))
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        painter.drawText(QRectF(30, 30, self.width() - 60, 40), self.feedback)
        painter.setBrush(QColor(3, 18, 34, 220))
        painter.setPen(QPen(QColor("#38bdf8"), 2))
        painter.drawRoundedRect(QRectF(28, self.height() - 92, 210, 58), 12, 12)
        painter.setPen(QColor("#bae6fd"))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(QRectF(44, self.height() - 82, 178, 22), self.phase_text)
        painter.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        painter.drawText(QRectF(44, self.height() - 61, 178, 28), f"REPS  {self.rep_text}")


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("card")
        self.setMinimumHeight(88)
        layout = QVBoxLayout(self)
        heading = QLabel(title.upper())
        heading.setObjectName("muted")
        self.value = QLabel(value)
        self.value.setStyleSheet(f"color:{accent}; font-size:30px; font-weight:700")
        layout.addWidget(heading)
        layout.addWidget(self.value)


class IntakeDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Personalize your NeuroFlex session")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        heading = QLabel("Before we begin")
        heading.setStyleSheet("font-size:26px;font-weight:700")
        explanation = QLabel(
            "These answers set up a comfortable baseline. They do not diagnose a condition."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("muted")
        form = QFormLayout()
        self.age = QSpinBox()
        self.age.setRange(5, 110)
        self.age.setValue(60)
        self.age.setSuffix(" years")
        self.area = QComboBox()
        self.area.addItems(
            ["Shoulder", "Elbow", "Hip", "Knee", "Ankle", "Trunk", "Full body"]
        )
        self.side = QComboBox()
        self.side.addItems(["Right", "Left", "Either"])
        self.mobility = QComboBox()
        self.mobility.addItems(["Limited", "Moderate", "Good"])
        self.mobility.setCurrentIndex(1)
        self.goal = QComboBox()
        self.goal.addItems(
            [
                "Improve comfortable range",
                "Improve bending or straightening",
                "Improve controlled reaching",
                "Improve lower-body mobility",
                "Improve sit-to-stand control",
                "Track progress over time",
            ]
        )
        self.discomfort = QSpinBox()
        self.discomfort.setRange(0, 10)
        self.discomfort.setSuffix(" / 10")
        form.addRow("Age band", self.age)
        form.addRow("Area you want help with", self.area)
        form.addRow("Side to work on", self.side)
        form.addRow("Current mobility", self.mobility)
        form.addRow("Main goal", self.goal)
        form.addRow("Discomfort right now", self.discomfort)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue to calibration")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(heading)
        layout.addWidget(explanation)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def intake(self) -> PatientIntake:
        return PatientIntake(
            self.age.value(), self.area.currentText(), self.side.currentText(),
            self.mobility.currentText(), self.goal.currentText(), self.discomfort.value(),
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NeuroFlex — Rehabilitation Assessment")
        self.resize(1360, 850)
        self.repo = SessionRepository(DATA_DIR / "neuroflex.db")
        self.patient_id = self.repo.ensure_demo_patient()
        self.state = SessionState.READY
        self.phase = 0.0
        self.frames: list[dict[str, object]] = []
        self.progress_history: dict[str, list[float]] = {"Left": [], "Right": []}
        self.active_side = "Left"
        self.previous_progress: float | None = None
        self.posture_history: list[bool] = []
        self.intake: PatientIntake | None = None
        self.baselines: dict[str, PersonalBaseline] = {}
        self.calibration_samples: list[float] = []
        self.calibration_rest_samples: list[float] = []
        self.calibration_posture_samples: list[float] = []
        self.calibration_started = 0.0
        self.calibration_valid_elapsed = 0.0
        self.calibration_last_valid_at: float | None = None
        self.calibration_armed = False
        self.tracking_lost_at: float | None = None
        self._load_personalization()
        self.started_at = datetime.now(UTC)
        self.coach: SessionCoach | None = None
        self.camera: LivePoseCamera | None = None
        self.camera_error: str | None = None
        try:
            self.camera = LivePoseCamera(MODEL_PATH)
            if not self.camera.available:
                self.camera_error = "Camera permission or device access is unavailable"
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.camera_error = str(error)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._safe_tick)
        self.timer.start(50)

    def _load_personalization(self) -> None:
        profile = self.repo.load_profile(self.patient_id)
        if profile:
            # Older prototypes stored only an age band and could silently select the wrong arm.
            # Require the clearer current intake once instead of migrating an unsafe assumption.
            if "age_band" in profile:
                return
            profile.setdefault("body_area", "Shoulder")
            self.intake = PatientIntake(**profile)
        for data in self.repo.load_baselines(self.patient_id):
            if "rest_value_deg" not in data or "posture_reference_deg" not in data:
                continue
            baseline = PersonalBaseline(**data)
            self.baselines[baseline.exercise_id] = baseline

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 22, 28, 24)
        header = QHBoxLayout()
        brand = QLabel("NEUROFLEX")
        brand.setObjectName("brand")
        warning = QLabel("INVESTIGATIONAL • NOT CLINICALLY VALIDATED")
        warning.setObjectName("warning")
        self.mode = QPushButton("Clinician view")
        self.mode.clicked.connect(self._toggle_view)
        header.addWidget(brand)
        header.addStretch()
        header.addWidget(warning)
        header.addWidget(self.mode)
        outer.addLayout(header)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._patient_view())
        self.stack.addWidget(self._clinician_view())
        outer.addWidget(self.stack)
        self.setCentralWidget(root)
        self.setStyleSheet(STYLES)

    def _patient_view(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 18, 0, 0)
        self.canvas = PoseCanvas()
        layout.addWidget(self.canvas, 3)
        right = QVBoxLayout()
        right.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_content = QWidget()
        side = QVBoxLayout(side_content)
        side.setContentsMargins(0, 0, 6, 0)
        self.stage = QLabel("1  PROFILE   →   2  CALIBRATE   →   3  MOVE   →   4  PROGRESS")
        self.stage.setObjectName("stage")
        self.stage.setWordWrap(True)
        side.addWidget(self.stage)
        welcome = QLabel("Your movement, your progress")
        welcome.setObjectName("sectionTitle")
        side.addWidget(welcome)
        title = QLabel("Today’s movement")
        title.setObjectName("muted")
        self.exercise = QComboBox()
        self.exercise.addItems([item.name for item in EXERCISES])
        self.exercise.setCurrentIndex(0)
        self.exercise.currentIndexChanged.connect(self._exercise_changed)
        self.exercise.setMinimumHeight(46)
        side.addWidget(title)
        side.addWidget(self.exercise)
        self.guidance = QLabel()
        self.guidance.setObjectName("guidance")
        self.guidance.setWordWrap(True)
        side.addWidget(self.guidance)
        self._refresh_guidance()
        self.profile_summary = QLabel("No personal profile yet • Start to answer setup questions")
        self.profile_summary.setObjectName("status")
        self.profile_summary.setWordWrap(True)
        side.addWidget(self.profile_summary)
        if self.intake is not None:
            self.profile_summary.setText(
                f"Age {self.intake.age_years} • {self.intake.body_area} • {self.intake.affected_side} side • "
                f"{self.intake.mobility_level} mobility • discomfort {self.intake.discomfort}/10"
            )
        edit_profile = QPushButton("Update profile / recalibrate")
        edit_profile.clicked.connect(self._update_profile)
        side.addWidget(edit_profile)
        metrics = QGridLayout()
        self.score_card = MetricCard("Live score", "—", "#60a5fa")
        self.angle_card = MetricCard("Joint angle", "—", "#34d399")
        self.rep_card = MetricCard("Repetitions", "0 / 8", "#fbbf24")
        metrics.addWidget(self.score_card, 0, 0)
        metrics.addWidget(self.angle_card, 0, 1)
        metrics.addWidget(self.rep_card, 1, 0, 1, 2)
        side.addLayout(metrics)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        side.addWidget(self.progress)
        self.motivation = QLabel("Every careful movement counts. You can do this.")
        self.motivation.setObjectName("motivation")
        self.motivation.setWordWrap(True)
        side.addWidget(self.motivation)
        self.status = QLabel("Ready when you are")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        buttons = QHBoxLayout()
        self.primary = QPushButton("Start session")
        self.primary.setObjectName("primary")
        self.primary.clicked.connect(self._primary_action)
        self.save_button = QPushButton("Save summary")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        buttons.addWidget(self.primary)
        buttons.addWidget(self.save_button)
        scroll.setWidget(side_content)
        right.addWidget(scroll, 1)
        right.addLayout(buttons)
        controls = QLabel("Use the large Start/Pause button. Touchless gestures remain a validation feature.")
        controls.setObjectName("muted")
        controls.setWordWrap(True)
        right.addWidget(controls)
        layout.addLayout(right, 2)
        return page

    def _clinician_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel("Clinical session review")
        heading.setStyleSheet("font-size:28px;font-weight:700")
        subtitle = QLabel("Pseudonymous local records • algorithm and export versions retained")
        subtitle.setObjectName("muted")
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Date", "Patient", "Exercise", "Target ROM", "Score", "Change", "Reps"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        refresh = QPushButton("Refresh sessions")
        refresh.clicked.connect(self._refresh_sessions)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addWidget(self.table)
        layout.addWidget(refresh, alignment=Qt.AlignmentFlag.AlignRight)
        return page

    def _toggle_view(self) -> None:
        next_index = 1 - self.stack.currentIndex()
        self.stack.setCurrentIndex(next_index)
        self.mode.setText("Patient view" if next_index else "Clinician view")
        if next_index:
            self._refresh_sessions()

    def _primary_action(self) -> None:
        if self.intake is None:
            dialog = IntakeDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            self.intake = dialog.intake()
            self.repo.save_profile(self.patient_id, asdict(self.intake))
            for index, candidate in enumerate(EXERCISES):
                if candidate.body_region == self.intake.body_area:
                    self.exercise.setCurrentIndex(index)
                    break
            self.profile_summary.setText(
                f"Age {self.intake.age_years} • {self.intake.body_area} • {self.intake.affected_side} side • "
                f"{self.intake.mobility_level} mobility • discomfort {self.intake.discomfort}/10"
            )
            self.status.setText(readiness_message(self.intake))
            self.stage.setText("✓  PROFILE   →   2  CALIBRATE   →   3  MOVE   →   4  PROGRESS")
        exercise_id = EXERCISES[self.exercise.currentIndex()].id
        if exercise_id not in self.baselines:
            if not self.calibration_armed:
                self.calibration_armed = True
                self.primary.setText("Begin calibration")
                self.status.setText(
                    "Read the START, MOVE, and FORM instructions above. When positioned safely, "
                    "press Begin calibration."
                )
                self.motivation.setText(
                    "There is no rush. Set up your space and understand the movement first."
                )
                return
            self.calibration_armed = False
            self.state = SessionState.CALIBRATING
            self.calibration_samples = []
            self.calibration_rest_samples = []
            self.calibration_posture_samples = []
            self.calibration_started = time.monotonic()
            self.calibration_valid_elapsed = 0.0
            self.calibration_last_valid_at = None
            self.progress_history = {"Left": [], "Right": []}
            self.primary.setEnabled(False)
            self.primary.setText("Calibrating…")
            self.status.setText(
                "Calibration: hold your comfortable starting pose, then follow the movement prompt"
            )
            self.stage.setText("✓  PROFILE   →   ●  CALIBRATE   →   3  MOVE   →   4  PROGRESS")
            return
        if self.state in {SessionState.READY, SessionState.PAUSED, SessionState.COMPLETE}:
            if self.state in {SessionState.READY, SessionState.COMPLETE}:
                self.frames = []
                self.progress_history = {"Left": [], "Right": []}
                self.previous_progress = None
                self.posture_history = []
                self.started_at = datetime.now(UTC)
                self.save_button.setEnabled(False)
            self.state = SessionState.ACTIVE
            if self.coach is None or self.coach.phase == CoachPhase.COMPLETE:
                exercise = EXERCISES[self.exercise.currentIndex()]
                self.coach = SessionCoach(exercise.repetitions)
                self._apply_coach_update(self.coach.start(time.monotonic()))
            else:
                self.coach.resume(time.monotonic())
            self.primary.setText("Pause session")
            self.stage.setText("✓  PROFILE   →   ✓  CALIBRATE   →   ●  MOVE   →   4  PROGRESS")
        else:
            self.state = SessionState.PAUSED
            if self.coach is not None:
                self.coach.pause(time.monotonic())
            self.primary.setText("Resume slowly")
            self.status.setText("Session paused safely. Resume only when you are ready.")
            self.canvas.phase_text = "PAUSED"

    def _update_profile(self) -> None:
        dialog = IntakeDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.intake = dialog.intake()
        self.repo.save_profile(self.patient_id, asdict(self.intake))
        self.baselines.clear()
        self.coach = None
        self.calibration_armed = False
        self.repo.clear_baselines(self.patient_id)
        self.profile_summary.setText(
            f"Age {self.intake.age_years} • {self.intake.body_area} • {self.intake.affected_side} side • "
            f"{self.intake.mobility_level} mobility • discomfort {self.intake.discomfort}/10"
        )
        self.state = SessionState.READY
        self.save_button.setEnabled(False)
        self.primary.setText("Start calibration")
        self.status.setText(f"{readiness_message(self.intake)} • previous baselines cleared")
        self.stage.setText("✓  PROFILE   →   ●  CALIBRATE   →   3  MOVE   →   4  PROGRESS")

    def _exercise_changed(self) -> None:
        self.frames = []
        self.progress_history = {"Left": [], "Right": []}
        self.coach = None
        self.calibration_armed = False
        self.state = SessionState.READY
        self.primary.setText("Start session")
        self.save_button.setEnabled(False)
        self._refresh_guidance()
        exercise_id = EXERCISES[self.exercise.currentIndex()].id
        if exercise_id in self.baselines:
            baseline = self.baselines[exercise_id]
            self.status.setText(
                f"Personal baseline ready • comfortable ROM {baseline.comfortable_rom_deg:.0f}°"
            )
        else:
            self.status.setText("Exercise changed • a short personal calibration is required")

    def _safe_tick(self) -> None:
        try:
            self._tick()
        except Exception as error:  # noqa: BLE001 - Qt callbacks need a crash boundary.
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with (DATA_DIR / "runtime-error.log").open("a", encoding="utf-8") as log:
                log.write(f"{datetime.now(UTC).isoformat()} {type(error).__name__}: {error}\n")
            self.state = SessionState.PAUSED
            self.canvas.good = False
            self.canvas.feedback = "Movement tracking had a problem. Your session is safely paused."
            self.canvas.phase_text = "PAUSED — RETRY"
            self.status.setText("Tracking paused safely. Press Resume slowly to try again.")
            self.primary.setEnabled(True)
            self.primary.setText("Resume slowly")
            self.canvas.update()

    def _tick(self) -> None:
        self.phase += 0.12
        landmarks: np.ndarray | None = None
        confidence = 0.0
        if self.camera is not None and self.camera.available:
            try:
                rgb, landmarks, confidence = self.camera.read()
                height, width, channels = rgb.shape
                self.canvas.frame = QImage(
                    rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
                ).copy()
                self.canvas.visibility = self.camera.last_visibility.copy()
                self.camera_error = None
            except (OSError, RuntimeError, ValueError) as error:
                self.camera_error = str(error)
        if landmarks is None:
            now = time.monotonic()
            if self.tracking_lost_at is None:
                self.tracking_lost_at = now
            missing_for = now - self.tracking_lost_at
            self.calibration_last_valid_at = None
            if self.camera_error:
                self.canvas.feedback = f"Camera unavailable: {self.camera_error}"
                self.canvas.update()
            elif self.canvas.frame is not None:
                self.canvas.feedback = "Hold still for a moment while I find your joints"
                self.canvas.update()
            if missing_for < 1.0:
                return
            self.canvas.landmarks = None
            self.canvas.visibility = None
            if self.state not in {SessionState.ACTIVE, SessionState.CALIBRATING}:
                return
            exercise = EXERCISES[self.exercise.currentIndex()]
            self.canvas.feedback = self._framing_cue(exercise)
            self.canvas.phase_text = "PAUSED — FINDING BODY"
            if self.coach is not None:
                self.coach.pause(time.monotonic())
            self.canvas.update()
            return
        else:
            self.canvas.landmarks = landmarks
            exercise = EXERCISES[self.exercise.currentIndex()]
            left_visibility = self._triplet_visibility(exercise.joint_triplet)
            right_visibility = self._triplet_visibility(
                tuple(index + 1 for index in exercise.joint_triplet)
            )
            if max(left_visibility, right_visibility) < 0.25:
                self.canvas.feedback = (
                    self._framing_cue(exercise)
                )
            else:
                self.canvas.feedback = (
                    "Required joints detected — press Start session"
                    if self.state not in {SessionState.ACTIVE, SessionState.CALIBRATING}
                    else "Tracking your movement"
                )
            self.canvas.update()
        if self.state not in {SessionState.ACTIVE, SessionState.CALIBRATING}:
            return
        exercise = EXERCISES[self.exercise.currentIndex()]
        geometry = (
            self.camera.last_world_landmarks
            if self.camera is not None and self.camera.last_world_landmarks is not None
            else landmarks
        )
        left_angle = extract_joint_angle(geometry, exercise.joint_triplet)
        right_triplet = tuple(index + 1 for index in exercise.joint_triplet)
        right_angle = extract_joint_angle(geometry, right_triplet)
        signals = {
            "Left": movement_signal(left_angle, exercise.movement_mode),
            "Right": movement_signal(right_angle, exercise.movement_mode),
        }
        for side, signal_value in signals.items():
            self.progress_history[side] = [*self.progress_history[side][-119:], signal_value]
        if self.intake is not None and self.intake.affected_side != "Either":
            self.active_side = self.intake.affected_side
        else:
            self.active_side = choose_active_side(self.progress_history)
        tracked_triplet = (
            exercise.joint_triplet
            if self.active_side == "Left"
            else tuple(index + 1 for index in exercise.joint_triplet)
        )
        self.canvas.focus_indices = tracked_triplet
        tracked_confidence = self._triplet_visibility(tracked_triplet)
        if tracked_confidence < 0.25:
            now = time.monotonic()
            if self.tracking_lost_at is None:
                self.tracking_lost_at = now
            if now - self.tracking_lost_at < 0.8:
                self.canvas.feedback = "Keep moving slowly — tracking is stabilizing"
                self.canvas.update()
                return
            self.canvas.good = False
            self.canvas.feedback = self._framing_cue(exercise)
            self.status.setText("Measurement paused until all required joints are visible")
            self.canvas.phase_text = "PAUSED — REPOSITION"
            if self.coach is not None:
                self.coach.pause(time.monotonic())
            self.canvas.update()
            return
        self.tracking_lost_at = None
        angle = left_angle if self.active_side == "Left" else right_angle
        signal_value = signals[self.active_side]
        if self.state == SessionState.CALIBRATING:
            now = time.monotonic()
            if self.calibration_last_valid_at is not None:
                self.calibration_valid_elapsed += min(0.12, now - self.calibration_last_valid_at)
            self.calibration_last_valid_at = now
            elapsed = self.calibration_valid_elapsed
            if elapsed <= 1.0:
                self.calibration_rest_samples.append(signal_value)
                self.calibration_posture_samples.append(torso_lean_degrees(geometry))
                phase_message = "Hold your comfortable starting pose"
            else:
                self.calibration_samples.append(signal_value)
                phase_message = "Move comfortably, then return"
            rest_value = float(np.median(self.calibration_rest_samples)) if self.calibration_rest_samples else signal_value
            posture_reference = (
                float(np.median(self.calibration_posture_samples))
                if self.calibration_posture_samples
                else torso_lean_degrees(geometry)
            )
            calibration_peak = max(
                (abs(sample - rest_value) for sample in self.calibration_samples), default=0.0
            )
            seconds_left = max(0.0, 5.0 - elapsed)
            self.canvas.good = confidence >= 0.5
            self.canvas.feedback = (
                f"{phase_message} • range {calibration_peak:.0f}° • {seconds_left:.1f}s"
            )
            self.canvas.phase_text = "CALIBRATION"
            self.canvas.rep_text = "—"
            self.angle_card.value.setText(f"{angle:.0f}°")
            self.progress.setValue(min(100, round(elapsed / 5.0 * 100)))
            self.canvas.update()
            if elapsed >= 5.0:
                self._finish_calibration(
                    exercise, calibration_peak, rest_value, posture_reference
                )
            return
        baseline = self.baselines[exercise.id]
        history = [
            abs(value - baseline.rest_value_deg)
            for value in self.progress_history[self.active_side]
        ]
        smoothed_history = smooth_angles(history) if history else np.asarray([], dtype=float)
        movement_progress = float(smoothed_history[-1]) if smoothed_history.size else 0.0
        achieved = float(np.max(smoothed_history)) if smoothed_history.size else 0.0
        self.previous_progress = movement_progress
        target_rom = baseline.target_rom_deg
        assessment = assess_live_movement(
            current_progress_deg=movement_progress,
            achieved_rom_deg=achieved,
            target_rom_deg=target_rom,
            confidence=tracked_confidence,
            continue_cue=exercise.cue_low,
        )
        live_score = assessment.score
        self.canvas.landmarks = landmarks
        self.canvas.feedback = f"{self.active_side} side • {assessment.current_ratio:.0%} of your target"
        assert self.coach is not None
        raw_posture_ok = self._posture_ok(geometry, exercise, baseline)
        self.posture_history = [*self.posture_history[-11:], raw_posture_ok]
        posture_ok = sum(self.posture_history) >= max(1, round(len(self.posture_history) * 0.75))
        coach_update = self.coach.update(
            progress_ratio=assessment.current_ratio,
            visible=True,
            posture_ok=posture_ok,
            now=time.monotonic(),
        )
        self.canvas.good = posture_ok
        self._apply_coach_update(coach_update)
        self.canvas.update()
        self.score_card.value.setText(f"{live_score:.0f}")
        self.angle_card.value.setText(f"{angle:.0f}°")
        self.rep_card.value.setText(
            f"{coach_update.repetitions} / {coach_update.target_repetitions}"
        )
        self.progress.setValue(round(live_score))
        self.frames.append({
            "timestamp": datetime.now(UTC).isoformat(), "frame_id": len(self.frames),
            "angle_deg": round(angle, 3), "progress_deg": round(movement_progress, 3),
            "score": round(live_score, 3), "confidence": round(confidence, 3),
        })
        if coach_update.phase == CoachPhase.COMPLETE:
            self.state = SessionState.COMPLETE
            self.primary.setText("Start another")
            self.save_button.setEnabled(True)
            self.status.setText(coach_update.instruction)

    def _finish_calibration(
        self,
        exercise,
        comfortable_rom: float,
        rest_value: float | None = None,
        posture_reference: float | None = None,
    ) -> None:
        assert self.intake is not None
        try:
            target = personalized_target(comfortable_rom, self.intake)
        except ValueError:
            self.state = SessionState.READY
            self.primary.setEnabled(True)
            self.primary.setText("Repeat calibration")
            self.status.setText(
                "Calibration did not capture enough movement. Keep the required joints visible and try again."
            )
            return
        baseline = PersonalBaseline(
            exercise.id, self.active_side, comfortable_rom, target,
            rest_value if rest_value is not None else 0.0,
            posture_reference if posture_reference is not None else 0.0,
            datetime.now(UTC).isoformat(),
        )
        self.baselines[exercise.id] = baseline
        self.repo.save_baseline(self.patient_id, exercise.id, asdict(baseline))
        self.state = SessionState.READY
        self.primary.setEnabled(True)
        self.primary.setText("Start personalized session")
        self.progress.setValue(0)
        self.status.setText(
            f"Baseline saved: comfortable ROM {comfortable_rom:.0f}° • session target {target:.0f}°"
        )
        self.stage.setText("✓  PROFILE   →   ✓  CALIBRATE   →   3  MOVE   →   4  PROGRESS")
        self.canvas.phase_text = "READY FOR SESSION"
        self.canvas.rep_text = f"0 / {exercise.repetitions}"

    def _triplet_visibility(self, triplet: tuple[int, int, int]) -> float:
        if self.camera is None:
            return 0.0
        return float(np.min(self.camera.last_visibility[np.asarray(triplet)]))

    @staticmethod
    def _framing_cue(exercise) -> str:
        if exercise.body_region in {"Shoulder", "Elbow"}:
            return "Show your working hip, shoulder and arm. You do not need to show your legs."
        if exercise.body_region in {"Hip", "Knee", "Ankle", "Full body"}:
            return "Step back until your working hip, knee and foot are visible."
        return "Keep your shoulders and hips visible in the camera."

    def _refresh_guidance(self) -> None:
        exercise = EXERCISES[self.exercise.currentIndex()]
        guide = guidance_for(exercise.id)
        self.guidance.setText(
            f"START  {guide.setup}\n\nMOVE  {guide.movement}\n\nFORM  {guide.posture}"
        )

    @staticmethod
    def _posture_ok(
        landmarks: np.ndarray, exercise, baseline: PersonalBaseline
    ) -> bool:
        if exercise.id == "trunk-side-bend":
            return True
        limit = 35 if exercise.id == "sit-to-stand" else 20
        return (
            abs(torso_lean_degrees(landmarks) - baseline.posture_reference_deg) <= limit
        )

    @staticmethod
    def _motivation_for_phase(phase: CoachPhase) -> str:
        return {
            CoachPhase.PREPARE: "Take your time. A calm start leads to better movement.",
            CoachPhase.MOVE_OUT: "You can do it — move slowly and never force through pain.",
            CoachPhase.HOLD: "Excellent control. Breathe normally and stay relaxed.",
            CoachPhase.RETURN: "Smooth on the way back — control matters in both directions.",
            CoachPhase.REST: "Well done. Relax, breathe, and let the muscles reset.",
            CoachPhase.COMPLETE: "Wonderful work! Consistency is how progress grows.",
            CoachPhase.IDLE: "Every careful movement counts. You can do this.",
        }[phase]

    def _apply_coach_update(self, update: CoachUpdate) -> None:
        phase_names = {
            CoachPhase.PREPARE: "GET READY",
            CoachPhase.MOVE_OUT: "MOVE SLOWLY",
            CoachPhase.HOLD: "HOLD GENTLY",
            CoachPhase.RETURN: "RETURN SLOWLY",
            CoachPhase.REST: "REST & BREATHE",
            CoachPhase.COMPLETE: "SESSION COMPLETE",
            CoachPhase.IDLE: "READY",
        }
        exercise = EXERCISES[self.exercise.currentIndex()]
        if update.phase == CoachPhase.MOVE_OUT:
            instruction = (
                f"{self.active_side.upper()} SIDE — "
                f"{exercise.cue_low.replace('your arm', f'your {self.active_side.lower()} arm')}. "
                "Move slowly and stop before pain."
            )
        elif update.phase == CoachPhase.HOLD:
            instruction = "That is enough. Hold gently and keep breathing."
        elif update.phase == CoachPhase.RETURN:
            instruction = "Now return slowly to your starting position to complete the repetition."
        else:
            instruction = update.instruction
        self.status.setText(instruction)
        self.motivation.setText(self._motivation_for_phase(update.phase))
        self.canvas.phase_text = phase_names[update.phase]
        self.canvas.rep_text = f"{update.repetitions} / {update.target_repetitions}"

    def _payload(self) -> dict[str, object]:
        scores = [float(frame["score"]) for frame in self.frames]
        progress_values = [float(frame.get("progress_deg", 0)) for frame in self.frames]
        exercise_id = EXERCISES[self.exercise.currentIndex()].id
        baseline = self.baselines.get(exercise_id)
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": "provisional-0.1",
            "exercise_id": exercise_id,
            "started_at": self.started_at.isoformat(),
            "score": round(float(np.mean(scores)), 2) if scores else 0.0,
            "achieved_rom_deg": round(max(progress_values), 2) if progress_values else 0.0,
            "repetitions": self.coach.repetitions if self.coach else 0,
            "patient_id": self.patient_id,
            "validation_status": "investigational",
            "personalization": {
                "age_years": self.intake.age_years if self.intake else None,
                "body_area": self.intake.body_area if self.intake else None,
                "affected_side": self.intake.affected_side if self.intake else None,
                "mobility_level": self.intake.mobility_level if self.intake else None,
                "goal": self.intake.goal if self.intake else None,
                "discomfort": self.intake.discomfort if self.intake else None,
                "comfortable_rom_deg": baseline.comfortable_rom_deg if baseline else None,
                "target_rom_deg": baseline.target_rom_deg if baseline else None,
                "rest_value_deg": baseline.rest_value_deg if baseline else None,
            },
            "frames": self.frames,
        }

    def _save(self) -> None:
        if self.state != SessionState.COMPLETE:
            self.status.setText("Complete the guided repetitions before saving the session summary.")
            return
        payload = self._payload()
        session_id = self.repo.save_session(self.patient_id, payload)
        payload["session_id"] = session_id
        export_session(payload, DATA_DIR / "exports")
        summary = summarize_progress(self.repo.list_sessions(), str(payload["exercise_id"]))
        if summary and summary.sessions > 1:
            direction = "improved" if summary.score_change >= 0 else "changed"
            message = (
                f"Saved • {direction} {abs(summary.score_change):.1f} score points and "
                f"{summary.rom_change:+.1f}° ROM since Day 1"
            )
        else:
            message = "Day-1 session saved — this is your personal progress reference"
        self.status.setText(message)
        self.motivation.setText("Great work showing up today. Recovery is built one session at a time.")
        self.save_button.setEnabled(False)
        self.stage.setText("✓  PROFILE   →   ✓  CALIBRATE   →   ✓  MOVE   →   ●  PROGRESS")

    def _refresh_sessions(self) -> None:
        rows = self.repo.list_sessions()
        first_scores: dict[str, float] = {}
        for historical in reversed(rows):
            exercise_id = str(historical.get("exercise_id", ""))
            first_scores.setdefault(exercise_id, float(historical.get("score", 0)))
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            personalization = row.get("personalization", {})
            target = personalization.get("target_rom_deg") if isinstance(personalization, dict) else None
            values = [
                str(row.get("started_at", ""))[:10],
                str(row.get("patient_id", "")),
                str(row.get("exercise_id", "")),
                f"{float(target):.0f}°" if target is not None else "—",
                f"{float(row.get('score', 0)):.0f}",
                f"{float(row.get('score', 0)) - first_scores.get(str(row.get('exercise_id', '')), 0):+.1f}",
                str(row.get("repetitions", 0)),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))

    def closeEvent(self, event) -> None:
        if self.camera is not None:
            self.camera.close()
        super().closeEvent(event)


STYLES = """
QWidget { background:#07101d; color:#e5edf8; font-family:'Segoe UI'; font-size:16px; }
QLabel#brand { color:#7dd3fc; font-size:24px; font-weight:800; letter-spacing:2px; }
QLabel#sectionTitle { color:#f8fafc; font-size:25px; font-weight:800; padding-top:4px; }
QLabel#stage { color:#93c5fd; background:#0c1d31; border:1px solid #234765; border-radius:10px; padding:10px; font-size:12px; font-weight:700; }
QLabel#warning { color:#fdba74; background:#3a2416; border:1px solid #9a5c28; border-radius:8px; padding:8px 14px; font-size:12px; font-weight:700; }
QLabel#muted { color:#8fa3ba; font-size:13px; }
QLabel#status { background:#10233a; border-left:4px solid #38bdf8; padding:14px; border-radius:5px; }
QLabel#motivation { color:#d1fae5; background:#0b2e2a; border:1px solid #176b5b; border-radius:10px; padding:14px; font-size:17px; font-weight:700; }
QLabel#guidance { color:#dbeafe; background:#0d2037; border:1px solid #294968; border-radius:10px; padding:12px; font-size:13px; }
QFrame#card { background:#0e1d30; border:1px solid #203a57; border-radius:12px; padding:8px; }
QPushButton { background:#172a43; border:1px solid #2a4765; border-radius:8px; padding:11px 16px; font-weight:600; }
QPushButton:hover { background:#203b5c; }
QPushButton#primary { background:#0284c7; border:none; color:white; }
QComboBox { background:#0e1d30; border:1px solid #284a6a; border-radius:8px; padding:8px; }
QProgressBar { background:#112238; border:none; border-radius:7px; min-height:14px; text-align:center; }
QProgressBar::chunk { background:#38bdf8; border-radius:7px; }
QTableWidget { background:#0b1727; alternate-background-color:#0f2035; gridline-color:#1d3650; border:1px solid #203a57; }
QHeaderView::section { background:#13263d; color:#b8c7d9; padding:10px; border:none; }
"""


def run() -> None:
    application = QApplication(sys.argv)
    application.setApplicationName("NeuroFlex")
    window = MainWindow()
    window.show()
    raise SystemExit(application.exec())
