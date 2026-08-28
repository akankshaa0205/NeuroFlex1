# What was built

NeuroFlex now has a reviewer-ready Windows desktop prototype rather than only a PRD.

## Demonstrable product

- A polished PySide6 desktop application with separate patient-session and clinician-review views.
- Live 1280×720 webcam video with a MediaPipe 33-landmark body skeleton drawn over the patient.
- Real-time joint-angle display, movement score, repetition count, progress bar, corrective status text, pause/resume, exercise selection, and session completion.
- MediaPipe's Full pose model, a high-contrast outlined skeleton, yellow highlighting for the three
  joints currently being measured, and a persistent camera-overlay repetition/phase display.
- Ten configured movements spanning shoulder, elbow, hip, knee, ankle, trunk, and full-body sit-to-stand work.
- A pre-session intake covering exact age, body area, affected side, mobility, goal, and current discomfort.
- A five-second rest-plus-movement calibration that measures the patient’s comfortable range relative to their actual starting pose. The target combines that measured capability with mobility, discomfort, and a modest age-aware effort factor, so an 80-year-old is not judged against the same absolute expectation as a healthy 20-year-old.
- Day-1 session performance is retained as the patient’s own reference. Later sessions show score and ROM changes from Day 1 in both the save confirmation and clinician history.
- A four-stage patient journey, large motivational card, progressive encouragement, safe visibility/form gating, and positive completion messages.
- A deterministic slow-session coach: five-second preparation, target dwell, gentle hold, controlled return, validated repetition, and two-second rest. Instructions remain stable within each phase instead of reacting to every camera frame.
- Exercise-specific START, MOVE, and FORM instructions for every supported movement, posture checks
  relative to the user's calibrated starting pose, and automatic session pause when required joints
  leave the frame.
- A prominent investigational warning so the display is not confused with a clinically validated release.
- Pseudonymous local session records, clinician history, SQLite storage, and atomic versioned JSON/CSV research exports. No webcam video is saved.

## Engineering foundation

- A reproducible `uv` project and repository-local `.venv`.
- Pure modules for vectorized 3D joint angles, Savitzky–Golay smoothing, exact reference DTW, bounded composite scoring, repetition detection, and gesture dwell/cooldown.
- Replaceable pose-provider boundary, a real MediaPipe camera adapter, deterministic synthetic fallback, and CUDA-oriented Colab validation notebook.
- Focused pytest/Hypothesis tests for clinical business logic, persistence, and full timed session-state simulations, plus Ruff static checks.
- A system-readiness checker and one-command PowerShell launcher.

## Honest remaining validation gates

This is a strong display prototype, not a medical-device release. The age factors and exercise cues are provisional configuration values requiring therapist validation. KIMORE/UI-PRMD accuracy benchmarking, impaired-user gesture validation with HaGRID, GTX-class latency profiling, encrypted-database key management, and clinical/regulatory review still require the corresponding hardware, datasets, and clinical oversight.
