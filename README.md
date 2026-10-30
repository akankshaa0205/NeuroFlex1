# NeuroFlex

NeuroFlex is an investigational, touchless physical-rehabilitation assessment prototype. It combines a live pose pipeline, deterministic kinematics, personalized scoring, trained gesture controls, local progress records, and reproducible validation tools.

> **Safety:** This build is not clinically validated and must not be used for diagnosis, treatment decisions, or unsupervised patient care.

## Run locally

```powershell
py -3.11 -m uv sync --dev --extra pose
.\.venv\Scripts\python.exe -m neuroflex
```

The project-local virtual environment is `.venv`. The Windows demo uses the included MediaPipe model for a live 33-landmark webcam overlay. Camera inference runs behind a bounded latest-frame worker so stale frames are discarded instead of delaying feedback. Five core exercises are presented first; five additional exercises remain clearly marked experimental.

The runtime uses CPU automatically on ordinary laptops. Installing the `cuda` extra enables the PyTorch CUDA path for batched kinematics when a compatible device is available. A Tesla T4 Colab benchmark verified CUDA execution and numerical compatibility: CPU remained preferable for small live-frame batches, while a 10,000-item batch was 2.48× faster end to end on the GPU. The UI reports the device actually selected rather than claiming inactive acceleration.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

Business tests cover vectorized joint angles, CPU fallback, bounded asynchronous processing, scoring bounds, DTW determinism, repetition state, gesture debounce/cooldown, KIMORE feature compatibility, UI flow, and transactional persistence/export. Prospective clinical validation and production-target hardware profiling remain explicit release gates.

## Colab

Open `colab/NeuroFlex_CUDA_Validation.ipynb` in a GPU runtime to verify PyTorch CUDA availability, run the suite, and benchmark vectorized angle computation.

## Local data

The desktop prototype stores pseudonymous records beneath `%USERPROFILE%/.neuroflex`. Raw webcam video is never stored. SQLite encryption and key lifecycle remain a release gate; this display build must not contain real patient data.

## KIMORE quality model

The compact standardized KIMORE derivative now trains a recording-level clinical-quality
regressor. Subject-grouped cross-validation and a held-out subject test prevent identity
leakage. The packaged model appears in the UI as **KIMORE quality** for compatible exercises.

This is an experimental cross-sensor estimate: KIMORE uses Kinect-derived angles while the
live application uses MediaPipe landmarks. It complements the personalized NeuroFlex score;
it is not a diagnosis or replacement for clinician assessment.

```powershell
python scripts/train_kimore_quality.py --frames data/raw/kimore_standardized/kimore_frames_standardized.csv --scores data/raw/kimore_standardized/kimore_scores_standardized.csv --model artifacts/kimore_quality.joblib --metrics data/processed/kimore/quality_metrics_20261029.json
```
