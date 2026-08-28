# NeuroFlex

NeuroFlex is an investigational, touchless physical-rehabilitation assessment prototype. It combines a live/synthetic pose pipeline, deterministic kinematics, personalized scoring, gesture-ready controls, local clinical records, and research exports.

> **Safety:** This build is not clinically validated and must not be used for diagnosis, treatment decisions, or unsupervised patient care.

## Run locally

```powershell
py -3.11 -m uv sync --dev --extra pose
.\.venv\Scripts\python.exe -m neuroflex
```

The project-local virtual environment is `.venv`. The Windows demo uses the included MediaPipe model for a live 33-landmark webcam overlay. A deterministic synthetic provider supports tests, while the CUDA validation path remains replaceable behind the pose boundary.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

Business tests cover vectorized joint angles, scoring bounds, DTW determinism, repetition state, gesture debounce/cooldown, and transactional persistence/export. GPU and clinical dataset validation are separate non-blocking gates.

## Colab

Open `colab/NeuroFlex_CUDA_Validation.ipynb` in a GPU runtime to verify PyTorch CUDA availability, run the suite, and benchmark vectorized angle computation.

## Local data

The desktop prototype stores pseudonymous records beneath `%USERPROFILE%/.neuroflex`. Raw webcam video is never stored. SQLite encryption and key lifecycle remain a release gate; this display build must not contain real patient data.
