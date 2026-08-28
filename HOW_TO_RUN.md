# How to run NeuroFlex

## Fastest option

From the repository folder, right-click `RUN_NEUROFLEX.ps1` and choose **Run with PowerShell**, or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\RUN_NEUROFLEX.ps1
```

Allow Windows camera access when prompted. Stand far enough back for all joints used by the selected exercise to remain visible. Your image is processed locally and is not recorded.

The joints used by the selected exercise are highlighted in yellow. The current phase and a large
`REPS completed / target` counter remain over the camera view. Older saved baselines are
recalibrated once so posture can be judged
relative to the user's natural starting pose instead of the camera placement.

After calibration, optionally enable **touch-free gesture controls**. Bring both hands together
over the center of your chest and hold for about 1.2 seconds to start, pause, or resume. Separate
your hands after each accepted command before using the gesture again. Gestures do not operate
during setup or calibration, and the normal buttons always remain available.

## First-time setup

If dependencies need to be restored:

```powershell
py -3.11 -m pip install --user uv
py -3.11 -m uv sync --dev --extra pose
.\.venv\Scripts\python.exe scripts\check_system.py
.\.venv\Scripts\python.exe -m neuroflex
```

In the app, select an exercise and press **Start session**. Complete the short intake, including exact age and the body area to work on. Read the visible START, MOVE, and FORM instructions before pressing **Begin calibration**. During the five-second calibration, hold the starting pose for one second, perform one slow comfortable movement, and return.

After the baseline is saved, press **Start personalized session**. NeuroFlex provides a five-second preparation, guides one direction at a time, asks for a gentle hold, guides the return, validates the repetition, and inserts a two-second rest. Instructions do not change until the required phase is completed. If posture or camera visibility is unsuitable, the session pauses instead of counting movement. **Save summary** becomes available only after all repetitions are complete.

The first saved session becomes the Day-1 performance reference; later saves report score and ROM changes from that personal reference. Use **Clinician view** to inspect the trend. Local data is placed in `%USERPROFILE%\.neuroflex`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

For GPU validation, upload `colab/NeuroFlex_CUDA_Validation.ipynb` to Google Colab, choose **Runtime → Change runtime type → GPU**, and run the cells in order.
