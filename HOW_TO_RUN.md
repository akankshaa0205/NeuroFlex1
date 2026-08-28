# How to run NeuroFlex

## Fastest option

From the repository folder, right-click `RUN_NEUROFLEX.ps1` and choose **Run with PowerShell**, or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\RUN_NEUROFLEX.ps1
```

Allow Windows camera access when prompted. Stand far enough back for all joints used by the selected exercise to remain visible. Your image is processed locally and is not recorded.

## First-time setup

If dependencies need to be restored:

```powershell
python -m pip install --user uv
python -m uv sync --dev --extra pose
python -m uv run python scripts/check_system.py
python -m uv run neuroflex
```

In the app, select an exercise and press **Start session**. Complete the short intake, including exact age and the body area to work on. Read the visible START, MOVE, and FORM instructions before pressing **Begin calibration**. During the five-second calibration, hold the starting pose for one second, perform one slow comfortable movement, and return.

After the baseline is saved, press **Start personalized session**. NeuroFlex provides a five-second preparation, guides one direction at a time, asks for a gentle hold, guides the return, validates the repetition, and inserts a two-second rest. Instructions do not change until the required phase is completed. If posture or camera visibility is unsuitable, the session pauses instead of counting movement. **Save summary** becomes available only after all repetitions are complete.

The first saved session becomes the Day-1 performance reference; later saves report score and ROM changes from that personal reference. Use **Clinician view** to inspect the trend. Local data is placed in `%USERPROFILE%\.neuroflex`.

## Verification

```powershell
python -m uv run pytest
python -m uv run ruff check .
```

For GPU validation, upload `colab/NeuroFlex_CUDA_Validation.ipynb` to Google Colab, choose **Runtime → Change runtime type → GPU**, and run the cells in order.
