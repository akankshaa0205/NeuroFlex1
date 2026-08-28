"""Print an honest NeuroFlex runtime readiness report."""

from __future__ import annotations

import platform
from pathlib import Path

import cv2


def yes_no(value: bool) -> str:
    return "READY" if value else "NOT AVAILABLE"


print("NeuroFlex system check")
print(f"Platform: {platform.platform()}")
print(f"Python: {platform.python_version()}")
model = Path("models/pose_landmarker_full.task")
print(f"Pose model: {yes_no(model.exists())} ({model})")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"Camera: {yes_no(camera.isOpened())}")
camera.release()
try:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {yes_no(torch.cuda.is_available())}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch/CUDA: optional extra not installed locally")
print("Clinical validation: NOT COMPLETE (display prototype only)")
