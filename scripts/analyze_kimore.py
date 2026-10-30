from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.stats import spearmanr

EXERCISE_NAMES = {
    1: "Arm lifting",
    2: "Lateral trunk tilt",
    3: "Trunk rotation",
    4: "Pelvis rotation",
    5: "Squatting",
}
EXERCISE_PATTERN = re.compile(r"Es([1-5])$")


def load_positions(path: Path) -> np.ndarray:
    raw = np.genfromtxt(path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] < 100:
        raise ValueError(f"Unexpected KIMORE joint-position shape {raw.shape}: {path}")
    # Kinect v2 stores x/y/z/tracking-state for each of 25 joints plus an empty final column.
    return raw[:, :100].reshape(len(raw), 25, 4)[:, :, :3]


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    left, right = a - b, c - b
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(
        np.sum(left * right, axis=1), denominator, out=np.ones_like(denominator), where=denominator > 1e-9
    )
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def movement_proxy(positions: np.ndarray, exercise: int) -> float:
    spine_base, spine_shoulder = positions[:, 0], positions[:, 20]
    torso = np.median(np.linalg.norm(spine_shoulder - spine_base, axis=1))
    torso = max(float(torso), 1e-6)
    if exercise == 1:  # bilateral wrist elevation relative to shoulder
        wrist_height = positions[:, [6, 10], 1].mean(axis=1) - spine_shoulder[:, 1]
        return float(np.ptp(wrist_height) / torso)
    if exercise == 2:  # lateral displacement of shoulder center relative to pelvis
        lateral = (spine_shoulder[:, 0] - spine_base[:, 0]) / torso
        return float(np.ptp(lateral))
    if exercise == 3:  # shoulder-line rotation in the transverse plane
        shoulder_line = positions[:, 8] - positions[:, 4]
        orientation = np.unwrap(np.arctan2(shoulder_line[:, 2], shoulder_line[:, 0]))
        return float(np.degrees(np.ptp(orientation)))
    if exercise == 4:  # pelvis-line rotation in the transverse plane
        hip_line = positions[:, 16] - positions[:, 12]
        orientation = np.unwrap(np.arctan2(hip_line[:, 2], hip_line[:, 0]))
        return float(np.degrees(np.ptp(orientation)))
    knees = np.column_stack(
        (angle(positions[:, 12], positions[:, 13], positions[:, 14]),
         angle(positions[:, 16], positions[:, 17], positions[:, 18]))
    )
    return float(180 - np.nanmin(knees.mean(axis=1)))


def clinical_score(path: Path, exercise: int) -> float:
    sheet = load_workbook(path, data_only=True, read_only=True).active
    rows = iter(sheet.values)
    headers = [str(value) for value in next(rows)]
    values = list(next(rows))
    suffix = f"Ex#{exercise}"
    selected = [values[index] for index, header in enumerate(headers) if header.endswith(suffix)]
    if len(selected) != 3 or any(value is None for value in selected):
        raise ValueError(f"Expected TS, PO and CF clinical values for exercise {exercise}: {path}")
    return float(sum(float(value) for value in selected))  # components sum to a 0–100 score


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and analyze locally acquired KIMORE data.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, failures = [], []
    for position_path in sorted(args.root.rglob("JointPosition*.csv")):
        exercise_dir = next((parent for parent in position_path.parents if EXERCISE_PATTERN.match(parent.name)), None)
        if exercise_dir is None:
            continue
        exercise = int(EXERCISE_PATTERN.match(exercise_dir.name).group(1))  # type: ignore[union-attr]
        label_files = list((exercise_dir / "Label").glob("ClinicalAssessment*.xlsx"))
        try:
            if not label_files:
                raise FileNotFoundError("clinical assessment missing")
            positions = load_positions(position_path)
            records.append(
                {
                    "group": position_path.relative_to(args.root).parts[0],
                    "cohort": position_path.relative_to(args.root).parts[1],
                    "subject": position_path.relative_to(args.root).parts[2],
                    "exercise": exercise,
                    "exercise_name": EXERCISE_NAMES[exercise],
                    "frames": len(positions),
                    "movement_proxy": movement_proxy(positions, exercise),
                    "clinical_score": clinical_score(label_files[0], exercise),
                }
            )
        except (OSError, ValueError, StopIteration) as error:
            failures.append({"file": str(position_path), "error": str(error)})

    exercise_metrics = {}
    for exercise, name in EXERCISE_NAMES.items():
        selected = [row for row in records if row["exercise"] == exercise]
        if len(selected) >= 3:
            correlation = spearmanr(
                [row["movement_proxy"] for row in selected],
                [row["clinical_score"] for row in selected],
            )
            rho, p_value = float(correlation.statistic), float(correlation.pvalue)
        else:
            rho, p_value = None, None
        exercise_metrics[str(exercise)] = {
            "name": name,
            "sessions": len(selected),
            "spearman_rho": rho,
            "p_value": p_value,
        }
    local_files = [path for path in args.root.rglob("*") if path.is_file() and not path.name.endswith(".part")]
    payload = {
        "acquisition": {
            "manifest_files": len(manifest),
            "local_files": len(local_files),
            "complete_pct": 100 * len(local_files) / len(manifest),
        },
        "records_analyzed": len(records),
        "failures": failures,
        "exercise_metrics": exercise_metrics,
        "records": records,
        "warning": "Movement proxies are engineering baselines, not clinical scores or trained models.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
