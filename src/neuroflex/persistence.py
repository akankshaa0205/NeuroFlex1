from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

SCHEMA_VERSION = "1.0"


class SessionRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY, alias TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, exercise_id TEXT NOT NULL,
                    started_at TEXT NOT NULL, score REAL NOT NULL, repetitions INTEGER NOT NULL,
                    algorithm_version TEXT NOT NULL, payload TEXT NOT NULL,
                    FOREIGN KEY(patient_id) REFERENCES patients(id)
                );
                CREATE TABLE IF NOT EXISTS patient_profiles (
                    patient_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    FOREIGN KEY(patient_id) REFERENCES patients(id)
                );
                CREATE TABLE IF NOT EXISTS baselines (
                    patient_id TEXT NOT NULL, exercise_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(patient_id, exercise_id),
                    FOREIGN KEY(patient_id) REFERENCES patients(id)
                );
                """
            )

    def ensure_demo_patient(self) -> str:
        patient_id = "demo-patient"
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO patients VALUES (?, ?, ?)",
                (patient_id, "Demo Patient", datetime.now(UTC).isoformat()),
            )
        return patient_id

    def save_session(self, patient_id: str, payload: dict[str, object]) -> str:
        session_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    patient_id,
                    payload["exercise_id"],
                    payload["started_at"],
                    payload["score"],
                    payload["repetitions"],
                    payload["algorithm_version"],
                    json.dumps(payload, separators=(",", ":")),
                ),
            )
        return session_id

    def save_profile(self, patient_id: str, payload: dict[str, object]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO patient_profiles VALUES (?, ?)",
                (patient_id, json.dumps(payload, separators=(",", ":"))),
            )

    def load_profile(self, patient_id: str) -> dict[str, object] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM patient_profiles WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def save_baseline(
        self, patient_id: str, exercise_id: str, payload: dict[str, object]
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO baselines VALUES (?, ?, ?)",
                (patient_id, exercise_id, json.dumps(payload, separators=(",", ":"))),
            )

    def load_baselines(self, patient_id: str) -> list[dict[str, object]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM baselines WHERE patient_id = ?", (patient_id,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def clear_baselines(self, patient_id: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM baselines WHERE patient_id = ?", (patient_id,))

    def list_sessions(self) -> list[dict[str, object]]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM sessions ORDER BY started_at DESC").fetchall()
        return [json.loads(row["payload"]) for row in rows]


def export_session(payload: dict[str, object], directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = str(payload.get("session_id", "session"))
    json_path, csv_path = directory / f"{stem}.json", directory / f"{stem}.csv"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    frames = payload.get("frames", [])
    if isinstance(frames, list):
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        columns = [
            "timestamp", "frame_id", "angle_deg", "progress_deg", "score", "confidence"
        ]
        with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in columns} for row in frames)
        temporary_csv.replace(csv_path)
    return json_path, csv_path
