from pathlib import Path

from neuroflex.persistence import SessionRepository, export_session


def test_session_round_trip_and_export(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "data.db")
    patient = repository.ensure_demo_patient()
    payload = {
        "exercise_id": "elbow-flexion", "started_at": "2026-01-01T00:00:00Z",
        "score": 88.5, "repetitions": 8, "algorithm_version": "test",
        "frames": [{"timestamp": "t", "frame_id": 1, "angle_deg": 90, "score": 88, "confidence": 1}],
    }
    session_id = repository.save_session(patient, payload)
    assert repository.list_sessions()[0]["score"] == 88.5
    assert repository.list_sessions()[0]["session_id"] == session_id
    payload["session_id"] = session_id
    json_path, csv_path = export_session(payload, tmp_path / "exports")
    assert json_path.exists() and csv_path.exists()
    assert not list((tmp_path / "exports").glob("*.tmp"))


def test_profile_and_baseline_round_trip(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "data.db")
    patient = repository.ensure_demo_patient()
    profile = {
        "age_years": 58, "body_area": "Shoulder", "affected_side": "Right", "mobility_level": "Moderate",
        "goal": "Reach overhead", "discomfort": 2,
    }
    baseline = {
        "exercise_id": "shoulder-flexion", "side": "Right", "comfortable_rom_deg": 90.0,
        "target_rom_deg": 73.8, "rest_value_deg": 10.0,
        "posture_reference_deg": 4.0,
        "calibrated_at": "2026-01-01T00:00:00Z",
    }
    repository.save_profile(patient, profile)
    repository.save_baseline(patient, "shoulder-flexion", baseline)
    assert repository.load_profile(patient) == profile
    assert repository.load_baselines(patient) == [baseline]
    repository.delete_baseline(patient, "shoulder-flexion")
    assert repository.load_baselines(patient) == []
