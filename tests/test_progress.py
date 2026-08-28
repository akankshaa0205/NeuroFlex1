from neuroflex.domain.progress import summarize_progress


def test_progress_compares_latest_session_to_day_one() -> None:
    rows = [
        {"exercise_id": "knee", "started_at": "2026-01-01", "score": 60, "achieved_rom_deg": 40},
        {"exercise_id": "knee", "started_at": "2026-01-08", "score": 76, "achieved_rom_deg": 52},
        {"exercise_id": "shoulder", "started_at": "2026-01-09", "score": 90},
    ]
    summary = summarize_progress(rows, "knee")
    assert summary is not None
    assert summary.sessions == 2
    assert summary.score_change == 16
    assert summary.rom_change == 12
