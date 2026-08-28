from neuroflex.domain.reports import build_session_report, recalibration_is_recommended


def session(rom: float, started: str, *, posture: float = 92, tracking: float = 95) -> dict:
    return {
        "patient_id": "p1",
        "exercise_id": "shoulder-flexion",
        "algorithm_version": "provisional-0.1",
        "started_at": started,
        "achieved_rom_deg": rom,
        "repetitions": 8,
        "prescribed_repetitions": 8,
        "posture_adherence_pct": posture,
        "tracking_coverage_pct": tracking,
        "personalization": {
            "affected_side": "Right",
            "target_rom_deg": 75,
            "comfortable_rom_deg": 80,
        },
        "frames": [
            {"posture_ok": True, "tracked_confidence": 0.9},
            {"posture_ok": True, "tracked_confidence": 0.8},
        ],
    }


def test_report_relates_performance_to_personal_baseline_and_day_one() -> None:
    first = session(70, "2026-01-01")
    latest = session(82, "2026-01-08")
    report = build_session_report(latest, [first])
    assert report.completed_repetitions == 8
    assert report.target_attainment_pct > 100
    assert report.baseline_attainment_pct == 102.5
    assert report.rom_change_from_day_one_deg == 12
    assert report.posture_adherence_pct == 100


def test_recalibration_requires_three_consistent_high_quality_sessions() -> None:
    first = session(85, "2026-01-01")
    second = session(87, "2026-01-04")
    latest = session(89, "2026-01-08")
    assert recalibration_is_recommended(latest, [first, second])
    low_posture = session(89, "2026-01-08", posture=70)
    assert not recalibration_is_recommended(low_posture, [first, second])
    assert not recalibration_is_recommended(second, [first])
