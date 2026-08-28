import pytest

from neuroflex.domain.feedback import assess_live_movement, choose_active_side


def test_live_score_responds_to_current_motion() -> None:
    low = assess_live_movement(
        current_progress_deg=20, achieved_rom_deg=30, target_rom_deg=100,
        confidence=1, continue_cue="Keep going",
    )
    high = assess_live_movement(
        current_progress_deg=80, achieved_rom_deg=80, target_rom_deg=100,
        confidence=1, continue_cue="Keep going",
    )
    assert high.score > low.score
    assert high.verdict != low.verdict


def test_live_score_reflects_tracking_confidence() -> None:
    tracked = assess_live_movement(
        current_progress_deg=90, achieved_rom_deg=90, target_rom_deg=100,
        confidence=1, continue_cue="Keep going",
    )
    uncertain = assess_live_movement(
        current_progress_deg=90, achieved_rom_deg=90, target_rom_deg=100,
        confidence=0.5, continue_cue="Keep going",
    )
    assert uncertain.score == pytest.approx(tracked.score / 2)


def test_active_side_uses_the_arm_with_greater_motion() -> None:
    assert choose_active_side({"Left": [0, 20], "Right": [0, 70]}) == "Right"
