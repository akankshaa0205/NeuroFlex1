from neuroflex.domain.coach import CoachPhase, SessionCoach


def test_full_slow_repetition_requires_dwell_hold_return_and_rest() -> None:
    coach = SessionCoach(2)
    assert coach.start(0).phase == CoachPhase.PREPARE
    assert coach.update(progress_ratio=0, visible=True, posture_ok=True, now=4).phase == CoachPhase.PREPARE
    assert coach.update(progress_ratio=0, visible=True, posture_ok=True, now=5).phase == CoachPhase.MOVE_OUT
    assert coach.update(progress_ratio=0.9, visible=True, posture_ok=True, now=6).phase == CoachPhase.MOVE_OUT
    assert coach.update(progress_ratio=0.9, visible=True, posture_ok=True, now=6.9).phase == CoachPhase.HOLD
    assert coach.update(progress_ratio=0.9, visible=True, posture_ok=True, now=8.2).phase == CoachPhase.RETURN
    assert coach.update(progress_ratio=0.1, visible=True, posture_ok=True, now=9).phase == CoachPhase.RETURN
    result = coach.update(progress_ratio=0.1, visible=True, posture_ok=True, now=9.7)
    assert result.phase == CoachPhase.REST
    assert result.repetitions == 1
    assert coach.update(progress_ratio=0.1, visible=True, posture_ok=True, now=11.8).phase == CoachPhase.MOVE_OUT
    coach.update(progress_ratio=0.9, visible=True, posture_ok=True, now=12)
    assert coach.update(progress_ratio=0.9, visible=True, posture_ok=True, now=13).phase == CoachPhase.HOLD
    assert coach.update(progress_ratio=0.9, visible=True, posture_ok=True, now=14.3).phase == CoachPhase.RETURN
    coach.update(progress_ratio=0.1, visible=True, posture_ok=True, now=15)
    finished = coach.update(progress_ratio=0.1, visible=True, posture_ok=True, now=15.7)
    assert finished.phase == CoachPhase.COMPLETE
    assert finished.repetitions == 2


def test_tracking_loss_pauses_phase_clock() -> None:
    coach = SessionCoach(1)
    coach.start(0)
    lost = coach.update(progress_ratio=0, visible=False, posture_ok=True, now=2)
    assert "paused" in lost.instruction.lower()
    resumed = coach.update(progress_ratio=0, visible=True, posture_ok=True, now=12)
    assert resumed.phase == CoachPhase.PREPARE


def test_bad_posture_blocks_progression() -> None:
    coach = SessionCoach(1, preparation_seconds=0)
    coach.start(0)
    coach.update(progress_ratio=0, visible=True, posture_ok=True, now=0)
    result = coach.update(progress_ratio=1, visible=True, posture_ok=False, now=3)
    assert result.phase == CoachPhase.MOVE_OUT
    assert "posture" in result.instruction.lower()


def test_instruction_is_stable_during_a_movement_phase() -> None:
    coach = SessionCoach(1, preparation_seconds=0)
    coach.start(0)
    coach.update(progress_ratio=0, visible=True, posture_ok=True, now=0)
    instructions = {
        coach.update(progress_ratio=value, visible=True, posture_ok=True, now=index).instruction
        for index, value in enumerate((0.1, 0.2, 0.35, 0.5, 0.7), start=1)
    }
    assert len(instructions) == 1
