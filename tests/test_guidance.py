from neuroflex.domain.exercises import EXERCISES
from neuroflex.domain.guidance import guidance_for


def test_every_exercise_has_complete_guidance() -> None:
    for exercise in EXERCISES:
        guidance = guidance_for(exercise.id)
        assert len(guidance.setup) > 20
        assert len(guidance.movement) > 20
        assert len(guidance.posture) > 20
