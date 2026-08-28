from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExerciseGuidance:
    setup: str
    movement: str
    posture: str


GUIDANCE = {
    "shoulder-flexion": ExerciseGuidance(
        "Sit or stand tall with the working arm relaxed by your side.",
        "Raise the straight arm forward only as far as comfortable, then lower it slowly.",
        "Keep your trunk upright and avoid shrugging the shoulder.",
    ),
    "shoulder-abduction": ExerciseGuidance(
        "Sit or stand tall with the working arm relaxed by your side.",
        "Raise the arm out to the side, then lower it with control.",
        "Keep the trunk centered; do not lean away from the moving arm.",
    ),
    "elbow-flexion": ExerciseGuidance(
        "Keep the upper arm close to your body with the elbow initially straight.",
        "Bend the elbow slowly, bringing the hand toward the shoulder, then straighten.",
        "Keep the shoulder relaxed and the upper arm still.",
    ),
    "hip-flexion": ExerciseGuidance(
        "Stand tall beside a stable support, with both feet on the floor.",
        "Lift one knee forward to a comfortable height, then lower it slowly.",
        "Keep the trunk upright and use support if balance is limited.",
    ),
    "hip-abduction": ExerciseGuidance(
        "Stand tall beside a stable support with toes pointing forward.",
        "Move one straight leg gently out to the side, then return.",
        "Keep the pelvis level and avoid leaning the trunk.",
    ),
    "knee-extension": ExerciseGuidance(
        "Sit securely with the back supported and both feet initially down.",
        "Straighten one knee slowly, pause comfortably, then lower the foot.",
        "Keep the thigh supported and avoid lifting the hip.",
    ),
    "knee-flexion": ExerciseGuidance(
        "Stand beside a stable support with knees aligned.",
        "Bend one knee, bringing the heel backward, then lower slowly.",
        "Keep both knees close and the trunk upright.",
    ),
    "sit-to-stand": ExerciseGuidance(
        "Sit near the front of a stable chair with feet hip-width apart.",
        "Lean forward slightly, stand with control, then sit down slowly.",
        "Keep knees aligned over the feet and use support when prescribed.",
    ),
    "trunk-side-bend": ExerciseGuidance(
        "Sit or stand tall with shoulders relaxed and weight centered.",
        "Slide one hand gently down the side, then return to center.",
        "Stay facing forward; avoid twisting or bending forward.",
    ),
    "ankle-mobility": ExerciseGuidance(
        "Sit securely with the working foot visible to the camera.",
        "Move the ankle slowly through a comfortable up-and-down range.",
        "Keep the knee and lower leg as still as possible.",
    ),
}


def guidance_for(exercise_id: str) -> ExerciseGuidance:
    return GUIDANCE[exercise_id]
