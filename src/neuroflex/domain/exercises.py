from neuroflex.domain.models import Exercise

# MediaPipe landmark indices. Thresholds are provisional and require clinician approval.
EXERCISES = (
    Exercise("shoulder-flexion", "Shoulder Flexion", (23, 11, 13), 110, 20, cue_low="Raise your arm forward", body_region="Shoulder"),
    Exercise("shoulder-abduction", "Shoulder Abduction", (23, 11, 13), 100, 20, cue_low="Lift your arm outward", body_region="Shoulder"),
    Exercise("elbow-flexion", "Elbow Flexion", (11, 13, 15), 120, 20, cue_low="Bend your elbow farther", body_region="Elbow", movement_mode="flexion"),
    Exercise("hip-flexion", "Standing Hip Flexion", (11, 23, 25), 80, 15, cue_low="Lift your knee comfortably", body_region="Hip", movement_mode="flexion"),
    Exercise("hip-abduction", "Standing Hip Abduction", (11, 23, 25), 45, 12, cue_low="Move your leg gently outward", body_region="Hip", movement_mode="flexion"),
    Exercise("knee-extension", "Seated Knee Extension", (23, 25, 27), 70, 15, cue_low="Straighten your knee farther", body_region="Knee"),
    Exercise("knee-flexion", "Standing Knee Flexion", (23, 25, 27), 80, 15, cue_low="Bend your knee farther", body_region="Knee", movement_mode="flexion"),
    Exercise("sit-to-stand", "Sit to Stand", (23, 25, 27), 75, 15, cue_low="Rise with steady control", body_region="Full body"),
    Exercise("trunk-side-bend", "Trunk Side Bend", (11, 23, 25), 35, 10, cue_low="Bend gently to the side", body_region="Trunk", movement_mode="flexion"),
    Exercise("ankle-mobility", "Ankle Mobility", (25, 27, 31), 30, 10, cue_low="Move through a comfortable ankle range", body_region="Ankle", movement_mode="flexion"),
)


def movement_signal(angle_deg: float, mode: str) -> float:
    if mode == "angle":
        return angle_deg
    if mode == "flexion":
        return 180.0 - angle_deg
    raise ValueError(f"Unsupported movement mode: {mode}")
