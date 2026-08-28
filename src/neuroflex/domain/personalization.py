from neuroflex.domain.models import PatientIntake


def effort_factor(intake: PatientIntake) -> float:
    """Provisional effort combining age with measured mobility and discomfort."""
    mobility_factor = {"Limited": 0.70, "Moderate": 0.82, "Good": 0.90}[
        intake.mobility_level
    ]
    if intake.age_years >= 80:
        age_factor = 0.85
    elif intake.age_years >= 65:
        age_factor = 0.90
    elif intake.age_years >= 45:
        age_factor = 0.95
    else:
        age_factor = 1.0
    factor = mobility_factor * age_factor
    if intake.discomfort >= 7:
        return min(factor, 0.60)
    if intake.discomfort >= 4:
        return min(factor, 0.72)
    return factor


def personalized_target(comfortable_rom_deg: float, intake: PatientIntake) -> float:
    if comfortable_rom_deg < 10:
        raise ValueError("Calibration range is too small; repeat calibration")
    return comfortable_rom_deg * effort_factor(intake)


def readiness_message(intake: PatientIntake) -> str:
    if intake.discomfort >= 7:
        return "High discomfort reported — use only with clinician guidance"
    if intake.discomfort >= 4:
        return "Moderate discomfort reported — target reduced for this session"
    return "Ready for a comfortable baseline calibration"
