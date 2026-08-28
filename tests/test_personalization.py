import pytest

from neuroflex.domain.models import PatientIntake
from neuroflex.domain.personalization import personalized_target, readiness_message


def intake(mobility: str = "Moderate", discomfort: int = 0) -> PatientIntake:
    return PatientIntake(58, "Shoulder", "Right", mobility, "Reach overhead", discomfort)


def test_target_uses_measured_baseline_and_mobility() -> None:
    assert personalized_target(100, intake("Limited")) == pytest.approx(66.5)
    assert personalized_target(100, intake("Good")) == pytest.approx(85.5)


def test_age_combines_with_personal_baseline_without_population_comparison() -> None:
    younger = PatientIntake(20, "Shoulder", "Right", "Moderate", "Reach", 0)
    older = PatientIntake(80, "Shoulder", "Right", "Moderate", "Reach", 0)
    assert personalized_target(100, older) < personalized_target(100, younger)
    assert personalized_target(100, older) == pytest.approx(69.7)


def test_high_discomfort_reduces_target_and_warns() -> None:
    person = intake("Good", 8)
    assert personalized_target(100, person) == pytest.approx(60)
    assert "clinician" in readiness_message(person).lower()


def test_invalid_calibration_rejected() -> None:
    with pytest.raises(ValueError, match="too small"):
        personalized_target(5, intake())
