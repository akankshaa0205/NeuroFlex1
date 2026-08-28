import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuroflex.domain.models import ScoreConfig
from neuroflex.domain.scoring import exact_dtw, movement_score


def test_identity_dtw_is_zero() -> None:
    assert exact_dtw([10, 20, 30], [10, 20, 30]) == 0


def test_perfect_score_is_100() -> None:
    result = movement_score(
        achieved_rom_deg=120, dtw_distance=0, theta_error_deg=0,
        config=ScoreConfig(120, 30, 15), confidence=1,
    )
    assert result.composite == 100


def test_invalid_denominator_rejected() -> None:
    with pytest.raises(ValueError):
        ScoreConfig(0, 1, 1)


@given(
    st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    st.floats(min_value=-2, max_value=2, allow_nan=False),
)
def test_score_is_always_bounded(rom: float, dtw: float, error: float, confidence: float) -> None:
    result = movement_score(
        achieved_rom_deg=rom, dtw_distance=dtw, theta_error_deg=error,
        config=ScoreConfig(120, 30, 15), confidence=confidence,
    )
    assert 0 <= result.composite <= 100
