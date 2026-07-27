import pytest

from analysis.metrics import latex_escape, mean, wilson_ci


def test_mean_empty_and_nonempty():
    assert mean([]) == 0.0
    assert mean([0.0, 1.0]) == 0.5


def test_wilson_interval_bounds():
    assert wilson_ci(0, 0) == {"lo": 0.0, "hi": 0.0}
    interval = wilson_ci(5, 10)
    assert 0.0 < interval["lo"] < 0.5 < interval["hi"] < 1.0


@pytest.mark.parametrize(("successes", "n"), [(-1, 10), (11, 10), (0, -1)])
def test_wilson_interval_rejects_invalid_counts(successes, n):
    with pytest.raises(ValueError):
        wilson_ci(successes, n)


def test_latex_escape():
    assert latex_escape("a_b & 50%") == r"a\_b \& 50\%"
    assert latex_escape("\\{}") == r"\textbackslash{}\{\}"
