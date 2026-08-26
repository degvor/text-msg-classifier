"""The perturbation engine is an experimental instrument, so its knob has to
behave: monotone in intensity, reproducible from a seed, and a strict no-op at
zero."""

import random

import pytest

from streamcls.corpus.perturb import OPERATORS, Perturber, perturb

SAMPLE = (
    "Redskins Top Giants, Break Losing Streak The Washington Redskins finally "
    "scored 20 points against the government of approximately three defenders."
)


def test_intensity_zero_is_exactly_the_identity():
    result = Perturber(intensity=0.0).apply(SAMPLE, random.Random(1))
    assert result.text == SAMPLE
    assert result.applied == ()


def test_same_seed_same_output():
    a = Perturber(intensity=0.7).apply(SAMPLE, random.Random(42))
    b = Perturber(intensity=0.7).apply(SAMPLE, random.Random(42))
    assert a == b


def test_different_seed_different_output():
    a = Perturber(intensity=0.7).apply(SAMPLE, random.Random(1)).text
    b = Perturber(intensity=0.7).apply(SAMPLE, random.Random(2)).text
    assert a != b


def test_more_intensity_means_more_operators_on_average():
    """The knob a covariate-shift scenario ramps must actually be monotone."""

    def mean_applied(intensity: float) -> float:
        p = Perturber(intensity=intensity)
        return sum(len(p.apply(SAMPLE, random.Random(s)).applied) for s in range(200)) / 200

    low, mid, high = mean_applied(0.15), mean_applied(0.5), mean_applied(1.0)
    assert low < mid < high


def test_output_stays_non_empty_and_is_stripped():
    for seed in range(50):
        text = Perturber(intensity=1.0).apply(SAMPLE, random.Random(seed)).text
        assert text
        assert text == text.strip()


def test_applied_names_are_real_operators():
    for seed in range(30):
        result = Perturber(intensity=0.9).apply(SAMPLE, random.Random(seed))
        assert set(result.applied) <= set(OPERATORS)


def test_operator_subset_is_respected():
    p = Perturber(intensity=1.0, enabled=("emoji",))
    for seed in range(20):
        assert set(p.apply(SAMPLE, random.Random(seed)).applied) <= {"emoji"}


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_each_operator_changes_something_and_never_crashes(name):
    operator, _ = OPERATORS[name]
    out = operator(SAMPLE, random.Random(3), 1.0)
    assert isinstance(out, str)
    assert out.strip()
    assert out != SAMPLE, f"{name} was a no-op at full intensity"


@pytest.mark.parametrize("text", ["", "a", "hi there", "12345 !!!"])
def test_operators_survive_degenerate_input(text):
    for name, (operator, _) in OPERATORS.items():
        assert isinstance(operator(text, random.Random(0), 0.8), str), name


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_intensity_is_range_checked(bad):
    with pytest.raises(ValueError, match="intensity"):
        Perturber(intensity=bad)


def test_unknown_operator_is_rejected():
    with pytest.raises(ValueError, match="unknown operators"):
        Perturber(enabled=("emoji", "teleport"))


def test_module_level_helper_matches_the_class():
    assert perturb(SAMPLE, random.Random(5), 0.6) == (
        Perturber(intensity=0.6).apply(SAMPLE, random.Random(5)).text
    )
