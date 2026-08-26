"""PSI and Jensen-Shannon, against hand-computed values and known identities."""

import math

import numpy as np
import pytest

from streamcls.drift.distributions import (
    PSI_MAJOR,
    PSI_MINOR,
    jensen_shannon,
    normalise,
)
from streamcls.drift.distributions import (
    population_stability_index as psi,
)


def test_identical_distributions_have_zero_distance():
    p = [0.25, 0.25, 0.25, 0.25]
    assert psi(p, p) == pytest.approx(0.0, abs=1e-9)
    assert jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-9)


def test_counts_and_proportions_agree():
    """Callers pass raw counts; scale must not change the statistic."""
    assert psi([100, 300], [200, 200]) == pytest.approx(psi([0.25, 0.75], [0.5, 0.5]), rel=1e-6)


def test_psi_matches_a_hand_computed_value():
    expected, observed = [0.5, 0.5], [0.9, 0.1]
    manual = (0.9 - 0.5) * math.log(0.9 / 0.5) + (0.1 - 0.5) * math.log(0.1 / 0.5)
    assert psi(expected, observed) == pytest.approx(manual, rel=1e-4)


def test_psi_is_symmetric():
    a, b = [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]
    assert psi(a, b) == pytest.approx(psi(b, a), rel=1e-9)


def test_psi_grows_monotonically_with_the_shift():
    reference = [0.25] * 4
    shifts = [
        [0.25, 0.25, 0.25, 0.25],
        [0.40, 0.20, 0.20, 0.20],
        [0.55, 0.15, 0.15, 0.15],
        [0.70, 0.10, 0.10, 0.10],
    ]
    values = [psi(reference, s) for s in shifts]
    assert values == sorted(values)
    assert values[0] < 1e-9


def test_psi_thresholds_bracket_a_realistic_prior_shift():
    """The prior_shift scenario moves 25/25/25/25 to 70/10/10/10; that has to
    read as a major shift or the threshold is useless."""
    assert psi([0.25] * 4, [0.7, 0.1, 0.1, 0.1]) > PSI_MAJOR
    # A mild wobble must not trip the minor threshold.
    assert psi([0.25] * 4, [0.27, 0.24, 0.25, 0.24]) < PSI_MINOR


def test_jensen_shannon_is_bounded_and_maximal_for_disjoint_support():
    disjoint_a, disjoint_b = [1.0, 0.0], [0.0, 1.0]
    value = jensen_shannon(disjoint_a, disjoint_b)
    assert 0.0 <= value <= 1.0
    assert value > 0.99, "disjoint distributions should be near the 1-bit maximum"


def test_jensen_shannon_is_symmetric():
    a, b = [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]
    assert jensen_shannon(a, b) == pytest.approx(jensen_shannon(b, a), rel=1e-9)


def test_empty_bins_do_not_produce_infinity():
    """Without smoothing this is a divide-by-zero, not a detection."""
    value = psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5])
    assert math.isfinite(value)
    assert jensen_shannon([1, 0, 0], [0, 0, 1]) <= 1.0


def test_an_empty_batch_is_not_reported_as_drift():
    """A micro-batch can legitimately be empty; that is not evidence of change."""
    assert psi([0.25] * 4, [0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-9)


def test_normalise_sums_to_one_and_rejects_bad_input():
    assert normalise([1, 2, 3]).sum() == pytest.approx(1.0)
    assert np.all(normalise([0, 0, 0]) == pytest.approx(1 / 3))
    with pytest.raises(ValueError, match="non-negative"):
        normalise([1, -1])
    with pytest.raises(ValueError, match="non-empty"):
        normalise([])


def test_mismatched_bin_counts_are_rejected():
    """A reference from one model against a histogram from another is a bug,
    not something to silently truncate."""
    with pytest.raises(ValueError, match="same number of bins"):
        psi([0.5, 0.5], [0.3, 0.3, 0.4])


def test_realistic_51_bin_vocabulary_shapes():
    rng = np.random.default_rng(0)
    reference = rng.dirichlet(np.ones(51))
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-9)
    # Resampling from the same reference should stay well under the minor
    # threshold: sampling noise must not read as drift.
    sampled = rng.multinomial(5000, reference)
    assert psi(reference, sampled) < PSI_MINOR
