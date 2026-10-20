"""The promotion rule and the record it leaves."""

from __future__ import annotations

import pytest

from streamcls.promotion import decide


def test_promotes_within_the_tolerance():
    decision = decide(0.805, 0.81, min_delta=0.01, rule="live")
    assert decision.promoted
    assert decision.margin == pytest.approx(0.005)


def test_rejects_below_the_tolerance():
    decision = decide(0.79, 0.81, min_delta=0.01, rule="matched_holdout")
    assert not decision.promoted
    assert decision.margin == pytest.approx(-0.01)


def test_the_boundary_promotes():
    assert decide(0.80, 0.81, min_delta=0.01, rule="live").promoted


def test_a_wide_tolerance_promotes_everything():
    """How the no-promotion-gate configuration is expressed."""
    assert decide(0.0, 1.0, min_delta=1.0, rule="live").promoted


def test_the_record_carries_both_scores_and_the_rule():
    record = decide(0.6, 0.5, min_delta=0.01, rule="recent").as_dict()
    assert record == {
        "promoted": True, "rule": "recent", "candidate_score": 0.6,
        "incumbent_score": 0.5, "min_delta": 0.01, "margin": pytest.approx(0.11),
    }


def test_unknown_rules_are_refused():
    with pytest.raises(ValueError):
        decide(0.5, 0.5, min_delta=0.0, rule="vibes")
