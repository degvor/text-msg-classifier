"""The schedule is the experiment's ground truth, so its semantics are pinned."""

import pytest

from streamcls.corpus.schedule import DriftSchedule, uniform_weights


def test_empty_schedule_is_the_stationary_control():
    sched = DriftSchedule()
    assert sched.change_points == ()
    state = sched.state_at(10_000)
    assert state.kind == "none"
    assert state.phase == 0
    assert state.concept_map == {}
    assert state.label_weights == uniform_weights()


def test_state_is_a_pure_function_of_the_index():
    """Wall-clock dependence is what made batch composition non-reproducible."""
    sched = DriftSchedule(events=[{"at": 100, "kind": "covariate_abrupt", "params": {"to": 0.8}}])
    assert sched.state_at(500) == sched.state_at(500)
    assert sched.state_at(99).intensity != sched.state_at(100).intensity


def test_events_apply_from_their_index_onward():
    sched = DriftSchedule(
        events=[{"at": 2000, "kind": "prior_shift", "params": {"weights": [0.7, 0.1, 0.1, 0.1]}}]
    )
    assert sched.state_at(1999).label_weights == uniform_weights()
    assert sched.state_at(2000).label_weights == pytest.approx((0.7, 0.1, 0.1, 0.1))
    assert sched.state_at(9999).label_weights == pytest.approx((0.7, 0.1, 0.1, 0.1))


def test_gradual_shift_ramps_linearly_then_clamps():
    sched = DriftSchedule(
        events=[{"at": 1000, "kind": "covariate_gradual", "params": {"to": 0.9, "over": 1000}}],
        base_intensity=0.1,
    )
    assert sched.state_at(1000).intensity == pytest.approx(0.1)
    assert sched.state_at(1500).intensity == pytest.approx(0.5)
    assert sched.state_at(2000).intensity == pytest.approx(0.9)
    assert sched.state_at(9000).intensity == pytest.approx(0.9)


def test_recurring_alternates_on_its_period():
    sched = DriftSchedule(
        events=[{"at": 0, "kind": "recurring", "params": {"period": 100, "to": 0.9}}],
        base_intensity=0.1,
    )
    assert sched.state_at(50).intensity == pytest.approx(0.9)
    assert sched.state_at(150).intensity == pytest.approx(0.1)
    assert sched.state_at(250).intensity == pytest.approx(0.9)


def test_concept_map_keys_survive_json_round_trip():
    """drift_schedule is stored as JSONB, where object keys become strings."""
    sched = DriftSchedule(events=[{"at": 10, "kind": "concept_abrupt", "params": {"map": {"2": 3}}}])
    assert sched.state_at(20).concept_map == {2: 3}


def test_reset_returns_to_the_baseline():
    sched = DriftSchedule(
        events=[
            {"at": 100, "kind": "covariate_abrupt", "params": {"to": 0.9}},
            {"at": 200, "kind": "reset"},
        ],
        base_intensity=0.2,
    )
    assert sched.state_at(150).intensity == pytest.approx(0.9)
    assert sched.state_at(250).intensity == pytest.approx(0.2)
    assert sched.state_at(250).kind == "none"


def test_events_are_sorted_so_hand_written_schedules_need_not_be():
    sched = DriftSchedule(
        events=[
            {"at": 300, "kind": "covariate_abrupt", "params": {"to": 0.9}},
            {"at": 100, "kind": "covariate_abrupt", "params": {"to": 0.5}},
        ]
    )
    assert sched.change_points == (100, 300)
    assert sched.state_at(150).intensity == pytest.approx(0.5)
    assert sched.state_at(350).intensity == pytest.approx(0.9)


def test_change_points_and_delay_reference_are_the_evaluation_ground_truth():
    sched = DriftSchedule(
        events=[
            {"at": 2000, "kind": "prior_shift", "params": {"weights": [0.7, 0.1, 0.1, 0.1]}},
            {"at": 5000, "kind": "concept_abrupt", "params": {"map": {"0": 1}}},
        ]
    )
    assert sched.change_points == (2000, 5000)
    assert sched.preceding_change_point(1999) is None
    assert sched.preceding_change_point(2000) == 2000
    assert sched.preceding_change_point(4999) == 2000
    assert sched.preceding_change_point(9999) == 5000
    assert sched.state_at(2000).is_drift_point
    assert not sched.state_at(2001).is_drift_point


def test_reset_is_not_a_change_point():
    """A reset restores the baseline; counting it would inflate the recall
    denominator when scoring detectors."""
    sched = DriftSchedule(
        events=[
            {"at": 100, "kind": "covariate_abrupt", "params": {"to": 0.9}},
            {"at": 200, "kind": "reset"},
        ]
    )
    assert sched.change_points == (100,)


def test_intensity_is_always_clamped_to_the_unit_interval():
    sched = DriftSchedule(events=[{"at": 0, "kind": "covariate_abrupt", "params": {"to": 5.0}}])
    assert sched.state_at(10).intensity == 1.0


@pytest.mark.parametrize(
    ("events", "match"),
    [
        ([{"at": 1, "kind": "not_a_kind"}], "not_a_kind"),
        ([{"kind": "reset"}], "needs an 'at'"),
    ],
)
def test_malformed_schedules_are_rejected_at_construction(events, match):
    with pytest.raises(ValueError, match=match):
        DriftSchedule(events=events)


@pytest.mark.parametrize("weights", [[1, 1], [0, 0, 0, 0], [-1, 1, 1, 1]])
def test_bad_prior_weights_are_rejected(weights):
    sched = DriftSchedule(events=[{"at": 0, "kind": "prior_shift", "params": {"weights": weights}}])
    with pytest.raises(ValueError):
        sched.state_at(1)


def test_weights_are_normalised_not_required_to_sum_to_one():
    sched = DriftSchedule(events=[{"at": 0, "kind": "prior_shift", "params": {"weights": [3, 1, 1, 1]}}])
    assert sum(sched.state_at(1).label_weights) == pytest.approx(1.0)
    assert sched.state_at(1).label_weights[0] == pytest.approx(0.5)
