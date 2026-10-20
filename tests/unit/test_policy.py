"""The retrain trigger rule and its controlled-experiment variants."""

from __future__ import annotations

import pytest

from streamcls.drift.monitor import DriftSignal
from streamcls.drift.policy import PERIODIC, RetrainTrigger, TriggerConfig

B = 500
ADWIN = DriftSignal("adwin_error", 0.4, None)


def seen_after(batch_id: int):
    """``messages_since`` for a stream whose batches 0..batch_id are processed."""
    return lambda origin: max(0, (batch_id + 1) * B - origin)


def run(trigger: RetrainTrigger, signals: dict[int, list[DriftSignal]], *, change_point, batches):
    """Feed batches in order; return {batch: action} for every batch that did something."""
    out = {}
    for b in batches:
        cp = change_point if change_point is not None and b * B >= change_point else None
        decision = trigger.decide(
            batch_id=b, actionable=signals.get(b, []), change_point=cp, messages_since=seen_after(b)
        )
        if decision.action == "queue":
            trigger.queued()
        if decision.action != "none":
            out[b] = (decision.action, decision.origin, decision.new_rows)
    return out


def test_default_rule_defers_until_enough_new_data_then_queues_once():
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=2000))
    # concept_abrupt: change at message 8000 (batch 16), ADWIN fires in batch 16 only
    got = run(trigger, {16: [ADWIN]}, change_point=8000, batches=range(16, 24))
    assert got == {
        16: ("defer", 8000, 500),
        17: ("defer", 8000, 1000),
        18: ("defer", 8000, 1500),
        19: ("queue", 8000, 2000),
    }
    assert trigger.pending is None


def test_without_a_change_point_the_signal_is_acted_on_at_once():
    """Stationary scenario, or a false alarm before the first change point."""
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=2000))
    got = run(trigger, {7: [ADWIN]}, change_point=None, batches=range(0, 10))
    assert got == {7: ("queue", None, None)}


def test_no_pending_drops_an_early_detection():
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=2000, retain_pending=False))
    got = run(trigger, {16: [ADWIN]}, change_point=8000, batches=range(16, 32))
    assert got == {16: ("drop", 8000, 500)}
    assert trigger.pending is None


def test_no_pending_still_acts_on_a_later_signal():
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=2000, retain_pending=False))
    got = run(trigger, {16: [ADWIN], 21: [ADWIN]}, change_point=8000, batches=range(16, 24))
    assert got == {16: ("drop", 8000, 500), 21: ("queue", 8000, 3000)}


def test_detection_origin_counts_from_the_detection_batch():
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=2000, origin="detection"))
    # covariate_gradual: change at 5333 (batch 10), detected only in batch 27
    got = run(trigger, {27: [ADWIN]}, change_point=5333, batches=range(10, 32))
    assert got == {
        27: ("defer", 13500, 500),
        28: ("defer", 13500, 1000),
        29: ("defer", 13500, 1500),
        30: ("queue", 13500, 2000),
    }


def test_detection_origin_matches_the_known_point_when_detected_at_once():
    known = run(RetrainTrigger(TriggerConfig(min_new_rows=2000)), {16: [ADWIN]},
                change_point=8000, batches=range(16, 24))
    detected = run(RetrainTrigger(TriggerConfig(min_new_rows=2000, origin="detection")),
                   {16: [ADWIN]}, change_point=8000, batches=range(16, 24))
    assert known == detected


def test_detection_origin_keeps_the_first_detection_when_signals_repeat():
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=2000, origin="detection"))
    got = run(trigger, {16: [ADWIN], 18: [ADWIN]}, change_point=8000, batches=range(16, 24))
    assert got[19] == ("queue", 8000, 2000)


def test_detection_origin_defers_a_false_alarm_too():
    """Without the oracle there is no 'no change point yet' shortcut."""
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=2000, origin="detection"))
    got = run(trigger, {7: [ADWIN]}, change_point=None, batches=range(0, 12))
    assert got[7] == ("defer", 3500, 500)
    assert got[10] == ("queue", 3500, 2000)


def test_zero_min_new_rows_queues_at_the_signal():
    """The simplified drift-triggered policy: no wait for new data."""
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=0))
    got = run(trigger, {16: [ADWIN]}, change_point=8000, batches=range(16, 20))
    assert got == {16: ("queue", 8000, None)}


def test_periodic_queues_on_the_schedule_and_ignores_signals():
    trigger = RetrainTrigger(TriggerConfig(trigger=PERIODIC, periodic_batches=8, min_new_rows=2000))
    got = run(trigger, {3: [ADWIN], 16: [ADWIN]}, change_point=8000, batches=range(0, 32))
    assert got == {8: ("queue", None, None), 16: ("queue", None, None), 24: ("queue", None, None)}
    decision = trigger.decide(
        batch_id=8, actionable=[], change_point=None, messages_since=seen_after(8)
    )
    assert decision.trigger == PERIODIC
    assert trigger.wants_every_batch


def test_a_held_detection_makes_the_trigger_want_every_batch():
    trigger = RetrainTrigger(TriggerConfig(min_new_rows=2000))
    assert not trigger.wants_every_batch
    trigger.decide(batch_id=16, actionable=[ADWIN], change_point=8000,
                   messages_since=seen_after(16))
    assert trigger.wants_every_batch


def test_from_stream_config_reads_the_knobs_and_keeps_defaults():
    config = {"max_offsets_per_trigger": 500, "retrain_min_new_rows": 2000}
    assert TriggerConfig.from_stream_config(config) == TriggerConfig(min_new_rows=2000)
    full = TriggerConfig.from_stream_config(config | {
        "retrain_trigger": "periodic", "periodic_retrain_batches": 8,
        "retain_pending_drift": False, "new_data_origin": "detection",
    })
    assert full == TriggerConfig(
        trigger="periodic", periodic_batches=8, min_new_rows=2000,
        retain_pending=False, origin="detection",
    )


@pytest.mark.parametrize("bad", [
    {"trigger": "weekly"},
    {"origin": "oracle"},
    {"trigger": "periodic", "periodic_batches": 0},
])
def test_invalid_knobs_fail_at_construction(bad):
    with pytest.raises(ValueError):
        TriggerConfig(**bad)
