"""Experiment scoring. Pure functions over plain rows, so no database here."""

from __future__ import annotations

import pytest

from streamcls.experiments import compare, format_table, score_run

BATCH = 500


def batches(spec):
    """spec: list of (batch_id, accuracy). 100 messages each."""
    return [
        {"batch_id": b, "n_received": 100, "n_classified": 100,
         "n_correct": round(a * 100), "accuracy": a}
        for b, a in spec
    ]


def event(batch_id, detector="adwin_error", *, suppressed=False, action="retrain_queued"):
    return {"batch_id": batch_id, "detector": detector, "statistic": 0.4,
            "suppressed": suppressed, "action_taken": action}


def score(*, spec, events=(), jobs=(), change_at=None, adaptation=True):
    schedule = [{"at": change_at, "kind": "concept_abrupt"}] if change_at is not None else []
    return score_run(
        run={"id": 1, "name": "r", "seed": 42, "adaptation_enabled": adaptation,
             "drift_schedule": schedule},
        batches=batches(spec), events=list(events), jobs=list(jobs), batch_size=BATCH,
    )


# --- accuracy windows --------------------------------------------------------
def test_accuracy_is_split_around_the_change_point():
    """Overall accuracy is dominated by the stable prefix both A/B arms share,
    so the split is what carries the comparison."""
    s = score(spec=[(i, 0.9) for i in range(6)] + [(i, 0.5) for i in range(6, 12)],
              change_at=6 * BATCH)
    assert s.accuracy_before_drift == pytest.approx(0.9)
    assert s.accuracy_after_drift == pytest.approx(0.5)
    assert s.accuracy_overall == pytest.approx(0.7)


def test_no_change_point_means_no_after_window():
    s = score(spec=[(i, 0.9) for i in range(5)])
    assert s.accuracy_before_drift == pytest.approx(0.9)
    assert s.accuracy_after_drift is None
    assert s.change_points == []


def test_change_point_is_converted_from_messages_to_batches():
    s = score(spec=[(i, 0.9) for i in range(10)], change_at=3000)
    assert s.change_points == [6]


# --- detection delay ---------------------------------------------------------
def test_detection_delay_is_measured_from_the_change_point():
    s = score(spec=[(i, 0.9) for i in range(12)], change_at=6 * BATCH, events=[event(8)])
    assert s.detection_delays == [2]
    assert s.mean_detection_delay == pytest.approx(2.0)
    assert s.missed_detections == 0


def test_detection_on_the_change_point_batch_is_zero_delay():
    s = score(spec=[(i, 0.9) for i in range(12)], change_at=6 * BATCH, events=[event(6)])
    assert s.detection_delays == [0]


def test_only_the_first_detection_after_a_change_point_counts():
    s = score(spec=[(i, 0.9) for i in range(14)], change_at=6 * BATCH,
              events=[event(8), event(9), event(10)])
    assert s.detection_delays == [2]


def test_a_change_point_with_no_detection_is_a_miss():
    s = score(spec=[(i, 0.9) for i in range(12)], change_at=6 * BATCH, events=[])
    assert s.detection_delays == [None]
    assert s.missed_detections == 1
    assert s.mean_detection_delay is None


def test_each_change_point_gets_its_own_detection():
    s = score(spec=[(i, 0.9) for i in range(40)], change_at=4 * BATCH,
              events=[event(5), event(21)])
    s2 = score_run(
        run={"id": 1, "name": "r", "seed": 1, "adaptation_enabled": True,
             "drift_schedule": [{"at": 4 * BATCH, "kind": "x"}, {"at": 20 * BATCH, "kind": "x"}]},
        batches=batches([(i, 0.9) for i in range(40)]),
        events=[event(5), event(21)], jobs=[], batch_size=BATCH,
    )
    assert s2.change_points == [4, 20]
    assert s2.detection_delays == [1, 1]
    assert s.missed_detections == 0


# --- false alarms ------------------------------------------------------------
def test_a_detection_with_no_change_point_is_a_false_alarm():
    s = score(spec=[(i, 0.9) for i in range(10)], events=[event(3)])
    assert s.false_alarms == 1


def test_a_detection_long_after_the_stream_settled_is_a_false_alarm():
    s = score(spec=[(i, 0.9) for i in range(60)], change_at=2 * BATCH,
              events=[event(3), event(55)])
    assert s.detection_delays == [1]
    assert s.false_alarms == 1


def test_suppressed_signals_are_neither_detections_nor_false_alarms():
    """Signals inside the post-swap grace window are recorded but must not
    count against the detector."""
    s = score(spec=[(i, 0.9) for i in range(10)], change_at=5 * BATCH,
              events=[event(2, suppressed=True), event(3, suppressed=True)])
    assert s.suppressed_signals == 2
    assert s.false_alarms == 0
    assert s.missed_detections == 1


def test_recorded_but_unactioned_signals_still_count_as_detections():
    """'The detector fired' and 'the system acted' are separate questions;
    conflating them would hide the false-alarm rate."""
    s = score(spec=[(i, 0.9) for i in range(10)], events=[event(3, action="recorded")])
    assert s.false_alarms == 1


# --- recovery ----------------------------------------------------------------
def test_recovery_is_measured_once_accuracy_returns_to_baseline():
    spec = [(i, 0.90) for i in range(5)] + [(5, 0.50), (6, 0.55), (7, 0.88), (8, 0.89)]
    s = score(spec=spec, change_at=5 * BATCH)
    # Batches 7 and 8 are the first consecutive pair within tolerance.
    assert s.recovery_batches == 2


def test_a_single_lucky_batch_is_not_recovery():
    spec = [(i, 0.90) for i in range(5)] + [(5, 0.50), (6, 0.89), (7, 0.50), (8, 0.51)]
    s = score(spec=spec, change_at=5 * BATCH)
    assert s.recovery_batches is None


def test_no_recovery_reports_none_rather_than_a_number():
    spec = [(i, 0.90) for i in range(5)] + [(i, 0.50) for i in range(5, 12)]
    assert score(spec=spec, change_at=5 * BATCH).recovery_batches is None


# --- A/B ---------------------------------------------------------------------
def test_compare_reports_the_delta_after_drift():
    baseline = score(spec=[(i, 0.9) for i in range(5)] + [(i, 0.50) for i in range(5, 12)],
                     change_at=5 * BATCH, adaptation=False)
    adaptive = score(spec=[(i, 0.9) for i in range(5)] + [(i, 0.70) for i in range(5, 12)],
                     change_at=5 * BATCH, adaptation=True)
    result = compare(baseline, adaptive)
    assert result["seed_matches"] is True
    assert result["accuracy_after_drift"]["delta"] == pytest.approx(0.20)


def test_compare_flags_a_seed_mismatch():
    """Without a shared seed the streams differ, so the gap is not attributable
    to adaptation alone."""
    import dataclasses

    a = score(spec=[(0, 0.9)])
    b = dataclasses.replace(a, seed=999)
    assert compare(a, b)["seed_matches"] is False


def test_format_table_renders_every_run():
    rows = [score(spec=[(i, 0.9) for i in range(3)]) for _ in range(2)]
    text = format_table(rows)
    assert text.count("\n") >= 3
    assert "delay" in text and "FA" in text


def test_empty_run_does_not_crash():
    s = score(spec=[])
    assert s.batches == 0
    assert s.accuracy_overall is None
    assert s.recovery_batches is None
