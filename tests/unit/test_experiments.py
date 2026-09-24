"""Experiment scoring. Pure functions over plain rows, so no database here."""

from __future__ import annotations

import csv
import dataclasses

import pytest

from streamcls.experiments import (
    METRICS,
    Result,
    RunScore,
    Stat,
    aggregate,
    compare,
    detector_table,
    format_table,
    pair,
    render_markdown,
    score_run,
    write_csv,
    write_pairs_csv,
)

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


# --- retrains and promotions -------------------------------------------------
def test_promotions_count_retrains_whose_model_was_activated():
    """The query joins model_version.activated_at: a retrain the promotion
    guard rejected trained a model but never served it."""
    jobs = [
        {"id": 1, "status": "succeeded", "model_version_id": 7, "promoted": True},
        {"id": 2, "status": "succeeded", "model_version_id": 8, "promoted": False},
        {"id": 3, "status": "failed", "model_version_id": None, "promoted": None},
    ]
    s = score(spec=[(i, 0.9) for i in range(3)], jobs=jobs)
    assert s.retrains == 3
    assert s.promotions == 1


# --- per-detector quality ----------------------------------------------------
def test_each_detector_is_scored_on_its_own_signals():
    """Every detector records its signals whether or not it is actionable, so
    the scenario x detector table comes out of the same runs."""
    s = score(
        spec=[(i, 0.9) for i in range(14)], change_at=6 * BATCH,
        events=[event(7, "ddm"), event(9, "adwin_error"), event(2, "psi_vocabulary"),
                event(8, "psi_vocabulary", suppressed=True)],
    )
    assert s.detection_delays == [1], "the run-level delay is the earliest detector"
    assert s.detectors["ddm"].detection_delays == [1]
    assert s.detectors["adwin_error"].detection_delays == [3]
    assert s.detectors["adwin_error"].mean_detection_delay == pytest.approx(3.0)
    psi = s.detectors["psi_vocabulary"]
    assert psi.detection_delays == [None] and psi.missed_detections == 1
    assert psi.false_alarms == 1 and psi.suppressed == 1 and psi.signals == 2
    assert set(s.detectors) == {"ddm", "adwin_error", "psi_vocabulary"}


def test_scenario_travels_with_the_score():
    s = score_run(
        run={"id": 1, "name": "r", "seed": 1, "adaptation_enabled": False,
             "drift_schedule": [], "scenario": "stationary"},
        batches=batches([(0, 0.9)]), events=[], jobs=[], batch_size=BATCH,
    )
    assert s.scenario == "stationary" and s.arm == "baseline"


def test_a_score_survives_the_ledger_round_trip():
    import json

    s = score(spec=[(i, 0.9) for i in range(14)], change_at=6 * BATCH,
              events=[event(7, "ddm"), event(9)], jobs=[{"id": 1, "promoted": True}])
    back = RunScore.from_dict(json.loads(json.dumps(s.as_dict())))
    assert back == s
    assert back.detectors["ddm"].mean_detection_delay == pytest.approx(1.0)


# --- aggregation -------------------------------------------------------------
def _result(scenario, arm, seed, *, post, recovery, ablation=None, run_id=None, detectors=()):
    """A ledger row for one completed cell, scored from a synthetic stream."""
    spec = [(i, 0.90) for i in range(6)] + [(i, post) for i in range(6, 12)]
    events = [event(6 + d, name) for name, d in detectors]
    s = score_run(
        run={"id": run_id or seed, "name": f"{scenario}/{arm}/seed{seed}", "seed": seed,
             "adaptation_enabled": arm == "adaptive",
             "drift_schedule": [{"at": 6 * BATCH, "kind": "concept_abrupt"}],
             "scenario": scenario},
        batches=batches(spec), events=events, jobs=[], batch_size=BATCH,
    )
    s.recovery_batches = recovery
    return {"scenario": scenario, "arm": arm, "ablation": ablation, "seed": seed,
            "run_id": s.run_id, "status": "completed", "score": s.as_dict()}


def test_stat_reports_mean_sd_and_how_many_runs_defined_the_metric():
    stat = Stat.of([2, 4, None])
    assert stat.mean == pytest.approx(3.0)
    assert stat.sd == pytest.approx(1.4142, abs=1e-3)
    assert (stat.n, stat.missing) == (2, 1)
    assert stat.fmt() == "3.0 ± 1.4 (2/3)"
    assert Stat.of([0.5]).fmt(percent=True) == "50.0"
    assert Stat.of([None, None]).fmt() == "—"
    assert Stat.of([]).n == 0


def test_aggregate_averages_over_seeds_per_cell():
    rows = [
        _result("concept_abrupt", "baseline", 1, post=0.50, recovery=None),
        _result("concept_abrupt", "baseline", 2, post=0.52, recovery=None),
        _result("concept_abrupt", "baseline", 3, post=0.54, recovery=None),
        _result("concept_abrupt", "adaptive", 1, post=0.80, recovery=2, run_id=11),
        _result("concept_abrupt", "adaptive", 2, post=0.82, recovery=None, run_id=12),
        _result("concept_abrupt", "adaptive", 3, post=0.84, recovery=4, run_id=13),
        _result("concept_abrupt", "adaptive", 1, post=0.70, recovery=3, ablation="base_rows_0",
                run_id=21),
    ]
    summaries = aggregate(Result.from_row(r) for r in rows)
    assert [(s.scenario, s.arm, s.ablation, s.n) for s in summaries] == [
        ("concept_abrupt", "baseline", None, 3),
        ("concept_abrupt", "adaptive", None, 3),
        ("concept_abrupt", "adaptive", "base_rows_0", 1),
    ]
    adaptive = summaries[1]
    assert adaptive.seeds == (1, 2, 3) and adaptive.run_ids == (11, 12, 13)
    assert adaptive.metrics["accuracy_after_drift"].mean == pytest.approx(0.82)
    assert adaptive.metrics["accuracy_after_drift"].sd == pytest.approx(0.02)
    # One seed never recovered: reported as 2 of 3, not averaged away.
    recovery = adaptive.metrics["recovery_batches"]
    assert (recovery.mean, recovery.n, recovery.missing) == (3.0, 2, 1)
    assert set(adaptive.metrics) == set(METRICS)
    single = summaries[2].metrics["accuracy_after_drift"]
    assert single.sd is None and single.fmt(percent=True) == "70.0"


def test_pair_matches_each_adaptive_run_to_the_baseline_of_its_seed():
    rows = [
        _result("concept_abrupt", "baseline", 1, post=0.50, recovery=None),
        _result("concept_abrupt", "baseline", 2, post=0.50, recovery=None),
        _result("concept_abrupt", "adaptive", 1, post=0.80, recovery=2, run_id=11),
        _result("concept_abrupt", "adaptive", 2, post=0.60, recovery=None, run_id=12),
        _result("concept_abrupt", "adaptive", 1, post=0.70, recovery=3, ablation="base_rows_0",
                run_id=21),
        _result("concept_abrupt", "adaptive", 9, post=0.90, recovery=1, run_id=99),  # no baseline
    ]
    pairs = pair(Result.from_row(r) for r in rows)
    assert [(p.seed, p.ablation, p.baseline_run_id, p.adaptive_run_id) for p in pairs] == [
        (1, None, 1, 11), (2, None, 2, 12), (1, "base_rows_0", 1, 21),
    ]
    assert pairs[0].delta_after_drift == pytest.approx(0.30)
    assert pairs[1].delta_after_drift == pytest.approx(0.10)
    assert (pairs[0].recovery_baseline, pairs[0].recovery_adaptive) == (None, 2)
    assert pairs[2].label == "adaptive/base_rows_0"


def test_detector_table_uses_the_baseline_arm_only():
    rows = [
        _result("concept_abrupt", "baseline", 1, post=0.5, recovery=None,
                detectors=[("ddm", 1), ("adwin_error", 3)]),
        _result("concept_abrupt", "baseline", 2, post=0.5, recovery=None,
                detectors=[("ddm", 2), ("adwin_error", 3)]),
        _result("concept_abrupt", "adaptive", 1, post=0.8, recovery=1, run_id=11,
                detectors=[("ddm", 0)]),
        _result("stationary", "baseline", 1, post=0.9, recovery=None, detectors=[]),
    ]
    table = detector_table(Result.from_row(r) for r in rows)
    assert [(d.scenario, d.detector, d.runs) for d in table] == [
        ("concept_abrupt", "adwin_error", 2), ("concept_abrupt", "ddm", 2),
    ]
    ddm = table[1]
    assert ddm.delay.mean == pytest.approx(1.5), "the adaptive run's 0 is not in here"
    assert ddm.missed.mean == 0 and ddm.false_alarms.mean == 0


def test_render_markdown_lists_every_scenario_and_section(tmp_path):
    rows = [
        _result(sc, arm, seed, post=0.5 if arm == "baseline" else 0.8,
                recovery=None if arm == "baseline" else 2, run_id=i * 10 + seed,
                detectors=[("adwin_error", 2)])
        for i, (sc, arm) in enumerate([
            ("stationary", "baseline"), ("stationary", "adaptive"),
            ("concept_abrupt", "baseline"), ("concept_abrupt", "adaptive"),
        ], start=1)
        for seed in (1, 2)
    ]
    rows.append(_result("concept_abrupt", "adaptive", 1, post=0.7, recovery=3,
                        ablation="min_new_500", run_id=91))
    results = [Result.from_row(r) for r in rows]
    summaries, pairs, detectors = aggregate(results), pair(results), detector_table(results)
    text = render_markdown(summaries, pairs, detectors, name="t")

    for heading in ("## Results by scenario", "## Seed-matched A/B", "### Mean Δ",
                    "## Ablations", "## Detectors"):
        assert heading in text
    assert text.count("| stationary |") >= 2 and text.count("| concept_abrupt |") >= 2
    assert "adaptive/min_new_500" in text
    assert "+30.0" in text, "A/B delta in percentage points"
    assert "adwin_error" in text
    assert "| 80.0 ± 0.0 |" in text or "| 80.0 |" in text

    write_csv(tmp_path / "summary.csv", summaries)
    write_pairs_csv(tmp_path / "pairs.csv", pairs)
    with (tmp_path / "summary.csv").open(newline="") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == len(summaries)
    assert table[0]["scenario"] == "stationary" and table[0]["n"] == "2"
    assert float(table[3]["accuracy_after_drift_mean"]) == pytest.approx(0.8)
    with (tmp_path / "pairs.csv").open(newline="") as fh:
        pair_rows = list(csv.DictReader(fh))
    assert len(pair_rows) == 5
    assert pair_rows[0]["recovery_baseline"] == ""  # None → empty, not "None"


def test_replacing_a_field_keeps_from_dict_in_step():
    """Adding a RunScore field must not break reading an older ledger row."""
    s = score(spec=[(0, 0.9)])
    data = s.as_dict()
    del data["scenario"]
    data["some_future_field"] = 1
    back = RunScore.from_dict(data)
    assert back.scenario == "" and dataclasses.replace(back, scenario="x").scenario == "x"
