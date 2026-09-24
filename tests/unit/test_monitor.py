"""The drift monitor: which signals fire, when, and what is allowed to act."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest

from streamcls.drift.monitor import DriftMonitor, MonitorConfig
from streamcls.labels import N_CLASSES


@dataclass
class FakeAggregate:
    """Stands in for a Spark BatchAggregate; the monitor only reads fields."""

    correct_sequence: list[int] = field(default_factory=list)
    confidence_hist: list[int] = field(default_factory=lambda: [0] * 10)
    predicted_hist: list[int] = field(default_factory=lambda: [125] * N_CLASSES)
    feature_sums: list[float] = field(default_factory=list)

    @property
    def n_classified(self) -> int:
        return len(self.correct_sequence)


def batch(error_rate: float, seed: int, n: int = 500, **kw) -> FakeAggregate:
    rng = random.Random(seed)
    return FakeAggregate(
        correct_sequence=[0 if rng.random() < error_rate else 1 for _ in range(n)], **kw
    )


def monitor(**kw) -> DriftMonitor:
    return DriftMonitor(config=MonitorConfig(grace_batches=0, **kw))


# --- error-stream detection ---------------------------------------------------
def test_stationary_batches_do_not_trigger_adaptation():
    """The whole loop depends on this: a detector that fires on noise would
    retrain forever on data that never changed."""
    m = monitor()
    triggered = sum(m.observe(batch(0.10, seed)).should_adapt for seed in range(12))
    assert triggered == 0


def test_ddm_is_recorded_but_may_not_spend_a_training_run():
    """DDM's measured false-alarm rate is 10-47 % on stationary streams, so it
    is reported for comparison and excluded from the actionable set."""
    m = monitor()
    ddm_fired = False
    for seed in range(12):
        result = m.observe(batch(0.10, seed))
        ddm_fired |= any(s.detector == "ddm" for s in result.signals)
        assert not result.should_adapt
    assert ddm_fired, "expected the documented DDM false alarm on a stationary stream"


def test_an_error_spike_triggers_adaptation():
    m = monitor()
    for seed in range(6):
        m.observe(batch(0.10, seed))
    results = [m.observe(batch(0.45, 100 + seed)) for seed in range(4)]
    assert any(r.should_adapt for r in results)


def test_the_actionable_set_is_an_experiment_variable():
    """The scenario x detector table widens the actionable set to DDM to
    measure what its false alarms would have cost. The same stationary stream
    that never triggers by default must trigger once DDM is allowed to act."""
    permissive = monitor(actionable_detectors=frozenset({"ddm", "adwin_error"}))
    strict = monitor()
    permissive_fired = strict_fired = False
    for seed in range(12):
        permissive_fired |= permissive.observe(batch(0.10, seed)).should_adapt
        strict_fired |= strict.observe(batch(0.10, seed)).should_adapt
    assert permissive_fired and not strict_fired


def test_an_empty_actionable_set_records_everything_and_acts_on_nothing():
    m = monitor(actionable_detectors=frozenset())
    for seed in range(6):
        m.observe(batch(0.10, seed))
    results = [m.observe(batch(0.45, 100 + seed)) for seed in range(4)]
    assert any(r.unsuppressed for r in results)
    assert not any(r.should_adapt for r in results)


def test_the_firing_detector_is_named():
    m = monitor()
    for seed in range(6):
        m.observe(batch(0.08, seed))
    for seed in range(4):
        result = m.observe(batch(0.50, 200 + seed))
        if result.should_adapt:
            assert {s.detector for s in result.actionable} & {"ddm", "adwin_error"}
            return
    pytest.fail("no detection")


def test_one_detector_firing_many_times_in_a_batch_is_one_signal():
    """500 messages must not produce 500 events, or the cooldown would be the
    only thing standing between drift and a retrain storm."""
    m = monitor()
    for seed in range(6):
        m.observe(batch(0.05, seed))
    for seed in range(5):
        result = m.observe(batch(0.60, 300 + seed))
        names = [s.detector for s in result.signals]
        assert len(names) == len(set(names))


# --- grace window --------------------------------------------------------------
def test_signals_inside_the_grace_window_are_suppressed_but_recorded():
    """After a start or a model swap the detectors are cold; acting on that
    would retrain immediately on nothing."""
    m = DriftMonitor(config=MonitorConfig(grace_batches=5))
    fired = []
    for seed in range(5):
        result = m.observe(batch(0.60, seed))
        fired.extend(result.signals)
    assert m.in_grace is False  # exhausted after 5 batches
    if fired:
        assert all(s.suppressed for s in fired)
        assert not any(r for r in fired if not r.suppressed)


def test_suppressed_signals_do_not_trigger_adaptation():
    m = DriftMonitor(config=MonitorConfig(grace_batches=10))
    results = [m.observe(batch(0.70, seed)) for seed in range(8)]
    assert not any(r.should_adapt for r in results)


# --- reset ---------------------------------------------------------------------
def test_reset_stops_a_single_drift_firing_forever():
    """Without a reset after adapting, every later batch would queue a retrain."""
    m = monitor()
    for seed in range(6):
        m.observe(batch(0.08, seed))
    for seed in range(6):
        if m.observe(batch(0.55, 400 + seed)).should_adapt:
            m.reset(grace=False)
            break
    else:
        pytest.fail("no detection")
    # The elevated rate is now the new normal, so it must stop being drift.
    later = [m.observe(batch(0.55, 500 + seed)).should_adapt for seed in range(6)]
    assert sum(later) <= 1


# --- label-free statistics ------------------------------------------------------
def test_label_free_signals_alone_never_trigger_a_retrain():
    """A covariate shift that does not hurt accuracy is not worth a training
    run, and vocabulary PSI over sparse bins is noisy."""
    m = monitor(psi_consecutive_batches=1)
    skewed = FakeAggregate(
        correct_sequence=[1] * 500,          # perfect accuracy: no error signal
        predicted_hist=[500, 0, 0, 0],       # but a wildly skewed prediction mix
    )
    result = m.observe(skewed)
    assert any(s.detector == "psi_predicted" for s in result.signals)
    assert result.should_adapt is False


def test_vocabulary_psi_requires_a_sustained_shift():
    """One spike over a 51-bin vocabulary is noise; three in a row is a shift."""
    reference = [0.5, 0.3, 0.2]
    m = DriftMonitor(
        config=MonitorConfig(grace_batches=0, psi_window_batches=1, psi_consecutive_batches=3),
        reference_histogram=reference,
        top_term_indices=[0, 1],
    )
    shifted = FakeAggregate(correct_sequence=[1] * 100, feature_sums=[1.0, 90.0, 9.0])
    fired = [
        any(s.detector == "psi_vocabulary" for s in m.observe(shifted).signals)
        for _ in range(4)
    ]
    assert fired == [False, False, True, False]


def test_confidence_psi_needs_a_matching_reference():
    """A reference of the wrong length means the bins do not line up; reporting
    a number anyway would be worse than reporting nothing."""
    m = monitor()
    aggregate = FakeAggregate(correct_sequence=[1] * 10, confidence_hist=[1] * 10)
    assert m.observe(aggregate).stats["psi_confidence"] is None

    m2 = DriftMonitor(
        config=MonitorConfig(grace_batches=0), confidence_reference=[0.1] * 10
    )
    assert m2.observe(aggregate).stats["psi_confidence"] is not None


def test_a_mismatched_vocabulary_reports_nothing_rather_than_nonsense():
    m = DriftMonitor(
        config=MonitorConfig(grace_batches=0, psi_window_batches=1),
        reference_histogram=[0.5, 0.5],
        top_term_indices=[0, 99],  # index beyond the observed vector
    )
    aggregate = FakeAggregate(correct_sequence=[1] * 10, feature_sums=[1.0, 2.0])
    assert m.observe(aggregate).stats["psi_vocabulary"] is None


# --- replay --------------------------------------------------------------------
def test_rebuild_reproduces_state_from_stored_bits():
    """This is what makes a consumer restart faithful without pickling
    detectors whose internals are not a stable contract."""
    from streamcls.spark.batch import pack_correct_bits

    batches = [batch(0.12, seed) for seed in range(8)]

    live = monitor()
    for b in batches:
        live.observe(b)

    rebuilt = monitor()
    rebuilt.rebuild(
        [
            {"correct_bits": pack_correct_bits(b.correct_sequence), "n_classified": len(b.correct_sequence)}
            for b in batches
        ]
    )
    assert rebuilt._ddm.statistic == pytest.approx(live._ddm.statistic)
    assert rebuilt._adwin.statistic == pytest.approx(live._adwin.statistic)
    assert rebuilt.batches_seen == live.batches_seen


def test_replayed_batches_do_not_feed_the_detectors_twice():
    """The sink is idempotent; the detectors are not."""
    m = monitor()
    b = batch(0.2, seed=1)
    m.observe(b)
    before = m._ddm.statistic
    result = m.observe(b, apply=False)
    assert m._ddm.statistic == pytest.approx(before)
    assert result.signals == ()
