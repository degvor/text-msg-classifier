"""The stream's guarantees are what make two experiment runs comparable."""

from collections import Counter

import pytest

from streamcls.corpus.agnews import Document
from streamcls.corpus.schedule import DriftSchedule
from streamcls.corpus.stream import CHANNELS, SeededStream
from streamcls.labels import N_CLASSES


@pytest.fixture
def pool():
    # Distinct, label-identifiable texts, so a message's origin pool is
    # recoverable by inspection.
    return [
        Document(f"label{label} document {i} about something newsworthy today", label)
        for label in range(N_CLASSES)
        for i in range(50)
    ]


def make(pool, *, seed=1234, run_id=7, total=400, schedule=None, **kw):
    return SeededStream(
        documents=pool,
        schedule=schedule or DriftSchedule(base_intensity=0.0),
        seed=seed,
        run_id=run_id,
        total_messages=total,
        **kw,
    )


def test_emits_exactly_total_messages_with_dense_indices(pool):
    msgs = list(make(pool, total=250))
    assert len(msgs) == 250
    assert [m.msg_index for m in msgs] == list(range(250))


def test_post_ids_are_run_scoped_and_unique(pool):
    """A bare post_id would collide when the holdout is replayed for run 2."""
    msgs = list(make(pool, run_id=3, total=100))
    assert msgs[42].post_id == "3-000042"
    assert len({m.post_id for m in msgs}) == 100
    other = list(make(pool, run_id=4, total=100))
    assert {m.post_id for m in msgs}.isdisjoint({m.post_id for m in other})


def test_same_seed_reproduces_the_stream_exactly(pool):
    """The headline A/B figure is only valid if this holds."""
    a = [(m.text, m.true_label, m.msg_index) for m in make(pool)]
    b = [(m.text, m.true_label, m.msg_index) for m in make(pool)]
    assert a == b


def test_different_seed_gives_a_different_stream(pool):
    a = [m.text for m in make(pool, seed=1)]
    b = [m.text for m in make(pool, seed=2)]
    assert a != b


def test_changing_intensity_does_not_disturb_the_label_sequence(pool):
    """One RNG per concern. With a shared generator, turning the perturbation
    knob would shift every later draw and an ablation would silently be
    comparing two different streams."""
    calm = [m.true_label for m in make(pool, schedule=DriftSchedule(base_intensity=0.0))]
    noisy = [m.true_label for m in make(pool, schedule=DriftSchedule(base_intensity=0.9))]
    assert calm == noisy


def test_prior_shift_changes_the_observed_class_mix(pool):
    sched = DriftSchedule(
        events=[{"at": 1000, "kind": "prior_shift", "params": {"weights": [0.7, 0.1, 0.1, 0.1]}}],
        base_intensity=0.0,
    )
    msgs = list(make(pool, total=3000, schedule=sched))
    before = Counter(m.true_label for m in msgs[:1000])
    after = Counter(m.true_label for m in msgs[1000:])
    assert 0.20 < before[0] / 1000 < 0.30
    assert 0.62 < after[0] / 2000 < 0.78


def test_concept_drift_gives_a_class_another_class_text(pool):
    """P(y|x) changes: the label stays, the text comes from a different pool.
    This is what the error detectors must catch while vocabulary PSI stays flat.
    """
    sched = DriftSchedule(
        events=[{"at": 500, "kind": "concept_abrupt", "params": {"map": {"2": 3}}}],
        base_intensity=0.0,
    )
    msgs = list(make(pool, total=1500, schedule=sched))
    before = [m for m in msgs[:500] if m.true_label == 2]
    after = [m for m in msgs[500:] if m.true_label == 2]
    assert before and after
    assert all(m.text.startswith("label2") for m in before)
    assert all(m.text.startswith("label3") for m in after)
    # Everything else keeps its own text.
    assert all(m.text.startswith("label1") for m in msgs[500:] if m.true_label == 1)


def test_ground_truth_fields_are_carried_for_scoring(pool):
    sched = DriftSchedule(
        events=[{"at": 200, "kind": "covariate_abrupt", "params": {"to": 0.9}}],
        base_intensity=0.1,
    )
    msgs = list(make(pool, total=400, schedule=sched))
    assert [m.msg_index for m in msgs if m.is_drift_point] == [200]
    assert msgs[100].drift_phase == 0
    assert msgs[300].drift_phase == 1
    assert msgs[100].perturbation_intensity == pytest.approx(0.1)
    assert msgs[300].perturbation_intensity == pytest.approx(0.9)


def test_view_counts_are_never_negative(pool):
    """The old generator computed base + randint(-100, 500) with base as low as
    80, writing negative counts into an unvalidated IntegerField."""
    assert min(m.view_count for m in make(pool, total=2000)) >= 0


def test_channel_is_assigned_per_class(pool):
    for msg in make(pool, total=200):
        handle, title = CHANNELS[msg.true_label]
        assert (msg.channel_id, msg.channel_title) == (handle, title)


def test_produced_at_is_monotone(pool):
    stamps = [m.produced_at for m in make(pool, total=100)]
    assert stamps == sorted(stamps)


def test_pool_wrapping_is_reported(pool):
    stream = make(pool, total=10)
    assert stream.wraps_at() == dict.fromkeys(range(N_CLASSES), 50)
    assert stream.describe()["change_points"] == []


def test_a_pool_missing_a_class_is_rejected(pool):
    with pytest.raises(ValueError, match="no documents for label"):
        make([d for d in pool if d.label != 3])


def test_empty_pool_and_zero_length_are_rejected(pool):
    with pytest.raises(ValueError, match="empty"):
        make([])
    with pytest.raises(ValueError, match="positive"):
        make(pool, total=0)
