"""The model registry and its promotion guard.

The guard matters more than it looks: without it, a retrain on an unlucky
window produces a worse model, the system promotes it anyway, accuracy drops,
drift fires again, and the loop oscillates -- which would show up in the thesis
as adaptation actively harming performance.
"""

from __future__ import annotations

import pytest

from dashboard.models import ModelStatus, ModelVersion, StreamStatus

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]


@pytest.fixture(autouse=True)
def _singleton_exists():
    """`transaction=True` truncates tables between tests, taking the migration's
    singleton row with it. Production always has it (the migration creates it,
    and a CheckConstraint keeps it unique), so restore it here."""
    StreamStatus.load()


@pytest.fixture
def repo():
    from django.conf import settings as django_settings

    from streamcls.config import PostgresSettings
    from streamcls.db.repo import Repository

    db = django_settings.DATABASES["default"]
    pg = PostgresSettings(
        _env_file=None,
        host=db["HOST"], port=int(db["PORT"]), db=db["NAME"],
        user=db["USER"], password=db["PASSWORD"],
    )
    with Repository(pg) as repository:
        yield repository


def _register(repo, version: str, accuracy: float, recent: float | None = None) -> int:
    return repo.register_model(
        version=version,
        artifact_path=f"/data/models/{version}",
        algorithm="logreg",
        hyperparams={"reg_param": 0.05},
        metrics={"test": {"accuracy": accuracy}},
        test_accuracy=accuracy,
        test_macro_f1=accuracy,
        train_rows=1000,
        vocab_size=20_000,
        top_term_indices=[1, 2, 3],
        reference_histogram=[0.5, 0.3, 0.2],
        confidence_reference=[0.1] * 10,
        recent_accuracy=recent,
    )


def test_registration_lands_as_ready_never_active(repo):
    """Finishing training must not be enough to take over serving."""
    model_id = _register(repo, "v1", 0.90)
    row = ModelVersion.objects.get(pk=model_id)
    assert row.status == ModelStatus.READY
    assert row.test_accuracy == pytest.approx(0.90)
    assert row.top_term_indices == [1, 2, 3]
    assert row.reference_histogram == [0.5, 0.3, 0.2]
    assert row.metrics["test"]["accuracy"] == pytest.approx(0.90)


def test_first_promotion_activates_and_signals_the_consumer(repo):
    model_id = _register(repo, "v1", 0.90)
    assert repo.promote_model(model_id) is True

    row = ModelVersion.objects.get(pk=model_id)
    assert row.status == ModelStatus.ACTIVE
    assert row.activated_at is not None
    # The whole hot-reload signal: the consumer's control poll sees
    # desired != active and restarts the query with the new model.
    assert StreamStatus.load().desired_model_version_id == model_id


def test_promotion_retires_the_incumbent_atomically(repo):
    old = _register(repo, "v1", 0.90)
    repo.promote_model(old)
    new = _register(repo, "v2", 0.93)
    assert repo.promote_model(new) is True

    assert ModelVersion.objects.get(pk=old).status == ModelStatus.RETIRED
    assert ModelVersion.objects.get(pk=new).status == ModelStatus.ACTIVE
    # The single-active constraint is real, so retire-then-activate must be one
    # transaction or the intermediate state would violate it.
    assert ModelVersion.objects.filter(status=ModelStatus.ACTIVE).count() == 1


def test_a_clearly_worse_model_is_rejected(repo):
    good = _register(repo, "v1", 0.90)
    repo.promote_model(good)
    bad = _register(repo, "v2", 0.70)

    assert repo.promote_model(bad, min_delta=0.01) is False
    assert ModelVersion.objects.get(pk=good).status == ModelStatus.ACTIVE
    assert ModelVersion.objects.get(pk=bad).status == ModelStatus.READY


def test_a_marginally_worse_model_within_the_tolerance_is_accepted(repo):
    """Adaptation should not be blocked by ordinary run-to-run noise."""
    good = _register(repo, "v1", 0.900)
    repo.promote_model(good)
    marginal = _register(repo, "v2", 0.895)

    assert repo.promote_model(marginal, min_delta=0.01) is True
    assert ModelVersion.objects.get(pk=marginal).status == ModelStatus.ACTIVE


def test_promoting_the_already_active_model_is_a_no_op(repo):
    model_id = _register(repo, "v1", 0.90)
    repo.promote_model(model_id)
    assert repo.promote_model(model_id) is True
    assert ModelVersion.objects.filter(status=ModelStatus.ACTIVE).count() == 1


def test_get_active_model_round_trips_the_drift_reference(repo):
    """The consumer needs the reference histogram to compute PSI, and it must
    be the one belonging to the model actually serving."""
    model_id = _register(repo, "v1", 0.90)
    repo.promote_model(model_id)

    active = repo.get_active_model()
    assert active is not None
    assert active.id == model_id
    assert active.version == "v1"
    assert active.reference_histogram == [0.5, 0.3, 0.2]
    assert active.top_term_indices == [1, 2, 3]
    # The confidence reference must travel with the model too: comparing a
    # batch against a uniform profile measures calibration, not drift.
    assert active.confidence_reference == [0.1] * 10
    assert active.vocab_size == 20_000


def test_no_active_model_returns_none_rather_than_raising(repo):
    _register(repo, "v1", 0.9)
    assert repo.get_active_model() is None


def test_unknown_model_raises(repo):
    with pytest.raises(LookupError, match="not found"):
        repo.get_model(9_999_999)
    with pytest.raises(LookupError, match="not found"):
        repo.promote_model(9_999_999)


def test_transaction_helper_rolls_back(repo):
    """promote_model relies on this; autocommit is normally on, where
    psycopg2's own `with connection:` does not open a transaction at all."""
    _register(repo, "v1", 0.9)
    before = ModelVersion.objects.count()
    with pytest.raises(RuntimeError), repo.transaction() as cur:
        cur.execute("UPDATE model_version SET status = 'retired'")
        raise RuntimeError("boom")
    assert ModelVersion.objects.count() == before
    assert ModelVersion.objects.get(version="v1").status == ModelStatus.READY


def test_promotion_compares_recent_scores_when_both_have_them(repo):
    """Under concept drift the fixed test set still encodes the old labelling,
    so a correctly adapted model scores worse on it. Comparing on it would
    reject every adaptation and make the loop structurally useless."""
    old = _register(repo, "v1", accuracy=0.90, recent=0.55)
    repo.promote_model(old)
    # Worse on the stale test set, much better on current data.
    new = _register(repo, "v2", accuracy=0.72, recent=0.87)

    assert repo.promote_model(new, min_delta=0.01) is True
    assert ModelVersion.objects.get(pk=new).status == ModelStatus.ACTIVE


def test_force_resets_to_a_worse_model_but_still_retires_the_incumbent(repo):
    """Between experiment cells the base model must come back regardless of
    how it scores against the retrained incumbent. Whether the guard would let
    it back depends on which scores happen to exist (a base model with no
    recent score is compared on the stale test set, where it wins); a reset
    must not depend on that, so it bypasses the guard. It must not bypass the
    single-active invariant."""
    base = _register(repo, "base", accuracy=0.90, recent=0.60)
    repo.promote_model(base)
    retrained = _register(repo, "retrained", accuracy=0.70, recent=0.95)
    assert repo.promote_model(retrained) is True
    assert repo.promote_model(base, min_delta=0.01) is False

    assert repo.promote_model(base, force=True) is True
    assert ModelVersion.objects.get(pk=base).status == ModelStatus.ACTIVE
    assert ModelVersion.objects.get(pk=retrained).status == ModelStatus.RETIRED
    assert ModelVersion.objects.filter(status=ModelStatus.ACTIVE).count() == 1
    assert StreamStatus.load().desired_model_version_id == base


def test_promotion_falls_back_to_measured_live_accuracy(repo):
    """The usual case: the incumbent was trained offline and has no recent
    score, so its measured live accuracy over recent batches stands in."""
    from datetime import UTC, datetime

    from dashboard.models import BatchMetric, ExperimentRun

    old = _register(repo, "v1", accuracy=0.91)
    repo.promote_model(old)
    run = ExperimentRun.objects.create(name="r", seed=1, total_messages=10)
    for batch_id in range(3):
        BatchMetric.objects.create(
            experiment_run=run, batch_id=batch_id, model_version_id=old,
            started_at=datetime.now(UTC), n_received=500, n_classified=500, n_correct=250,
        )
    assert repo.live_accuracy() == pytest.approx(0.5)

    new = _register(repo, "v2", accuracy=0.70, recent=0.80)
    assert repo.promote_model(new, min_delta=0.01) is True


def test_live_accuracy_can_be_scoped_to_the_current_concept(repo):
    """Drift is detected within a batch or two, so an unscoped window is mostly
    pre-drift data and reports the incumbent as far healthier than it is."""
    from datetime import UTC, datetime

    from dashboard.models import BatchMetric, ExperimentRun

    model_id = _register(repo, "v1", accuracy=0.9)
    run = ExperimentRun.objects.create(name="r", seed=1, total_messages=10)
    for batch_id, correct in enumerate([450, 450, 450, 250, 250]):
        BatchMetric.objects.create(
            experiment_run=run, batch_id=batch_id, model_version_id=model_id,
            started_at=datetime.now(UTC), n_received=500, n_classified=500, n_correct=correct,
        )
    assert repo.live_accuracy(run_id=run.pk) == pytest.approx((450 * 3 + 250 * 2) / 2500)
    assert repo.live_accuracy(run_id=run.pk, since_batch=3) == pytest.approx(0.5)


def test_every_guarded_decision_is_recorded_on_the_candidate(repo):
    """Accepted or rejected, the scores and the rule survive for the analysis."""
    good = _register(repo, "v1", 0.90)
    repo.promote_model(good)
    bad = _register(repo, "v2", 0.70)
    assert repo.promote_model(bad, min_delta=0.01) is False

    record = ModelVersion.objects.get(pk=bad).metrics["promotion"]
    assert record["promoted"] is False
    assert record["rule"] == "test"
    assert record["candidate_score"] == pytest.approx(0.70)
    assert record["incumbent_score"] == pytest.approx(0.90)
    assert record["incumbent_id"] == good
    assert record["margin"] == pytest.approx(0.70 - 0.89)
    # the existing metrics are kept, the decision is merged in
    assert ModelVersion.objects.get(pk=bad).metrics["test"]["accuracy"] == pytest.approx(0.70)


def test_a_forced_reset_records_no_decision(repo):
    first = _register(repo, "v1", 0.90)
    repo.promote_model(first)
    second = _register(repo, "v2", 0.50)
    assert repo.promote_model(second, force=True) is True
    assert "promotion" not in ModelVersion.objects.get(pk=second).metrics


def test_the_matched_holdout_rule_compares_on_the_same_messages(repo):
    """The incumbent's live accuracy would let this candidate through; on the
    candidate's own holdout the incumbent is better, so the matched rule refuses."""
    incumbent = _register(repo, "v1", 0.90)
    repo.promote_model(incumbent)
    candidate = _register(repo, "v2", 0.80, recent=0.62)

    assert repo.promote_model(candidate, min_delta=0.01, incumbent_score=0.70) is False
    record = ModelVersion.objects.get(pk=candidate).metrics["promotion"]
    assert record["rule"] == "matched_holdout"
    assert record["incumbent_score"] == pytest.approx(0.70)
    assert record["candidate_score"] == pytest.approx(0.62)

    assert repo.promote_model(candidate, min_delta=0.01, incumbent_score=0.625) is True
    assert ModelVersion.objects.get(pk=candidate).status == ModelStatus.ACTIVE


def test_annotate_merges_top_level_keys(repo):
    model_id = _register(repo, "v1", 0.90)
    repo.annotate_model(model_id, timings={"corpus_s": 1.5}, incumbent_holdout_accuracy=0.5)
    metrics = ModelVersion.objects.get(pk=model_id).metrics
    assert metrics["timings"] == {"corpus_s": 1.5}
    assert metrics["incumbent_holdout_accuracy"] == pytest.approx(0.5)
    assert metrics["test"]["accuracy"] == pytest.approx(0.90)


def test_an_orphaned_running_job_is_requeued_after_its_lease(repo):
    """A trainer killed mid-job must not block the run forever."""
    from dashboard.models import ExperimentRun, TrainingJob

    run = ExperimentRun.objects.create(name="r", seed=1, total_messages=10)
    job = TrainingJob.objects.create(experiment_run=run, status="running")
    TrainingJob.objects.filter(pk=job.pk).update(started_at="2020-01-01T00:00:00Z")
    fresh = TrainingJob.objects.create(experiment_run=run, status="running")
    TrainingJob.objects.filter(pk=fresh.pk).update(started_at="2999-01-01T00:00:00Z")

    assert repo.requeue_stale_jobs(lease_s=120) == [job.pk]
    job.refresh_from_db()
    assert job.status == "queued"
    assert job.started_at is None
    assert TrainingJob.objects.get(pk=fresh.pk).status == "running"


def test_a_job_failed_by_the_trainer_is_requeued_once(repo):
    """A dead or out-of-memory JVM is the trainer's failure, not the job's: the
    job goes back to the queue once, and a second such failure is final, so a
    job that itself breaks the JVM cannot loop forever."""
    from dashboard.models import ExperimentRun, TrainingJob

    run = ExperimentRun.objects.create(name="r", seed=1, total_messages=10)
    job = TrainingJob.objects.create(experiment_run=run, status="running")

    assert repo.requeue_training_job(job.pk, reason="ConnectionRefusedError") is True
    job.refresh_from_db()
    assert job.status == "queued"
    assert job.started_at is None
    assert job.error.startswith("infrastructure:")

    TrainingJob.objects.filter(pk=job.pk).update(status="running")
    assert repo.requeue_training_job(job.pk, reason="ConnectionRefusedError") is False
    job.refresh_from_db()
    assert job.status == "running"
