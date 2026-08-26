"""The upsert statement itself, exercised directly against Postgres.

Spark commits Kafka offsets *after* foreachBatch returns, so a crash between
writing and committing replays the batch. These tests pin the behaviour that
turns that at-least-once delivery into an effectively-once result.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from psycopg2.extras import Json, execute_values

from dashboard.models import ClassifiedMessage, ExperimentRun, ModelVersion
from streamcls.spark.sink import UPSERT_SQL

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]


@pytest.fixture(autouse=True)
def _reference_data():
    """`transaction=True` truncates every table, taking the seed migration's
    Category rows with it. Production always has them (seeded by migration
    0002), and ClassifiedMessage has FKs onto them."""
    from dashboard.models import Category
    from streamcls.labels import LABELS

    for label in LABELS:
        Category.objects.update_or_create(
            id=label.id,
            defaults={"key": label.key, "display_name": label.display_name},
        )


@pytest.fixture
def repo():
    from django.conf import settings as django_settings

    from streamcls.config import PostgresSettings
    from streamcls.db.repo import Repository

    db = django_settings.DATABASES["default"]
    pg = PostgresSettings(
        _env_file=None, host=db["HOST"], port=int(db["PORT"]), db=db["NAME"],
        user=db["USER"], password=db["PASSWORD"],
    )
    with Repository(pg) as repository:
        yield repository


@pytest.fixture
def fixtures():
    run = ExperimentRun.objects.create(name="sink", seed=1, total_messages=10)
    other = ExperimentRun.objects.create(name="sink-2", seed=1, total_messages=10)
    model_a = ModelVersion.objects.create(version="m1", artifact_path="/p1")
    model_b = ModelVersion.objects.create(version="m2", artifact_path="/p2")
    return run, other, model_a, model_b


def _row(run, model, post_id, *, batch_id=0, predicted=2, msg_index=0, correct=True):
    now = datetime.now(UTC)
    return (
        run.pk, post_id, msg_index, "some text", "Market Desk", "market_desk", 10,
        2, predicted, 0.9, Json([0.05, 0.02, 0.9, 0.03]), correct,
        model.pk, batch_id, "none", 0, now, now,
    )


def _write(repo, rows):
    with repo.cursor() as cur:
        execute_values(cur, UPSERT_SQL, rows, page_size=100)


def test_writing_the_same_batch_twice_is_idempotent(repo, fixtures):
    """The exact scenario after a checkpoint replay."""
    run, _, model, _ = fixtures
    rows = [_row(run, model, f"{run.pk}-{i:06d}", msg_index=i) for i in range(50)]
    _write(repo, rows)
    _write(repo, rows)
    assert ClassifiedMessage.objects.filter(experiment_run=run).count() == 50


def test_a_replay_after_a_model_swap_updates_the_prediction(repo, fixtures):
    """`DO NOTHING` would leave the old model's prediction in place while the
    BatchMetric row attributed the batch to the new model -- a silent
    inconsistency in the accuracy accounting."""
    run, _, model_a, model_b = fixtures
    _write(repo, [_row(run, model_a, "r-1", batch_id=5, predicted=2, correct=True)])
    _write(repo, [_row(run, model_b, "r-1", batch_id=6, predicted=3, correct=False)])

    row = ClassifiedMessage.objects.get(post_id="r-1")
    assert row.predicted_label_id == 3
    assert row.model_version_id == model_b.pk
    assert row.batch_id == 6
    assert row.is_correct is False


def test_an_out_of_order_older_batch_cannot_overwrite_a_newer_result(repo, fixtures):
    """The WHERE clause keeps the upsert monotone."""
    run, _, model_a, model_b = fixtures
    _write(repo, [_row(run, model_b, "r-1", batch_id=9, predicted=3)])
    _write(repo, [_row(run, model_a, "r-1", batch_id=2, predicted=0)])

    row = ClassifiedMessage.objects.get(post_id="r-1")
    assert row.batch_id == 9
    assert row.predicted_label_id == 3


def test_the_same_post_id_in_two_runs_is_two_rows(repo, fixtures):
    """The held-out split is replayed for every run. A bare UNIQUE(post_id)
    would make run 2 silently overwrite run 1."""
    run, other, model, _ = fixtures
    _write(repo, [_row(run, model, "1-000001")])
    _write(repo, [_row(other, model, "1-000001")])
    assert ClassifiedMessage.objects.filter(post_id="1-000001").count() == 2


def test_batch_metric_upsert_does_not_double_count_on_replay(repo, fixtures):
    run, _, model, _ = fixtures
    from dashboard.models import BatchMetric

    payload = {
        "experiment_run_id": run.pk,
        "batch_id": 3,
        "model_version_id": model.pk,
        "started_at": datetime.now(UTC),
        "n_received": 500,
        "n_classified": 500,
        "n_correct": 450,
        "accuracy": 0.9,
    }
    repo.write_batch_metric(**payload)
    repo.write_batch_metric(**{**payload, "n_correct": 455, "accuracy": 0.91})

    assert BatchMetric.objects.filter(experiment_run=run, batch_id=3).count() == 1
    assert BatchMetric.objects.get(experiment_run=run, batch_id=3).n_correct == 455


def test_batch_metric_exists_detects_a_replay(repo, fixtures):
    run, _, model, _ = fixtures
    assert repo.batch_metric_exists(run.pk, 7) is False
    repo.write_batch_metric(
        experiment_run_id=run.pk, batch_id=7, model_version_id=model.pk,
        started_at=datetime.now(UTC), n_received=1, n_classified=1,
    )
    assert repo.batch_metric_exists(run.pk, 7) is True


def test_raw_writes_do_not_need_django_level_defaults(repo, fixtures):
    """Django's `default=` is applied in Python, so a raw INSERT that omits a
    NOT NULL column fails unless the column has a *database* default. This is
    how the consumer first died on `ddm_warning`."""
    run, _, model, _ = fixtures
    repo.write_batch_metric(
        experiment_run_id=run.pk, batch_id=11, model_version_id=model.pk,
        started_at=datetime.now(UTC), n_received=10, n_classified=10,
    )
    from dashboard.models import BatchMetric

    row = BatchMetric.objects.get(experiment_run=run, batch_id=11)
    assert row.ddm_warning is False
    assert row.duration_ms == 0
    assert row.n_correct == 0
