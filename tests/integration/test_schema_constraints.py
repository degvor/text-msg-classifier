"""The database constraints that protect the experimental results.

Each test here corresponds to a way the previous design lost or corrupted data
silently. They need a real Postgres because the guarantees are enforced by
Postgres, not by Django.
"""

from __future__ import annotations

import pytest
from django.db import IntegrityError, connection, transaction
from django.utils import timezone

from dashboard.models import (
    BatchMetric,
    Category,
    ClassifiedMessage,
    ExperimentRun,
    ModelStatus,
    ModelVersion,
    StreamConfig,
    StreamStatus,
)
from streamcls.db.tables import Table
from streamcls.labels import LABELS

pytestmark = [pytest.mark.integration, pytest.mark.django_db]


@pytest.fixture
def model_version():
    return ModelVersion.objects.create(
        version="test-v1", artifact_path="/data/models/test-v1", test_accuracy=0.91
    )


@pytest.fixture
def run():
    return ExperimentRun.objects.create(name="run A", seed=42)


def _message(run, model_version, post_id, msg_index=1):
    now = timezone.now()
    return ClassifiedMessage.objects.create(
        experiment_run=run,
        post_id=post_id,
        msg_index=msg_index,
        post_text="Oil prices climb",
        true_label_id=2,
        predicted_label_id=2,
        confidence=0.87,
        is_correct=True,
        model_version=model_version,
        batch_id=0,
        produced_at=now,
        classified_at=now,
    )


# ---------------------------------------------------------------------------
def test_tables_use_pinned_names():
    """Spark and psycopg2 address these by name; a Django rename must not move them."""
    with connection.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        present = {row[0] for row in cur.fetchall()}
    missing = {t.value for t in Table} - present
    assert not missing, f"tables missing or renamed: {sorted(missing)}"
    assert not {t for t in present if t.startswith("dashboard_")}


def test_categories_are_seeded_from_the_pinned_label_map():
    assert Category.objects.count() == len(LABELS)
    assert dict(Category.objects.values_list("id", "key")) == {lab.id: lab.key for lab in LABELS}


# --- the trap --------------------------------------------------------------
def test_same_post_id_is_allowed_across_runs(run, model_version):
    """The held-out split is replayed for every run.

    With a bare ``UNIQUE(post_id)`` the second run would either overwrite the
    first or write nothing at all, invisibly invalidating both.
    """
    other = ExperimentRun.objects.create(name="run B", seed=42)
    _message(run, model_version, "1-000001")
    _message(other, model_version, "1-000001")
    assert ClassifiedMessage.objects.filter(post_id="1-000001").count() == 2


def test_duplicate_post_id_within_a_run_is_rejected(run, model_version):
    """This uniqueness is what makes the foreachBatch upsert idempotent, so a
    Spark checkpoint replay cannot double-count a batch."""
    _message(run, model_version, "1-000001")
    with pytest.raises(IntegrityError), transaction.atomic():
        _message(run, model_version, "1-000001", msg_index=2)


def test_batch_metric_is_unique_per_run(run, model_version):
    """Without this, cumulative accuracy drifts on every consumer restart."""
    kwargs = {
        "experiment_run": run,
        "batch_id": 7,
        "model_version": model_version,
        "started_at": timezone.now(),
        "n_received": 10,
        "n_classified": 10,
    }
    BatchMetric.objects.create(**kwargs)
    with pytest.raises(IntegrityError), transaction.atomic():
        BatchMetric.objects.create(**kwargs)


# --- invariants ------------------------------------------------------------
def test_classified_cannot_exceed_received(run, model_version):
    """The old metrics incremented both counters by the same value, so the
    drop count was structurally unobservable."""
    with pytest.raises(IntegrityError), transaction.atomic():
        BatchMetric.objects.create(
            experiment_run=run,
            batch_id=1,
            model_version=model_version,
            started_at=timezone.now(),
            n_received=10,
            n_classified=11,
        )


def test_only_one_model_version_can_be_active(model_version):
    model_version.status = ModelStatus.ACTIVE
    model_version.save()
    with pytest.raises(IntegrityError), transaction.atomic():
        ModelVersion.objects.create(
            version="test-v2", artifact_path="/p", status=ModelStatus.ACTIVE
        )


def test_several_ready_versions_are_fine(model_version):
    ModelVersion.objects.create(version="test-v2", artifact_path="/p", status=ModelStatus.READY)
    ModelVersion.objects.create(version="test-v3", artifact_path="/p", status=ModelStatus.READY)
    assert ModelVersion.objects.filter(status=ModelStatus.READY).count() == 2


@pytest.mark.parametrize("model", [StreamConfig, StreamStatus])
def test_control_tables_are_singletons(model):
    """Two rows are how Django (.first()) and Spark (ORDER BY id DESC) came to
    disagree about which one was authoritative."""
    assert model.objects.count() == 1
    with pytest.raises(IntegrityError), transaction.atomic():
        model.objects.create(pk=2)


def test_singleton_save_always_targets_pk_1():
    cfg = StreamConfig.load()
    cfg.pk = 99
    cfg.max_message_length = 1234
    cfg.save()
    assert StreamConfig.objects.count() == 1
    assert StreamConfig.load().max_message_length == 1234
