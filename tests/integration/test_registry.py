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


def _register(repo, version: str, accuracy: float) -> int:
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
