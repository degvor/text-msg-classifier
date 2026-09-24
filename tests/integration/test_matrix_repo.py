"""The repository surface the unattended matrix runner drives a cell through.

The runner has no Django, so every row it writes lands via psycopg2 with the
columns it names and nothing else. These tests read the rows back through the
ORM to pin that the two agree -- a Python ``default=`` the ORM would have
filled in is exactly what psycopg2 does not.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dashboard.models import (
    BatchMetric,
    ExperimentRun,
    RunStatus,
    StreamConfig,
    StreamMode,
    StreamStatus,
    TrainingJob,
)

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]


@pytest.fixture(autouse=True)
def _singletons_exist():
    """`transaction=True` truncates tables between tests, taking the migration's
    singleton rows with it. Production always has them."""
    StreamStatus.load()
    StreamConfig.load()


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


def _create(repo, **overrides) -> int:
    fields = {
        "name": "thesis-v1/concept_abrupt/adaptive/seed1",
        "scenario": "concept_abrupt",
        "seed": 1,
        "mode": "replay",
        "adaptation_enabled": True,
        "total_messages": 2000,
        "drift_schedule": [{"at": 1000, "kind": "concept_abrupt", "params": {"map": {"2": 3}}}],
        "dataset_split": "agnews-stream-holdout-v1",
        "base_model_version_id": None,
        "consumer_config": {"matrix": "thesis-v1", "arm": "adaptive"},
        "producer_config": {},
        "git_commit": "abc1234",
        "topic_for_run": lambda run_id: f"posts.run-{run_id}",
    }
    fields.update(overrides)
    return repo.create_run(**fields)


# --- create_run ------------------------------------------------------------


def test_create_run_round_trips_through_the_orm(repo):
    run_id = _create(repo)
    run = ExperimentRun.objects.get(pk=run_id)

    assert run.name == "thesis-v1/concept_abrupt/adaptive/seed1"
    assert run.scenario == "concept_abrupt"
    assert run.seed == 1
    assert run.mode == StreamMode.REPLAY
    assert run.status == RunStatus.DRAFT
    assert run.adaptation_enabled is True
    assert run.total_messages == 2000
    assert run.drift_schedule == [
        {"at": 1000, "kind": "concept_abrupt", "params": {"map": {"2": 3}}}
    ]
    assert run.dataset_split == "agnews-stream-holdout-v1"
    assert run.git_commit == "abc1234"
    # Columns the INSERT did not name must still be populated: this is the
    # ``db_default`` rule from CLAUDE.md, now applying to this table too.
    assert run.notes == ""
    assert run.rate_msgs_per_sec == pytest.approx(50.0)
    assert run.producer_config == {}
    assert run.created_at is not None
    assert run.started_at is None and run.ended_at is None


def test_create_run_records_its_own_topic_in_the_same_transaction(repo):
    """The topic is named after the id, which exists only after the INSERT;
    the caller's snapshot must survive the merge."""
    run_id = _create(repo)
    run = ExperimentRun.objects.get(pk=run_id)
    assert run.consumer_config == {
        "matrix": "thesis-v1",
        "arm": "adaptive",
        "topic": f"posts.run-{run_id}",
    }


def test_create_run_without_a_topic_factory_leaves_the_snapshot_alone(repo):
    run_id = _create(repo, topic_for_run=None, consumer_config={"k": 1})
    assert ExperimentRun.objects.get(pk=run_id).consumer_config == {"k": 1}


def test_create_run_can_pin_a_base_model(repo):
    from dashboard.models import ModelVersion

    model = ModelVersion.objects.create(
        version="base", artifact_path="/data/models/base", algorithm="logreg"
    )
    run_id = _create(repo, base_model_version_id=model.pk)
    assert ExperimentRun.objects.get(pk=run_id).base_model_version_id == model.pk


def test_the_consumer_can_load_a_runner_created_run(repo):
    """The other side of the contract: the row the runner writes is the row
    the producer and consumer read via ``load_run``."""
    run_id = _create(repo)
    spec = repo.load_run(run_id)
    assert spec.status == "draft"
    assert spec.seed == 1
    assert spec.total_messages == 2000
    assert spec.drift_schedule[0]["kind"] == "concept_abrupt"


# --- find_runs_by_name -----------------------------------------------------


def test_runs_are_found_by_name_oldest_first(repo):
    first = _create(repo)
    second = _create(repo)
    _create(repo, name="something else")

    found = repo.find_runs_by_name("thesis-v1/concept_abrupt/adaptive/seed1")
    assert [r["id"] for r in found] == [first, second]
    assert found[0]["status"] == "draft"
    assert found[0]["consumer_config"]["topic"] == f"posts.run-{first}"
    assert repo.find_runs_by_name("never created") == []


# --- update_config ---------------------------------------------------------


def test_update_config_writes_scalars_and_json(repo):
    repo.update_config(
        trigger_seconds=5,
        retrain_min_new_rows=500,
        actionable_detectors=["ddm", "adwin_error"],
    )
    config = StreamConfig.load()
    assert config.trigger_seconds == 5
    assert config.retrain_min_new_rows == 500
    assert config.actionable_detectors == ["ddm", "adwin_error"]
    # And the consumer sees the same thing through its own read path.
    assert repo.read_config()["actionable_detectors"] == ["ddm", "adwin_error"]


def test_update_config_rejects_an_unknown_column(repo):
    """A misspelt ablation knob must fail loudly, not run the default config
    under the ablation's name."""
    before = StreamConfig.load().trigger_seconds
    with pytest.raises(ValueError, match="retrain_min_rows"):
        repo.update_config(trigger_seconds=3, retrain_min_rows=1)
    assert StreamConfig.load().trigger_seconds == before


def test_update_config_refuses_the_bookkeeping_columns(repo):
    with pytest.raises(ValueError, match="id"):
        repo.update_config(id=2)
    with pytest.raises(ValueError, match="updated_at"):
        repo.update_config(updated_at=None)


def test_update_config_with_nothing_is_a_no_op(repo):
    stamp = StreamConfig.load().updated_at
    repo.update_config()
    assert StreamConfig.load().updated_at == stamp


# --- activate_run ----------------------------------------------------------


def test_activate_run_mirrors_the_dashboard_endpoint(repo):
    run_id = _create(repo)
    StreamStatus.objects.filter(pk=1).update(last_error="stale failure from run 3")

    repo.activate_run(run_id, producer=True, consumer=False)

    status = StreamStatus.load()
    assert status.active_experiment_run_id == run_id
    assert status.producer_active is True
    assert status.is_active is False
    assert status.last_error == ""

    repo.activate_run(run_id, producer=False, consumer=True)
    status.refresh_from_db()
    assert (status.producer_active, status.is_active) == (False, True)


def test_activating_an_unknown_run_raises(repo):
    with pytest.raises(LookupError, match="not found"):
        repo.activate_run(9_999_999, producer=True, consumer=True)


# --- run_progress ----------------------------------------------------------


def test_run_progress_on_a_fresh_run_is_all_zeros(repo):
    run_id = _create(repo)
    progress = repo.run_progress(run_id)
    assert progress.status == "draft"
    assert progress.classified == 0
    assert progress.last_batch_id is None
    assert progress.active_jobs == 0
    assert progress.is_active is False
    assert progress.consumer_query_run_id is None
    assert progress.last_error == ""


def test_run_progress_counts_only_this_run(repo):
    from dashboard.models import ModelVersion

    model = ModelVersion.objects.create(
        version="v1", artifact_path="/data/models/v1", algorithm="logreg"
    )
    mine = _create(repo)
    other = _create(repo, name="other")
    for run, batches in ((mine, 3), (other, 5)):
        for batch_id in range(batches):
            BatchMetric.objects.create(
                experiment_run_id=run, batch_id=batch_id, model_version=model,
                started_at=datetime.now(UTC), n_received=500, n_classified=500, n_correct=400,
            )
    TrainingJob.objects.create(experiment_run_id=mine, status="queued")
    TrainingJob.objects.create(experiment_run_id=mine, status="succeeded")
    TrainingJob.objects.create(experiment_run_id=other, status="running")
    repo.mark_run_started(mine)
    repo.activate_run(mine, producer=False, consumer=True)
    repo.set_consumer_query(mine)

    progress = repo.run_progress(mine)
    assert progress.status == "running"
    assert progress.classified == 1500
    assert progress.last_batch_id == 2
    assert progress.active_jobs == 1
    assert progress.is_active is True
    assert progress.active_experiment_run_id == mine
    assert progress.consumer_query_run_id == mine


def test_run_progress_of_an_unknown_run_raises(repo):
    with pytest.raises(LookupError, match="not found"):
        repo.run_progress(9_999_999)


# --- what the scorer reads back ----------------------------------------------


def test_run_bundle_carries_the_matrix_fields_and_flags_promoted_retrains(repo):
    """``promotions`` was always 0 before: the scorer read a column the query
    did not select. Now the query says, per retrain, whether the model it
    produced was ever activated -- which only ``promote_model`` sets."""
    from dashboard.models import ModelVersion
    from streamcls.db.queries import fetch_run_bundle
    from streamcls.experiments import score_run

    base = ModelVersion.objects.create(version="base", artifact_path="/m/base", algorithm="logreg")
    run_id = _create(repo, base_model_version_id=base.pk)
    kept = ModelVersion.objects.create(version="r1", artifact_path="/m/r1", algorithm="logreg")
    rejected = ModelVersion.objects.create(version="r2", artifact_path="/m/r2", algorithm="logreg")
    assert repo.promote_model(kept.pk, force=True) is True
    TrainingJob.objects.create(experiment_run_id=run_id, status="succeeded", model_version=kept)
    TrainingJob.objects.create(experiment_run_id=run_id, status="succeeded", model_version=rejected)
    TrainingJob.objects.create(experiment_run_id=run_id, status="failed")
    BatchMetric.objects.create(
        experiment_run_id=run_id, batch_id=0, model_version=base,
        started_at=datetime.now(UTC), n_received=500, n_classified=500, n_correct=450,
    )

    bundle = fetch_run_bundle(repo, run_id)
    assert bundle["run"]["scenario"] == "concept_abrupt"
    assert bundle["run"]["base_model_version_id"] == base.pk
    assert bundle["run"]["consumer_config"]["topic"] == f"posts.run-{run_id}"
    assert [j["promoted"] for j in bundle["jobs"]] == [True, False, False]

    score = score_run(**bundle, batch_size=500)
    assert score.retrains == 3
    assert score.promotions == 1
    assert score.scenario == "concept_abrupt"
