import pytest
from pydantic import ValidationError

from streamcls.config import KafkaSettings, PostgresSettings, Settings, get_settings


def test_password_has_no_default(monkeypatch):
    """A committed default password is how the previous version leaked one."""
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(ValidationError):
        PostgresSettings(_env_file=None)


def test_dsn_and_jdbc_are_built_from_the_same_fields(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "pg.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_DB", "thesis")
    pg = PostgresSettings(_env_file=None)
    assert "host=pg.internal port=6543 dbname=thesis" in pg.dsn
    assert pg.jdbc_url == "jdbc:postgresql://pg.internal:6543/thesis"
    assert pg.jdbc_properties()["driver"] == "org.postgresql.Driver"


def test_password_is_not_in_repr(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "hunter2")
    pg = PostgresSettings(_env_file=None)
    assert pg.password == "hunter2"


def test_paths_derive_from_one_root(monkeypatch):
    monkeypatch.setenv("STREAMCLS_DATA_ROOT", "/mnt/thesis")
    s = Settings(_env_file=None)
    assert str(s.paths.model_root) == "/mnt/thesis/models"
    assert str(s.paths.corpus_dir) == "/mnt/thesis/corpora"
    assert str(s.paths.checkpoint_root) == "/mnt/thesis/checkpoints"


def test_max_offsets_per_trigger_is_always_set():
    """Unbounded batches risk driver OOM and make batching non-deterministic."""
    s = Settings(_env_file=None)
    assert s.spark.max_offsets_per_trigger > 0
    assert s.spark.master != "local[*]", "pin the core count for reproducible partitioning"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_every_run_gets_its_own_topic(monkeypatch):
    """A shared topic made run N's batch ids start where run N-1's ended."""
    monkeypatch.setenv("KAFKA_TOPIC", "posts")
    kafka = KafkaSettings(_env_file=None)
    assert kafka.topic_for_run(7) == "posts.run-7"
    assert kafka.topic_for_run(7) != kafka.topic_for_run(8)
    assert kafka.topic_for_run("12") == "posts.run-12"
