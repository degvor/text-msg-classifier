"""One definition of every runtime setting, sourced from the environment.

Replaces credentials and endpoints that were previously hardcoded in five
places (``docker-compose.yaml``, both Spark scripts, Django ``settings.py``) and
drifted independently.

Settings are built lazily through :func:`get_settings` rather than at import
time, so a module that never touches Postgres does not fail because
``POSTGRES_PASSWORD`` is unset.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["KafkaSettings", "PathSettings", "PostgresSettings", "Settings", "get_settings"]

_ENV_FILE = (".env", "../.env")


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_", env_file=_ENV_FILE, extra="ignore"
    )

    host: str = "db"
    port: int = 5432
    db: str = "streamdb"
    user: str = "streamuser"
    # No default: a committed password is how the previous version leaked one
    # into git. `make env` writes a dev value into .env.
    password: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        """libpq connection string, for psycopg2."""
        return (
            f"host={self.host} port={self.port} dbname={self.db} "
            f"user={self.user} password={self.password}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def jdbc_url(self) -> str:
        return f"jdbc:postgresql://{self.host}:{self.port}/{self.db}"

    def jdbc_properties(self) -> dict[str, str]:
        return {
            "user": self.user,
            "password": self.password,
            "driver": "org.postgresql.Driver",
        }


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_", env_file=_ENV_FILE, extra="ignore")

    bootstrap_servers: str = "kafka:9092"
    #: Prefix, not a topic. Every run gets its own topic (see
    #: :meth:`topic_for_run`) so Spark's batch ids start at 0 for each run.
    topic: str = "posts"
    #: Experiments run on a single partition so batch composition is
    #: deterministic. Scalability benchmarks override this.
    num_partitions: int = 1

    def topic_for_run(self, run_id: int) -> str:
        """The Kafka topic carrying exactly one experiment run.

        A shared topic plus a per-run checkpoint made run N first replay every
        earlier run as filtered-empty micro-batches, so its batch ids started
        at roughly N x 32 while the scorer, the consumer's detection-delay
        arithmetic and the dashboard all assume 0.
        """
        return f"{self.topic}.run-{int(run_id)}"


class PathSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STREAMCLS_", env_file=_ENV_FILE, extra="ignore")

    data_root: Path = Path("/data")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def corpus_dir(self) -> Path:
        return self.data_root / "corpora"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_root(self) -> Path:
        return self.data_root / "models"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checkpoint_root(self) -> Path:
        return self.data_root / "checkpoints"

    def checkpoint_for_run(self, run_id: int) -> Path:
        """Where the consumer checkpoints one run. The matrix runner checks it
        from the host, so both must derive it here."""
        return self.checkpoint_root / f"run-{int(run_id)}"


class SparkSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPARK_", env_file=_ENV_FILE, extra="ignore")

    #: Pinned rather than left to `local[*]` so partitioning -- and therefore
    #: model fitting -- is reproducible across machines with different core
    #: counts.
    master: str = "local[4]"
    driver_memory: str = "2g"
    executor_memory: str = "2g"
    shuffle_partitions: int = 8

    #: Mandatory, not optional. Without a cap a micro-batch is bounded only by
    #: Kafka backlog, which risks driver OOM and makes batch composition
    #: non-deterministic.
    max_offsets_per_trigger: int = 500
    trigger_seconds: int = 10


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    environment: str = Field(default="dev", description="dev | test | prod")

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)  # type: ignore[arg-type]
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    spark: SparkSettings = Field(default_factory=SparkSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
