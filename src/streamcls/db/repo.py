"""Direct Postgres access for the services that are not Django.

The producer, consumer and trainer read the control plane every couple of
seconds. The previous consumer did this by running a full Spark JDBC job --
``read().orderBy().limit(1).collect()`` -- every ten seconds, forever, just to
fetch one boolean. psycopg2 does it in about a millisecond.

It also caught every exception and returned ``False``, so "Postgres is down",
"the table is missing" and "the user turned the stream off" were indistinguishable
and all produced a silent idle. Here a control read that fails raises, and the
caller decides.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2.extras import Json

from streamcls.config import PostgresSettings
from streamcls.db.tables import Table

__all__ = ["ControlState", "ModelRecord", "Repository", "RunSpec"]


@dataclass(frozen=True, slots=True)
class ControlState:
    """One poll of the control plane."""

    is_active: bool
    producer_active: bool
    active_experiment_run_id: int | None
    active_model_version_id: int | None
    desired_model_version_id: int | None

    @property
    def model_swap_pending(self) -> bool:
        return (
            self.desired_model_version_id is not None
            and self.desired_model_version_id != self.active_model_version_id
        )


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """A registered model, as the consumer and trainer see it."""

    id: int
    version: str
    status: str
    artifact_path: str
    algorithm: str
    test_accuracy: float | None
    vocab_size: int | None
    top_term_indices: list[int]
    reference_histogram: list[float]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything needed to reproduce a run's stream."""

    id: int
    name: str
    seed: int
    mode: str
    status: str
    adaptation_enabled: bool
    total_messages: int
    rate_msgs_per_sec: float
    drift_schedule: list[dict[str, Any]]
    dataset_split: str


class Repository:
    """Thin, explicit data access. No ORM, no magic, reconnects on demand."""

    def __init__(self, settings: PostgresSettings, *, connect_timeout: int = 10) -> None:
        self._settings = settings
        self._connect_timeout = connect_timeout
        self._conn: psycopg2.extensions.connection | None = None

    # --- connection ---------------------------------------------------------
    @property
    def connection(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                self._settings.dsn, connect_timeout=self._connect_timeout
            )
            self._conn.autocommit = True
        return self._conn

    @contextlib.contextmanager
    def cursor(self) -> Iterator[psycopg2.extensions.cursor]:
        try:
            with self.connection.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                yield cur
        except psycopg2.OperationalError:
            # Drop the handle so the next call reconnects rather than reusing a
            # socket that is already gone.
            self.close()
            raise

    @contextlib.contextmanager
    def transaction(self) -> Iterator[psycopg2.extensions.cursor]:
        """Run several statements atomically.

        The connection is normally in autocommit mode, where psycopg2's own
        ``with connection:`` block does *not* open a transaction -- every
        statement has already been committed by the time the block ends. So
        autocommit is turned off for the duration and restored afterwards.
        """
        conn = self.connection
        previous = conn.autocommit
        conn.autocommit = False
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.autocommit = previous

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            with contextlib.suppress(Exception):
                self._conn.close()
        self._conn = None

    def __enter__(self) -> Repository:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- control plane ------------------------------------------------------
    def read_control(self) -> ControlState:
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT is_active, producer_active, active_experiment_run_id,
                       active_model_version_id, desired_model_version_id
                FROM {Table.STREAM_STATUS} WHERE id = 1
                """
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                f"{Table.STREAM_STATUS} has no row with id=1; run Django migrations first"
            )
        return ControlState(**row)

    def read_config(self) -> dict[str, Any]:
        with self.cursor() as cur:
            cur.execute(f"SELECT * FROM {Table.STREAM_CONFIG} WHERE id = 1")
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"{Table.STREAM_CONFIG} has no row with id=1")
        return dict(row)

    def load_run(self, run_id: int) -> RunSpec:
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, name, seed, mode, status, adaptation_enabled,
                       total_messages, rate_msgs_per_sec, drift_schedule, dataset_split
                FROM {Table.EXPERIMENT_RUN} WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"experiment run {run_id} not found")
        data = dict(row)
        schedule = data.get("drift_schedule") or []
        if isinstance(schedule, str):  # older drivers hand back raw JSON
            schedule = json.loads(schedule)
        data["drift_schedule"] = schedule
        return RunSpec(**data)

    def mark_run_started(self, run_id: int) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {Table.EXPERIMENT_RUN}
                SET status = 'running', started_at = COALESCE(started_at, now())
                WHERE id = %s
                """,
                (run_id,),
            )

    def mark_run_finished(self, run_id: int, status: str = "completed") -> None:
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.EXPERIMENT_RUN} SET status = %s, ended_at = now() WHERE id = %s",
                (status, run_id),
            )

    # --- heartbeats ---------------------------------------------------------
    def producer_heartbeat(self) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.STREAM_STATUS} SET producer_heartbeat_at = now() WHERE id = 1"
            )

    def consumer_heartbeat(self, *, last_batch_id: int | None = None) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {Table.STREAM_STATUS}
                SET consumer_heartbeat_at = now(),
                    last_batch_id = COALESCE(%s, last_batch_id)
                WHERE id = 1
                """,
                (last_batch_id,),
            )

    # --- model registry -----------------------------------------------------
    def register_model(
        self,
        *,
        version: str,
        artifact_path: str,
        algorithm: str,
        hyperparams: dict[str, Any],
        metrics: dict[str, Any],
        test_accuracy: float,
        test_macro_f1: float,
        train_rows: int,
        vocab_size: int,
        top_term_indices: list[int],
        reference_histogram: list[float],
        created_by_run: int | None = None,
        git_commit: str = "",
    ) -> int:
        """Insert a ``ready`` model version and return its id.

        Status is ``ready``, never ``active``: promotion is a separate,
        guarded transaction, so an unverified model can never take over serving
        merely by finishing training.
        """
        with self.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {Table.MODEL_VERSION}
                    (version, status, artifact_path, algorithm, hyperparams,
                     test_accuracy, test_macro_f1, metrics, train_rows,
                     vocab_size, top_term_indices, reference_histogram,
                     created_by_run_id, git_commit, trained_at)
                VALUES (%s, 'ready', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                RETURNING id
                """,
                (
                    version, artifact_path, algorithm, Json(hyperparams),
                    test_accuracy, test_macro_f1, Json(metrics), train_rows,
                    vocab_size, Json(top_term_indices), Json(reference_histogram),
                    created_by_run, git_commit,
                ),
            )
            return int(cur.fetchone()["id"])

    def get_model(self, model_id: int) -> ModelRecord:
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, version, status, artifact_path, algorithm, test_accuracy,
                       vocab_size, top_term_indices, reference_histogram
                FROM {Table.MODEL_VERSION} WHERE id = %s
                """,
                (model_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"model version {model_id} not found")
        return ModelRecord(**row)

    def get_active_model(self) -> ModelRecord | None:
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, version, status, artifact_path, algorithm, test_accuracy,
                       vocab_size, top_term_indices, reference_histogram
                FROM {Table.MODEL_VERSION} WHERE status = 'active'
                """
            )
            row = cur.fetchone()
        return ModelRecord(**row) if row else None

    def promote_model(self, model_id: int, *, min_delta: float = 0.0) -> bool:
        """Make ``model_id`` the active version, if it is not clearly worse.

        The guard exists because without it a retrain on an unlucky window
        makes the system worse and it oscillates -- which would show up in the
        thesis as adaptation actively hurting. With it, adaptation is monotone
        by construction, and a rejected promotion is itself evidence the guard
        works.

        Retire-then-activate happens in one transaction because
        ``only_one_active_model_version`` is a real constraint.
        """
        with self.transaction() as cur:
            # FOR UPDATE: two concurrent promotions must not both pass the
            # guard and then collide on the single-active constraint.
            cur.execute(
                f"SELECT id, test_accuracy FROM {Table.MODEL_VERSION} WHERE id = %s FOR UPDATE",
                (model_id,),
            )
            candidate = cur.fetchone()
            if candidate is None:
                raise LookupError(f"model version {model_id} not found")

            cur.execute(
                f"""
                SELECT id, test_accuracy FROM {Table.MODEL_VERSION}
                WHERE status = 'active' FOR UPDATE
                """
            )
            incumbent = cur.fetchone()

            if incumbent is not None and incumbent["id"] != model_id:
                new_score = candidate["test_accuracy"] or 0.0
                old_score = incumbent["test_accuracy"] or 0.0
                if new_score < old_score - min_delta:
                    return False
                cur.execute(
                    f"UPDATE {Table.MODEL_VERSION} SET status = 'retired' WHERE id = %s",
                    (incumbent["id"],),
                )

            cur.execute(
                f"""
                UPDATE {Table.MODEL_VERSION}
                SET status = 'active', activated_at = now() WHERE id = %s
                """,
                (model_id,),
            )
            cur.execute(
                f"UPDATE {Table.STREAM_STATUS} SET desired_model_version_id = %s WHERE id = 1",
                (model_id,),
            )
            if cur.rowcount != 1:
                # Without this the promotion "succeeds" while the consumer is
                # never told, so the new model is active in the registry and
                # never actually serves. Silent, and very hard to spot.
                raise RuntimeError(
                    f"{Table.STREAM_STATUS} has no row with id=1; cannot signal the consumer"
                )
        return True

    def set_producer_active(self, active: bool) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.STREAM_STATUS} SET producer_active = %s WHERE id = 1",
                (active,),
            )
