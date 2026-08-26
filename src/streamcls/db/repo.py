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

from streamcls.config import PostgresSettings
from streamcls.db.tables import Table

__all__ = ["ControlState", "Repository", "RunSpec"]


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

    def set_producer_active(self, active: bool) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.STREAM_STATUS} SET producer_active = %s WHERE id = 1",
                (active,),
            )
