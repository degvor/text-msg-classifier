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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2.extras import Json

from streamcls.config import PostgresSettings
from streamcls.db.tables import Table
from streamcls.promotion import decide

__all__ = ["ControlState", "ModelRecord", "Repository", "RunProgress", "RunSpec"]


@dataclass(frozen=True, slots=True)
class ControlState:
    """One poll of the control plane."""

    is_active: bool
    producer_active: bool
    active_experiment_run_id: int | None
    active_model_version_id: int | None
    desired_model_version_id: int | None
    #: The run the consumer currently holds a streaming query for; ``None``
    #: when it holds none. What the consumer *is doing*, as opposed to
    #: ``is_active``, which is what it has been asked to do.
    consumer_query_run_id: int | None = None

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
    recent_accuracy: float | None
    vocab_size: int | None
    top_term_indices: list[int]
    reference_histogram: list[float]
    confidence_reference: list[float]


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


@dataclass(frozen=True, slots=True)
class RunProgress:
    """One poll of a run, as the matrix runner sees it.

    ``classified`` is summed from ``batch_metric``, which the consumer writes
    *after* the messages of that batch are committed -- so reaching
    ``total_messages`` means every row is in the sink, not merely received.
    """

    status: str
    classified: int
    last_batch_id: int | None
    active_jobs: int
    is_active: bool
    producer_active: bool
    active_experiment_run_id: int | None
    consumer_query_run_id: int | None
    last_error: str


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
                       active_model_version_id, desired_model_version_id,
                       consumer_query_run_id
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

    def set_consumer_query(self, run_id: int | None) -> None:
        """Record which run the consumer has a live streaming query for.

        ``is_active`` is what the operator wants; this is what the consumer is
        doing. NULL means no query. The matrix runner waits on it before
        touching the active model, because a query holds its model until it has
        actually stopped.
        """
        with self.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {Table.STREAM_STATUS}
                SET consumer_query_run_id = %s, consumer_heartbeat_at = now()
                WHERE id = 1
                """,
                (run_id,),
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
        recent_accuracy: float | None,
        train_rows: int,
        vocab_size: int,
        top_term_indices: list[int],
        reference_histogram: list[float],
        confidence_reference: list[float],
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
                     test_accuracy, test_macro_f1, recent_accuracy, metrics, train_rows,
                     vocab_size, top_term_indices, reference_histogram,
                     confidence_reference, created_by_run_id, git_commit, trained_at)
                VALUES (%s, 'ready', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                RETURNING id
                """,
                (
                    version, artifact_path, algorithm, Json(hyperparams),
                    test_accuracy, test_macro_f1, recent_accuracy, Json(metrics), train_rows,
                    vocab_size, Json(top_term_indices), Json(reference_histogram),
                    Json(confidence_reference), created_by_run, git_commit,
                ),
            )
            return int(cur.fetchone()["id"])

    def get_model(self, model_id: int) -> ModelRecord:
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, version, status, artifact_path, algorithm, test_accuracy,
                       recent_accuracy, vocab_size, top_term_indices,
                       reference_histogram, confidence_reference
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
                       recent_accuracy, vocab_size, top_term_indices,
                       reference_histogram, confidence_reference
                FROM {Table.MODEL_VERSION} WHERE status = 'active'
                """
            )
            row = cur.fetchone()
        return ModelRecord(**row) if row else None

    def live_accuracy(
        self, *, run_id: int | None = None, since_batch: int | None = None, batches: int = 5
    ) -> float | None:
        """The active model's measured accuracy on data from the current concept.

        Already computed every micro-batch, so the promotion guard can compare
        like with like for free rather than reloading the old model to re-score
        it on the candidate's holdout.

        ``since_batch`` matters more than it looks. Drift is detected within a
        batch or two of the change point, so a plain "last N batches" window
        still contains mostly *pre-drift* data and reports the incumbent as far
        healthier than it currently is -- which rejects every adapted model and
        silently makes adaptation impossible. Scoping to batches at or after the
        detection point measures the incumbent under the concept the candidate
        was trained for.
        """
        clauses, params = [], []
        if run_id is not None:
            clauses.append("experiment_run_id = %s")
            params.append(run_id)
        if since_batch is not None:
            clauses.append("batch_id >= %s")
            params.append(since_batch)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(batches)

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT sum(n_correct)::float / NULLIF(sum(n_classified), 0) AS accuracy
                FROM (
                    SELECT n_correct, n_classified FROM {Table.BATCH_METRIC}
                    {where}
                    ORDER BY experiment_run_id DESC, batch_id DESC LIMIT %s
                ) recent
                """,
                tuple(params),
            )
            row = cur.fetchone()
        return float(row["accuracy"]) if row and row["accuracy"] is not None else None

    def promote_model(
        self,
        model_id: int,
        *,
        min_delta: float = 0.0,
        run_id: int | None = None,
        since_batch: int | None = None,
        force: bool = False,
        incumbent_score: float | None = None,
    ) -> bool:
        """Make ``model_id`` the active version, if it is not clearly worse.

        The guard exists because without it a retrain on an unlucky window
        makes the system worse and it oscillates -- which would show up in the
        thesis as adaptation actively hurting.

        **What it compares matters more than the threshold.** It scores both
        models on ``recent_accuracy`` -- performance on recent stream data --
        falling back to ``test_accuracy`` only when the candidate has no recent
        score. Comparing on the fixed test set would be wrong under concept
        drift: that set still encodes the pre-drift labelling, so a correctly
        adapted model scores *worse* on it and would be rejected every time,
        making adaptation structurally impossible.

        ``force`` skips the guard but still retires the incumbent. It is for
        resetting to the base model between experiment cells: that is not a
        promotion, and the guard would rightly refuse it -- the retrained
        incumbent usually *is* better on recent data.

        ``incumbent_score`` is the incumbent's accuracy on the candidate's own
        holdout, scored by the trainer: given, the two are compared on the same
        messages (the ``matched_holdout`` rule) instead of the default above.
        Every guarded decision, accepted or not, is recorded on the candidate
        as ``metrics.promotion`` (see ``streamcls.promotion``).

        Retire-then-activate happens in one transaction because
        ``only_one_active_model_version`` is a real constraint.
        """
        with self.transaction() as cur:
            # FOR UPDATE: two concurrent promotions must not both pass the
            # guard and then collide on the single-active constraint.
            cur.execute(
                f"""
                SELECT id, test_accuracy, recent_accuracy FROM {Table.MODEL_VERSION}
                WHERE id = %s FOR UPDATE
                """,
                (model_id,),
            )
            candidate = cur.fetchone()
            if candidate is None:
                raise LookupError(f"model version {model_id} not found")

            cur.execute(
                f"""
                SELECT id, test_accuracy, recent_accuracy FROM {Table.MODEL_VERSION}
                WHERE status = 'active' FOR UPDATE
                """
            )
            incumbent = cur.fetchone()

            if incumbent is not None and incumbent["id"] != model_id:
                if not force:
                    if incumbent_score is not None and candidate["recent_accuracy"] is not None:
                        new_score, old_score, rule = (
                            candidate["recent_accuracy"], incumbent_score, "matched_holdout"
                        )
                    else:
                        new_score, old_score, rule = self._comparable_scores(
                            candidate, incumbent, run_id=run_id, since_batch=since_batch
                        )
                    decision = decide(new_score, old_score, min_delta=min_delta, rule=rule)
                    cur.execute(
                        f"""
                        UPDATE {Table.MODEL_VERSION}
                        SET metrics = metrics || jsonb_build_object('promotion', %s::jsonb)
                        WHERE id = %s
                        """,
                        (json.dumps(decision.as_dict() | {"incumbent_id": incumbent["id"]}),
                         model_id),
                    )
                    if not decision.promoted:
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

    def annotate_model(self, model_id: int, **entries: Any) -> None:
        """Merge ``entries`` into the model's ``metrics`` as top-level keys.

        For what is known only after registration: the job's stage timings and
        the incumbent's score on the candidate's holdout.
        """
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.MODEL_VERSION} SET metrics = metrics || %s::jsonb WHERE id = %s",
                (json.dumps(entries), model_id),
            )

    # --- stream output ------------------------------------------------------
    def write_batch_metric(self, **fields: Any) -> None:
        """Upsert one row per micro-batch.

        Upsert rather than insert because a checkpoint replay re-runs
        ``foreachBatch`` for a batch id that already has a row. Without the
        conflict clause, cumulative accuracy would drift on every restart.
        """
        columns = list(fields)
        placeholders = ", ".join(["%s"] * len(columns))
        updatable = [c for c in columns if c not in ("experiment_run_id", "batch_id")]
        assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)
        with self.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {Table.BATCH_METRIC} ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (experiment_run_id, batch_id) DO UPDATE SET {assignments}
                """,
                tuple(fields[c] for c in columns),
            )

    def batch_metric_exists(self, run_id: int, batch_id: int) -> bool:
        """Has this batch already been processed? Distinguishes a replay."""
        with self.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {Table.BATCH_METRIC} WHERE experiment_run_id = %s AND batch_id = %s",
                (run_id, batch_id),
            )
            return cur.fetchone() is not None

    def batch_metrics_since(
        self, run_id: int, *, after_batch: int = -1
    ) -> list[dict[str, Any]]:
        """Ordered history, for rebuilding detector state after a restart.

        Replay beats pickling detector objects: river's internals are not a
        documented, version-stable contract, and a stored blob cannot be
        inspected or reasoned about. ``correct_bits`` makes the replay exact.
        """
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT batch_id, n_classified, n_correct, mean_confidence, correct_bits
                FROM {Table.BATCH_METRIC}
                WHERE experiment_run_id = %s AND batch_id > %s
                ORDER BY batch_id
                """,
                (run_id, after_batch),
            )
            return [dict(row) for row in cur.fetchall()]

    def record_drift_event(
        self,
        *,
        run_id: int,
        batch_id: int,
        detector: str,
        statistic: float | None,
        threshold: float | None,
        accuracy_before: float | None,
        suppressed: bool,
        ground_truth_index: int | None,
        detection_delay_batches: int | None,
        action_taken: str,
        training_job_id: int | None = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {Table.DRIFT_EVENT}
                    (experiment_run_id, batch_id, detector, statistic, threshold,
                     accuracy_before, suppressed, ground_truth_drift_index,
                     detection_delay_batches, action_taken, training_job_id, detected_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                RETURNING id
                """,
                (run_id, batch_id, detector, statistic, threshold, accuracy_before,
                 suppressed, ground_truth_index, detection_delay_batches,
                 action_taken, training_job_id),
            )
            return int(cur.fetchone()["id"])

    # --- training job queue ---------------------------------------------------
    def enqueue_training_job(
        self,
        *,
        run_id: int,
        drift_event_id: int | None,
        params: dict[str, Any],
        min_batches_between: int,
        current_batch: int,
        max_per_run: int,
    ) -> int | None:
        """Queue a retrain, unless a guard says no. Returns the job id or None.

        Drift typically fires on several consecutive batches, so without these
        guards a single change point would queue dozens of training runs. The
        checks are done in one transaction with the insert so two batches
        cannot both pass them.
        """
        with self.transaction() as cur:
            cur.execute(
                f"""
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE status IN ('queued','running')) AS active
                FROM {Table.TRAINING_JOB} WHERE experiment_run_id = %s
                """,
                (run_id,),
            )
            counts = cur.fetchone()
            if counts["active"]:
                return None
            if counts["total"] >= max_per_run:
                return None

            # Cooldown measured against the last batch that produced a job.
            cur.execute(
                f"""
                SELECT max(d.batch_id) AS last_batch
                FROM {Table.TRAINING_JOB} j
                JOIN {Table.DRIFT_EVENT} d ON d.id = j.trigger_drift_event_id
                WHERE j.experiment_run_id = %s
                """,
                (run_id,),
            )
            last = cur.fetchone()["last_batch"]
            if last is not None and current_batch - last < min_batches_between:
                return None

            cur.execute(
                f"""
                INSERT INTO {Table.TRAINING_JOB}
                    (experiment_run_id, trigger_drift_event_id, status, params, requested_at)
                VALUES (%s, %s, 'queued', %s, now())
                RETURNING id
                """,
                (run_id, drift_event_id, Json(params)),
            )
            return int(cur.fetchone()["id"])

    def requeue_stale_jobs(self, *, lease_s: float) -> list[int]:
        """Put back in the queue any job ``running`` for longer than ``lease_s``.

        A trainer killed mid-job leaves its job ``running`` forever, and an
        active job blocks every later retrain of that run (one job at a time)
        and keeps the run from ever finishing. A retrain takes seconds, so a
        lease of minutes only ever catches an orphan -- and it stays safe with
        several workers, which a requeue-everything-at-startup rule would not.
        """
        with self.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {Table.TRAINING_JOB}
                SET status = 'queued', started_at = NULL,
                    error = 'requeued: running longer than the lease'
                WHERE status = 'running' AND started_at < now() - make_interval(secs => %s)
                RETURNING id
                """,
                (lease_s,),
            )
            return [row["id"] for row in cur.fetchall()]

    def claim_training_job(self) -> dict[str, Any] | None:
        """Atomically take the oldest queued job.

        ``FOR UPDATE SKIP LOCKED`` is the right primitive: it never loses a job,
        never hands the same one to two workers, and scales to several trainer
        replicas without changing anything.
        """
        with self.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {Table.TRAINING_JOB} SET status = 'running', started_at = now()
                WHERE id = (
                    SELECT id FROM {Table.TRAINING_JOB}
                    WHERE status = 'queued' ORDER BY id LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, experiment_run_id, params, trigger_drift_event_id
                """
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def finish_training_job(
        self,
        job_id: int,
        status: str,
        *,
        model_version_id: int | None = None,
        error: str = "",
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {Table.TRAINING_JOB}
                SET status = %s, finished_at = now(), model_version_id = %s, error = %s
                WHERE id = %s
                """,
                (status, model_version_id, error[:4000], job_id),
            )

    def messages_since(self, run_id: int, msg_index: int) -> int:
        """How many messages of the current concept have arrived so far."""
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT count(*) AS n FROM {Table.CLASSIFIED_MESSAGE}
                WHERE experiment_run_id = %s AND msg_index >= %s
                """,
                (run_id, msg_index),
            )
            return int(cur.fetchone()["n"])

    def recent_labelled_messages(self, run_id: int, limit: int) -> list[tuple[str, int]]:
        """The sliding window of stream data a retrain learns the new concept from."""
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT post_text, true_label_id FROM {Table.CLASSIFIED_MESSAGE}
                WHERE experiment_run_id = %s ORDER BY msg_index DESC LIMIT %s
                """,
                (run_id, limit),
            )
            return [(r["post_text"], r["true_label_id"]) for r in cur.fetchall()]

    def record_stream_error(self, message: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.STREAM_STATUS} SET last_error = %s WHERE id = 1",
                (message[:2000],),
            )

    def set_active_model(self, model_id: int | None) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.STREAM_STATUS} SET active_model_version_id = %s WHERE id = 1",
                (model_id,),
            )

    def set_stream_active(self, active: bool) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.STREAM_STATUS} SET is_active = %s WHERE id = 1",
                (active,),
            )

    def _comparable_scores(
        self,
        candidate: dict,
        incumbent: dict,
        *,
        run_id: int | None = None,
        since_batch: int | None = None,
    ) -> tuple[float, float, str]:
        """Score both models on the same kind of data; also say which data.

        Order of preference:

        1. both have a recent-data score -- use those;
        2. the candidate has one and the incumbent does not (the usual case:
           the incumbent was trained offline before the stream existed) -- use
           the incumbent's *measured live accuracy* over the last few batches,
           which is its performance on exactly the same distribution;
        3. neither -- fall back to the fixed test set.

        Never mix: comparing a candidate's recent-data score against an
        incumbent's fixed-test score rejects every adapted model, because under
        concept drift the fixed set still encodes the old labelling.
        """
        if candidate["recent_accuracy"] is None:
            return (candidate["test_accuracy"] or 0.0, incumbent["test_accuracy"] or 0.0, "test")

        new_score = candidate["recent_accuracy"]
        if incumbent["recent_accuracy"] is not None:
            return (new_score, incumbent["recent_accuracy"], "recent")

        measured = self.live_accuracy(run_id=run_id, since_batch=since_batch)
        if measured is not None:
            return (new_score, measured, "live")
        return (new_score, incumbent["test_accuracy"] or 0.0, "test")

    def set_producer_active(self, active: bool) -> None:
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE {Table.STREAM_STATUS} SET producer_active = %s WHERE id = 1",
                (active,),
            )

    # --- experiment matrix --------------------------------------------------
    # What the unattended runner needs and the dashboard does through Django:
    # write the knobs, create a run, point the control plane at it, and watch
    # it finish. Kept here so the runner has no Django and no HTTP dependency.

    def update_config(self, **fields: Any) -> None:
        """Overwrite ``stream_config`` knobs by column name.

        The allowlist is the table itself, so a knob added to the Django model
        is writable here without a code change. An unknown name raises rather
        than being skipped: an ablation whose override silently did nothing
        would be indistinguishable from a null result.
        """
        if not fields:
            return
        allowed = set(self.read_config()) - {"id", "updated_at"}
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise ValueError(f"unknown {Table.STREAM_CONFIG} columns: {', '.join(unknown)}")
        columns = list(fields)
        assignments = ", ".join(f"{c} = %s" for c in columns)
        values = [
            Json(fields[c]) if isinstance(fields[c], list | dict) else fields[c] for c in columns
        ]
        with self.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {Table.STREAM_CONFIG} SET {assignments}, updated_at = now()
                WHERE id = 1
                """,
                tuple(values),
            )

    def create_run(
        self,
        *,
        name: str,
        scenario: str,
        seed: int,
        mode: str,
        adaptation_enabled: bool,
        total_messages: int,
        drift_schedule: list[dict[str, Any]],
        dataset_split: str,
        base_model_version_id: int | None,
        consumer_config: dict[str, Any],
        producer_config: dict[str, Any],
        git_commit: str,
        notes: str = "",
        rate_msgs_per_sec: float = 50.0,
        topic_for_run: Callable[[int], str] | None = None,
    ) -> int:
        """Insert a ``draft`` run and return its id.

        The run's Kafka topic is named after the id, which does not exist
        until the INSERT returns, so it is merged into ``consumer_config`` in
        the same transaction: a run row either records its topic or does not
        exist.
        """
        with self.transaction() as cur:
            cur.execute(
                f"""
                INSERT INTO {Table.EXPERIMENT_RUN}
                    (name, notes, scenario, seed, mode, status, adaptation_enabled,
                     total_messages, rate_msgs_per_sec, drift_schedule, dataset_split,
                     base_model_version_id, producer_config, consumer_config, git_commit,
                     created_at)
                VALUES (%s, %s, %s, %s, %s, 'draft', %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                RETURNING id
                """,
                (
                    name, notes, scenario, seed, mode, adaptation_enabled,
                    total_messages, rate_msgs_per_sec, Json(drift_schedule), dataset_split,
                    base_model_version_id, Json(producer_config), Json(consumer_config),
                    git_commit,
                ),
            )
            run_id = int(cur.fetchone()["id"])
            if topic_for_run is not None:
                cur.execute(
                    f"""
                    UPDATE {Table.EXPERIMENT_RUN}
                    SET consumer_config = consumer_config || %s::jsonb WHERE id = %s
                    """,
                    (Json({"topic": topic_for_run(run_id)}), run_id),
                )
        return run_id

    def find_runs_by_name(self, name: str) -> list[dict[str, Any]]:
        """Every run with this name, oldest first. Names are not unique: a
        redone matrix cell is a new row with the same name."""
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, name, scenario, seed, status, adaptation_enabled, total_messages,
                       base_model_version_id, consumer_config, git_commit, created_at
                FROM {Table.EXPERIMENT_RUN} WHERE name = %s ORDER BY id
                """,
                (name,),
            )
            return [dict(r) for r in cur.fetchall()]

    def activate_run(self, run_id: int, *, producer: bool, consumer: bool) -> None:
        """Point the control plane at ``run_id`` -- what the dashboard's
        activate endpoint does. ``last_error`` is cleared so a stale failure
        from an earlier run cannot be read as this one's."""
        with self.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {Table.EXPERIMENT_RUN} WHERE id = %s", (run_id,)
            )
            if cur.fetchone() is None:
                raise LookupError(f"experiment run {run_id} not found")
            cur.execute(
                f"""
                UPDATE {Table.STREAM_STATUS}
                SET active_experiment_run_id = %s, producer_active = %s, is_active = %s,
                    last_error = ''
                WHERE id = 1
                """,
                (run_id, producer, consumer),
            )

    def heartbeat_ages(self) -> dict[str, float | None]:
        """Seconds since each service last checked in; ``None`` if it never
        has. Both services heartbeat while idle, so a stale value means the
        container is not running, not merely that nothing is happening."""
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT extract(epoch FROM now() - producer_heartbeat_at) AS producer,
                       extract(epoch FROM now() - consumer_heartbeat_at) AS consumer
                FROM {Table.STREAM_STATUS} WHERE id = 1
                """
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"{Table.STREAM_STATUS} has no row with id=1")
        return {k: (float(v) if v is not None else None) for k, v in row.items()}

    def run_progress(self, run_id: int) -> RunProgress:
        """One consistent snapshot of a run and the control plane."""
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT r.status,
                       COALESCE(m.classified, 0)::bigint AS classified,
                       m.last_batch_id,
                       COALESCE(j.active_jobs, 0)::int AS active_jobs,
                       s.is_active, s.producer_active, s.active_experiment_run_id,
                       s.consumer_query_run_id, s.last_error
                FROM {Table.EXPERIMENT_RUN} r
                CROSS JOIN {Table.STREAM_STATUS} s
                LEFT JOIN (
                    SELECT sum(n_classified) AS classified, max(batch_id) AS last_batch_id
                    FROM {Table.BATCH_METRIC} WHERE experiment_run_id = %s
                ) m ON true
                LEFT JOIN (
                    SELECT count(*) AS active_jobs FROM {Table.TRAINING_JOB}
                    WHERE experiment_run_id = %s AND status IN ('queued', 'running')
                ) j ON true
                WHERE r.id = %s AND s.id = 1
                """,
                (run_id, run_id, run_id),
            )
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"experiment run {run_id} not found")
        data = dict(row)
        data["classified"] = int(data["classified"])
        data["active_jobs"] = int(data["active_jobs"])
        return RunProgress(**data)
