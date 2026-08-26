"""Idempotent Postgres sink for classified messages.

The previous consumer had a ``write_to_postgres`` method that was never
attached to a ``writeStream`` at all -- dead code -- and it used
``mode("append")``, which issues plain INSERTs. With ``post_id`` unique, the
first checkpoint replay would have raised a unique violation and failed the
batch permanently, since it would replay again.

Spark commits Kafka offsets *after* ``foreachBatch`` returns, so a crash between
writing and committing replays the batch. That makes the delivery guarantee
at-least-once; the unique key plus this upsert makes the observable result
effectively-once. Those are the words to use in the thesis -- not "exactly-once",
which Spark does not provide here.

Rows are written from the **executors** via ``foreachPartition``: high
cardinality, naturally parallel. Metrics and drift events are written from the
driver, because that is where the detector state lives.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

from streamcls.db.tables import Table

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

__all__ = ["UPSERT_SQL", "upsert_messages"]

#: `DO UPDATE` rather than `DO NOTHING`: after a model swap, a replayed row must
#: end up holding the prediction from the model the BatchMetric row attributes
#: it to. `DO NOTHING` would leave the old prediction in place while the metrics
#: recorded the new model -- a silent inconsistency in the accuracy accounting.
#:
#: The `WHERE` clause keeps it monotone, so an out-of-order replay of an older
#: batch cannot overwrite a newer result.
UPSERT_SQL = f"""
INSERT INTO {Table.CLASSIFIED_MESSAGE}
    (experiment_run_id, post_id, msg_index, post_text, channel_title, channel_id,
     view_count, true_label_id, predicted_label_id, confidence, probabilities,
     is_correct, model_version_id, batch_id, perturbation_kind, drift_phase,
     produced_at, classified_at)
VALUES %s
ON CONFLICT (experiment_run_id, post_id) DO UPDATE SET
    predicted_label_id = EXCLUDED.predicted_label_id,
    confidence         = EXCLUDED.confidence,
    probabilities      = EXCLUDED.probabilities,
    is_correct         = EXCLUDED.is_correct,
    model_version_id   = EXCLUDED.model_version_id,
    batch_id           = EXCLUDED.batch_id,
    classified_at      = EXCLUDED.classified_at
WHERE {Table.CLASSIFIED_MESSAGE}.batch_id <= EXCLUDED.batch_id
"""

SINK_COLUMNS = (
    "experiment_run_id",
    "post_id",
    "msg_index",
    "text",
    "channel_title",
    "channel_id",
    "view_count",
    "true_label",
    "prediction",
    "confidence",
    "probability",
    "is_correct",
    "perturbation_kind",
    "drift_phase",
    "produced_at",
)

_PAGE_SIZE = 1000


def upsert_messages(
    scored_df: DataFrame,
    *,
    dsn: str,
    model_version_id: int,
    batch_id: int,
    max_partitions: int = 4,
) -> None:
    """Write a scored batch to Postgres, idempotently.

    Only plain values are captured by the closure. ``foreachPartition``'s
    function *is* pickled and shipped to executors, unlike ``foreachBatch``'s,
    so capturing anything holding a ``SparkSession`` (or the drift detectors)
    raises at runtime rather than at import.
    """
    from datetime import UTC, datetime

    columns = list(SINK_COLUMNS)
    projected = scored_df.select(*columns)
    partitions = min(max(projected.rdd.getNumPartitions(), 1), max_partitions)

    def write_partition(rows: Any) -> None:  # pragma: no cover - runs on executors
        import psycopg2
        from psycopg2.extras import Json, execute_values

        buffer: list[tuple] = []
        connection = _connection(dsn)
        classified_at = datetime.now(UTC)
        try:
            with connection.cursor() as cursor:
                for row in rows:
                    buffer.append(
                        (
                            row["experiment_run_id"],
                            row["post_id"],
                            row["msg_index"],
                            row["text"],
                            row["channel_title"],
                            row["channel_id"],
                            row["view_count"],
                            int(row["true_label"]),
                            int(row["prediction"]),
                            float(row["confidence"]),
                            Json([float(v) for v in row["probability"].toArray()]),
                            bool(row["is_correct"]),
                            model_version_id,
                            batch_id,
                            row["perturbation_kind"],
                            row["drift_phase"],
                            row["produced_at"],
                            classified_at,
                        )
                    )
                    if len(buffer) >= _PAGE_SIZE:
                        execute_values(cursor, UPSERT_SQL, buffer, page_size=_PAGE_SIZE)
                        buffer.clear()
                if buffer:
                    execute_values(cursor, UPSERT_SQL, buffer, page_size=_PAGE_SIZE)
            connection.commit()
        except psycopg2.OperationalError:
            # A dead socket must not be reused by the next task on this worker.
            _drop_connection()
            raise
        except Exception:
            if not connection.closed:
                connection.rollback()
            raise

    projected.coalesce(partitions).rdd.foreachPartition(write_partition)


# --- per-worker connection reuse --------------------------------------------
# Spark reuses Python worker processes across batches, so a module-level handle
# keyed by pid genuinely persists and avoids reconnecting per partition per
# batch. `psycopg2.pool` would be ceremony: an executor task is single-threaded.
_CONNECTIONS: dict[int, Any] = {}


def _connection(dsn: str):  # pragma: no cover - runs on executors
    import psycopg2

    pid = os.getpid()
    existing = _CONNECTIONS.get(pid)
    if existing is not None and not existing.closed:
        return existing
    connection = psycopg2.connect(dsn, connect_timeout=10)
    _CONNECTIONS[pid] = connection
    return connection


def _drop_connection() -> None:  # pragma: no cover - runs on executors
    connection = _CONNECTIONS.pop(os.getpid(), None)
    if connection is not None and not connection.closed:
        with contextlib.suppress(Exception):
            connection.close()
