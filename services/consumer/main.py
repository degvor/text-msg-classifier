"""Spark Structured Streaming consumer.

This replaces the file with the largest concentration of defects in the
project. What changes, and why each one mattered:

* **The stop button worked on nothing.** ``query.awaitTermination()`` blocks
  forever, so the control-flag check ran exactly once and the entire Django
  control surface was decorative. Here ``awaitTermination(timeout=POLL)`` *is*
  the sleep: it parks the thread with no busy-wait but returns immediately if
  the query dies, giving sub-second failure detection and ~2 s control latency
  from a single call.
* **Offsets were never recovered.** ``checkpointLocation`` was set at *session*
  level, so Spark appended a random UUID leaf per query per start -- 88 orphan
  directories accumulated, and with ``startingOffsets: latest`` every restart
  silently discarded whatever was produced while the consumer was down. The
  checkpoint is now per-run and stable, which is also what makes a model swap
  lossless.
* **A second ``writeStream``** ran the whole NLP and inference pipeline a second
  time, on independent offsets, purely to print to console. Deleted.
* **The length filter rejected every multi-line post.** ``rlike("^.{0,N}$")``
  uses Java regex, where ``.`` excludes line terminators without ``DOTALL``. For
  a pipeline about channel posts that is categorical data loss, and the known
  ``max_len``/``max_message_length`` key mismatch had been masking it.
* **Metrics were fiction.** ``received`` and ``classified`` were incremented by
  the same value so drops were unobservable, and throughput was derived from a
  timestamp the same function overwrote -- measuring the producer's emission
  rate rather than system capacity, which is why status read "slow" forever.

Adaptation is deliberately not wired up here. This phase establishes that
accuracy tracks the offline score and that the drift statistics move under an
injected shift; acting on them is Phase 5.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import FrameType

from streamcls.config import Settings, get_settings
from streamcls.db.repo import ControlState, ModelRecord, Repository
from streamcls.drift.distributions import jensen_shannon, population_stability_index
from streamcls.labels import N_CLASSES
from streamcls.logging import configure, get_logger
from streamcls.schemas import spark_schema
from streamcls.spark.batch import (
    CONFIDENCE_BINS,
    aggregate_batch,
    pack_correct_bits,
    score,
)
from streamcls.spark.session import build_session
from streamcls.spark.sink import upsert_messages

log = get_logger(__name__)

POLL_SECONDS = 2.0
#: Vocabulary PSI is accumulated over this many batches before being reported.
#: A 20 000-term space against a 500-message batch is almost all empty bins, and
#: PSI over sparse bins is noise rather than signal.
PSI_WINDOW_BATCHES = 10


class Consumer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repo = Repository(settings.postgres)
        self.stop_event = threading.Event()
        self.spark = None
        self._model = None
        self._model_record: ModelRecord | None = None
        self._feature_window: list[list[float]] = []
        self._failures = 0

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        log.info("shutdown requested", extra={"signal": signum})
        self.stop_event.set()

    # --- model ---------------------------------------------------------------
    def _load_model(self, record: ModelRecord) -> None:
        from pyspark.ml import PipelineModel

        started = time.monotonic()
        self._model = PipelineModel.load(record.artifact_path)
        self._model_record = record
        self._feature_window.clear()
        self.repo.set_active_model(record.id)
        log.info(
            "model loaded",
            extra={
                "version": record.version,
                "model_id": record.id,
                "seconds": round(time.monotonic() - started, 2),
            },
        )

    # --- per-batch -----------------------------------------------------------
    def _make_foreach_batch(self, run_id: int):
        """Build the ``foreachBatch`` callback.

        This closure runs on the **driver** and is not serialized, so capturing
        ``self`` -- the repository, the model record, the sliding window -- is
        safe. The sink's ``foreachPartition`` closure *is* serialized, which is
        why it takes only plain values.
        """
        dsn = self.settings.postgres.dsn

        def process(batch_df, batch_id: int) -> None:
            started = time.monotonic()
            record = self._model_record
            assert record is not None

            scored = score(batch_df).persist()
            try:
                n_received = batch_df.count()
                replay = self.repo.batch_metric_exists(run_id, batch_id)
                if replay:
                    log.warning("replaying batch after restart", extra={"batch_id": batch_id})

                aggregate = aggregate_batch(scored)
                if aggregate.is_empty:
                    return

                upsert_messages(
                    scored, dsn=dsn, model_version_id=record.id, batch_id=batch_id
                )

                stats = self._distribution_stats(aggregate, record)
                self.repo.write_batch_metric(
                    experiment_run_id=run_id,
                    batch_id=batch_id,
                    model_version_id=record.id,
                    started_at=_utcnow(),
                    duration_ms=int((time.monotonic() - started) * 1000),
                    n_received=n_received,
                    n_classified=aggregate.n_classified,
                    n_correct=aggregate.n_correct,
                    accuracy=aggregate.accuracy,
                    mean_confidence=aggregate.mean_confidence,
                    std_confidence=aggregate.std_confidence,
                    confidence_hist=_json(aggregate.confidence_hist),
                    predicted_hist=_json(aggregate.predicted_hist),
                    true_hist=_json(aggregate.true_hist),
                    correct_bits=_binary(pack_correct_bits(aggregate.correct_sequence)),
                    throughput_msgs_s=_throughput(aggregate.n_classified, started),
                    **stats,
                )
                self.repo.consumer_heartbeat(last_batch_id=batch_id)
                log.info(
                    "batch",
                    extra={
                        "batch_id": batch_id,
                        "received": n_received,
                        "classified": aggregate.n_classified,
                        "accuracy": round(aggregate.accuracy or 0.0, 4),
                        "mean_confidence": round(aggregate.mean_confidence or 0.0, 4),
                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()},
                    },
                )
            finally:
                scored.unpersist()

        return process

    def _distribution_stats(self, aggregate, record: ModelRecord) -> dict[str, float | None]:
        """Label-free drift statistics for this batch.

        Confidence and predicted-class histograms are well populated at any
        realistic batch size, so those are per-batch signals. Vocabulary PSI is
        accumulated over a window first, because sparse bins make it noise.
        """
        uniform_classes = [1.0 / N_CLASSES] * N_CLASSES
        stats: dict[str, float | None] = {
            "psi_confidence": None,
            "psi_predicted": None,
            "psi_vocabulary": None,
            "js_vocabulary": None,
        }

        confidence_reference = record.confidence_reference or []
        if any(aggregate.confidence_hist) and len(confidence_reference) == CONFIDENCE_BINS:
            # Against the model's own held-out confidence profile, so movement
            # means "less sure than this model normally is" -- a label-free
            # proxy for real drift. Against a uniform reference it would just
            # measure how confident the model is in general.
            stats["psi_confidence"] = population_stability_index(
                confidence_reference, aggregate.confidence_hist
            )
        if any(aggregate.predicted_hist):
            stats["psi_predicted"] = population_stability_index(
                uniform_classes, aggregate.predicted_hist
            )

        reference = record.reference_histogram or []
        top_terms = record.top_term_indices or []
        if reference and top_terms and aggregate.feature_sums:
            self._feature_window.append(aggregate.feature_sums)
            if len(self._feature_window) > PSI_WINDOW_BATCHES:
                self._feature_window.pop(0)
            if len(self._feature_window) == PSI_WINDOW_BATCHES:
                observed = _fold(self._feature_window, top_terms)
                if observed is not None:
                    stats["psi_vocabulary"] = population_stability_index(reference, observed)
                    stats["js_vocabulary"] = jensen_shannon(reference, observed)
        return stats

    # --- streaming query ------------------------------------------------------
    def _start_query(self, run_id: int, config: dict):
        from pyspark.sql import functions as F

        checkpoint = Path(self.settings.paths.checkpoint_root) / f"run-{run_id}"
        max_length = int(config["max_message_length"])

        raw = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.settings.kafka.bootstrap_servers)
            .option("subscribe", self.settings.kafka.topic)
            .option("startingOffsets", "earliest")
            # Mandatory: without it a batch is bounded only by Kafka backlog,
            # which is both a driver-OOM risk and the reason batch composition
            # would not be reproducible between runs.
            .option("maxOffsetsPerTrigger", int(config["max_offsets_per_trigger"]))
            .option("failOnDataLoss", "false")
            .load()
        )

        parsed = (
            raw.select(
                F.col("offset").alias("_offset"),
                F.from_json(F.col("value").cast("string"), spark_schema()).alias("d"),
            )
            .select("_offset", "d.*")
            .filter(F.col("experiment_run_id") == run_id)
            .filter(F.col("text").isNotNull())
            # length(), not a regex. `rlike("^.{0,N}$")` silently dropped every
            # multi-line post, and is a backtracking match per row besides.
            .filter(F.length(F.col("text")) <= max_length)
        )

        predictions = self._model.transform(parsed)
        return (
            predictions.writeStream.foreachBatch(self._make_foreach_batch(run_id))
            .outputMode("append")
            .option("checkpointLocation", str(checkpoint))
            .queryName(f"classify-run-{run_id}")
            .trigger(processingTime=f"{int(config['trigger_seconds'])} seconds")
            .start()
        )

    # --- control loop ---------------------------------------------------------
    def run(self) -> None:
        self.spark = build_session("consumer", settings=self.settings.spark)
        query = None  # bound before the try: the old `finally` referenced an
        # unbound name behind a bare `except:` that also swallowed KeyboardInterrupt
        log.info("consumer starting", extra={"poll_seconds": POLL_SECONDS})

        try:
            while not self.stop_event.is_set():
                try:
                    control = self.repo.read_control()
                except Exception:
                    log.exception("control read failed; retrying")
                    self.stop_event.wait(POLL_SECONDS)
                    continue

                desired = self._desired_model(control)
                swap_needed = (
                    query is not None
                    and desired is not None
                    and self._model_record is not None
                    and desired.id != self._model_record.id
                )

                if query is not None and (not control.is_active or swap_needed):
                    reason = "model swap" if swap_needed else "deactivated"
                    log.info("stopping query", extra={"reason": reason})
                    self._stop_query(query)
                    query = None
                    if swap_needed and desired is not None:
                        # Same checkpoint, so offsets carry across: the swap is
                        # lossless. A PipelineModel is bound into the query plan
                        # at start(), so an in-place swap is not possible --
                        # stop-and-restart is the honest mechanism, and its
                        # latency is a measurement rather than a limitation.
                        self._load_model(desired)
                    continue

                if query is None and control.is_active:
                    if control.active_experiment_run_id is None:
                        log.warning("stream active with no experiment run; idling")
                        self.stop_event.wait(POLL_SECONDS)
                        continue
                    if desired is None:
                        log.warning("no active model version; idling")
                        self.stop_event.wait(POLL_SECONDS)
                        continue
                    if self._model_record is None or self._model_record.id != desired.id:
                        self._load_model(desired)
                    query = self._start_query(
                        control.active_experiment_run_id, self.repo.read_config()
                    )
                    self._failures = 0
                    log.info("query started", extra={"run_id": control.active_experiment_run_id})
                    continue

                if query is not None:
                    # This *is* the sleep: parks the thread, but returns at once
                    # if the query dies. One call gives both control latency and
                    # fast failure detection.
                    if query.awaitTermination(timeout=POLL_SECONDS):
                        error = query.exception()
                        query = None
                        if error is not None and not self._backoff(error):
                            self.repo.set_stream_active(False)
                else:
                    self.stop_event.wait(POLL_SECONDS)
        finally:
            if query is not None:
                try:
                    self._stop_query(query)
                except Exception:
                    log.exception("failed to stop query cleanly")
            self.repo.close()
            if self.spark is not None:
                self.spark.stop()

    def _desired_model(self, control: ControlState) -> ModelRecord | None:
        target = control.desired_model_version_id
        if target is not None:
            try:
                return self.repo.get_model(target)
            except LookupError:
                log.exception("desired model version missing", extra={"model_id": target})
        return self.repo.get_active_model()

    @staticmethod
    def _stop_query(query) -> None:
        query.stop()
        # Bounded: an unbounded await here would reintroduce the original hang,
        # just somewhere else, and SIGTERM would time out into a SIGKILL.
        query.awaitTermination(timeout=60)

    def _backoff(self, error: BaseException) -> bool:
        self._failures += 1
        self.repo.record_stream_error(str(error))
        if self._failures > 3:
            log.error("giving up after repeated query failures", extra={"error": str(error)[:500]})
            return False
        delay = min(2**self._failures, 30)
        log.warning(
            "query failed; retrying",
            extra={"attempt": self._failures, "delay": delay, "error": str(error)[:500]},
        )
        self.stop_event.wait(delay)
        return True


# --- helpers -----------------------------------------------------------------
def _utcnow():
    from datetime import UTC, datetime

    return datetime.now(UTC)


def _throughput(count: int, started: float) -> float:
    elapsed = max(time.monotonic() - started, 1e-6)
    return round(count / elapsed, 3)


def _json(value):
    from psycopg2.extras import Json

    return Json(value)


def _binary(value: bytes):
    import psycopg2

    return psycopg2.Binary(value)


def _fold(window: list[list[float]], top_terms: list[int]) -> list[float] | None:
    """Sum a window of feature vectors and fold to top terms plus 'other'."""
    import numpy as np

    stacked = np.sum(np.asarray(window, dtype=float), axis=0)
    total = float(stacked.sum())
    if total <= 0:
        return None
    indices = [i for i in top_terms if i < stacked.size]
    if len(indices) != len(top_terms):
        # A reference built against a different vocabulary is a bug, not
        # something to silently truncate: the bins would not line up.
        log.error(
            "reference vocabulary does not match the model",
            extra={"reference_terms": len(top_terms), "vocab": int(stacked.size)},
        )
        return None
    selected = stacked[indices]
    return [*selected.tolist(), max(0.0, total - float(selected.sum()))]


def main() -> int:
    configure("consumer")
    consumer = Consumer(get_settings())
    try:
        consumer.run()
    except Exception:
        log.exception("consumer failed")
        return 1
    log.info("consumer stopped")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    sys.exit(main())
