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

The adaptation loop closes here: detect -> record a DriftEvent -> enqueue a
guarded TrainingJob -> reset the detectors. The trainer service does the fitting
and the promotion; the consumer notices the new ``desired_model_version`` on its
next control poll and restarts the query against the same checkpoint, so the
swap is lossless.

A ``PipelineModel`` is bound into the query plan at ``start()``, so an in-place
swap is not possible. Stop-and-restart is the honest mechanism, and its latency
is reported as a measurement rather than hidden.
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
from streamcls.corpus.schedule import DriftSchedule
from streamcls.db.repo import ControlState, ModelRecord, Repository
from streamcls.drift.monitor import DEFAULT_ACTIONABLE, DriftMonitor, DriftSignal, MonitorConfig
from streamcls.logging import configure, get_logger
from streamcls.schemas import spark_schema
from streamcls.spark.batch import aggregate_batch, pack_correct_bits, score
from streamcls.spark.session import build_session
from streamcls.spark.sink import upsert_messages

log = get_logger(__name__)

POLL_SECONDS = 2.0


class Consumer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repo = Repository(settings.postgres)
        self.stop_event = threading.Event()
        self.spark = None
        self._model = None
        self._model_record: ModelRecord | None = None
        self._monitor: DriftMonitor | None = None
        #: A detection that fired before enough of the new concept had arrived.
        #: Held because ADWIN detects *changes*: once the stream settles at the
        #: new error rate it is stationary again and will never fire a second
        #: time, so a deferred drift that is not remembered is lost forever.
        self._pending_drift: DriftSignal | None = None
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
        self.repo.set_active_model(record.id)
        log.info(
            "model loaded",
            extra={
                "version": record.version,
                "model_id": record.id,
                "seconds": round(time.monotonic() - started, 2),
            },
        )

    def _new_monitor(self, record: ModelRecord, config: dict) -> DriftMonitor:
        """A cold monitor for one (run, model) pair.

        Built at every query start, not only when the model changes. PSI is
        defined relative to *this* model's training distribution, so the
        references travel with the model; and the detectors must start cold
        behind a grace window for every run, or the second run on the same
        model would inherit the first run's ADWIN window and skip its grace
        period. Any history for *this* run is replayed on top by the caller.
        """
        actionable = config.get("actionable_detectors")
        monitor = DriftMonitor(
            config=MonitorConfig(
                adwin_delta=float(config["adwin_delta"]),
                psi_threshold=float(config["psi_threshold"]),
                psi_consecutive_batches=int(config["psi_consecutive_batches"]),
                grace_batches=int(config["grace_batches"]),
                actionable_detectors=(
                    frozenset(actionable) if actionable is not None else DEFAULT_ACTIONABLE
                ),
            ),
            reference_histogram=list(record.reference_histogram or []),
            confidence_reference=list(record.confidence_reference or []),
            top_term_indices=list(record.top_term_indices or []),
        )
        self._pending_drift = None
        return monitor

    # --- per-batch -----------------------------------------------------------
    def _make_foreach_batch(self, run_id: int):
        """Build the ``foreachBatch`` callback.

        This closure runs on the **driver** and is not serialized, so capturing
        ``self`` -- the repository, the model record, the sliding window -- is
        safe. The sink's ``foreachPartition`` closure *is* serialized, which is
        why it takes only plain values.
        """
        dsn = self.settings.postgres.dsn
        config = self.repo.read_config()
        run = self.repo.load_run(run_id)
        schedule = DriftSchedule(events=list(run.drift_schedule))
        adaptation_on = bool(run.adaptation_enabled) and bool(config.get("adaptation_enabled", True))
        config = dict(config) | {"_adaptation_on": adaptation_on}

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

                monitor = self._monitor
                assert monitor is not None
                result = monitor.observe(aggregate, apply=not replay)
                stats = result.stats
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
                    adwin_estimate=monitor._adwin.estimation,
                    adwin_width=monitor._adwin.width,
                    ddm_warning=monitor._ddm.warning_detected,
                    **stats,
                )
                if result.signals or self._pending_drift is not None:
                    self._handle_drift(run_id, batch_id, aggregate, result, schedule, config)
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

    def _handle_drift(self, run_id, batch_id, aggregate, result, schedule, config) -> None:
        """Record what fired and, if permitted, queue exactly one retrain."""
        change_point = schedule.preceding_change_point(
            batch_id * int(config["max_offsets_per_trigger"])
        )
        # A drift held over from an earlier batch counts as actionable now.
        first_actionable = result.actionable[0] if result.actionable else self._pending_drift
        adaptation_on = config.get("_adaptation_on", False)
        job_id = None

        # Retraining on the first few hundred rows of a new concept produces a
        # worse model and spends one of the run's retrain budget, so wait until
        # enough of the new concept exists to learn from.
        enough_new_data = True
        if first_actionable is not None and change_point is not None:
            required = int(config.get("retrain_min_new_rows", 0))
            if required:
                seen = self.repo.messages_since(run_id, change_point)
                enough_new_data = seen >= required

        if first_actionable is not None and not enough_new_data:
            if self._pending_drift is None:
                log.info(
                    "drift detected; deferring retrain until enough new data",
                    extra={"batch_id": batch_id, "required": required},
                )
            # Held, not dropped. ADWIN detects *changes*: once the stream
            # settles at the new error rate it is stationary again and will
            # never fire a second time, so a deferred drift that is forgotten
            # is lost for the rest of the run.
            self._pending_drift = first_actionable
            self._record_events(run_id, batch_id, aggregate, result, change_point, config, None)
            return

        if first_actionable is not None and adaptation_on:
            # Guarded inside one transaction: cooldown, no concurrent job, and a
            # per-run cap. Drift fires on several consecutive batches, so without
            # these a single change point would queue a dozen training runs.
            job_id = self.repo.enqueue_training_job(
                run_id=run_id,
                drift_event_id=None,
                params={"trigger": first_actionable.detector, "batch_id": batch_id},
                min_batches_between=int(config["min_batches_between_retrains"]),
                current_batch=batch_id,
                max_per_run=int(config["max_retrains_per_run"]),
            )

        self._record_events(
            run_id, batch_id, aggregate, result, change_point, config, job_id,
            trigger=first_actionable if job_id else None,
        )

        if job_id:
            self._pending_drift = None
            log.warning(
                "drift detected; retrain queued",
                extra={
                    "batch_id": batch_id,
                    "detector": first_actionable.detector,
                    "accuracy": round(aggregate.accuracy or 0.0, 4),
                    "job_id": job_id,
                },
            )
            # Reset, or this same drift fires on every later batch and the
            # cooldown becomes the only thing preventing a retrain storm.
            assert self._monitor is not None
            self._monitor.reset()

    def _record_events(
        self, run_id, batch_id, aggregate, result, change_point, config, job_id, *, trigger=None
    ) -> None:
        """Persist every signal that fired, actionable or not.

        ``trigger`` is the signal that actually caused the retrain. It may not
        appear in ``result.signals`` at all: a deferred drift fires on one batch
        and is acted on several batches later, and the link between detection
        and action has to survive that or the results chapter cannot attribute
        a retrain to anything.
        """
        per_batch = int(config["max_offsets_per_trigger"])
        signals = list(result.signals)
        if trigger is not None and trigger not in signals:
            signals.append(trigger)
        for fired in signals:
            actionable = trigger is not None and fired is trigger
            self.repo.record_drift_event(
                run_id=run_id,
                batch_id=batch_id,
                detector=fired.detector,
                statistic=fired.statistic,
                threshold=fired.threshold,
                accuracy_before=aggregate.accuracy,
                suppressed=fired.suppressed,
                ground_truth_index=change_point,
                # Delay in batches against the injected change point. Only
                # computable because the producer ships the ground truth.
                detection_delay_batches=(
                    batch_id - change_point // per_batch if change_point is not None else None
                ),
                action_taken=(
                    "retrain_queued" if (actionable and job_id) else
                    "suppressed" if fired.suppressed else
                    "recorded"
                ),
                training_job_id=job_id if actionable else None,
            )

    # --- streaming query ------------------------------------------------------
    def _start_query(self, run_id: int, config: dict):
        from pyspark.sql import functions as F

        checkpoint = Path(self.settings.paths.checkpoint_root) / f"run-{run_id}"
        max_length = int(config["max_message_length"])

        raw = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.settings.kafka.bootstrap_servers)
            # One topic per run. With a shared topic and a per-run checkpoint,
            # `earliest` made every new run first drain all earlier runs as
            # filtered-empty batches, so batch ids no longer started at 0 and
            # every batch-denominated figure (detection delay, recovery) was
            # offset by the size of history. The topic is created by whoever
            # starts the run (producer or matrix runner); see kafka_admin.
            .option("subscribe", self.settings.kafka.topic_for_run(run_id))
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
            # Belt and braces: the topic already carries only this run.
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
        # A consumer that died hard (OOM, SIGKILL) never cleared its flag; a
        # fresh process holds no query, so the flag is wrong until it says so.
        self.repo.set_consumer_query(None)

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
                        # The monitor is rebuilt when the query restarts.
                        self._load_model(desired)
                    continue

                if query is None and control.is_active:
                    if control.active_experiment_run_id is None:
                        log.warning("stream active with no experiment run; idling")
                        self._idle()
                        continue
                    if desired is None:
                        log.warning("no active model version; idling")
                        self._idle()
                        continue
                    run_id = control.active_experiment_run_id
                    if self.repo.load_run(run_id).status == "draft":
                        # The producer creates the run's topic before it marks
                        # the run started; subscribing before that fails.
                        log.info("run not started yet; waiting for producer", extra={"run_id": run_id})
                        self._idle()
                        continue
                    if self._model_record is None or self._model_record.id != desired.id:
                        self._load_model(desired)
                    config = self.repo.read_config()
                    # Always a cold monitor, then replay -- never unpickle:
                    # detector internals are not a version-stable contract, and
                    # correct_bits makes the replay exact.
                    self._monitor = self._new_monitor(desired, config)
                    history = self.repo.batch_metrics_since(run_id)
                    if history:
                        self._monitor.rebuild(history)
                        log.info(
                            "detector state rebuilt by replay",
                            extra={"batches": len(history)},
                        )
                    query = self._start_query(run_id, config)
                    self.repo.set_consumer_query(run_id)
                    self._failures = 0
                    log.info("query started", extra={"run_id": run_id})
                    continue

                if query is not None:
                    # This *is* the sleep: parks the thread, but returns at once
                    # if the query dies. One call gives both control latency and
                    # fast failure detection.
                    if query.awaitTermination(timeout=POLL_SECONDS):
                        error = query.exception()
                        query = None
                        self.repo.set_consumer_query(None)
                        if error is not None and not self._backoff(error):
                            self.repo.set_stream_active(False)
                else:
                    self._idle()
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

    def _stop_query(self, query) -> None:
        try:
            query.stop()
            # Bounded: an unbounded await here would reintroduce the original
            # hang, just somewhere else, and SIGTERM would time out into a
            # SIGKILL.
            query.awaitTermination(timeout=60)
        finally:
            # Cleared even if stop() raised: a flag that never clears would
            # block the matrix runner forever, which is the worse failure.
            self.repo.set_consumer_query(None)

    def _idle(self) -> None:
        """Sleep one poll, but stay visibly alive: the dashboard and the matrix
        runner read the heartbeat, and until now it was only written from
        inside a batch, so an idle consumer looked identical to a dead one."""
        self.repo.consumer_heartbeat()
        self.stop_event.wait(POLL_SECONDS)

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
