"""Kafka producer: replays a held-out corpus split as a drift-scheduled stream.

Three defects in the previous producer are fixed here by construction.

1. **It stopped after 100 messages and never recovered.** ``send_test_data``
   closed the ``KafkaProducer`` in a ``finally``, then the next iteration's
   ``if not self.producer`` guard passed -- a *closed* producer is not ``None``
   -- so every later batch blocked for 60 s in ``_wait_on_metadata`` and raised.
   The process never exited, so ``restart: unless-stopped`` never fired and the
   failure was completely silent. The connection is now opened once and owned
   by a context manager.
2. **No ground truth.** Nothing in the payload said which class a post really
   belonged to, so live accuracy was unmeasurable; the only way to score the
   old system at all was to abuse ``ChannelTitle`` as a proxy. Every message now
   carries ``true_label`` plus the drift phase it was produced in.
3. **Configuration was decorative.** ``KAFKA_BOOTSTRAP_SERVERS`` was set in
   compose and never read -- the class hardcoded its default and was constructed
   with no arguments.
"""

from __future__ import annotations

import signal
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

from kafka import KafkaProducer
from kafka.errors import KafkaError

from streamcls.config import Settings, get_settings
from streamcls.corpus.schedule import DriftSchedule
from streamcls.corpus.splits import build_splits
from streamcls.corpus.stream import SeededStream
from streamcls.db.repo import Repository, RunSpec
from streamcls.kafka_admin import ensure_topic
from streamcls.logging import configure, get_logger

log = get_logger(__name__)

POLL_SECONDS = 2.0
#: How often to flush and heartbeat while streaming.
PROGRESS_EVERY = 500


class Producer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repo = Repository(settings.postgres)
        self.stop_event = threading.Event()
        self._producer: KafkaProducer | None = None
        self._splits_cache: tuple[str, list] | None = None

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        log.info("shutdown requested", extra={"signal": signum})
        self.stop_event.set()

    # --- kafka --------------------------------------------------------------
    @property
    def producer(self) -> KafkaProducer:
        """One long-lived producer. Never closed between runs."""
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.settings.kafka.bootstrap_servers.split(","),
                value_serializer=lambda value: value.encode("utf-8"),
                key_serializer=lambda key: key,
                # at-least-once from the producer side; the consumer's upsert
                # makes the end-to-end result effectively-once.
                acks="all",
                retries=5,
                linger_ms=20,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True,
            )
            log.info("connected to kafka", extra={"servers": self.settings.kafka.bootstrap_servers})
        return self._producer

    def close(self) -> None:
        """Close once, at process shutdown -- never between runs.

        Closing per batch is precisely the bug that silenced the old producer:
        the handle stayed non-``None`` after ``close()``, so the reconnect guard
        never fired again.
        """
        producer, self._producer = self._producer, None
        if producer is not None:
            try:
                producer.flush(timeout=10)
            finally:
                producer.close(timeout=10)
        self.repo.close()

    # --- corpus -------------------------------------------------------------
    def _stream_pool(self, dataset_split: str) -> list:
        """Load and cache the held-out documents for a split name."""
        if self._splits_cache is not None and self._splits_cache[0] == dataset_split:
            return self._splits_cache[1]

        corpus_dir = Path(self.settings.paths.corpus_dir) / "agnews"
        log.info("loading corpus", extra={"dir": str(corpus_dir), "split": dataset_split})
        splits = build_splits(corpus_dir)
        pool = splits.stream_holdout
        self._splits_cache = (dataset_split, pool)
        log.info("corpus ready", extra={"documents": len(pool), "counts": splits.counts()})
        return pool

    # --- main loop ----------------------------------------------------------
    def run(self) -> None:
        log.info("producer starting", extra={"poll_seconds": POLL_SECONDS})
        while not self.stop_event.is_set():
            try:
                control = self.repo.read_control()
            except Exception:
                # A control read that fails is a real error, not "inactive".
                log.exception("control read failed; retrying")
                self.stop_event.wait(POLL_SECONDS)
                continue

            if not control.producer_active or control.active_experiment_run_id is None:
                self.repo.producer_heartbeat()
                self.stop_event.wait(POLL_SECONDS)
                continue

            try:
                run = self.repo.load_run(control.active_experiment_run_id)
            except LookupError:
                log.exception("active run does not exist; idling")
                self.stop_event.wait(POLL_SECONDS)
                continue

            self._emit_run(run)

            if not self.stop_event.is_set():
                # One run per activation. Turning producer_active back off stops
                # the loop from immediately replaying the same run.
                self.repo.set_producer_active(False)

    def _emit_run(self, run: RunSpec) -> None:
        pool = self._stream_pool(run.dataset_split)
        schedule = DriftSchedule(events=list(run.drift_schedule))
        # Live mode stamps each message with the moment it is due
        # (start + index / rate) and sends it then, so produced_at is the send
        # time and classified_at - produced_at the end-to-end latency. Replay
        # mode keeps the stream's nominal stamps and does not wait.
        live = run.mode == "live" and run.rate_msgs_per_sec > 0
        stream = SeededStream(
            documents=pool,
            schedule=schedule,
            seed=run.seed,
            run_id=run.id,
            total_messages=run.total_messages,
            **({"interval_seconds": 1.0 / run.rate_msgs_per_sec} if live else {}),
        )

        # One topic per run, so Spark's batch ids start at 0 for this run.
        # Created before the run is marked started: a run whose status is no
        # longer "draft" therefore always has a topic the consumer can subscribe
        # to, which is what the consumer's draft guard relies on.
        topic = self.settings.kafka.topic_for_run(run.id)
        try:
            ensure_topic(
                self.settings.kafka.bootstrap_servers,
                topic,
                partitions=self.settings.kafka.num_partitions,
            )
        except KafkaError:
            log.exception("could not create topic", extra={"run_id": run.id, "topic": topic})
            self.repo.mark_run_finished(run.id, status="aborted")
            return

        log.info("run starting", extra=dict(stream.describe()) | {"mode": run.mode, "topic": topic})
        self.repo.mark_run_started(run.id)

        # Replay mode writes everything as fast as Kafka accepts it, so the
        # consumer always has a full backlog and every micro-batch holds exactly
        # maxOffsetsPerTrigger messages in offset order. That is what makes
        # batch composition -- and therefore every per-batch chart -- identical
        # between two runs with the same seed.
        if live:
            stream.start_time = datetime.now(UTC)  # the schedule starts with the first send
        sent = 0
        try:
            for message in stream:
                if self.stop_event.is_set():
                    log.warning("run interrupted", extra={"run_id": run.id, "sent": sent})
                    self.repo.mark_run_finished(run.id, status="aborted")
                    return

                if live:
                    # Pace by the absolute schedule, so a slow send does not
                    # accumulate into drift between stamps and real time.
                    due = (message.produced_at - datetime.now(UTC)).total_seconds()
                    if due > 0:
                        self.stop_event.wait(due)
                self.producer.send(
                    topic, key=message.kafka_key(), value=message.model_dump_json()
                )
                sent += 1

                if sent % PROGRESS_EVERY == 0:
                    self.producer.flush(timeout=30)
                    self.repo.producer_heartbeat()
                    log.info(
                        "progress",
                        extra={"run_id": run.id, "sent": sent, "total": run.total_messages},
                    )

            self.producer.flush(timeout=60)
        except KafkaError:
            log.exception("kafka send failed", extra={"run_id": run.id, "sent": sent})
            self.repo.mark_run_finished(run.id, status="aborted")
            return

        self.repo.mark_run_finished(run.id, status="completed")
        log.info("run complete", extra={"run_id": run.id, "sent": sent})


def main() -> int:
    configure("producer")
    settings = get_settings()
    producer = Producer(settings)
    try:
        producer.run()
    except Exception:
        log.exception("producer failed")
        return 1
    finally:
        producer.close()
    log.info("producer stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
