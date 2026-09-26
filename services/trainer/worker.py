"""Long-lived trainer service: claims retrain jobs and publishes new models.

Why a separate service rather than the consumer shelling out to
``spark-submit``: the consumer already holds a Spark driver running a streaming
query, so forking a second JVM inside that container would contend for the same
memory and cores *while the throughput it competes with is being measured*. It
would also couple lifecycles -- a consumer restart would orphan the training
job -- and the exit code would still need plumbing back through a table, which
is most of this file anyway.

The queue is claimed with ``FOR UPDATE SKIP LOCKED``: atomic, never loses a
job, never hands one to two workers, and scales to several replicas unchanged.

One long-lived ``SparkSession`` is held across jobs, because paying ~30 s of
session startup per retrain would dominate the adaptation-latency measurement.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import FrameType

from services.trainer.train import register, train
from streamcls.config import Settings, get_settings
from streamcls.corpus.agnews import Document
from streamcls.corpus.splits import build_splits, split_retrain_window
from streamcls.db.repo import Repository
from streamcls.logging import configure, get_logger
from streamcls.spark.session import build_session

log = get_logger(__name__)

POLL_SECONDS = 5.0


class TrainerWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repo = Repository(settings.postgres)
        self.stop_event = threading.Event()
        self._spark = None
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        log.info("shutdown requested", extra={"signal": signum})
        self.stop_event.set()

    def run(self) -> None:
        log.info("trainer worker starting", extra={"poll_seconds": POLL_SECONDS})
        while not self.stop_event.is_set():
            try:
                job = self.repo.claim_training_job()
            except Exception:
                log.exception("failed to claim a job; retrying")
                self.stop_event.wait(POLL_SECONDS)
                continue

            if job is None:
                self.stop_event.wait(POLL_SECONDS)
                continue

            self._run_job(job)
        self._shutdown()

    @property
    def spark(self):
        """One session for the worker's lifetime, created on first use."""
        if self._spark is None:
            self._spark = build_session("trainer-worker", settings=self.settings.spark)
        return self._spark

    def _shutdown(self) -> None:
        self.repo.close()
        if self._spark is not None:
            self._spark.stop()
            self._spark = None

    def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        run_id = job["experiment_run_id"]
        params = job.get("params") or {}
        log.info("training job claimed", extra={"job_id": job_id, "run_id": run_id})
        started = time.monotonic()

        try:
            config = self.repo.read_config()
            documents, holdout = self._training_corpus(run_id, config)
            version = f"adapt-run{run_id}-job{job_id}"

            result = train(
                settings=self.settings,
                algorithm=params.get("algorithm", "logreg"),
                seed=params.get("seed", 20260827),
                version=version,
                documents=documents,
                holdout=holdout,
                spark=self.spark,
            )
            model_id = register(
                result,
                self.settings,
                promote=bool(config.get("auto_promote", True)),
                min_delta=float(config.get("promote_min_delta", 0.01)),
                run_id=run_id,
                # The batch the drift fired on: everything from here is the
                # current concept, and the only fair basis for comparison.
                since_batch=params.get("batch_id"),
            )
            self.repo.finish_training_job(job_id, "succeeded", model_version_id=model_id)
            log.info(
                "training job finished",
                extra={
                    "job_id": job_id,
                    "model_id": model_id,
                    "test_accuracy": round(result.test.accuracy, 4),
                    "seconds": round(time.monotonic() - started, 1),
                },
            )
        except Exception as error:
            log.exception("training job failed", extra={"job_id": job_id})
            self.repo.finish_training_job(job_id, "failed", error=str(error))

    def _training_corpus(self, run_id: int, config: dict) -> tuple[list[Document], list[Document]]:
        """Base sample plus a sliding window of recent labelled stream data.

        The base sample is what stops catastrophic forgetting: training only on
        the recent window would make the model excellent at the new concept and
        useless at everything that did not change. The ratio between the two is
        a knob worth ablating, and it is recorded on the job.

        Honest caveat for the write-up: using ground-truth labels off the stream
        is *labelled-feedback adaptation*, an upper bound on what is achievable
        rather than a deployable method. ``StreamConfig.label_delay_batches``
        exists to model the realistic delayed-feedback case.
        """
        window_rows = int(config.get("retrain_window_rows", 20_000))
        base_rows = int(config.get("retrain_base_rows", 20_000))

        recent = self.repo.recent_labelled_messages(run_id, window_rows)
        window = [Document(text=text, label=label) for text, label in recent]

        # The holdout is the only fair basis for deciding whether this model
        # is an improvement, so it is drawn from the current concept -- but
        # interleaved, so the rest of the new concept is trained on.
        window, holdout = split_retrain_window(
            window, new_rows=int(config.get("retrain_min_new_rows", 0))
        )

        corpus_dir = Path(self.settings.paths.corpus_dir) / "agnews"
        base = build_splits(corpus_dir).train[:base_rows] if base_rows else []

        log.info(
            "training corpus assembled",
            extra={
                "window_rows": len(window),
                "holdout_rows": len(holdout),
                "base_rows": len(base),
                "run_id": run_id,
            },
        )
        return [*base, *window], holdout


def main() -> int:
    configure("trainer-worker")
    worker = TrainerWorker(get_settings())
    try:
        worker.run()
    except Exception:
        log.exception("trainer worker failed")
        return 1
    log.info("trainer worker stopped")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    sys.exit(main())
