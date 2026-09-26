# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `README.md` first — it carries the architecture diagram, the measured
results and the design rationale. This file covers only what is specific to
working *in* the repo.

## Commands

```bash
make help              # every target
make env               # .env from .env.example (required before anything)
make data              # download AG News, checksum-verified, idempotent
make up                # db, kafka, web, producer, consumer
make train PROMOTE=1   # train and activate a model
make score             # results table
make compare BASELINE=1 ADAPTIVE=2
make matrix SPEC=experiments/smoke.toml ARGS=--dry-run   # cell table, no writes
make matrix            # experiments/matrix.toml, resumable, ~5.5 h
make report NAME=smoke # results/<NAME>/runs.jsonl -> summary.md + CSVs, no DB

make test-unit         # no JVM, no DB, no Docker
make test-integration  # needs db + kafka up
make test-spark        # runs inside the Spark image
make lint
```

Run a single test: `.venv/bin/python -m pytest tests/unit/test_detectors.py::test_matches_river -q`.
Integration tests need `--ds=config.settings.dev` and `POSTGRES_HOST=localhost
POSTGRES_PORT=55432` (the dev override publishes Postgres there). `make
score/compare/matrix` run on the host and get the same override from
`HOST_ENV` in the Makefile.

## Constraints that shape everything

**There is no JRE on the dev machine.** Anything importing pyspark runs in
Docker. This is why `streamcls.drift`, `streamcls.corpus`, `streamcls.metrics`,
`streamcls.experiments` and `streamcls.matrix` are pure Python with no Spark
dependency — they hold the logic the thesis actually reports on, and they must
be testable locally (the runner is exercised against a scripted fake world in
`tests/unit/test_matrix.py`).
`streamcls.spark.*` is the only place pyspark may be imported, and never at
`streamcls/__init__` level (the Django image installs `streamcls[web]` and must
not pull a 300 MB wheel).

**`src/`, `services/` and `web/` are bind-mounted in dev** via
`docker-compose.override.yml`. Editing them does not need a rebuild — but
gunicorn and the Spark drivers do need a container restart to re-import.

## Rules that are easy to break

**A Django `default=` is not a database default.** `batch_metric`,
`classified_message`, `drift_event`, `training_job`, `model_version`,
`stream_status` and `experiment_run` are all written by raw psycopg2 — the
first six from the Spark services, `experiment_run` by the matrix runner's
`Repository.create_run`. Any field on those with a Python `default=` also needs
`db_default=`, or an INSERT that omits the column hits a NOT NULL violation at
runtime. This has bitten three times.

**Every run has its own Kafka topic**, `posts.run-<id>`
(`KafkaSettings.topic_for_run`). `KAFKA_TOPIC` is the prefix, not a topic. The
producer creates it before `mark_run_started`, the runner before activating
anything; the consumer never creates topics (its image has no kafka-python) and
subscribes by run id. This is what makes `batch_id` start at 0 in every run —
the scorer, the consumer's delay arithmetic and the dashboard all assume it.
Checkpoints are per run too (`PathSettings.checkpoint_for_run`, on the host
bind mount), so neither they nor topics die with the Postgres volume. The
runner's `_claim_run_id` fails a cell whose new id already has either one.
`HOST_ENV` sets `STREAMCLS_DATA_ROOT` so the host sees `./data`.

**The consumer rebuilds its `DriftMonitor` at every query start**, not only on
a model change: fresh detectors, a fresh grace window, `_pending_drift` cleared,
then replay of *this* run's history. Without that, run N+1 on the same model
inherits run N's ADWIN window. `stream_status.consumer_query_run_id` is the
only reliable "the query has actually stopped" signal — it is set after
`start()`, cleared after every stop and unconditionally at consumer startup.

**During a matrix the runner owns `stream_status.is_active` and
`producer_active`.** It reads a dropped consumer flag as the consumer's own
surrender and fails the cell. The matrix ledger `results/<name>/runs.jsonl` is
the resume state: a cell whose last line is `completed` is never redone, and
`make report` takes the last line per cell. `--only` re-runs create a new
`experiment_run` row with the same name. `MatrixRunner.run()` re-activates the
base model in its `finally`, because `base_model = "active"` is resolved afresh
per invocation, and it refuses to resume a ledger whose completed cells used a
different base.

**Retrains use the trainer's default seed and algorithm**
(`services/trainer/worker.py`, `params.get("seed", 20260827)`), not the run's.
Known limitation; document it, do not paper over it in results.

**Table names are pinned** in `streamcls/db/tables.py` and set via
`Meta.db_table`. Spark and psycopg2 address them by name.

**Label ids are constants** in `streamcls/labels.py`, never derived from a
fitted `StringIndexer`. Deriving them makes the integer for each class a
function of training class balance, so a retrain silently renumbers every class
and changes the meaning of every stored prediction.

**`foreachBatch` runs on the driver and is not serialized** — capturing `self`,
the repository and the detectors is fine. **`foreachPartition` is pickled and
shipped**, so its closure may capture only plain values; capturing anything
holding a `SparkSession` fails at runtime, not at import.

**`maxOffsetsPerTrigger` is a correctness requirement**, not a tuning knob:
without it a batch is bounded only by Kafka backlog, which risks driver OOM and
makes batch composition non-reproducible.

**Multi-line Django template comments need `{% comment %}`.** `{# #}` is
single-line, so a multi-line one leaks its tail into the rendered page.

## Conventions

- English throughout, including the UI.
- The JSON API uses one envelope: `{"ok": bool, "data": ..., "error": ...}`.
  Never return `str(exception)` to a caller.
- The client builds DOM with `createElement` + `textContent`; a test forbids
  `innerHTML` assignment.
- Mutating endpoints are POST-only, CSRF-enforced and staff-gated. Nothing is
  `@csrf_exempt`.
- Claims about guarantees should be exact: delivery is *at-least-once made
  effectively-once*, not exactly-once; model swap is *stop-and-restart*, not a
  hot swap; splits are byte-reproducible, fitted models are reproducible only to
  within floating-point aggregation noise.
