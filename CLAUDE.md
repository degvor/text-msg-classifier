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

make test-unit         # no JVM, no DB, no Docker
make test-integration  # needs db + kafka up
make test-spark        # runs inside the Spark image
make lint
```

Run a single test: `.venv/bin/python -m pytest tests/unit/test_detectors.py::test_matches_river -q`.
Integration tests need `--ds=config.settings.dev` and `POSTGRES_HOST=localhost
POSTGRES_PORT=55432` (the dev override publishes Postgres there).

## Constraints that shape everything

**There is no JRE on the dev machine.** Anything importing pyspark runs in
Docker. This is why `streamcls.drift`, `streamcls.corpus`, `streamcls.metrics`
and `streamcls.experiments` are pure Python with no Spark dependency — they hold
the logic the thesis actually reports on, and they must be testable locally.
`streamcls.spark.*` is the only place pyspark may be imported, and never at
`streamcls/__init__` level (the Django image installs `streamcls[web]` and must
not pull a 300 MB wheel).

**`src/`, `services/` and `web/` are bind-mounted in dev** via
`docker-compose.override.yml`. Editing them does not need a rebuild — but
gunicorn and the Spark drivers do need a container restart to re-import.

## Rules that are easy to break

**A Django `default=` is not a database default.** `batch_metric`,
`classified_message`, `drift_event`, `training_job`, `model_version` and
`stream_status` are all written by raw psycopg2 from the Spark services. Any
field on those with a Python `default=` also needs `db_default=`, or an INSERT
that omits the column hits a NOT NULL violation at runtime. This has bitten
three times.

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
