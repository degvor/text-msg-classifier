# Streaming text classification with online drift detection and adaptation

A Kafka → Spark Structured Streaming → Postgres → Django pipeline that classifies
a stream of short posts, **detects concept drift while it runs**, and retrains
and swaps the model automatically.

The system is an experimental platform, not a demo: drift is *injected* under
controlled parameters, *detected* by competing detectors, and *responded to* by
a guarded retraining loop — and detection delay, false-alarm rate and recovery
are measured against known change points.

---

## Quick start

```bash
make install        # uv venv + dev extras (host-side CLI, tests)
make env            # write .env from .env.example
make data           # download AG News (~31 MB, checksum-verified)
make up             # db, kafka (KRaft), web, producer, consumer, trainer-worker
make train PROMOTE=1  # train the initial model and activate it
open http://localhost:8000
```

`make up` builds whatever image is missing, including the shared Spark base
image the consumer and trainer build `FROM`; `make build` rebuilds all of them.
`make help` lists everything.

---

## Architecture

```
                    ┌──────────────┐
                    │   producer   │  seeded replay of a held-out corpus split,
                    │              │  Telegram-style perturbation, drift schedule
                    └──────┬───────┘
                           │ Kafka (KRaft, 1 partition for determinism)
                           ▼
                    ┌──────────────┐
                    │   consumer   │  Spark Structured Streaming
                    │              │  ├─ classify        (PipelineModel)
                    │              │  ├─ upsert rows     (foreachPartition)
                    │              │  ├─ one aggregation (driver-side scalars)
                    │              │  └─ DriftMonitor    (DDM · ADWIN · PSI · JS)
                    └──────┬───────┘
                           │ detect → guarded TrainingJob
                           ▼
   ┌────────────┐   ┌──────────────┐   ┌──────────────┐
   │  Postgres  │◄──│trainer-worker│──►│model registry│
   │            │   │ FOR UPDATE   │   │  + promotion │
   │ single     │   │ SKIP LOCKED  │   │    guard     │
   │ source of  │   └──────────────┘   └──────┬───────┘
   │ truth      │                             │ desired_model_version
   │            │◄────────────────────────────┘
   └─────┬──────┘        consumer swaps on its next control poll
         │
         ▼
   ┌────────────┐
   │  Django    │  dashboard + control plane + JSON API
   └────────────┘
```

Every cross-service contract is a Postgres table. There is no shared-file bus.

### Repository layout

| Path | What |
|---|---|
| `src/streamcls/` | Shared library. `drift/` and `corpus/` are pure Python — no Spark, no JVM. |
| `services/producer/` | Seeded, drift-scheduled stream generator |
| `services/consumer/` | Spark streaming job, drift monitor, idempotent sink |
| `services/trainer/` | Offline training (`train.py`) and the retrain worker (`worker.py`) |
| `web/` | Django project: dashboard, JSON API, admin |
| `tests/unit` | No JVM, no database, no Docker |
| `tests/integration` | Real Postgres; the producer suite also needs Kafka |

---

## Running an experiment

An experiment is one `ExperimentRun` row: a seed, a message count, and a drift
schedule. Same seed → byte-identical stream, which is what makes two runs
comparable. Each run consumes its own Kafka topic, `posts.run-<id>`, so batch
ids start at 0 for every run and no volume reset is needed between them.

### One run by hand

```bash
# A drift schedule to paste into a run
python -m streamcls.corpus schedule concept_abrupt --total 16000
```

```json
[{"at": 8000, "kind": "concept_abrupt", "params": {"map": {"2": 3, "3": 2}}}]
```

Create the run in the admin (`/admin/dashboard/experimentrun/`) or via the API,
set it active, and start the stream. Then:

```bash
make score                         # results table for recent runs
make compare BASELINE=1 ADAPTIVE=2 # the A/B figure
```

```
 run  name                             seed  adapt batches     acc     pre    post  delay  FA miss  recov retrain
   6  adaptive recovery            20260827     on      32   65.2%   90.0%   59.5%    0.0   0    0      —       1
```

### The matrix

The thesis numbers come from `experiments/matrix.toml`: 5 scenarios × 3 seeds
× 2 arms, plus 4 ablations on two scenarios — 54 cells, about 6.5 hours. The
runner performs every cell identically and unattended:

```bash
make matrix SPEC=experiments/smoke.toml ARGS=--dry-run   # the cell table and an estimate; touches nothing
make matrix SPEC=experiments/smoke.toml                  # 2 cells, ~5 min: proves the lifecycle end to end
make matrix                                              # the real one; leave it overnight
make report                                              # results/thesis-v1/summary.md, summary.csv, pairs.csv, detectors.csv
```

Per cell the runner stops both services and waits for the consumer to release
its query, writes the spec's `[config]` block into `stream_config`, re-activates
the base model, creates the run and its topic, lets the producer write the whole
stream, *then* starts the consumer against the complete backlog — so every batch
is exactly `max_offsets_per_trigger` rows and batch composition is a pure
function of message index. The cell is done when every message is classified
and no retrain is queued or running; it is then scored and one JSON line is
appended to `results/<name>/runs.jsonl`.

A spec is TOML:

| Block | What |
|---|---|
| `[matrix]` | `name`, `seeds`, `arms ⊆ {baseline, adaptive}` |
| `[defaults]` | `total_messages`, `mode`, `dataset_split`, `timeout_minutes`, `base_model` (`"active"` or a version id, resolved once and pinned for every cell) |
| `[config]` | the full `stream_config` every cell runs with; written before each cell so the matrix is reproducible from the file plus the commit it records |
| `[scenarios.<preset>]` | a preset from `streamcls.corpus.schedule`, or an explicit `schedule = [...]` |
| `[[ablations]]` | `name`, `scenarios`, `config` overrides; adaptive arm only |

`trigger_seconds` is pinned in `[config]` on purpose: detection delay, recovery
and swap latency are all measured in batches, and how many batches a retrain
spans depends on how fast batches arrive. Halving it halves the wall clock and
changes those numbers.

Things to know while it runs:

- **The ledger is the state.** `runs.jsonl` is append-only and one line per
  attempt; a cell whose last line says `completed` is skipped on re-run, so a
  crash or Ctrl-C (which stops the stream, marks the run `aborted`, records the
  cell and exits 130) is resumed by running the same command again. To redo a
  cell, delete its line or pass `ARGS="--only 'concept_abrupt/*'"`. `make report`
  takes the last line per cell.
- **The base model is restored on exit.** Adaptive cells promote retrained
  models; when the loop ends, finished, failed or interrupted, the runner
  re-activates the base, so the next `base_model = "active"` resolves to the
  same model. A crash gets no cleanup, so a resume refuses a ledger whose
  completed cells ran on a different base and tells you which
  `ARGS="--base-model <id>"` to pass.
- **The runner owns the stream flags.** Do not toggle the producer or consumer
  from the dashboard during a matrix; a dropped consumer flag is exactly what
  the consumer's own surrender looks like, so the cell is recorded as failed.
- **Never `make down` mid-matrix** — it is `docker compose down -v` and drops
  Postgres. `docker compose stop` / `docker compose start` keeps everything.
- `make matrix`, `make score` and `make compare` run on the host and reach the
  stack through the ports `docker-compose.override.yml` publishes
  (`HOST_ENV` in the Makefile); the services inside Docker keep using `db` and
  `kafka`.
- Retrains inside a run use the trainer's default seed and algorithm, not the
  run's — a known limitation, recorded as such.

### Drift scenarios

| Kind | Changes | Which signal should fire |
|---|---|---|
| `covariate_abrupt` / `covariate_gradual` | P(x) — perturbation intensity | vocabulary PSI |
| `prior_shift` | P(y) — the class mix | predicted-class PSI |
| `concept_abrupt` | P(y\|x) — the labelling rule | error-rate detectors **only** |
| `recurring` | alternates | tests whether adaptation over-fits |

**Use a bijection for concept drift** (`{"2": 3, "3": 2}`, not `{"2": 3}`). A
one-directional map makes two classes draw from the same text pool with
different labels, which is irreducibly ambiguous — accuracy is then capped by
Bayes error and no amount of adaptation can recover it.

---

## Measured results

Numbers from the runs in this repository, not illustrations.

**Classifier.** AG News, TF-IDF + logistic regression, 89,911 training rows:

```
TEST  n=7594  accuracy=0.9089  macro-F1=0.9088
  world 0.9083 · sports 0.9645 · business 0.8747 · sci_tech 0.8875   (per-class F1)
```

In line with the published ~92% baseline for this model class. Anything near
100% on AG News means leakage.

**Drift detection**, concept swap injected at message 3000 (batch 6):

| batches | accuracy | PSI vocabulary | what happened |
|---|---|---|---|
| 0–5 | ~0.90 | — | baseline, tracks the offline score |
| 6 | 0.51 | ~0.003 | ADWIN fires, **detection delay 0 batches** |
| 6–12 | ~0.50 | ~0.003 | retrain deferred until enough new-concept data |
| 13–31 | ~0.63 | ~0.003 | retrained, promoted, model swapped |

The flat PSI column is the point: under concept drift the *input* distribution
does not change, so the label-free detectors stay silent while the error rate
moves. The model is confidently wrong. That is a real limit of unsupervised
drift detection, measured rather than asserted.

**Detector comparison**, 40 seeded stationary streams of 3000 observations:

| error rate | DDM false alarms | ADWIN |
|---|---|---|
| 0.05 | 47.5% | 0% |
| 0.10 | 37.5% | 0% |
| 0.20 | 25.0% | 0% |
| 0.40 | 10.0% | 0% |

DDM's threshold is `p_min + 3·s_min`, and `s` shrinks without bound as the run
grows, so the band eventually becomes tighter than ordinary binomial
fluctuation. These figures match `river`'s implementation exactly (asserted in
`tests/unit/test_detectors.py::test_matches_river`), so it is the algorithm's
behaviour, not this implementation's. Adaptation is therefore gated on ADWIN;
DDM is recorded for comparison, and the gate is configurable for the ablation.

---

## Design decisions worth knowing

**Promotion is gated on recent-data accuracy, not the fixed test set.** Under
concept drift the held-out set still encodes the *old* labelling, so a correctly
adapted model necessarily scores worse on it. Gating there rejects every
adaptation and makes the loop structurally useless. The final adapted model in
this repo shows it plainly: `test=0.7257`, `recent=0.5967` on the concept it
actually serves.

**Delivery is at-least-once made effectively-once.** Spark commits Kafka offsets
*after* `foreachBatch` returns, so a crash between writing and committing
replays the batch. `UNIQUE (experiment_run_id, post_id)` plus an upsert absorbs
that. Verified by `SIGKILL`-ing the consumer mid-stream: batch ids resumed
mid-sequence, 10,000/10,000 rows, zero duplicates, no gaps. Not exactly-once —
Spark does not provide that here.

**"Hot reload" is stop-and-restart.** A `PipelineModel` is bound into the query
plan at `start()`; there is no supported mutation point, and the `pandas_udf`
alternative would need a second feature pipeline and therefore a second source
of train/serve skew. The query restarts against the same checkpoint, so the swap
is lossless, and its latency is reported as a measurement.

**Detector state is rebuilt by replay, never pickled.** `BatchMetric` stores one
bit per message (63 bytes per 500-message batch), so a restart replays the exact
error sequence.

**Reproducibility, precisely.** Splits and the stream are byte-reproducible from
a seed and asserted so by fingerprint. Fitted models are *not* bit-identical
across runs: L-BFGS `treeAggregate` sums gradients in task-completion order and
floating-point addition is not associative. Measured spread ≈ 3e-4 in accuracy.

**Labelled-feedback adaptation.** Retraining uses ground-truth labels from the
stream, which is not available in production. It is an upper bound on achievable
adaptation, reported as such; `StreamConfig.label_delay_batches` models the
delayed-feedback case.

---

## Development

```bash
make install          # uv venv + dev extras
make test-unit        # pure Python: no JVM, no database, no Docker
make test-integration # needs db + kafka
make test-spark       # runs inside the Spark image
make lint
```

Unit tests cover the drift maths, the perturbation engine, the schema and the
scorer, and need no container. **There is no JRE on a typical dev machine**, so
anything Spark-touching runs in Docker; `src/` is bind-mounted in
`docker-compose.override.yml`, so editing the library does not require a rebuild.

### A real Spark cluster

The default is `local[4]` — a pinned core count, because partitioning affects
model fitting. For scalability measurements:

```bash
make cluster N=3      # spark-master + 3 workers, UI on :8080
```

Jobs then need `SPARK_MASTER=spark://spark-master:7077` **and**
`SPARK_DRIVER_HOST=<resolvable name>`: Spark runs the driver in client mode, so
executors dial back to it. Without a resolvable host they retry forever while
the job makes no progress, so the session builder refuses to start rather than
hanging.

---

## What this replaces

This began as a bachelor's diploma project. The audit that motivated the rework
found, among other things:

- the classifier scored **65.2%** in production with `politics` recall at
  **0.00** — training and serving text shared *zero* sentences and 70.5% of
  production vocabulary was unseen, so 7 of 25 inputs produced an all-zero
  feature vector and fell through to the intercepts;
- the training corpus was 500 rows from 50 templates with 24% exact duplicates
  and no seed; leave-one-out Naive Bayes scored 100% on it, and there was no
  train/test split, evaluator or confusion matrix anywhere;
- `checkpointLocation` was set at session level, so offsets were never
  recovered and every restart silently discarded whatever arrived while down;
- the stop button did nothing — `awaitTermination()` blocked forever, so the
  control flag was read exactly once;
- `GET /stream/toggle/` had no method guard, and Django does not CSRF-check GET;
- and it did not build at all: `bitnami/spark:3.4.1` has been removed from
  Docker Hub.

`git log` has the original.
