"""The matrix runner, against a scripted world.

The runner's value is the *order* it does things in -- each step closes a
race that bit the manual procedure -- so these tests assert on call order and
on the state of the fake control plane at the moment each call is made, not
just on the final result.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from streamcls.corpus.schedule import PRESETS
from streamcls.db.repo import ControlState, RunProgress
from streamcls.matrix import (
    MAX_CONSECUTIVE_FAILURES,
    Cell,
    MatrixRunner,
    Phase,
    RunnerError,
    SpecError,
    append_row,
    completed_keys,
    estimate_seconds,
    expand,
    parse_spec,
    phase,
    read_ledger,
    result_row,
    select_cells,
)

# ---------------------------------------------------------------------------
# Spec fixtures
# ---------------------------------------------------------------------------

CONFIG = {
    "trigger_seconds": 10,
    "max_offsets_per_trigger": 500,
    "grace_batches": 3,
    "retrain_base_rows": 20_000,
    "retrain_min_new_rows": 2_000,
    "actionable_detectors": ["adwin_error"],
}


def thesis_spec() -> dict[str, Any]:
    """The shape of ``experiments/matrix.toml``: 5 x 3 x 2 + 4 x 2 x 3 = 54."""
    return {
        "matrix": {"name": "thesis-v1", "seeds": [20260827, 20260828, 20260829]},
        "defaults": {"total_messages": 16_000, "timeout_minutes": 25},
        "config": dict(CONFIG),
        "scenarios": {name: {} for name in PRESETS},
        "ablations": [
            {
                "name": "base_rows_0",
                "scenarios": ["concept_abrupt", "covariate_gradual"],
                "config": {"retrain_base_rows": 0},
            },
            {
                "name": "base_rows_5000",
                "scenarios": ["concept_abrupt", "covariate_gradual"],
                "config": {"retrain_base_rows": 5_000},
            },
            {
                "name": "min_new_500",
                "scenarios": ["concept_abrupt", "covariate_gradual"],
                "config": {"retrain_min_new_rows": 500},
            },
            {
                "name": "min_new_4000",
                "scenarios": ["concept_abrupt", "covariate_gradual"],
                "config": {"retrain_min_new_rows": 4_000},
            },
        ],
    }


def small_spec(**overrides: Any) -> dict[str, Any]:
    """One scenario, one seed, both arms: two cells."""
    data: dict[str, Any] = {
        "matrix": {"name": "smoke", "seeds": [7]},
        "defaults": {"total_messages": 2_000, "timeout_minutes": 10},
        "config": dict(CONFIG),
        "scenarios": {"concept_abrupt": {}},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


# ---------------------------------------------------------------------------
# Parsing and expansion
# ---------------------------------------------------------------------------


def test_thesis_matrix_expands_to_54_cells():
    cells = expand(parse_spec(thesis_spec()))
    assert len(cells) == 54
    assert sum(1 for c in cells if c.ablation is None) == 30
    assert sum(1 for c in cells if c.ablation is not None) == 24
    assert all(c.arm == "adaptive" for c in cells if c.ablation is not None)
    assert len({c.key for c in cells}) == 54, "keys must be unique"


def test_expansion_order_is_scenario_seed_arm_then_ablations():
    cells = expand(parse_spec(thesis_spec()))
    assert cells[0].key == "stationary/baseline/seed20260827"
    assert cells[1].key == "stationary/adaptive/seed20260827"
    assert cells[2].key == "stationary/baseline/seed20260828"
    assert cells[6].key == "prior_shift/baseline/seed20260827"
    assert cells[30].key == "concept_abrupt/adaptive/seed20260827/base_rows_0"


def test_cell_carries_everything_the_run_needs():
    cell = expand(parse_spec(thesis_spec()))[30]
    assert cell.run_name == "thesis-v1/concept_abrupt/adaptive/seed20260827/base_rows_0"
    assert cell.adaptation_enabled is True
    assert cell.total_messages == 16_000
    assert cell.timeout_s == 25 * 60
    assert cell.batch_size == 500 and cell.batches == 32
    assert cell.config["retrain_base_rows"] == 0, "ablation override applied"
    assert cell.config["retrain_min_new_rows"] == 2_000, "other knobs kept"
    assert cell.schedule[0]["kind"] == "concept_abrupt"
    assert cell.schedule[0]["at"] == 8_000, "preset scaled to the run length"


def test_a_scenario_may_declare_its_own_schedule():
    data = small_spec(
        scenarios={
            "custom": {
                "schedule": [{"at": 500, "kind": "prior_shift", "params": {"weights": [1, 1, 1, 5]}}]
            }
        }
    )
    spec = parse_spec(data)
    assert spec.scenarios["custom"][0]["at"] == 500


def test_arms_default_to_both_and_may_be_restricted():
    assert parse_spec(small_spec()).arms == ("baseline", "adaptive")
    spec = parse_spec(small_spec(matrix={"name": "x", "seeds": [1], "arms": ["adaptive"]}))
    assert [c.arm for c in expand(spec)] == ["adaptive"]


def test_base_model_is_active_or_an_id():
    assert parse_spec(small_spec()).base_model is None
    assert parse_spec(small_spec(defaults={"base_model": 12})).base_model == 12
    with pytest.raises(SpecError, match="base_model"):
        parse_spec(small_spec(defaults={"base_model": "latest"}))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(default={"total_messages": 1}), "unknown keys"),
        (lambda d: d["matrix"].pop("name"), "name"),
        (lambda d: d["matrix"].update(name="thesis v1"), "letters, digits"),
        (lambda d: d["matrix"].update(seeds=[]), "seeds"),
        (lambda d: d["matrix"].update(seeds=[1, 1]), "duplicates"),
        (lambda d: d["matrix"].update(arms=["adaptive", "control"]), "arms"),
        (lambda d: d["defaults"].update(total_messages=4), "at least 8"),
        (lambda d: d["defaults"].update(mode="batch"), "mode"),
        (lambda d: d["defaults"].update(timeout_minutes=0), "timeout_minutes"),
        (lambda d: d.pop("config"), r"\[config\]"),
        (lambda d: d["config"].pop("trigger_seconds"), "trigger_seconds"),
        (lambda d: d["config"].update(max_offsets_per_trigger=0), "max_offsets_per_trigger"),
        (lambda d: d.update(scenarios={}), "no scenarios"),
        (lambda d: d.update(scenarios={"made_up": {}}), "not a preset"),
        (lambda d: d.update(scenarios={"x": {"schedule": [{"kind": "concept_abrupt"}]}}), "invalid"),
        (lambda d: d.update(scenarios={"concept_abrupt": {"sched": []}}), "unknown keys"),
        (
            lambda d: d.update(ablations=[{"name": "a", "scenarios": ["recurring"], "config": {"x": 1}}]),
            "undeclared scenarios",
        ),
        (
            lambda d: d.update(ablations=[{"name": "adaptive", "scenarios": ["concept_abrupt"], "config": {"x": 1}}]),
            "collides",
        ),
        (
            lambda d: d.update(ablations=[{"name": "a", "scenarios": ["concept_abrupt"], "config": {}}]),
            "non-empty config",
        ),
        (
            lambda d: d.update(
                ablations=[
                    {"name": "a", "scenarios": ["concept_abrupt"], "config": {"x": 1}},
                    {"name": "a", "scenarios": ["concept_abrupt"], "config": {"x": 2}},
                ]
            ),
            "declared twice",
        ),
    ],
)
def test_malformed_specs_are_rejected_by_name(mutate, message):
    data = small_spec()
    mutate(data)
    with pytest.raises(SpecError, match=message):
        parse_spec(data)


def test_ablations_need_the_adaptive_arm():
    data = small_spec(
        matrix={"name": "x", "seeds": [1], "arms": ["baseline"]},
        ablations=[{"name": "a", "scenarios": ["concept_abrupt"], "config": {"x": 1}}],
    )
    with pytest.raises(SpecError, match="adaptive arm"):
        parse_spec(data)


def test_a_run_name_that_would_not_fit_the_column_is_rejected_up_front():
    data = small_spec(matrix={"name": "m" * 100, "seeds": [1]})
    with pytest.raises(SpecError, match="limit is 120"):
        expand(parse_spec(data))


def test_select_cells_matches_keys_with_shell_patterns():
    cells = expand(parse_spec(thesis_spec()))
    assert select_cells(cells, []) == cells
    only = select_cells(cells, ["concept_abrupt/*"])
    assert len(only) == 6 + 12 and all(c.scenario == "concept_abrupt" for c in only)
    one = select_cells(cells, ["stationary/adaptive/seed20260828"])
    assert [c.key for c in one] == ["stationary/adaptive/seed20260828"]
    ablations = select_cells(cells, ["*/min_new_*"])
    assert len(ablations) == 12
    assert select_cells(cells, ["nothing/*"]) == []


def test_estimate_is_triggers_plus_overhead():
    baseline, adaptive = expand(parse_spec(small_spec()))
    assert baseline.batches == 4
    assert estimate_seconds(baseline) == pytest.approx(4 * 10 + 60)
    assert estimate_seconds(adaptive) > estimate_seconds(baseline)


# ---------------------------------------------------------------------------
# The decision table
# ---------------------------------------------------------------------------


def progress(**overrides: Any) -> RunProgress:
    fields: dict[str, Any] = {
        "status": "completed",
        "classified": 0,
        "last_batch_id": None,
        "active_jobs": 0,
        "is_active": True,
        "producer_active": False,
        "active_experiment_run_id": 1,
        "consumer_query_run_id": 1,
        "last_error": "",
    }
    fields.update(overrides)
    return RunProgress(**fields)


@pytest.mark.parametrize(
    ("snapshot", "elapsed", "expected"),
    [
        (progress(classified=500), 10, Phase.RUNNING),
        (progress(classified=2000), 10, Phase.COMPLETE),
        (progress(classified=2000, active_jobs=1), 10, Phase.RUNNING),
        (progress(classified=2000, is_active=False), 10, Phase.COMPLETE),
        (progress(classified=500, is_active=False), 10, Phase.CONSUMER_GAVE_UP),
        (progress(classified=500), 600, Phase.TIMED_OUT),
        (progress(classified=2000), 600, Phase.COMPLETE),
        (progress(classified=500, is_active=False), 600, Phase.CONSUMER_GAVE_UP),
    ],
)
def test_phase_prefers_completeness_then_surrender_then_the_clock(snapshot, elapsed, expected):
    assert phase(snapshot, total_messages=2000, elapsed_s=elapsed, timeout_s=600) is expected


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class FakeScore:
    def __init__(self, run_id: int, batch_size: int) -> None:
        self.run_id = run_id
        self.batch_size = batch_size

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "accuracy_overall": 0.9, "batch_size": self.batch_size}


def test_ledger_round_trips_and_only_completed_rows_count(tmp_path):
    cells = expand(parse_spec(small_spec()))
    ledger = tmp_path / "runs.jsonl"
    t0 = datetime(2026, 8, 30, 22, 0, tzinfo=UTC)
    append_row(
        ledger,
        result_row(
            cells[0], status="completed", run_id=1, base_model_version_id=3,
            started_at=t0, ended_at=t0 + timedelta(seconds=90), score=FakeScore(1, 500),
        ),
    )
    append_row(
        ledger,
        result_row(
            cells[1], status="timed_out", run_id=2, base_model_version_id=3,
            started_at=t0, ended_at=t0 + timedelta(minutes=10), error="stalled",
        ),
    )
    rows = read_ledger(ledger)
    assert [r["key"] for r in rows] == [cells[0].key, cells[1].key]
    assert rows[0]["score"]["accuracy_overall"] == 0.9
    assert rows[0]["wall_s"] == 90.0
    assert rows[1]["score"] is None and rows[1]["error"] == "stalled"
    assert rows[0]["config"] == CONFIG
    assert completed_keys(ledger) == {cells[0].key}


def test_a_damaged_ledger_line_is_skipped_not_fatal(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    ledger.write_text('{"key": "a/baseline/seed1", "status": "completed"}\n{"key": "b/adap', "utf-8")
    assert completed_keys(ledger) == {"a/baseline/seed1"}
    assert completed_keys(tmp_path / "missing.jsonl") == set()


# ---------------------------------------------------------------------------
# A scripted world for the runner
# ---------------------------------------------------------------------------


@dataclass
class FakeModel:
    id: int
    status: str


@dataclass
class FakeRun:
    id: int
    name: str
    total_messages: int
    adaptation_enabled: bool
    status: str = "draft"
    classified: int = 0
    last_batch_id: int | None = None
    active_jobs: int = 0
    consumer_config: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)


class FakeWorld:
    """A Repository stand-in that also *is* the producer, consumer and trainer.

    ``sleep`` advances the world one tick: the producer finishes a draft run,
    the consumer classifies ``per_tick`` messages of an active one, a queued
    retrain completes, a stopped query is released. Every repository call is
    recorded together with the control-plane state at that instant, so tests
    can assert not just *that* something was called but *when*.
    """

    def __init__(
        self,
        *,
        per_tick: int = 1_000,
        release_ticks: int = 0,
        retrain_at_end: bool = False,
        enqueue_delay_ticks: int = 1,
        job_ticks: int = 2,
        gives_up_after: int | None = None,
        stalls: bool = False,
        producer_aborts: bool = False,
        active_model: int | None = 1,
        heartbeats: dict[str, float | None] | None = None,
    ) -> None:
        self.per_tick = per_tick
        self.release_ticks = release_ticks
        self.retrain_at_end = retrain_at_end
        self.enqueue_delay_ticks = enqueue_delay_ticks
        self.job_ticks = job_ticks
        self.gives_up_after = gives_up_after
        self.stalls = stalls
        self.producer_aborts = producer_aborts

        self.calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.runs: dict[int, FakeRun] = {}
        self.config: dict[str, Any] = {"id": 1, "updated_at": "t", **CONFIG}
        self.control: dict[str, Any] = {
            "is_active": False,
            "producer_active": False,
            "active_experiment_run_id": None,
            "active_model_version_id": None,
            "desired_model_version_id": active_model,
            "consumer_query_run_id": None,
        }
        self.last_error = ""
        self.models = {1: FakeModel(1, "active" if active_model == 1 else "ready"),
                       2: FakeModel(2, "ready")}
        self.active_model = active_model
        self.heartbeats = heartbeats if heartbeats is not None else {"producer": 1.0, "consumer": 2.0}
        self.time = 0.0
        self.ticks = 0
        self._release_countdown: int | None = None
        self._enqueue_countdown: int | None = None
        self._job_countdown: int | None = None
        self.jobs_finished = 0
        self.query_started_with_model: list[int | None] = []
        #: Ticks on which Ctrl-C is pressed, once each.
        self.interrupt_at_ticks: set[int] = set()
        self.topics: list[str] = []
        #: Topics on the broker, including any left over from a previous database.
        self.existing_topics: set[str] = set()
        self.scored: list[tuple[int, int]] = []

    # --- recording ---------------------------------------------------------
    def _record(self, method: str, arg: Any = None) -> None:
        run = self.runs.get(self.control["active_experiment_run_id"])
        self.calls.append(
            (method, arg, {**self.control, "run_status": run.status if run else None,
                           "active_jobs": run.active_jobs if run else None,
                           "jobs_finished": self.jobs_finished,
                           "active_model": self.active_model})
        )

    def calls_named(self, *names: str) -> list[tuple[str, Any, dict[str, Any]]]:
        return [c for c in self.calls if c[0] in names]

    def order(self) -> list[str]:
        return [c[0] for c in self.calls]

    # --- Repository API ----------------------------------------------------
    def read_control(self) -> ControlState:
        return ControlState(**self.control)

    def read_config(self) -> dict[str, Any]:
        return dict(self.config)

    def update_config(self, **fields: Any) -> None:
        unknown = sorted(set(fields) - (set(self.config) - {"id", "updated_at"}))
        if unknown:
            raise ValueError(f"unknown columns {unknown}")
        self._record("update_config", dict(fields))
        self.config.update(fields)

    def promote_model(self, model_id: int, *, force: bool = False, **_: Any) -> bool:
        self._record("promote_model", (model_id, force))
        for m in self.models.values():
            if m.status == "active":
                m.status = "retired"
        self.models[model_id].status = "active"
        self.active_model = model_id
        self.control["desired_model_version_id"] = model_id
        return True

    def get_model(self, model_id: int) -> FakeModel:
        if model_id not in self.models:
            raise LookupError(f"model version {model_id} not found")
        return self.models[model_id]

    def get_active_model(self) -> FakeModel | None:
        return next((m for m in self.models.values() if m.status == "active"), None)

    def heartbeat_ages(self) -> dict[str, float | None]:
        return dict(self.heartbeats)

    def create_run(self, *, name: str, total_messages: int, adaptation_enabled: bool,
                   consumer_config: dict[str, Any], topic_for_run=None, **fields: Any) -> int:
        run_id = max(self.runs, default=0) + 1
        merged = dict(consumer_config)
        if topic_for_run is not None:
            merged["topic"] = topic_for_run(run_id)
        self.runs[run_id] = FakeRun(
            run_id, name, total_messages, adaptation_enabled, consumer_config=merged, fields=fields
        )
        self._record("create_run", run_id)
        return run_id

    def find_runs_by_name(self, name: str) -> list[dict[str, Any]]:
        return [
            {"id": r.id, "name": r.name, "status": r.status, "consumer_config": r.consumer_config}
            for r in sorted(self.runs.values(), key=lambda r: r.id)
            if r.name == name
        ]

    def activate_run(self, run_id: int, *, producer: bool, consumer: bool) -> None:
        if run_id not in self.runs:
            raise LookupError("not found")
        self.control.update(
            active_experiment_run_id=run_id, producer_active=producer, is_active=consumer
        )
        self.last_error = ""
        self._record("activate_run", (run_id, producer, consumer))

    def set_stream_active(self, active: bool) -> None:
        self._record("set_stream_active", active)
        self.control["is_active"] = active

    def set_producer_active(self, active: bool) -> None:
        self._record("set_producer_active", active)
        self.control["producer_active"] = active

    def mark_run_finished(self, run_id: int, status: str = "completed") -> None:
        self._record("mark_run_finished", (run_id, status))
        self.runs[run_id].status = status

    def run_progress(self, run_id: int) -> RunProgress:
        run = self.runs[run_id]
        return RunProgress(
            status=run.status,
            classified=run.classified,
            last_batch_id=run.last_batch_id,
            active_jobs=run.active_jobs,
            is_active=self.control["is_active"],
            producer_active=self.control["producer_active"],
            active_experiment_run_id=self.control["active_experiment_run_id"],
            consumer_query_run_id=self.control["consumer_query_run_id"],
            last_error=self.last_error,
        )

    # --- injected collaborators ------------------------------------------
    def topic_for_run(self, run_id: int) -> str:
        return f"posts.run-{run_id}"

    def ensure_topic(self, name: str) -> bool:
        self._record("ensure_topic", name)
        self.topics.append(name)
        created = name not in self.existing_topics
        self.existing_topics.add(name)
        return created

    def score(self, run_id: int, batch_size: int) -> FakeScore:
        self._record("score", (run_id, batch_size))
        self.scored.append((run_id, batch_size))
        return FakeScore(run_id, batch_size)

    def clock(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.time += seconds
        self.tick()

    # --- the world ---------------------------------------------------------
    def tick(self) -> None:
        self.ticks += 1
        if self.ticks in self.interrupt_at_ticks:
            raise KeyboardInterrupt
        c = self.control
        run = self.runs.get(c["active_experiment_run_id"])

        # Producer: a replay run is written in one tick.
        if c["producer_active"] and run is not None and run.status == "draft":
            run.status = "aborted" if self.producer_aborts else "completed"
            c["producer_active"] = False

        # Consumer.
        if c["is_active"] and run is not None and run.status != "draft":
            if c["consumer_query_run_id"] is None:
                c["consumer_query_run_id"] = run.id
                self.query_started_with_model.append(self.active_model)
            elif self.gives_up_after is not None and run.classified >= self.gives_up_after:
                c["is_active"] = False
                c["consumer_query_run_id"] = None
                self.last_error = "Query terminated with exception: boom"
            elif not self.stalls and run.classified < run.total_messages:
                run.classified = min(run.total_messages, run.classified + self.per_tick)
                run.last_batch_id = math.ceil(run.classified / CONFIG["max_offsets_per_trigger"]) - 1
                if (
                    run.classified == run.total_messages
                    and self.retrain_at_end
                    and run.adaptation_enabled
                ):
                    self._enqueue_countdown = self.enqueue_delay_ticks

        # Trainer.
        if run is not None and self._enqueue_countdown is not None:
            self._enqueue_countdown -= 1
            if self._enqueue_countdown <= 0:
                self._enqueue_countdown = None
                run.active_jobs = 1
                self._job_countdown = self.job_ticks
        if run is not None and self._job_countdown is not None:
            self._job_countdown -= 1
            if self._job_countdown <= 0:
                self._job_countdown = None
                run.active_jobs = 0
                self.jobs_finished += 1

        # A stopped query is released after `release_ticks`.
        if not c["is_active"] and c["consumer_query_run_id"] is not None:
            if self._release_countdown is None:
                self._release_countdown = self.release_ticks
            if self._release_countdown <= 0:
                c["consumer_query_run_id"] = None
                self._release_countdown = None
            else:
                self._release_countdown -= 1
        elif c["is_active"]:
            self._release_countdown = None


#: A checkpoint root with nothing left over in it.
NO_CHECKPOINTS = Path("/nonexistent/checkpoints")


def runner(world: FakeWorld, **kwargs: Any) -> MatrixRunner:
    kwargs.setdefault("checkpoint_for_run", lambda run_id: NO_CHECKPOINTS / f"run-{run_id}")
    return MatrixRunner(
        world,
        topic_for_run=world.topic_for_run,
        ensure_topic=world.ensure_topic,
        score=world.score,
        git_commit="deadbeef",
        poll_s=5.0,
        sleep=world.sleep,
        clock=world.clock,
        now=lambda: datetime(2026, 8, 30, tzinfo=UTC) + timedelta(seconds=world.time),
        **kwargs,
    )


def two_cells(**overrides: Any) -> tuple[Any, list[Cell]]:
    spec = parse_spec(small_spec(**overrides))
    return spec, expand(spec)


def retrain_promotes_during(world: FakeWorld, run_id: int) -> None:
    """Have the trainer promote model 2 halfway through ``run_id``, as an
    adaptive run's retrain does."""
    original_tick = world.tick

    def tick() -> None:
        original_tick()
        run = world.runs.get(world.control["active_experiment_run_id"])
        if run and run.id == run_id and run.classified >= 1000 and world.active_model == 1:
            world.promote_model(2)

    world.tick = tick  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# The lifecycle
# ---------------------------------------------------------------------------


def test_a_cell_runs_the_lifecycle_in_order(tmp_path):
    world = FakeWorld()
    spec, cells = two_cells()
    code = runner(world).run(spec, cells[:1], ledger=tmp_path / "runs.jsonl")

    assert code == 0
    names = [n for n in world.order() if n not in ("set_producer_active",)]
    assert names == [
        "set_stream_active",      # 1. quiesce
        "update_config",          # 2. knobs
        "promote_model",          # 3. base model
        "create_run",             # 4. run row
        "ensure_topic",           #    and its topic
        "activate_run",           # 5. producer only
        "set_stream_active",      # 6. consumer on, after the producer finished
        "set_stream_active",      # 7. consumer off
        "score",                  #    then score
        "set_stream_active",      # matrix over: leave both flags off
        "promote_model",          #    and the base model active
    ]
    rows = read_ledger(tmp_path / "runs.jsonl")
    assert len(rows) == 1 and rows[0]["status"] == "completed"
    assert rows[0]["run_id"] == 1 and rows[0]["score"]["run_id"] == 1
    assert rows[0]["git_commit"] == "deadbeef"
    assert world.topics == ["posts.run-1"]
    assert world.runs[1].consumer_config["topic"] == "posts.run-1"
    assert world.runs[1].consumer_config["cell_key"] == cells[0].key
    assert world.runs[1].consumer_config["stream_config"]["trigger_seconds"] == 10
    assert "updated_at" not in world.runs[1].consumer_config["stream_config"]
    assert world.runs[1].fields["git_commit"] == "deadbeef"
    assert world.runs[1].fields["scenario"] == "concept_abrupt"
    assert world.control["is_active"] is False and world.control["producer_active"] is False


def test_the_consumer_is_started_only_after_the_producer_has_finished(tmp_path):
    world = FakeWorld()
    spec, cells = two_cells()
    runner(world).run(spec, cells[:1], ledger=tmp_path / "runs.jsonl")

    (_, producer, _), = world.calls_named("activate_run")
    assert producer == (1, True, False), "producer only; consumer stays off"
    starts = [s for _, active, s in world.calls_named("set_stream_active") if active]
    assert len(starts) == 1
    assert starts[0]["run_status"] == "completed", "backlog is complete before the query starts"
    assert starts[0]["producer_active"] is False


def test_the_base_model_is_reset_only_once_the_query_is_released(tmp_path):
    """A query holds its model until it stops; promoting underneath it would
    race the swap. Here the consumer takes three polls to let go."""
    world = FakeWorld(release_ticks=3)
    world.control["consumer_query_run_id"] = 99  # a previous cell's query, still alive
    world.control["is_active"] = True
    spec, cells = two_cells()
    runner(world).run(spec, cells[:1], ledger=tmp_path / "runs.jsonl")

    (_, (model_id, force), state), = world.calls_named("promote_model")[:1]
    assert (model_id, force) == (1, True)
    assert state["consumer_query_run_id"] is None
    assert state["is_active"] is False
    # And the new query was started with the base model in place.
    assert world.query_started_with_model == [1]


def test_every_cell_starts_from_the_base_model_even_after_a_retrain(tmp_path):
    world = FakeWorld()
    spec, cells = two_cells()
    retrain_promotes_during(world, run_id=1)
    runner(world).run(spec, cells, ledger=tmp_path / "runs.jsonl")

    resets = [arg for name, arg, _ in world.calls if name == "promote_model" and arg[1]]
    assert resets == [(1, True), (1, True), (1, True)], "one per cell, one on the way out"
    assert world.query_started_with_model == [1, 1]
    assert world.runs[2].fields["base_model_version_id"] == 1


def test_the_matrix_leaves_the_base_model_active_after_a_retrain(tmp_path):
    """``base_model = "active"`` is resolved afresh by every invocation, so a
    matrix that ends on a retrained model would hand it to the next one as
    its base."""
    world = FakeWorld()
    spec, cells = two_cells()
    retrain_promotes_during(world, run_id=2)
    code = runner(world).run(spec, cells, ledger=tmp_path / "runs.jsonl")

    assert code == 0
    assert world.active_model == 1
    assert world.models[1].status == "active" and world.models[2].status == "retired"


def test_completion_waits_for_a_retrain_enqueued_after_the_last_batch(tmp_path):
    """The last batch can enqueue a retrain *after* its metric row lands, so
    a single complete-looking poll is not enough. Here the job appears one
    poll after the counts first look complete."""
    world = FakeWorld(retrain_at_end=True, enqueue_delay_ticks=2, job_ticks=2)
    spec, cells = two_cells()
    adaptive = [c for c in cells if c.adaptation_enabled]
    code = runner(world).run(spec, adaptive, ledger=tmp_path / "runs.jsonl")

    assert code == 0
    (_, _, at_score), = world.calls_named("score")
    assert at_score["jobs_finished"] == 1, "the late retrain ran to completion before scoring"
    assert at_score["active_jobs"] == 0
    rows = read_ledger(tmp_path / "runs.jsonl")
    assert rows[0]["status"] == "completed"


def test_a_stalled_cell_times_out_is_marked_aborted_and_the_matrix_continues(tmp_path):
    world = FakeWorld(stalls=True)
    spec, cells = two_cells(defaults={"timeout_minutes": 1})
    code = runner(world).run(spec, cells, ledger=tmp_path / "runs.jsonl")

    assert code == 1
    rows = read_ledger(tmp_path / "runs.jsonl")
    assert [r["status"] for r in rows] == ["timed_out", "timed_out"]
    assert [r["run_id"] for r in rows] == [1, 2]
    assert "0/2000 classified" in rows[0]["error"]
    assert all(r.status == "aborted" for r in world.runs.values())
    assert world.scored == []
    assert world.control["is_active"] is False


def test_stop_on_timeout_ends_the_matrix_at_the_first_timeout(tmp_path):
    world = FakeWorld(stalls=True)
    spec, cells = two_cells(defaults={"timeout_minutes": 1})
    code = runner(world).run(spec, cells, ledger=tmp_path / "runs.jsonl", stop_on_timeout=True)

    assert code == 1
    assert [r["status"] for r in read_ledger(tmp_path / "runs.jsonl")] == ["timed_out"]
    assert len(world.runs) == 1


def test_too_many_consecutive_failures_stop_the_night(tmp_path):
    world = FakeWorld(stalls=True)
    spec = parse_spec(small_spec(matrix={"name": "m", "seeds": [1, 2, 3]}, defaults={"timeout_minutes": 1}))
    cells = expand(spec)
    assert len(cells) == 6
    code = runner(world).run(spec, cells, ledger=tmp_path / "runs.jsonl")

    assert code == 1
    assert len(read_ledger(tmp_path / "runs.jsonl")) == MAX_CONSECUTIVE_FAILURES


def test_a_consumer_that_gives_up_is_recorded_with_its_error(tmp_path):
    world = FakeWorld(gives_up_after=1000)
    spec, cells = two_cells()
    code = runner(world).run(spec, cells[:1], ledger=tmp_path / "runs.jsonl")

    assert code == 1
    (row,) = read_ledger(tmp_path / "runs.jsonl")
    assert row["status"] == "consumer_gave_up"
    assert "boom" in row["error"]
    assert world.runs[1].status == "aborted"


def test_a_producer_failure_is_recorded_without_starting_the_consumer(tmp_path):
    world = FakeWorld(producer_aborts=True)
    spec, cells = two_cells()
    code = runner(world).run(spec, cells[:1], ledger=tmp_path / "runs.jsonl")

    assert code == 1
    (row,) = read_ledger(tmp_path / "runs.jsonl")
    assert row["status"] == "producer_failed"
    assert not [s for _, active, s in world.calls_named("set_stream_active") if active]


def test_a_repository_error_mid_cell_becomes_an_error_row_not_a_crash(tmp_path):
    world = FakeWorld()
    spec, cells = two_cells()
    original = world.ensure_topic

    def flaky(name: str) -> bool:
        if name == "posts.run-1":
            raise ConnectionError("kafka unreachable")
        return original(name)

    world.ensure_topic = flaky  # type: ignore[method-assign]
    code = runner(world).run(spec, cells, ledger=tmp_path / "runs.jsonl")

    rows = read_ledger(tmp_path / "runs.jsonl")
    assert [r["status"] for r in rows] == ["error", "completed"]
    assert "kafka unreachable" in rows[0]["error"]
    assert world.runs[1].status == "aborted"
    assert code == 1


# ---------------------------------------------------------------------------
# Leftovers from a previous database: run ids restart at 1, host-side
# checkpoints and Kafka topics do not
# ---------------------------------------------------------------------------


def test_a_leftover_checkpoint_for_a_new_run_id_fails_the_cell_before_the_producer(tmp_path):
    """A query resuming a leftover checkpoint believes the stream is already
    consumed, reads nothing and stalls, silently, until the cell times out."""
    world = FakeWorld()
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "run-1" / "commits").mkdir(parents=True)
    spec, cells = two_cells()
    code = runner(world, checkpoint_for_run=lambda run_id: checkpoints / f"run-{run_id}").run(
        spec, cells, ledger=tmp_path / "runs.jsonl"
    )

    assert code == 1
    rows = read_ledger(tmp_path / "runs.jsonl")
    assert [r["status"] for r in rows] == ["error", "completed"]
    assert str(checkpoints / "run-1") in rows[0]["error"]
    assert world.runs[1].status == "aborted"
    assert [arg for _, arg, _ in world.calls_named("activate_run")] == [(2, True, False)]
    assert world.topics == ["posts.run-2"], "no topic is created for the failed run"


def test_every_ledger_row_records_whether_the_tree_was_dirty(tmp_path):
    """The ledger is the evidence ``make report`` reads; the flag only lived
    in Postgres before."""
    world = FakeWorld()
    world.existing_topics.add("posts.run-1")  # one failed row, one completed
    spec, cells = two_cells()
    runner(world, git_dirty=True).run(spec, cells, ledger=tmp_path / "runs.jsonl")

    rows = read_ledger(tmp_path / "runs.jsonl")
    assert [(r["status"], r["git_dirty"]) for r in rows] == [("error", True), ("completed", True)]


def test_a_topic_that_existed_before_its_run_fails_the_cell_before_the_producer(tmp_path):
    """The Kafka side of the same collision: the producer would append a
    second copy of the stream to the leftover one."""
    world = FakeWorld()
    world.existing_topics.add("posts.run-1")
    spec, cells = two_cells()
    code = runner(world).run(spec, cells, ledger=tmp_path / "runs.jsonl")

    assert code == 1
    rows = read_ledger(tmp_path / "runs.jsonl")
    assert [r["status"] for r in rows] == ["error", "completed"]
    assert "posts.run-1" in rows[0]["error"]
    assert world.runs[1].status == "aborted"
    assert [arg for _, arg, _ in world.calls_named("activate_run")] == [(2, True, False)]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_completed_cells_in_the_ledger_are_skipped(tmp_path):
    world = FakeWorld()
    spec, cells = two_cells()
    ledger = tmp_path / "runs.jsonl"
    t0 = datetime(2026, 8, 30, tzinfo=UTC)
    append_row(
        ledger,
        result_row(cells[0], status="completed", run_id=41, base_model_version_id=1,
                   started_at=t0, ended_at=t0, score=FakeScore(41, 500)),
    )
    code = runner(world).run(spec, cells, ledger=ledger)

    assert code == 0
    assert [r.name for r in world.runs.values()] == [cells[1].run_name]
    rows = read_ledger(ledger)
    assert [r["key"] for r in rows] == [cells[0].key, cells[1].key]
    assert rows[0]["run_id"] == 41


def test_a_finished_run_that_never_reached_the_ledger_is_adopted_not_redone(tmp_path):
    world = FakeWorld()
    spec, cells = two_cells()
    cell = cells[0]
    # As left by a runner that crashed between step 7 and the append.
    world.runs[5] = FakeRun(
        5, cell.run_name, cell.total_messages, cell.adaptation_enabled, status="completed",
        classified=cell.total_messages, last_batch_id=3,
        consumer_config={"matrix": cell.matrix, "cell_key": cell.key},
    )
    # A half-consumed run with the same name must not be adopted.
    world.runs[4] = FakeRun(
        4, cell.run_name, cell.total_messages, cell.adaptation_enabled, status="completed",
        classified=1000, consumer_config={"matrix": cell.matrix, "cell_key": cell.key},
    )
    code = runner(world).run(spec, cells[:1], ledger=tmp_path / "runs.jsonl")

    assert code == 0
    assert not world.calls_named("create_run")
    assert world.scored == [(5, 500)]
    (row,) = read_ledger(tmp_path / "runs.jsonl")
    assert row["run_id"] == 5 and row["status"] == "completed"


def test_a_run_from_another_matrix_with_the_same_name_is_not_adopted(tmp_path):
    world = FakeWorld()
    spec, cells = two_cells()
    cell = cells[0]
    world.runs[5] = FakeRun(
        5, cell.run_name, cell.total_messages, cell.adaptation_enabled, status="completed",
        classified=cell.total_messages, consumer_config={"matrix": "other", "cell_key": cell.key},
    )
    runner(world).run(spec, cells[:1], ledger=tmp_path / "runs.jsonl")
    assert [arg for name, arg, _ in world.calls if name == "create_run"] == [6]


def test_a_resume_against_a_different_base_model_is_refused(tmp_path):
    """A crash leaves a retrained model active with no ``finally`` to undo
    it. Resuming on it would mix two base models in one ledger."""
    world = FakeWorld()
    world.promote_model(2)
    world.calls.clear()
    spec, cells = two_cells()
    ledger = tmp_path / "runs.jsonl"
    t0 = datetime(2026, 8, 30, tzinfo=UTC)
    append_row(
        ledger,
        result_row(cells[0], status="completed", run_id=41, base_model_version_id=1,
                   started_at=t0, ended_at=t0, score=FakeScore(41, 500)),
    )
    with pytest.raises(RunnerError, match="--base-model 1"):
        runner(world).run(spec, cells, ledger=ledger)
    assert world.calls == []
    assert len(read_ledger(ledger)) == 1

    pinned = parse_spec(small_spec(defaults={"base_model": 1}))
    assert runner(world).run(pinned, expand(pinned), ledger=ledger) == 0
    assert [r.name for r in world.runs.values()] == [cells[1].run_name]
    assert world.runs[1].fields["base_model_version_id"] == 1


# ---------------------------------------------------------------------------
# Ctrl-C
# ---------------------------------------------------------------------------


def test_ctrl_c_stops_the_stream_marks_the_run_aborted_and_writes_a_row(tmp_path):
    world = FakeWorld()
    world.interrupt_at_ticks = {3}  # during the consumer phase of cell 1
    spec, cells = two_cells()
    code = runner(world).run(spec, cells, ledger=tmp_path / "runs.jsonl")

    assert code == 130
    (row,) = read_ledger(tmp_path / "runs.jsonl")
    assert row["status"] == "interrupted" and row["run_id"] == 1
    assert world.runs[1].status == "aborted"
    assert world.control["is_active"] is False
    assert world.control["producer_active"] is False
    assert len(world.runs) == 1, "the second cell never started"


def interrupted_after_a_retrain(world: FakeWorld, *presses: int) -> list[Cell]:
    """The adaptive cell alone, with model 2 promoted at tick 3 and Ctrl-C
    pressed on ``presses``; tick 4 is inside the consumer phase."""
    retrain_promotes_during(world, run_id=1)
    world.interrupt_at_ticks = set(presses)
    _, cells = two_cells()
    return [c for c in cells if c.adaptation_enabled]


def test_ctrl_c_puts_the_base_model_back_once_the_query_has_let_go(tmp_path):
    """Resuming resolves ``base_model = "active"`` again: without the reset
    the remaining cells would run from the retrained model."""
    world = FakeWorld(release_ticks=2)
    spec, _ = two_cells()
    adaptive = interrupted_after_a_retrain(world, 4)
    code = runner(world).run(spec, adaptive, ledger=tmp_path / "runs.jsonl")

    assert code == 130
    promotes = world.calls_named("promote_model")
    assert [arg for _, arg, _ in promotes] == [(1, True), (2, False), (1, True)]
    assert world.active_model == 1
    (_, _, at_reset) = promotes[-1]
    assert at_reset["consumer_query_run_id"] is None, "not underneath a query still stopping"
    assert at_reset["is_active"] is False


def test_a_second_ctrl_c_during_the_reset_still_exits_130_and_says_how_to_resume(
    tmp_path, caplog
):
    world = FakeWorld(release_ticks=2)
    spec, _ = two_cells()
    adaptive = interrupted_after_a_retrain(world, 4, 5)
    with caplog.at_level(logging.WARNING, logger="streamcls.matrix"):
        code = runner(world).run(spec, adaptive, ledger=tmp_path / "runs.jsonl")

    assert code == 130
    assert world.active_model == 2, "the reset was abandoned"
    assert any("--base-model 1" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def test_preflight_requires_a_base_model():
    world = FakeWorld(active_model=None)
    spec, cells = two_cells()
    with pytest.raises(RunnerError, match="no active model"):
        runner(world).preflight(spec, cells)

    spec_pinned = parse_spec(small_spec(defaults={"base_model": 2}))
    assert runner(world).preflight(spec_pinned, cells) == 2
    with pytest.raises(LookupError):
        runner(world).preflight(parse_spec(small_spec(defaults={"base_model": 77})), cells)


def test_preflight_refuses_to_start_against_dead_services():
    world = FakeWorld(heartbeats={"producer": 3.0, "consumer": 900.0})
    spec, cells = two_cells()
    with pytest.raises(RunnerError, match="consumer \\(900 s ago\\)"):
        runner(world).preflight(spec, cells)
    world.heartbeats = {"producer": None, "consumer": 3.0}
    with pytest.raises(RunnerError, match="producer \\(never\\)"):
        runner(world).preflight(spec, cells)
    assert runner(world).preflight(spec, cells, require_services=False) == 1


def test_preflight_rejects_a_knob_the_table_does_not_have():
    world = FakeWorld()
    spec = parse_spec(small_spec(config={"retrain_min_rows": 1}))
    with pytest.raises(SpecError, match="retrain_min_rows"):
        runner(world).preflight(spec, expand(spec))
    assert not world.calls, "preflight must not write"


def test_preflight_writes_nothing(tmp_path):
    world = FakeWorld()
    spec, cells = two_cells()
    runner(world).preflight(spec, cells)
    assert world.calls == []
    assert json.dumps(world.config, default=str)  # unchanged and serialisable


# ---------------------------------------------------------------------------
# The shipped specs and the CLI's dry run
# ---------------------------------------------------------------------------

SPECS = Path(__file__).resolve().parents[2] / "experiments"


def test_the_shipped_thesis_spec_is_the_54_cell_matrix():
    from streamcls.matrix import load_spec

    spec = load_spec(SPECS / "matrix.toml")
    cells = expand(spec)
    assert spec.name == "thesis-v1" and len(cells) == 54
    assert spec.config["adaptation_enabled"] is True, "the arm decides; config must not veto"
    assert spec.config["label_delay_batches"] == 0


def test_the_shipped_smoke_spec_is_one_pair():
    from streamcls.matrix import load_spec

    cells = expand(load_spec(SPECS / "smoke.toml"))
    assert [c.key for c in cells] == [
        "concept_abrupt/baseline/seed20260827",
        "concept_abrupt/adaptive/seed20260827",
    ]


def test_cli_dry_run_prints_the_plan_and_opens_no_connection(tmp_path, monkeypatch, capsys):
    import streamcls.__main__ as cli

    def no_db(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("--dry-run must not touch the database")

    monkeypatch.setattr(cli, "Repository", no_db)
    monkeypatch.setattr(cli, "get_settings", no_db)
    code = cli.main(
        ["matrix", str(SPECS / "smoke.toml"), "--dry-run", "--results", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "2 cell(s), 2 pending" in out
    assert "concept_abrupt/adaptive/seed20260827" in out
    assert "estimated wall clock" in out


def test_cli_refuses_to_run_when_the_data_root_is_not_on_this_host(
    tmp_path, monkeypatch, capsys
):
    """The leftover-checkpoint check looks under the data root. Pointed at the
    containers' ``/data`` on the host, it would never find anything."""
    import types

    import streamcls.__main__ as cli
    from streamcls.config import PathSettings

    def no_db(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must refuse before connecting")

    missing = tmp_path / "not-mounted"
    monkeypatch.setattr(cli, "Repository", no_db)
    monkeypatch.setattr(
        cli, "get_settings", lambda: types.SimpleNamespace(paths=PathSettings(data_root=missing))
    )
    code = cli.main(["matrix", str(SPECS / "smoke.toml"), "--results", str(tmp_path)])

    assert code == 2
    err = capsys.readouterr().err
    assert "STREAMCLS_DATA_ROOT" in err and str(missing) in err


def test_cli_dry_run_reports_completed_cells_from_the_ledger(tmp_path, capsys):
    import streamcls.__main__ as cli
    from streamcls.matrix import load_spec

    cells = expand(load_spec(SPECS / "smoke.toml"))
    t0 = datetime(2026, 8, 30, tzinfo=UTC)
    append_row(
        tmp_path / "smoke" / "runs.jsonl",
        result_row(cells[0], status="completed", run_id=1, base_model_version_id=1,
                   started_at=t0, ended_at=t0, score=FakeScore(1, 500)),
    )
    code = cli.main(["matrix", str(SPECS / "smoke.toml"), "--dry-run", "--results", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "2 cell(s), 1 pending" in out
    assert "baseline/seed20260827" in out and "  done" in out


def test_cli_rejects_a_bad_spec_and_an_empty_selection(tmp_path, capsys):
    import streamcls.__main__ as cli

    bad = tmp_path / "bad.toml"
    bad.write_text('[matrix]\nname = "x"\nseeds = [1]\n', "utf-8")
    assert cli.main(["matrix", str(bad), "--dry-run"]) == 2
    assert "[config]" in capsys.readouterr().err

    code = cli.main(["matrix", str(SPECS / "smoke.toml"), "--dry-run", "--only", "nothing/*"])
    assert code == 2
    assert "no cell matches" in capsys.readouterr().err


def scored(cell: Cell, run_id: int, post: float):
    """A real ``RunScore`` for a 4-batch run at 0.9 accuracy, then ``post``."""
    from streamcls.experiments import score_run

    rows = [
        {"batch_id": b, "n_received": 500, "n_classified": 500,
         "n_correct": round((0.9 if b < 2 else post) * 500), "accuracy": 0.9 if b < 2 else post}
        for b in range(4)
    ]
    return score_run(
        run={"id": run_id, "name": cell.run_name, "seed": cell.seed,
             "adaptation_enabled": cell.adaptation_enabled, "scenario": cell.scenario,
             "drift_schedule": list(cell.schedule)},
        batches=rows, events=[], jobs=[], batch_size=cell.batch_size,
    )


def test_cli_report_writes_the_tables_and_notes_unfinished_cells(tmp_path, capsys):
    import streamcls.__main__ as cli
    from streamcls.matrix import load_spec

    cells = expand(load_spec(SPECS / "smoke.toml"))
    ledger = tmp_path / "smoke" / "runs.jsonl"
    t0 = datetime(2026, 8, 30, tzinfo=UTC)

    # A first attempt at the adaptive cell timed out; the re-run completed.
    append_row(ledger, result_row(cells[0], status="completed", run_id=1, base_model_version_id=1,
                                  started_at=t0, ended_at=t0, score=scored(cells[0], 1, 0.5)))
    append_row(ledger, result_row(cells[1], status="timed_out", run_id=2, base_model_version_id=1,
                                  started_at=t0, ended_at=t0, error="stalled"))
    append_row(ledger, result_row(cells[1], status="completed", run_id=3, base_model_version_id=1,
                                  started_at=t0, ended_at=t0, score=scored(cells[1], 3, 0.8)))

    assert cli.main(["report", str(tmp_path / "smoke")]) == 0
    out = capsys.readouterr().out
    assert "# Matrix `smoke`" in out
    assert "| concept_abrupt | adaptive | 1 |" in out
    assert "+30.0" in out
    assert "not completed" not in out, "the superseded timed-out attempt does not count"
    for name in ("summary.md", "summary.csv", "pairs.csv", "detectors.csv"):
        assert (tmp_path / "smoke" / name).exists()

    # A cell whose last attempt failed is listed, not silently missing.
    append_row(ledger, result_row(cells[0], status="error", run_id=4, base_model_version_id=1,
                                  started_at=t0, ended_at=t0, error="boom"))
    assert cli.main(["report", str(tmp_path / "smoke")]) == 0
    assert "1 cell(s) not completed: `concept_abrupt/baseline/seed20260827`" in capsys.readouterr().out

    assert cli.main(["report", str(tmp_path / "nothing")]) == 1


def test_cli_report_states_the_commit_and_flags_a_dirty_tree(tmp_path, capsys):
    """The report is what the thesis cites, so it says which code produced it."""
    import streamcls.__main__ as cli
    from streamcls.matrix import load_spec

    cells = expand(load_spec(SPECS / "smoke.toml"))
    ledger = tmp_path / "smoke" / "runs.jsonl"
    t0 = datetime(2026, 8, 30, tzinfo=UTC)
    for index, cell in enumerate(cells, start=1):
        append_row(ledger, result_row(cell, status="completed", run_id=index, base_model_version_id=1,
                                      started_at=t0, ended_at=t0, score=scored(cell, index, 0.8),
                                      git_commit="a" * 40, git_dirty=False))
    assert cli.main(["report", str(tmp_path / "smoke")]) == 0
    out = capsys.readouterr().out
    assert "a" * 8 in out
    assert "dirty" not in out

    append_row(ledger, result_row(cells[1], status="completed", run_id=3, base_model_version_id=1,
                                  started_at=t0, ended_at=t0, score=scored(cells[1], 3, 0.8),
                                  git_commit="a" * 40, git_dirty=True))
    assert cli.main(["report", str(tmp_path / "smoke")]) == 0
    assert "dirty" in capsys.readouterr().out


def test_git_state_leaves_the_results_directory_out_of_the_dirty_check(tmp_path, monkeypatch):
    """A resumed matrix always finds its own ledger under results/: that is
    output, not a change to the code being measured."""
    import subprocess

    from streamcls.__main__ import _git_state

    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-qm", "init")
    (tmp_path / "results" / "m").mkdir(parents=True)
    (tmp_path / "results" / "m" / "runs.jsonl").write_text("{}\n")
    monkeypatch.chdir(tmp_path)

    commit, dirty = _git_state(Path("results"))
    assert len(commit) == 40
    assert dirty is False

    (tmp_path / "code.py").write_text("x = 2\n")
    assert _git_state(Path("results")) == (commit, True)
