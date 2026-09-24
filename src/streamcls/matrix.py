"""The unattended experiment matrix: declare cells in TOML, run them, resume.

A *cell* is one experiment run -- (scenario, seed, arm, optional ablation) --
and the matrix is their cross product. Running one by hand is a ritual with a
dozen steps and three races; this module makes it a loop the machine performs
identically 54 times overnight.

Everything here is plain Python. The runner is handed a repository, a topic
namer, a topic creator and a scorer, so the whole lifecycle is unit-testable
against a scripted fake with a fake clock. The only side effects are Postgres
writes through the repository and one JSON line per finished cell appended to
``results/<matrix>/runs.jsonl`` -- the ledger that makes a crashed or
interrupted matrix resumable: a cell recorded as ``completed`` is never redone.

Why the lifecycle is ordered the way it is (each step closes a race):

1. stop both services and wait until the consumer has *actually* released its
   streaming query -- a query holds its model until it stops, so promoting the
   base model before that would race the swap;
2. write the cell's ``stream_config`` knobs -- the consumer reads them once,
   at query start;
3. re-activate the base model (``force``: the retrained incumbent is usually
   better on recent data and the guard would rightly refuse);
4. create the run row and its Kafka topic;
5. start only the producer and wait for it to finish -- the consumer then
   starts against a complete backlog, so every batch is exactly
   ``max_offsets_per_trigger`` rows and batch composition is a pure function
   of message index;
6. start the consumer and poll; the cell is complete when every message is
   classified and no retrain is queued or running, on two consecutive polls
   (a retrain may be enqueued just after the last batch's metric row);
7. stop the consumer, wait for the query to release, score, append.

When the loop ends -- finished, failed or Ctrl-C -- the base model is made
active again. ``base_model = "active"`` is resolved afresh by every
invocation, so a matrix that left a retrained model active would hand it to
the next matrix, or to its own resume, as the base. A crash gets no such
cleanup, so a resume also refuses a ledger whose completed cells ran on a
different base.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import logging
import math
import os
import re
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from streamcls.corpus.schedule import PRESETS, DriftSchedule, preset

if TYPE_CHECKING:
    from streamcls.db.repo import RunProgress

__all__ = [
    "ARMS",
    "Ablation",
    "Cell",
    "MatrixRunner",
    "Phase",
    "RunnerError",
    "Spec",
    "SpecError",
    "append_row",
    "completed_keys",
    "estimate_seconds",
    "expand",
    "load_spec",
    "parse_spec",
    "phase",
    "read_ledger",
    "result_row",
    "select_cells",
]

log = logging.getLogger(__name__)

#: Both arms, in the order a cell pair is run. ``baseline`` never retrains;
#: ``adaptive`` is the system under test. The default when a spec omits ``arms``.
ARMS: tuple[str, ...] = ("baseline", "adaptive")
DEFAULT_DATASET_SPLIT = "agnews-stream-holdout-v1"
#: ``experiment_run.name`` is ``CharField(max_length=120)``.
MAX_RUN_NAME = 120
#: How long a replay producer may take to write a whole run. 16k messages
#: take ~15 s; this is generous so a cold corpus load does not count as failure.
PRODUCER_TIMEOUT_S = 300.0
#: The consumer bounds ``query.stop()`` at 60 s; this is that plus a poll.
QUIESCE_TIMEOUT_S = 90.0
#: A service whose heartbeat is older than this is treated as not running.
HEARTBEAT_FRESH_S = 60.0
#: Stop after this many cells in a row that did not complete: something is
#: systematically wrong and the night is better spent not running.
MAX_CONSECUTIVE_FAILURES = 3
#: Complete-looking snapshots needed before a cell is called done.
SETTLE_POLLS = 2

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOP_LEVEL_KEYS = frozenset({"matrix", "defaults", "config", "scenarios", "ablations"})
_MATRIX_KEYS = frozenset({"name", "seeds", "arms"})
_DEFAULTS_KEYS = frozenset(
    {"total_messages", "mode", "dataset_split", "timeout_minutes", "base_model"}
)
_SCENARIO_KEYS = frozenset({"schedule"})
_ABLATION_KEYS = frozenset({"name", "scenarios", "config"})
_REQUIRED_CONFIG = ("max_offsets_per_trigger", "trigger_seconds")


class SpecError(ValueError):
    """The spec is malformed. Raised before anything is written anywhere."""


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Ablation:
    """A named ``stream_config`` override, run on the adaptive arm only."""

    name: str
    scenarios: tuple[str, ...]
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Spec:
    name: str
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    total_messages: int
    mode: str
    dataset_split: str
    timeout_minutes: float
    #: A model version id, or ``None`` for "whichever is active when the
    #: matrix starts" -- resolved once in preflight and pinned for every cell.
    base_model: int | None
    #: ``stream_config`` knobs written before every cell.
    config: Mapping[str, Any]
    #: Scenario name -> drift schedule, in declaration order.
    scenarios: Mapping[str, tuple[dict[str, Any], ...]]
    ablations: tuple[Ablation, ...]


def load_spec(path: str | Path) -> Spec:
    with Path(path).open("rb") as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as error:
            raise SpecError(f"{path}: {error}") from None
    return parse_spec(data)


def parse_spec(data: Mapping[str, Any]) -> Spec:
    """Validate a decoded spec. Every problem is reported by name.

    Unknown keys are errors rather than ignored: ``[default]`` for
    ``[defaults]`` would otherwise silently run every cell at the built-in
    length, and a misspelt ablation knob would run the default config under
    the ablation's name.
    """
    _reject_unknown(data, _TOP_LEVEL_KEYS, "top level")

    matrix = _table(data, "matrix")
    _reject_unknown(matrix, _MATRIX_KEYS, "[matrix]")
    name = _string(matrix, "name", "[matrix]")
    if not _SAFE_NAME.match(name):
        raise SpecError(f"[matrix] name {name!r} may only contain letters, digits, . _ -")
    seeds = _int_list(matrix, "seeds", "[matrix]")
    arms = tuple(_str_list(matrix, "arms", "[matrix]", default=list(ARMS)))
    if unknown := sorted(set(arms) - set(ARMS)):
        raise SpecError(f"[matrix] arms must be a subset of {list(ARMS)}; got {unknown}")

    defaults = _table(data, "defaults", required=False)
    _reject_unknown(defaults, _DEFAULTS_KEYS, "[defaults]")
    total_messages = _int(defaults, "total_messages", "[defaults]", default=16_000)
    if total_messages < 8:
        raise SpecError("[defaults] total_messages must be at least 8")
    mode = str(defaults.get("mode", "replay"))
    if mode not in ("replay", "live"):
        raise SpecError(f"[defaults] mode must be 'replay' or 'live'; got {mode!r}")
    dataset_split = str(defaults.get("dataset_split", DEFAULT_DATASET_SPLIT))
    timeout_minutes = float(defaults.get("timeout_minutes", 25))
    if timeout_minutes <= 0:
        raise SpecError("[defaults] timeout_minutes must be positive")
    base_model = _base_model(defaults.get("base_model", "active"))

    config = _table(data, "config")
    for key in _REQUIRED_CONFIG:
        # Batch-denominated latencies (detection delay, recovery, swap) mean
        # nothing unless the batch is pinned; the scorer needs the size too.
        if not isinstance(config.get(key), int) or isinstance(config.get(key), bool):
            raise SpecError(f"[config] {key} must be set to an integer")
        if config[key] <= 0:
            raise SpecError(f"[config] {key} must be positive")

    scenarios_table = _table(data, "scenarios")
    if not scenarios_table:
        raise SpecError("[scenarios] declares no scenarios")
    scenarios: dict[str, tuple[dict[str, Any], ...]] = {}
    for scenario_name, body in scenarios_table.items():
        if not isinstance(body, Mapping):
            raise SpecError(f"[scenarios.{scenario_name}] must be a table")
        _reject_unknown(body, _SCENARIO_KEYS, f"[scenarios.{scenario_name}]")
        if not _SAFE_NAME.match(scenario_name):
            raise SpecError(f"scenario name {scenario_name!r} may only contain letters, digits, . _ -")
        scenarios[scenario_name] = _schedule(scenario_name, body, total_messages)

    raw_ablations = data.get("ablations", [])
    if not isinstance(raw_ablations, list):
        raise SpecError("[[ablations]] must be an array of tables")
    ablations: list[Ablation] = []
    for index, body in enumerate(raw_ablations):
        where = f"[[ablations]] #{index + 1}"
        if not isinstance(body, Mapping):
            raise SpecError(f"{where} must be a table")
        _reject_unknown(body, _ABLATION_KEYS, where)
        ablation_name = _string(body, "name", where)
        if not _SAFE_NAME.match(ablation_name):
            raise SpecError(f"{where} name {ablation_name!r} may only contain letters, digits, . _ -")
        if ablation_name in ARMS:
            raise SpecError(f"{where} name {ablation_name!r} collides with an arm name")
        if any(a.name == ablation_name for a in ablations):
            raise SpecError(f"{where} name {ablation_name!r} is declared twice")
        targets = tuple(_str_list(body, "scenarios", where))
        if missing := sorted(set(targets) - set(scenarios)):
            raise SpecError(f"{where} targets undeclared scenarios: {missing}")
        overrides = body.get("config")
        if not isinstance(overrides, Mapping) or not overrides:
            raise SpecError(f"{where} needs a non-empty config table")
        ablations.append(Ablation(ablation_name, targets, dict(overrides)))
    if ablations and "adaptive" not in arms:
        raise SpecError("[[ablations]] run on the adaptive arm, which [matrix] arms excludes")

    return Spec(
        name=name,
        seeds=seeds,
        arms=arms,
        total_messages=total_messages,
        mode=mode,
        dataset_split=dataset_split,
        timeout_minutes=timeout_minutes,
        base_model=base_model,
        config=dict(config),
        scenarios=scenarios,
        ablations=tuple(ablations),
    )


def _schedule(name: str, body: Mapping[str, Any], total: int) -> tuple[dict[str, Any], ...]:
    if "schedule" in body:
        events = body["schedule"]
        if not isinstance(events, list):
            raise SpecError(f"[scenarios.{name}] schedule must be an array of events")
        try:
            DriftSchedule(events=[dict(e) for e in events])
        except (ValueError, TypeError) as error:
            raise SpecError(f"[scenarios.{name}] schedule is invalid: {error}") from None
        return tuple(dict(e) for e in events)
    if name not in PRESETS:
        raise SpecError(
            f"[scenarios.{name}] is not a preset ({', '.join(PRESETS)}) and has no schedule"
        )
    return tuple(preset(name, total))


def _base_model(value: Any) -> int | None:
    if value == "active":
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise SpecError(f"[defaults] base_model must be 'active' or a model version id; got {value!r}")


def _reject_unknown(table: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    if unknown := sorted(set(table) - allowed):
        raise SpecError(f"{where}: unknown keys {unknown}; allowed: {sorted(allowed)}")


def _table(data: Mapping[str, Any], key: str, *, required: bool = True) -> Mapping[str, Any]:
    value = data.get(key)
    if value is None:
        if required:
            raise SpecError(f"[{key}] table is required")
        return {}
    if not isinstance(value, Mapping):
        raise SpecError(f"[{key}] must be a table")
    return value


def _string(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise SpecError(f"{where} {key} must be a non-empty string")
    return value


def _int(table: Mapping[str, Any], key: str, where: str, *, default: int) -> int:
    value = table.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpecError(f"{where} {key} must be an integer")
    return value


def _int_list(table: Mapping[str, Any], key: str, where: str) -> tuple[int, ...]:
    value = table.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(v, int) or isinstance(v, bool) for v in value)
    ):
        raise SpecError(f"{where} {key} must be a non-empty list of integers")
    if len(set(value)) != len(value):
        raise SpecError(f"{where} {key} contains duplicates")
    return tuple(value)


def _str_list(
    table: Mapping[str, Any], key: str, where: str, *, default: list[str] | None = None
) -> list[str]:
    value = table.get(key, default)
    if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
        raise SpecError(f"{where} {key} must be a non-empty list of strings")
    if len(set(value)) != len(value):
        raise SpecError(f"{where} {key} contains duplicates")
    return value


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Cell:
    """One experiment run, fully specified. Nothing is looked up later."""

    matrix: str
    scenario: str
    seed: int
    arm: str
    ablation: str | None
    total_messages: int
    mode: str
    dataset_split: str
    timeout_s: float
    config: Mapping[str, Any]
    schedule: tuple[dict[str, Any], ...]

    @property
    def key(self) -> str:
        """Stable identity within a matrix: what the ledger and ``--only`` use."""
        base = f"{self.scenario}/{self.arm}/seed{self.seed}"
        return f"{base}/{self.ablation}" if self.ablation else base

    @property
    def run_name(self) -> str:
        return f"{self.matrix}/{self.key}"

    @property
    def adaptation_enabled(self) -> bool:
        return self.arm == "adaptive"

    @property
    def batch_size(self) -> int:
        return int(self.config["max_offsets_per_trigger"])

    @property
    def batches(self) -> int:
        return math.ceil(self.total_messages / self.batch_size)


def expand(spec: Spec) -> list[Cell]:
    """Scenario -> seed -> arm, then each ablation over its scenarios and
    seeds on the adaptive arm. Deterministic, so ``--dry-run`` shows exactly
    the order the night will follow."""

    def make(scenario: str, seed: int, arm: str, ablation: Ablation | None) -> Cell:
        config = dict(spec.config)
        if ablation is not None:
            config.update(ablation.config)
        return Cell(
            matrix=spec.name,
            scenario=scenario,
            seed=seed,
            arm=arm,
            ablation=ablation.name if ablation else None,
            total_messages=spec.total_messages,
            mode=spec.mode,
            dataset_split=spec.dataset_split,
            timeout_s=spec.timeout_minutes * 60.0,
            config=config,
            schedule=spec.scenarios[scenario],
        )

    cells = [
        make(scenario, seed, arm, None)
        for scenario in spec.scenarios
        for seed in spec.seeds
        for arm in spec.arms
    ]
    cells += [
        make(scenario, seed, "adaptive", ablation)
        for ablation in spec.ablations
        for scenario in ablation.scenarios
        for seed in spec.seeds
    ]
    for c in cells:
        if len(c.run_name) > MAX_RUN_NAME:
            raise SpecError(
                f"run name {c.run_name!r} is {len(c.run_name)} characters; "
                f"the limit is {MAX_RUN_NAME}"
            )
    return cells


def select_cells(cells: Sequence[Cell], patterns: Iterable[str]) -> list[Cell]:
    """Keep cells whose key matches any shell-style pattern; none means all."""
    patterns = list(patterns)
    if not patterns:
        return list(cells)
    return [c for c in cells if any(fnmatch.fnmatchcase(c.key, p) for p in patterns)]


def estimate_seconds(cell: Cell) -> float:
    """A planning figure, not a measurement: one trigger per batch plus start
    and stop overhead, plus a retrain-and-swap allowance on the adaptive arm."""
    triggers = cell.batches * float(cell.config["trigger_seconds"])
    overhead = 60.0
    adaptation = 90.0 if cell.adaptation_enabled else 0.0
    return triggers + overhead + adaptation


# ---------------------------------------------------------------------------
# Progress -> decision
# ---------------------------------------------------------------------------
class Phase(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    CONSUMER_GAVE_UP = "consumer_gave_up"
    TIMED_OUT = "timed_out"


def phase(
    progress: RunProgress, *, total_messages: int, elapsed_s: float, timeout_s: float
) -> Phase:
    """The decision table for one poll of a running cell.

    Completeness wins over everything: if every message is classified and no
    retrain is in flight the data is whole, whatever else happened. Then the
    consumer's own surrender (it clears ``is_active`` after repeated query
    failures), then the clock.
    """
    if progress.classified >= total_messages and progress.active_jobs == 0:
        return Phase.COMPLETE
    if not progress.is_active:
        return Phase.CONSUMER_GAVE_UP
    if elapsed_s >= timeout_s:
        return Phase.TIMED_OUT
    return Phase.RUNNING


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
def result_row(
    cell: Cell,
    *,
    status: str,
    run_id: int | None,
    base_model_version_id: int | None,
    started_at: datetime,
    ended_at: datetime,
    score: Any = None,
    error: str = "",
    git_commit: str = "",
) -> dict[str, Any]:
    """One ledger line. ``status == "completed"`` is the only value the
    resume logic trusts; everything else is diagnostic."""
    return {
        "matrix": cell.matrix,
        "key": cell.key,
        "scenario": cell.scenario,
        "arm": cell.arm,
        "seed": cell.seed,
        "ablation": cell.ablation,
        "run_id": run_id,
        "run_name": cell.run_name,
        "status": status,
        "base_model_version_id": base_model_version_id,
        "total_messages": cell.total_messages,
        "config": dict(cell.config),
        "git_commit": git_commit,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "wall_s": round((ended_at - started_at).total_seconds(), 1),
        "error": error,
        "score": score.as_dict() if score is not None else None,
    }


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one line and force it to disk before returning: the ledger is
    what a crash resumes from, so a buffered line is a line that never was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Every parseable row, oldest first. A damaged line (a crash mid-write)
    is skipped with a warning: at worst its cell is run again, which is safe."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("ledger %s line %d is not valid JSON; skipping", path, number)
    return rows


def completed_keys(path: Path) -> set[str]:
    return {r["key"] for r in read_ledger(path) if r.get("status") == "completed" and "key" in r}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class RunnerError(RuntimeError):
    """The environment, not the spec, is wrong: a service is missing, the
    consumer will not release its query, no base model exists."""


class MatrixRunner:
    """Drives cells through the lifecycle described in the module docstring.

    Every collaborator is injected so the whole thing runs against a fake:
    ``repo`` is a :class:`streamcls.db.repo.Repository` (or a stand-in with
    the same methods), ``topic_for_run`` names a run's topic,
    ``ensure_topic`` creates it, ``score`` turns a finished run id into a
    ``RunScore``. ``sleep`` and ``clock`` exist so tests do not wait.
    """

    def __init__(
        self,
        repo: Any,
        *,
        topic_for_run: Callable[[int], str],
        ensure_topic: Callable[[str], Any],
        score: Callable[[int, int], Any],
        git_commit: str = "",
        git_dirty: bool = False,
        poll_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repo = repo
        self._topic_for_run = topic_for_run
        self._ensure_topic = ensure_topic
        self._score = score
        self._git_commit = git_commit
        self._git_dirty = git_dirty
        self._poll_s = poll_s
        self._sleep = sleep
        self._clock = clock
        self._now = now
        #: (cell, run_id, started_at) of the cell in flight, for the Ctrl-C path.
        self._current: tuple[Cell, int | None, datetime] | None = None

    # --- preflight ----------------------------------------------------------
    def preflight(self, spec: Spec, cells: Sequence[Cell], *, require_services: bool = True) -> int:
        """Fail before the first write. Returns the base model id every cell
        will be reset to."""
        self._repo.read_control()  # raises if the singleton row is missing

        allowed = set(self._repo.read_config()) - {"id", "updated_at"}
        for cell in cells:
            if unknown := sorted(set(cell.config) - allowed):
                raise SpecError(f"cell {cell.key}: unknown stream_config knobs {unknown}")

        if spec.base_model is not None:
            record = self._repo.get_model(spec.base_model)  # LookupError if missing
        else:
            record = self._repo.get_active_model()
            if record is None:
                raise RunnerError("no active model version; run `make train PROMOTE=1` first")
        base_id = int(record.id)

        if require_services:
            ages = self._repo.heartbeat_ages()
            stale = [
                f"{service} ({'never' if age is None else f'{age:.0f} s ago'})"
                for service, age in sorted(ages.items())
                if age is None or age > HEARTBEAT_FRESH_S
            ]
            if stale:
                raise RunnerError(
                    "services not heartbeating: " + ", ".join(stale) + " -- is `make up` running?"
                )
        return base_id

    # --- the whole matrix ---------------------------------------------------
    def run(
        self,
        spec: Spec,
        cells: Sequence[Cell],
        *,
        ledger: Path,
        stop_on_timeout: bool = False,
    ) -> int:
        """Run every cell not already completed in ``ledger``. Returns a
        process exit code: 0 all completed, 1 some did not, 130 interrupted."""
        base_id = self.preflight(spec, cells)
        self._check_ledger_base(ledger, base_id)
        done = completed_keys(ledger)
        failures = 0
        consecutive = 0
        try:
            for index, cell in enumerate(cells, start=1):
                if cell.key in done:
                    log.info("[%d/%d] %s: already completed, skipping", index, len(cells), cell.key)
                    continue
                adopted = self._adopt(cell)
                if adopted is not None:
                    log.info("[%d/%d] %s: adopting finished run %d", index, len(cells), cell.key, adopted)
                    row = self._score_row(cell, run_id=adopted, base_id=base_id, started=self._now())
                else:
                    log.info("[%d/%d] %s: starting", index, len(cells), cell.key)
                    row = self.run_cell(cell, base_model_id=base_id)
                    self._current = None
                append_row(ledger, row)
                if row["status"] == "completed":
                    consecutive = 0
                    log.info("[%d/%d] %s: completed in %.0f s", index, len(cells), cell.key, row["wall_s"])
                    continue
                failures += 1
                consecutive += 1
                log.error("[%d/%d] %s: %s %s", index, len(cells), cell.key, row["status"], row["error"])
                if stop_on_timeout and row["status"] == Phase.TIMED_OUT:
                    log.error("stopping on timeout as requested")
                    break
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    log.error("%d cells failed in a row; stopping", consecutive)
                    break
        except KeyboardInterrupt:
            self._interrupted(ledger, base_id)
            return 130
        finally:
            with _cleanup("stopping the stream"):
                self._quiet()
            self._restore_base(base_id)
        return 1 if failures else 0

    def _check_ledger_base(self, ledger: Path, base_id: int) -> None:
        bases = {
            r["base_model_version_id"]
            for r in read_ledger(ledger)
            if r.get("status") == "completed" and r.get("base_model_version_id") is not None
        }
        if bases and bases != {base_id}:
            hint = f"; resume with --base-model {next(iter(bases))}" if len(bases) == 1 else ""
            raise RunnerError(
                f"{ledger}: completed cells ran on base model "
                f"{', '.join(map(str, sorted(bases)))}, this run would start from {base_id}{hint}"
            )

    def _restore_base(self, base_id: int) -> None:
        """Leave the registry as the matrix found it. Not ``_cleanup``: a
        second Ctrl-C here must not turn exit code 130 into a traceback."""
        try:
            self._wait_for_release()
            self._repo.promote_model(base_id, force=True)
            return
        except KeyboardInterrupt:
            reason = "interrupted"
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
        log.warning(
            "base model %d was not made active again (%s); resume with --base-model %d",
            base_id, reason, base_id,
        )

    # --- one cell -----------------------------------------------------------
    def run_cell(self, cell: Cell, *, base_model_id: int) -> dict[str, Any]:
        started = self._now()
        self._current = (cell, None, started)
        run_id: int | None = None
        try:
            # 1. Nothing may be running, and the consumer must have let go of
            #    its query, before the model underneath it is changed.
            self._quiesce()
            # 2. The consumer reads its knobs once, at query start.
            self._repo.update_config(**cell.config)
            # 3. Not a promotion: a reset, so the guard does not apply.
            self._repo.promote_model(base_model_id, force=True)
            # 4. The run row carries a full snapshot of what it was run with.
            run_id = self._repo.create_run(
                name=cell.run_name,
                scenario=cell.scenario,
                seed=cell.seed,
                mode=cell.mode,
                adaptation_enabled=cell.adaptation_enabled,
                total_messages=cell.total_messages,
                drift_schedule=[dict(e) for e in cell.schedule],
                dataset_split=cell.dataset_split,
                base_model_version_id=base_model_id,
                consumer_config=self._snapshot(cell, base_model_id),
                producer_config={},
                git_commit=self._git_commit,
                notes=f"matrix {cell.matrix}",
                topic_for_run=self._topic_for_run,
            )
            self._current = (cell, run_id, started)
            self._ensure_topic(self._topic_for_run(run_id))
            # 5. Producer first, alone.
            self._repo.activate_run(run_id, producer=True, consumer=False)
            status = self._wait_for_status(run_id, ("completed", "aborted"), PRODUCER_TIMEOUT_S)
            if status != "completed":
                self._repo.set_producer_active(False)
                if status != "aborted":
                    self._repo.mark_run_finished(run_id, status="aborted")
                return self._failed_row(
                    cell, run_id, base_model_id, started,
                    status="producer_failed" if status == "aborted" else "producer_timed_out",
                    error=f"run status {status!r} after the producer phase",
                )
            # 6. Consumer against the full backlog.
            self._repo.set_stream_active(True)
            outcome, progress = self._watch(run_id, cell)
            # 7. Stop, release, score.
            self._repo.set_stream_active(False)
            self._wait_for_release()
            if outcome is Phase.COMPLETE:
                return self._score_row(cell, run_id=run_id, base_id=base_model_id, started=started)
            self._repo.mark_run_finished(run_id, status="aborted")
            return self._failed_row(
                cell, run_id, base_model_id, started,
                status=str(outcome),
                error=progress.last_error
                or f"{progress.classified}/{cell.total_messages} classified, "
                f"{progress.active_jobs} job(s) active",
            )
        except KeyboardInterrupt:
            raise  # ``_current`` stays set so ``_interrupted`` can record it
        except Exception as error:  # one bad cell must not end the night
            log.exception("cell %s failed", cell.key)
            with _cleanup("aborting the cell"):
                self._quiet()
                if run_id is not None:
                    self._repo.mark_run_finished(run_id, status="aborted")
            return self._failed_row(
                cell, run_id, base_model_id, started, status="error",
                error=f"{type(error).__name__}: {error}",
            )

    # --- steps ------------------------------------------------------------
    def _quiet(self) -> None:
        self._repo.set_stream_active(False)
        self._repo.set_producer_active(False)

    def _quiesce(self) -> None:
        self._quiet()
        self._wait_for_release()

    def _wait_for_release(self) -> None:
        if not self._wait(
            lambda: self._repo.read_control().consumer_query_run_id is None, QUIESCE_TIMEOUT_S
        ):
            raise RunnerError(
                f"consumer did not release its query within {QUIESCE_TIMEOUT_S:.0f} s"
            )

    def _wait_for_status(self, run_id: int, wanted: tuple[str, ...], timeout_s: float) -> str:
        status = ""

        def reached() -> bool:
            nonlocal status
            status = self._repo.run_progress(run_id).status
            return status in wanted

        self._wait(reached, timeout_s)
        return status

    def _watch(self, run_id: int, cell: Cell) -> tuple[Phase, Any]:
        """Poll until the cell is complete on ``SETTLE_POLLS`` consecutive
        snapshots, or something ends it."""
        t0 = self._clock()
        settled = 0
        last_report = t0
        while True:
            progress = self._repo.run_progress(run_id)
            elapsed = self._clock() - t0
            current = phase(
                progress,
                total_messages=cell.total_messages,
                elapsed_s=elapsed,
                timeout_s=cell.timeout_s,
            )
            if current is Phase.COMPLETE:
                settled += 1
                if settled >= SETTLE_POLLS:
                    return current, progress
            else:
                settled = 0
                if current is not Phase.RUNNING:
                    return current, progress
            if self._clock() - last_report >= 60.0:
                last_report = self._clock()
                log.info(
                    "%s: %d/%d classified, batch %s, %d job(s) active, %.0f s",
                    cell.key, progress.classified, cell.total_messages,
                    progress.last_batch_id, progress.active_jobs, elapsed,
                )
            self._sleep(self._poll_s)

    def _wait(self, predicate: Callable[[], bool], timeout_s: float) -> bool:
        deadline = self._clock() + timeout_s
        while True:
            if predicate():
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self._poll_s)

    def _snapshot(self, cell: Cell, base_model_id: int) -> dict[str, Any]:
        """What ``consumer_config`` records: enough to know, from the row
        alone, exactly which configuration produced it."""
        config = {
            k: v for k, v in self._repo.read_config().items() if k not in ("id", "updated_at")
        }
        return {
            "matrix": cell.matrix,
            "cell_key": cell.key,
            "arm": cell.arm,
            "ablation": cell.ablation,
            "base_model_version_id": base_model_id,
            "stream_config": config,
            "git_dirty": self._git_dirty,
        }

    def _adopt(self, cell: Cell) -> int | None:
        """A run with this cell's name that finished but never reached the
        ledger (a crash between step 7 and the append) is scored, not redone.
        Anything less than fully consumed is redone under a fresh id."""
        candidates = self._repo.find_runs_by_name(cell.run_name)
        for run in reversed(candidates):
            tag = run.get("consumer_config") or {}
            if tag.get("matrix") != cell.matrix or tag.get("cell_key") != cell.key:
                continue
            if run.get("status") != "completed":
                continue
            progress = self._repo.run_progress(run["id"])
            if progress.classified >= cell.total_messages and progress.active_jobs == 0:
                return int(run["id"])
        return None

    def _score_row(self, cell: Cell, *, run_id: int, base_id: int, started: datetime) -> dict:
        score = self._score(run_id, cell.batch_size)
        return result_row(
            cell,
            status="completed",
            run_id=run_id,
            base_model_version_id=base_id,
            started_at=started,
            ended_at=self._now(),
            score=score,
            git_commit=self._git_commit,
        )

    def _failed_row(
        self, cell: Cell, run_id: int | None, base_id: int, started: datetime, *, status: str, error: str
    ) -> dict[str, Any]:
        return result_row(
            cell,
            status=status,
            run_id=run_id,
            base_model_version_id=base_id,
            started_at=started,
            ended_at=self._now(),
            error=error,
            git_commit=self._git_commit,
        )

    def _interrupted(self, ledger: Path, base_id: int) -> None:
        """Ctrl-C: leave the control plane off, the run marked aborted, and a
        row in the ledger saying so -- never a run that looks in progress."""
        log.warning("interrupted; stopping the stream")
        current, self._current = self._current, None
        with _cleanup("stopping the stream"):
            self._quiet()
        if current is None:
            return
        cell, run_id, started = current
        with _cleanup("marking the run aborted"):
            if run_id is not None:
                self._repo.mark_run_finished(run_id, status="aborted")
        append_row(
            ledger,
            self._failed_row(
                cell, run_id, base_id, started, status="interrupted", error="KeyboardInterrupt"
            ),
        )


@contextlib.contextmanager
def _cleanup(what: str) -> Iterator[None]:
    """``contextlib.suppress(Exception)`` with a log line: on a cleanup path
    a second failure must be seen, but must not hide the first."""
    try:
        yield
    except KeyboardInterrupt:
        raise
    except Exception as error:
        log.warning("%s failed: %s: %s", what, type(error).__name__, error)
