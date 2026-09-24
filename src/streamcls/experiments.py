"""Experiment scoring: turn a finished run into the numbers a thesis reports.

Everything here is computed *after* a run, from ``BatchMetric``, ``DriftEvent``
and the run's own drift schedule. Nothing is measured live and remembered, so a
result can be recomputed from the database at any time and a scoring bug can be
fixed without re-running the stream.

The metrics are the standard concept-drift evaluation set, and each is only
computable because the producer ships ground truth:

``detection_delay``
    Batches from an injected change point to the first unsuppressed detection.
``false_alarms``
    Detections in a stable phase -- one with no preceding change point, or far
    enough past one that the stream has settled.
``missed_detections``
    Change points with no detection before the next one.
``recovery_batches``
    Batches from the change point until accuracy returns to within a tolerance
    of the pre-drift baseline. ``None`` means it never did.
``accuracy_after_drift``
    The number the A/B comparison turns on. Overall accuracy is dominated by
    the stable prefix both arms share, so comparing it understates the effect.

The second half of the module aggregates many scored runs -- the matrix
ledger -- into the tables the thesis prints: per-cell means over seeds,
seed-matched A/B deltas, and a scenario x detector table.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = [
    "METRICS",
    "CellSummary",
    "DetectorScore",
    "DetectorSummary",
    "PairResult",
    "Result",
    "RunScore",
    "Stat",
    "aggregate",
    "compare",
    "detector_table",
    "format_table",
    "pair",
    "render_markdown",
    "score_run",
    "write_csv",
    "write_detectors_csv",
    "write_pairs_csv",
]

#: A detection this far past a change point counts as a false alarm rather than
#: a late detection: the stream has settled and this is a new claim.
SETTLE_BATCHES = 25
#: Accuracy counts as recovered within this margin of the pre-drift baseline.
RECOVERY_TOLERANCE = 0.03


@dataclass(slots=True)
class DetectorScore:
    """Detection quality of one detector on one run, suppressed or not acted
    on notwithstanding: every detector always records its signals, so the
    baseline runs alone give the scenario x detector table."""

    detector: str
    signals: int
    detection_delays: list[int | None] = field(default_factory=list)
    false_alarms: int = 0
    missed_detections: int = 0
    suppressed: int = 0

    @property
    def mean_detection_delay(self) -> float | None:
        return _mean_or_none(self.detection_delays)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"mean_detection_delay": self.mean_detection_delay}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectorScore:
        return cls(
            detector=data["detector"],
            signals=int(data["signals"]),
            detection_delays=list(data.get("detection_delays") or []),
            false_alarms=int(data.get("false_alarms") or 0),
            missed_detections=int(data.get("missed_detections") or 0),
            suppressed=int(data.get("suppressed") or 0),
        )


@dataclass(slots=True)
class RunScore:
    run_id: int
    name: str
    seed: int
    adaptation_enabled: bool
    batches: int
    messages: int

    accuracy_overall: float | None
    accuracy_before_drift: float | None
    accuracy_after_drift: float | None
    accuracy_final: float | None

    change_points: list[int] = field(default_factory=list)
    detection_delays: list[int | None] = field(default_factory=list)
    false_alarms: int = 0
    missed_detections: int = 0
    suppressed_signals: int = 0
    recovery_batches: int | None = None
    retrains: int = 0
    promotions: int = 0
    detector_counts: dict[str, int] = field(default_factory=dict)
    detectors: dict[str, DetectorScore] = field(default_factory=dict)
    scenario: str = ""

    @property
    def mean_detection_delay(self) -> float | None:
        return _mean_or_none(self.detection_delays)

    @property
    def arm(self) -> str:
        return "adaptive" if self.adaptation_enabled else "baseline"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["detectors"] = {k: d.as_dict() for k, d in self.detectors.items()}
        return data | {"mean_detection_delay": self.mean_detection_delay}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunScore:
        """Inverse of :meth:`as_dict`, for reading scores back from a ledger."""
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names}
        kwargs["detectors"] = {
            k: DetectorScore.from_dict(d) for k, d in (data.get("detectors") or {}).items()
        }
        return cls(**kwargs)


def score_run(
    *,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
    events: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    batch_size: int,
) -> RunScore:
    """Score one run. Pure: takes plain rows, so it is testable without a DB."""
    ordered = sorted(batches, key=lambda b: b["batch_id"])
    change_points = sorted(
        int(e["at"]) // batch_size for e in (run.get("drift_schedule") or []) if "at" in e
    )

    classified = sum(b["n_classified"] or 0 for b in ordered)
    correct = sum(b["n_correct"] or 0 for b in ordered)
    first_change = change_points[0] if change_points else None

    before = [b for b in ordered if first_change is None or b["batch_id"] < first_change]
    after = [b for b in ordered if first_change is not None and b["batch_id"] >= first_change]

    # Scoring counts every unsuppressed signal, including ones that were only
    # recorded: "the detector fired" and "the system acted" are separate
    # questions, and conflating them would hide the false-alarm rate.
    delays, false_alarms = _detection_quality(events, change_points)

    detector_counts: dict[str, int] = {}
    for event in events:
        detector_counts[event["detector"]] = detector_counts.get(event["detector"], 0) + 1
    detectors = {
        name: _score_detector(name, [e for e in events if e["detector"] == name], change_points)
        for name in sorted(detector_counts)
    }

    return RunScore(
        run_id=run["id"],
        name=run["name"],
        seed=run["seed"],
        adaptation_enabled=bool(run.get("adaptation_enabled")),
        batches=len(ordered),
        messages=classified,
        accuracy_overall=(correct / classified) if classified else None,
        accuracy_before_drift=_accuracy(before),
        accuracy_after_drift=_accuracy(after),
        accuracy_final=_accuracy(ordered[-5:]),
        change_points=change_points,
        detection_delays=delays,
        false_alarms=false_alarms,
        missed_detections=sum(1 for d in delays if d is None),
        suppressed_signals=sum(1 for e in events if e.get("suppressed")),
        recovery_batches=_recovery(ordered, first_change),
        retrains=len(jobs),
        # Set by the query from model_version.activated_at: a retrain whose
        # model the promotion guard rejected trained but did not promote.
        promotions=sum(1 for j in jobs if j.get("promoted")),
        detector_counts=detector_counts,
        detectors=detectors,
        scenario=str(run.get("scenario") or ""),
    )


def _detection_quality(
    events: Iterable[dict[str, Any]], change_points: Sequence[int]
) -> tuple[list[int | None], int]:
    """Per change point, batches to the first unsuppressed detection (``None``
    for a miss); and the count of detections that explain no change point."""
    unsuppressed = [e for e in events if not e.get("suppressed")]

    delays: list[int | None] = []
    used: set[int] = set()
    for index, change in enumerate(change_points):
        # A detection belongs to this change point if it lands at or after it
        # and before the next one.
        upper = change_points[index + 1] if index + 1 < len(change_points) else None
        candidates = [
            e
            for e in unsuppressed
            if e["batch_id"] >= change and (upper is None or e["batch_id"] < upper)
        ]
        if candidates:
            first = min(candidates, key=lambda e: e["batch_id"])
            delays.append(first["batch_id"] - change)
            used.add(id(first))
        else:
            delays.append(None)

    false_alarms = sum(
        1
        for e in unsuppressed
        if id(e) not in used and not _near_change_point(e["batch_id"], change_points)
    )
    return delays, false_alarms


def _score_detector(
    name: str, events: list[dict[str, Any]], change_points: Sequence[int]
) -> DetectorScore:
    delays, false_alarms = _detection_quality(events, change_points)
    return DetectorScore(
        detector=name,
        signals=len(events),
        detection_delays=delays,
        false_alarms=false_alarms,
        missed_detections=sum(1 for d in delays if d is None),
        suppressed=sum(1 for e in events if e.get("suppressed")),
    )


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    classified = sum(r["n_classified"] or 0 for r in rows)
    if not classified:
        return None
    return sum(r["n_correct"] or 0 for r in rows) / classified


def _near_change_point(batch_id: int, change_points: Sequence[int]) -> bool:
    return any(0 <= batch_id - cp <= SETTLE_BATCHES for cp in change_points)


def _recovery(
    ordered: list[dict[str, Any]], first_change: int | None, *, tolerance: float = RECOVERY_TOLERANCE
) -> int | None:
    """Batches until accuracy returns to within ``tolerance`` of the baseline.

    Requires two consecutive batches above the bar, so a single lucky batch
    does not read as recovery.
    """
    if first_change is None:
        return None
    baseline = _accuracy([b for b in ordered if b["batch_id"] < first_change])
    if baseline is None:
        return None

    target = baseline - tolerance
    streak = 0
    for row in ordered:
        if row["batch_id"] < first_change or row["accuracy"] is None:
            continue
        streak = streak + 1 if row["accuracy"] >= target else 0
        if streak >= 2:
            return row["batch_id"] - first_change - 1
    return None


def _mean_or_none(values: Iterable[float | int | None]) -> float | None:
    seen = [v for v in values if v is not None]
    return sum(seen) / len(seen) if seen else None


def compare(baseline: RunScore, adaptive: RunScore) -> dict[str, Any]:
    """The headline A/B figure.

    Only valid when both runs share a seed: the streams are then byte-identical
    and any difference is attributable to adaptation alone.
    """
    return {
        "seed_matches": baseline.seed == adaptive.seed,
        "baseline": baseline.name,
        "adaptive": adaptive.name,
        "accuracy_after_drift": {
            "baseline": baseline.accuracy_after_drift,
            "adaptive": adaptive.accuracy_after_drift,
            "delta": _delta(adaptive.accuracy_after_drift, baseline.accuracy_after_drift),
        },
        "accuracy_final": {
            "baseline": baseline.accuracy_final,
            "adaptive": adaptive.accuracy_final,
            "delta": _delta(adaptive.accuracy_final, baseline.accuracy_final),
        },
        "recovery_batches": {
            "baseline": baseline.recovery_batches,
            "adaptive": adaptive.recovery_batches,
        },
        "retrains": {"baseline": baseline.retrains, "adaptive": adaptive.retrains},
    }


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def format_table(scores: list[RunScore]) -> str:
    """Fixed-width results table, for the console and the thesis appendix."""
    header = (
        f"{'run':>4}  {'name':<28} {'seed':>8} {'adapt':>6} {'batches':>7} "
        f"{'acc':>7} {'pre':>7} {'post':>7} {'delay':>6} {'FA':>3} {'miss':>4} "
        f"{'recov':>6} {'retrain':>7}"
    )
    lines = [header, "-" * len(header)]
    for s in scores:
        lines.append(
            f"{s.run_id:>4}  {s.name[:28]:<28} {s.seed:>8} "
            f"{('on' if s.adaptation_enabled else 'off'):>6} {s.batches:>7} "
            f"{_pct(s.accuracy_overall):>7} {_pct(s.accuracy_before_drift):>7} "
            f"{_pct(s.accuracy_after_drift):>7} "
            f"{_num(s.mean_detection_delay):>6} {s.false_alarms:>3} {s.missed_detections:>4} "
            f"{_num(s.recovery_batches):>6} {s.retrains:>7}"
        )
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


# ---------------------------------------------------------------------------
# Aggregation over a matrix
# ---------------------------------------------------------------------------

#: The per-run numbers that are averaged over seeds, in table order.
METRICS: tuple[str, ...] = (
    "accuracy_overall",
    "accuracy_before_drift",
    "accuracy_after_drift",
    "accuracy_final",
    "mean_detection_delay",
    "false_alarms",
    "missed_detections",
    "recovery_batches",
    "retrains",
    "promotions",
)
_PERCENT = frozenset(m for m in METRICS if m.startswith("accuracy"))


@dataclass(frozen=True, slots=True)
class Result:
    """One completed cell as the ledger records it: where it sits in the
    matrix, and its score."""

    scenario: str
    arm: str
    ablation: str | None
    seed: int
    run_id: int
    score: RunScore

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Result:
        score = RunScore.from_dict(row["score"])
        return cls(
            scenario=str(row.get("scenario") or score.scenario),
            arm=str(row.get("arm") or score.arm),
            ablation=row.get("ablation") or None,
            seed=int(row.get("seed", score.seed)),
            run_id=int(row.get("run_id", score.run_id)),
            score=score,
        )

    @property
    def cell(self) -> tuple[str, str, str | None]:
        return (self.scenario, self.arm, self.ablation)


@dataclass(frozen=True, slots=True)
class Stat:
    """Mean and sample standard deviation over seeds. ``sd`` is ``None``
    below two values; ``missing`` counts runs where the metric was undefined
    (no drift to recover from, no detection to time) and is reported rather
    than silently dropped."""

    mean: float | None
    sd: float | None
    n: int
    missing: int = 0

    @classmethod
    def of(cls, values: Iterable[float | int | None]) -> Stat:
        values = list(values)
        seen = [float(v) for v in values if v is not None]
        missing = len(values) - len(seen)
        if not seen:
            return cls(None, None, 0, missing)
        sd = statistics.stdev(seen) if len(seen) >= 2 else None
        return cls(statistics.fmean(seen), sd, len(seen), missing)

    def fmt(self, *, percent: bool = False, digits: int = 1) -> str:
        if self.mean is None:
            return "—"
        scale = 100.0 if percent else 1.0
        text = f"{self.mean * scale:.{digits}f}"
        if self.sd is not None:
            text += f" ± {self.sd * scale:.{digits}f}"
        if self.missing:
            text += f" ({self.n}/{self.n + self.missing})"
        return text


@dataclass(frozen=True, slots=True)
class CellSummary:
    scenario: str
    arm: str
    ablation: str | None
    seeds: tuple[int, ...]
    run_ids: tuple[int, ...]
    metrics: Mapping[str, Stat]

    @property
    def n(self) -> int:
        return len(self.run_ids)

    @property
    def label(self) -> str:
        return self.arm if self.ablation is None else f"{self.arm}/{self.ablation}"


def aggregate(results: Iterable[Result]) -> list[CellSummary]:
    """Mean ± sd of every metric per (scenario, arm, ablation), in the order
    the cells first appear."""
    groups: dict[tuple[str, str, str | None], list[Result]] = defaultdict(list)
    for result in results:
        groups[result.cell].append(result)
    summaries = []
    for (scenario, arm, ablation), members in groups.items():
        members.sort(key=lambda r: (r.seed, r.run_id))
        summaries.append(
            CellSummary(
                scenario=scenario,
                arm=arm,
                ablation=ablation,
                seeds=tuple(r.seed for r in members),
                run_ids=tuple(r.run_id for r in members),
                metrics={
                    metric: Stat.of([getattr(r.score, metric) for r in members])
                    for metric in METRICS
                },
            )
        )
    return summaries


@dataclass(frozen=True, slots=True)
class PairResult:
    """One seed-matched A/B: an adaptive run (plain or ablation) against the
    baseline run of the same scenario and seed."""

    scenario: str
    seed: int
    ablation: str | None
    baseline_run_id: int
    adaptive_run_id: int
    delta_after_drift: float | None
    delta_final: float | None
    recovery_baseline: int | None
    recovery_adaptive: int | None
    retrains: int
    promotions: int

    @property
    def label(self) -> str:
        return "adaptive" if self.ablation is None else f"adaptive/{self.ablation}"


def pair(results: Iterable[Result]) -> list[PairResult]:
    """Pair every adaptive result with the baseline of its scenario and seed.
    Adaptive results with no baseline partner are left out: without the
    shared seed the difference is not attributable to adaptation."""
    baselines: dict[tuple[str, int], Result] = {}
    adaptives: list[Result] = []
    for result in results:
        if result.arm == "baseline" and result.ablation is None:
            baselines[(result.scenario, result.seed)] = result
        elif result.arm == "adaptive":
            adaptives.append(result)
    pairs = []
    for adaptive in adaptives:
        baseline = baselines.get((adaptive.scenario, adaptive.seed))
        if baseline is None:
            continue
        ab = compare(baseline.score, adaptive.score)
        pairs.append(
            PairResult(
                scenario=adaptive.scenario,
                seed=adaptive.seed,
                ablation=adaptive.ablation,
                baseline_run_id=baseline.run_id,
                adaptive_run_id=adaptive.run_id,
                delta_after_drift=ab["accuracy_after_drift"]["delta"],
                delta_final=ab["accuracy_final"]["delta"],
                recovery_baseline=baseline.score.recovery_batches,
                recovery_adaptive=adaptive.score.recovery_batches,
                retrains=adaptive.score.retrains,
                promotions=adaptive.score.promotions,
            )
        )
    pairs.sort(key=lambda p: (p.scenario, p.ablation or "", p.seed))
    return pairs


@dataclass(frozen=True, slots=True)
class DetectorSummary:
    scenario: str
    detector: str
    runs: int
    delay: Stat
    false_alarms: Stat
    missed: Stat
    signals: Stat
    suppressed: Stat


def detector_table(results: Iterable[Result], *, arm: str = "baseline") -> list[DetectorSummary]:
    """Scenario x detector, over the seeds of one arm. The baseline arm by
    default: it never swaps models, so no grace window suppresses signals and
    every detector sees the same uninterrupted stream."""
    groups: dict[tuple[str, str], list[DetectorScore]] = defaultdict(list)
    for result in results:
        if result.arm != arm or result.ablation is not None:
            continue
        for name, detector in result.score.detectors.items():
            groups[(result.scenario, name)].append(detector)
    table = []
    for (scenario, name), members in groups.items():
        table.append(
            DetectorSummary(
                scenario=scenario,
                detector=name,
                runs=len(members),
                delay=Stat.of([d.mean_detection_delay for d in members]),
                false_alarms=Stat.of([d.false_alarms for d in members]),
                missed=Stat.of([d.missed_detections for d in members]),
                signals=Stat.of([d.signals for d in members]),
                suppressed=Stat.of([d.suppressed for d in members]),
            )
        )
    table.sort(key=lambda d: (d.scenario, d.detector))
    return table


# --- rendering ---------------------------------------------------------------

_HEADINGS = {
    "accuracy_overall": "acc %",
    "accuracy_before_drift": "pre %",
    "accuracy_after_drift": "post %",
    "accuracy_final": "final %",
    "mean_detection_delay": "delay",
    "false_alarms": "FA",
    "missed_detections": "miss",
    "recovery_batches": "recovery",
    "retrains": "retrains",
    "promotions": "promoted",
}


def render_markdown(
    summaries: Sequence[CellSummary],
    pairs: Sequence[PairResult],
    detectors: Sequence[DetectorSummary],
    *,
    name: str,
) -> str:
    """The three thesis tables as Markdown: main results (mean ± sd over
    seeds), seed-matched A/B deltas, ablations, and scenario x detector."""
    main = [s for s in summaries if s.ablation is None]
    ablations = [s for s in summaries if s.ablation is not None]
    out = [f"# Matrix `{name}`", ""]

    out += ["## Results by scenario and arm (mean ± sd over seeds)", ""]
    out += _md_table(
        ["scenario", "arm", "n", *(_HEADINGS[m] for m in METRICS)],
        [
            [s.scenario, s.label, str(s.n), *(_fmt_stat(m, s.metrics[m]) for m in METRICS)]
            for s in main
        ],
    )
    out += ["", "delay / recovery in batches; `(k/n)` = defined in k of n runs.", ""]

    out += ["## Seed-matched A/B (adaptive - baseline)", ""]
    out += _md_table(
        ["scenario", "arm", "seed", "Δ post (pp)", "Δ final (pp)", "recovery b→a",
         "retrains", "promoted", "runs b/a"],
        [
            [p.scenario, p.label, str(p.seed), _pp(p.delta_after_drift), _pp(p.delta_final),
             f"{_num(p.recovery_baseline)} → {_num(p.recovery_adaptive)}",
             str(p.retrains), str(p.promotions), f"{p.baseline_run_id}/{p.adaptive_run_id}"]
            for p in pairs
        ],
    )
    if pairs:
        out += ["", "### Mean Δ per scenario", ""]
        by_cell: dict[tuple[str, str | None], list[PairResult]] = defaultdict(list)
        for p in pairs:
            by_cell[(p.scenario, p.ablation)].append(p)
        out += _md_table(
            ["scenario", "arm", "n", "Δ post (pp)", "Δ final (pp)"],
            [
                [scenario, members[0].label, str(len(members)),
                 Stat.of([p.delta_after_drift for p in members]).fmt(percent=True),
                 Stat.of([p.delta_final for p in members]).fmt(percent=True)]
                for (scenario, _ablation), members in by_cell.items()
            ],
        )
    out.append("")

    if ablations:
        out += ["## Ablations (adaptive arm; default adaptive row for reference)", ""]
        rows = []
        for scenario in dict.fromkeys(s.scenario for s in ablations):
            rows += [
                [s.scenario, s.label, str(s.n), *(_fmt_stat(m, s.metrics[m]) for m in METRICS)]
                for s in summaries
                if s.scenario == scenario and s.arm == "adaptive"
            ]
        out += _md_table(["scenario", "arm", "n", *(_HEADINGS[m] for m in METRICS)], rows)
        out.append("")

    out += ["## Detectors by scenario (baseline arm; every detector records its signals)", ""]
    out += _md_table(
        ["scenario", "detector", "n", "delay", "FA", "miss", "signals", "suppressed"],
        [
            [d.scenario, d.detector, str(d.runs), d.delay.fmt(), d.false_alarms.fmt(),
             d.missed.fmt(), d.signals.fmt(), d.suppressed.fmt()]
            for d in detectors
        ],
    )
    out.append("")
    return "\n".join(out)


def _fmt_stat(metric: str, stat: Stat) -> str:
    return stat.fmt(percent=metric in _PERCENT)


def _pp(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.1f}"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return ["_no data_"]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def write_csv(path: Path, summaries: Sequence[CellSummary]) -> None:
    """One row per cell; every metric as ``<metric>_mean``, ``_sd``, ``_n``."""
    header = ["scenario", "arm", "ablation", "n", "seeds", "run_ids"]
    for metric in METRICS:
        header += [f"{metric}_mean", f"{metric}_sd", f"{metric}_n"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for s in summaries:
            row: list[Any] = [
                s.scenario, s.arm, s.ablation or "", s.n,
                " ".join(map(str, s.seeds)), " ".join(map(str, s.run_ids)),
            ]
            for metric in METRICS:
                stat = s.metrics[metric]
                row += [_csv_num(stat.mean), _csv_num(stat.sd), stat.n]
            writer.writerow(row)


def write_pairs_csv(path: Path, pairs: Sequence[PairResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "scenario", "seed", "ablation", "baseline_run_id", "adaptive_run_id",
            "delta_after_drift", "delta_final", "recovery_baseline", "recovery_adaptive",
            "retrains", "promotions",
        ])
        for p in pairs:
            writer.writerow([
                p.scenario, p.seed, p.ablation or "", p.baseline_run_id, p.adaptive_run_id,
                _csv_num(p.delta_after_drift), _csv_num(p.delta_final),
                _csv_num(p.recovery_baseline), _csv_num(p.recovery_adaptive),
                p.retrains, p.promotions,
            ])


def write_detectors_csv(path: Path, detectors: Sequence[DetectorSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "scenario", "detector", "runs",
            "delay_mean", "delay_sd", "delay_n",
            "false_alarms_mean", "false_alarms_sd",
            "missed_mean", "missed_sd", "signals_mean", "suppressed_mean",
        ])
        for d in detectors:
            writer.writerow([
                d.scenario, d.detector, d.runs,
                _csv_num(d.delay.mean), _csv_num(d.delay.sd), d.delay.n,
                _csv_num(d.false_alarms.mean), _csv_num(d.false_alarms.sd),
                _csv_num(d.missed.mean), _csv_num(d.missed.sd),
                _csv_num(d.signals.mean), _csv_num(d.suppressed.mean),
            ])


def _csv_num(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}" if isinstance(value, float) else str(value)
