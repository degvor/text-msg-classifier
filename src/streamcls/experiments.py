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
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["RunScore", "compare", "score_run"]

#: A detection this far past a change point counts as a false alarm rather than
#: a late detection: the stream has settled and this is a new claim.
SETTLE_BATCHES = 25
#: Accuracy counts as recovered within this margin of the pre-drift baseline.
RECOVERY_TOLERANCE = 0.03


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

    @property
    def mean_detection_delay(self) -> float | None:
        seen = [d for d in self.detection_delays if d is not None]
        return sum(seen) / len(seen) if seen else None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"mean_detection_delay": self.mean_detection_delay}


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

    detector_counts: dict[str, int] = {}
    for event in events:
        detector_counts[event["detector"]] = detector_counts.get(event["detector"], 0) + 1

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
        promotions=sum(1 for j in jobs if j.get("status") == "succeeded" and j.get("model_version")),
        detector_counts=detector_counts,
    )


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    classified = sum(r["n_classified"] or 0 for r in rows)
    if not classified:
        return None
    return sum(r["n_correct"] or 0 for r in rows) / classified


def _near_change_point(batch_id: int, change_points: list[int]) -> bool:
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
