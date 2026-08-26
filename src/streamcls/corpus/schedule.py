"""Drift schedules: what changes in the stream, when, and by how much.

The schedule is a **pure function of the message index**, not of wall-clock
time. That is what makes a run reproducible: "drift at message 4000" always
lands at the same point, in the same micro-batch, on any machine.

It is also the experiment's ground truth. Each message carries the phase it was
produced in and whether it sits exactly on a change point, so detection delay,
false-alarm rate and missed detections are computable rather than eyeballed off
a chart. The consumer must use those fields **only for scoring** -- they are
deliberately kept out of every detector's signature.

The distinction the schedule exists to express:

* **covariate shift** changes P(x) -- the text looks different, the labelling
  rule does not. Vocabulary PSI should fire; the error rate need not.
* **prior shift** changes P(y) -- the class mix moves.
* **concept drift** changes P(y|x) -- the same kind of text now carries a
  different label. The error detectors should fire and vocabulary PSI should
  not.

Being able to produce both halves of that last pair, on demand, is what makes
comparing supervised against label-free detection a real result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from streamcls.labels import N_CLASSES

__all__ = ["DriftKind", "DriftSchedule", "StreamState", "uniform_weights"]


class DriftKind(StrEnum):
    COVARIATE_ABRUPT = "covariate_abrupt"
    COVARIATE_GRADUAL = "covariate_gradual"
    PRIOR_SHIFT = "prior_shift"
    CONCEPT_ABRUPT = "concept_abrupt"
    RECURRING = "recurring"
    RESET = "reset"


def uniform_weights(n: int = N_CLASSES) -> tuple[float, ...]:
    return tuple(1.0 / n for _ in range(n))


@dataclass(frozen=True, slots=True)
class StreamState:
    """Everything the producer needs to emit message ``index``."""

    #: Sampling prior over classes.
    label_weights: tuple[float, ...]
    #: Perturbation intensity in [0, 1].
    intensity: float
    #: Which pool a message's text is drawn from, per assigned label. Identity
    #: unless a concept drift is active. ``{2: 3}`` means "messages labelled
    #: Business are given Sci/Tech text".
    concept_map: Mapping[int, int]
    phase: int
    kind: str
    is_drift_point: bool


def _normalise(weights: Sequence[float]) -> tuple[float, float, ...]:  # type: ignore[misc]
    if len(weights) != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} weights, got {len(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError(f"weights must be non-negative: {weights}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to more than zero")
    return tuple(w / total for w in weights)


@dataclass(slots=True)
class DriftSchedule:
    """A list of change points, evaluated at any message index.

    ``events`` is the JSON stored on ``ExperimentRun.drift_schedule``::

        [{"at": 4000, "kind": "concept_abrupt", "params": {"map": {"2": 3}}},
         {"at": 8000, "kind": "covariate_gradual",
          "params": {"to": 0.9, "over": 2000}}]

    An empty list is the stationary control condition.
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    base_intensity: float = 0.15
    base_weights: tuple[float, ...] = field(default_factory=uniform_weights)

    def __post_init__(self) -> None:
        self.base_weights = _normalise(self.base_weights)
        if not 0.0 <= self.base_intensity <= 1.0:
            raise ValueError(f"base_intensity must be in [0, 1], got {self.base_intensity}")
        for event in self.events:
            if "at" not in event:
                raise ValueError(f"drift event needs an 'at' index: {event}")
            DriftKind(event.get("kind", ""))  # raises on an unknown kind
        # Evaluation walks these in order; sorting here means a hand-written
        # schedule does not have to be.
        self.events.sort(key=lambda e: int(e["at"]))

    @property
    def change_points(self) -> tuple[int, ...]:
        """Message indices where a drift begins. The evaluation ground truth."""
        return tuple(int(e["at"]) for e in self.events if e.get("kind") != DriftKind.RESET)

    def preceding_change_point(self, index: int) -> int | None:
        """Latest change point at or before ``index``, for detection delay."""
        candidates = [cp for cp in self.change_points if cp <= index]
        return max(candidates) if candidates else None

    def state_at(self, index: int) -> StreamState:
        """Resolve the stream's state at ``index``. Pure; no side effects."""
        weights = self.base_weights
        intensity = self.base_intensity
        concept: dict[int, int] = {}
        phase = 0
        kind = "none"

        for position, event in enumerate(self.events, start=1):
            at = int(event["at"])
            if index < at:
                break
            params = event.get("params", {}) or {}
            event_kind = DriftKind(event["kind"])
            phase = position
            kind = str(event_kind)

            if event_kind is DriftKind.RESET:
                weights, intensity, concept, kind = (
                    self.base_weights,
                    self.base_intensity,
                    {},
                    "none",
                )

            elif event_kind is DriftKind.COVARIATE_ABRUPT:
                intensity = float(params.get("to", 0.9))

            elif event_kind is DriftKind.COVARIATE_GRADUAL:
                target = float(params.get("to", 0.9))
                over = max(1, int(params.get("over", 2000)))
                # Linear ramp, clamped once the window closes.
                progress = min(1.0, (index - at) / over)
                intensity = intensity + (target - intensity) * progress

            elif event_kind is DriftKind.PRIOR_SHIFT:
                weights = _normalise(params.get("weights", self.base_weights))

            elif event_kind is DriftKind.CONCEPT_ABRUPT:
                # JSON object keys are strings; normalise to ints.
                concept = {int(k): int(v) for k, v in (params.get("map") or {}).items()}

            elif event_kind is DriftKind.RECURRING:
                # Alternates between the base state and an alternate one, so a
                # detector that adapts can be tested for over-fitting to a
                # transient concept.
                period = max(1, int(params.get("period", 2000)))
                alternate = ((index - at) // period) % 2 == 0
                if alternate:
                    weights = _normalise(params.get("weights", self.base_weights))
                    intensity = float(params.get("to", intensity))
                    concept = {int(k): int(v) for k, v in (params.get("map") or {}).items()}
                else:
                    weights, intensity, concept = self.base_weights, self.base_intensity, {}
                kind = f"{event_kind}:{'on' if alternate else 'off'}"

        return StreamState(
            label_weights=weights,
            intensity=max(0.0, min(1.0, intensity)),
            concept_map=concept,
            phase=phase,
            kind=kind,
            is_drift_point=index in set(self.change_points),
        )
