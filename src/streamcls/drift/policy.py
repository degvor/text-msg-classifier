"""When a detected drift -- or a schedule -- becomes a training job.

The consumer used to inline this rule; it lives here so the rule and its
variants are pure Python, testable without Spark, and so a controlled
experiment can switch off exactly one part of it:

=====================  =========================================================
knob                   what it changes
=====================  =========================================================
``trigger``            ``drift`` (default): an actionable signal queues a job;
                       ``periodic``: a job every ``periodic_batches`` batches,
                       whatever the detectors say (blind retraining)
``min_new_rows``       wait until this many messages of the new concept exist
``retain_pending``     hold a detection made too early until the data exists
                       (default), or drop it
``origin``             count new messages from the injected change point
                       (default; only an experiment knows it) or from the first
                       message of the batch the drift was detected in
=====================  =========================================================

With every default the decisions are exactly those of the consumer the thesis
matrices ran, which is what makes the variants comparable with them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from streamcls.drift.monitor import DriftSignal

__all__ = ["PERIODIC", "RetrainTrigger", "TriggerConfig", "TriggerDecision"]

#: The trigger recorded on a job queued by the schedule rather than a detector.
PERIODIC = "periodic"

Action = Literal["none", "defer", "drop", "queue"]


@dataclass(frozen=True, slots=True)
class TriggerConfig:
    trigger: str = "drift"
    periodic_batches: int = 0
    min_new_rows: int = 0
    retain_pending: bool = True
    origin: str = "change_point"
    batch_size: int = 500

    def __post_init__(self) -> None:
        if self.trigger not in ("drift", PERIODIC):
            raise ValueError(f"trigger must be 'drift' or 'periodic'; got {self.trigger!r}")
        if self.origin not in ("change_point", "detection"):
            raise ValueError(f"origin must be 'change_point' or 'detection'; got {self.origin!r}")
        if self.trigger == PERIODIC and self.periodic_batches <= 0:
            raise ValueError("the periodic trigger needs periodic_batches > 0")

    @classmethod
    def from_stream_config(cls, config: Mapping[str, Any]) -> TriggerConfig:
        """Read the knobs off a ``stream_config`` row; absent ones keep the defaults."""
        return cls(
            trigger=str(config.get("retrain_trigger") or "drift"),
            periodic_batches=int(config.get("periodic_retrain_batches") or 0),
            min_new_rows=int(config.get("retrain_min_new_rows") or 0),
            retain_pending=bool(config.get("retain_pending_drift", True)),
            origin=str(config.get("new_data_origin") or "change_point"),
            batch_size=int(config["max_offsets_per_trigger"]),
        )


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    action: Action
    #: What the job, if one is queued, is credited to: a detector name or ``periodic``.
    trigger: str | None = None
    #: The signal acted on -- possibly one held over from an earlier batch.
    signal: DriftSignal | None = None
    #: First message index counted as new-concept data, when one applies.
    origin: int | None = None
    #: Messages seen since ``origin`` when the rule was evaluated.
    new_rows: int | None = None


class RetrainTrigger:
    """Per-run trigger state: the held detection and where its data starts.

    Rebuilt with the drift monitor at every query start, so a detection never
    outlives the query it was made in -- as before.
    """

    def __init__(self, config: TriggerConfig) -> None:
        self.config = config
        self._pending: DriftSignal | None = None
        self._pending_origin: int | None = None

    @property
    def pending(self) -> DriftSignal | None:
        return self._pending

    @property
    def wants_every_batch(self) -> bool:
        """True when a batch without signals can still queue a job."""
        return self.config.trigger == PERIODIC or self._pending is not None

    def decide(
        self,
        *,
        batch_id: int,
        actionable: Sequence[DriftSignal],
        change_point: int | None,
        messages_since: Callable[[int], int],
    ) -> TriggerDecision:
        """The rule for one processed batch.

        ``change_point`` is the injected change point preceding this batch, or
        ``None`` before the first one; ``messages_since(i)`` counts the run's
        messages with index >= ``i``.
        """
        config = self.config
        if config.trigger == PERIODIC:
            if batch_id > 0 and batch_id % config.periodic_batches == 0:
                return TriggerDecision("queue", trigger=PERIODIC)
            return TriggerDecision("none")

        # A drift held over from an earlier batch counts as actionable now.
        signal = actionable[0] if actionable else self._pending
        if signal is None:
            return TriggerDecision("none")

        if config.origin == "detection":
            # Only what the system itself observed: the first message of the
            # batch the held detection was made in, or of this batch.
            origin = (
                self._pending_origin
                if self._pending is not None and self._pending_origin is not None
                else batch_id * config.batch_size
            )
        else:
            origin = change_point

        seen: int | None = None
        if origin is not None and config.min_new_rows:
            seen = messages_since(origin)
            if seen < config.min_new_rows:
                if config.retain_pending:
                    # Held, not dropped. ADWIN detects *changes*: once the
                    # stream settles at the new error rate it is stationary
                    # again and will not fire a second time, so a deferred
                    # drift that is forgotten is lost for the rest of the run.
                    self._pending = signal
                    self._pending_origin = origin
                    return TriggerDecision(
                        "defer", trigger=signal.detector, signal=signal, origin=origin, new_rows=seen
                    )
                self._clear()
                return TriggerDecision(
                    "drop", trigger=signal.detector, signal=signal, origin=origin, new_rows=seen
                )

        return TriggerDecision(
            "queue", trigger=signal.detector, signal=signal, origin=origin, new_rows=seen
        )

    def queued(self) -> None:
        """A job was created: the held detection is spent."""
        self._clear()

    def _clear(self) -> None:
        self._pending = None
        self._pending_origin = None
