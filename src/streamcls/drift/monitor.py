"""Combines the detectors and the distribution statistics into one decision.

Three signals, deliberately kept separate because they detect different things
and the comparison between them is a result in its own right:

===================  ==========================  ==========  ===================
signal               detects                     labels?     detector
===================  ==========================  ==========  ===================
error stream         real drift, ΔP(y|x)         yes         DDM + ADWIN
feature histogram    virtual drift, ΔP(x)        no          PSI / JS
confidence profile   label-free proxy            no          ADWIN + PSI
===================  ==========================  ==========  ===================

Measured in Phase 4 on an injected concept drift: accuracy fell ~20 points while
vocabulary PSI stayed at ~0.003 and confidence PSI at ~0.02. The model was
*confidently wrong* -- the text still looked like Sci/Tech and the model still
said Sci/Tech, only the label moved. So the label-free signals miss this class
of drift entirely, which is exactly the limitation worth reporting rather than
papering over.

**State lives on the driver and is rebuilt by replay, never pickled.** Detector
internals are not a version-stable contract, and a stored blob cannot be
inspected or reasoned about. ``BatchMetric.correct_bits`` stores one bit per
message, so a restart replays the exact error sequence for a few KB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from streamcls.drift.detectors import ADWIN, DDM
from streamcls.drift.distributions import (
    PSI_MAJOR,
    jensen_shannon,
    population_stability_index,
)
from streamcls.labels import N_CLASSES

__all__ = ["DriftMonitor", "DriftSignal", "MonitorConfig", "MonitorResult"]


@dataclass(frozen=True, slots=True)
class DriftSignal:
    detector: str
    statistic: float
    threshold: float | None
    #: Fired inside the post-start / post-swap grace window. Recorded rather
    #: than dropped, so suppression is auditable.
    suppressed: bool = False


#: Detectors permitted to spend a training run.
#:
#: DDM is deliberately *not* here by default. Measured over 40 seeded
#: stationary streams of 3000 observations, its false-alarm rate ran from 10 %
#: at a 40 % error rate to 47.5 % at 5 %, while ADWIN's was 0 % throughout --
#: and those figures match ``river``'s implementation exactly, so this is the
#: algorithm's behaviour, not a bug. Letting DDM trigger adaptation would mean
#: retraining repeatedly on data that never changed.
#:
#: It is still computed, recorded and charted, because "the classic baseline
#: detector would have fired N times spuriously here" is a result. The
#: experiment runner can widen this set to quantify exactly that.
DEFAULT_ACTIONABLE = frozenset({"adwin_error"})


@dataclass(frozen=True, slots=True)
class MonitorResult:
    signals: tuple[DriftSignal, ...]
    #: Statistics to persist on the BatchMetric row.
    stats: dict[str, float | None]
    actionable_detectors: frozenset[str] = DEFAULT_ACTIONABLE

    @property
    def actionable(self) -> tuple[DriftSignal, ...]:
        return tuple(
            s
            for s in self.signals
            if not s.suppressed and s.detector in self.actionable_detectors
        )

    @property
    def unsuppressed(self) -> tuple[DriftSignal, ...]:
        """Everything that fired outside the grace window, actionable or not."""
        return tuple(s for s in self.signals if not s.suppressed)

    @property
    def should_adapt(self) -> bool:
        return bool(self.actionable)


@dataclass(slots=True)
class MonitorConfig:
    adwin_delta: float = 0.002
    psi_threshold: float = PSI_MAJOR
    psi_consecutive_batches: int = 3
    grace_batches: int = 5
    psi_window_batches: int = 10
    #: Which detectors may trigger a retrain. See DEFAULT_ACTIONABLE.
    actionable_detectors: frozenset[str] = DEFAULT_ACTIONABLE


@dataclass(slots=True)
class DriftMonitor:
    """Per-run drift state. Owned by the driver, captured by ``foreachBatch``."""

    config: MonitorConfig = field(default_factory=MonitorConfig)
    reference_histogram: list[float] = field(default_factory=list)
    confidence_reference: list[float] = field(default_factory=list)
    top_term_indices: list[int] = field(default_factory=list)

    _ddm: DDM = field(init=False, default_factory=DDM)
    _adwin: ADWIN = field(init=False, default=None)  # type: ignore[assignment]
    _feature_window: list[list[float]] = field(init=False, default_factory=list)
    _psi_streak: int = field(init=False, default=0)
    _batches_seen: int = field(init=False, default=0)
    _grace_remaining: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._adwin = ADWIN(delta=self.config.adwin_delta)
        self._grace_remaining = self.config.grace_batches

    # --- lifecycle ------------------------------------------------------------
    def reset(self, *, grace: bool = True) -> None:
        """Forget everything. Called after an adaptation.

        Resetting is what stops a single drift producing a detection on every
        subsequent batch, which would otherwise queue a retrain per batch.
        """
        self._ddm.reset()
        self._adwin.reset()
        self._feature_window.clear()
        self._psi_streak = 0
        if grace:
            self._grace_remaining = self.config.grace_batches

    def rebuild(self, history: list[dict[str, Any]]) -> None:
        """Replay detector state from stored per-batch error bits."""
        from streamcls.spark.batch import unpack_correct_bits

        self.reset(grace=True)
        for row in history:
            errors = [
                1.0 - bit
                for bit in unpack_correct_bits(row.get("correct_bits"), row["n_classified"] or 0)
            ]
            self._ddm.extend(errors)
            self._adwin.extend(errors)
            self._batches_seen += 1

    @property
    def batches_seen(self) -> int:
        return self._batches_seen

    @property
    def in_grace(self) -> bool:
        return self._grace_remaining > 0

    # --- per batch ------------------------------------------------------------
    def observe(self, aggregate, *, apply: bool = True) -> MonitorResult:
        """Fold one batch in and report what fired.

        ``apply=False`` on a checkpoint replay: the sink is idempotent, but
        feeding the same errors to the detectors twice is not.
        """
        stats = self._distribution_stats(aggregate, record_window=apply)
        if not apply:
            return MonitorResult(
                signals=(), stats=stats,
                actionable_detectors=self.config.actionable_detectors,
            )

        self._batches_seen += 1
        signals: list[DriftSignal] = []

        for error in (1.0 - bit for bit in aggregate.correct_sequence):
            self._ddm.update(error)
            self._adwin.update(error)
            if self._ddm.drift_detected:
                signals.append(
                    DriftSignal("ddm", self._ddm.statistic, None, self.in_grace)
                )
            if self._adwin.drift_detected:
                signals.append(
                    DriftSignal("adwin_error", self._adwin.statistic, None, self.in_grace)
                )

        signals.extend(self._distribution_signals(stats))

        if self._grace_remaining > 0:
            self._grace_remaining -= 1

        # Collapse repeats within a batch: one detector firing 40 times across
        # 500 messages is one event, not 40.
        seen: set[str] = set()
        unique: list[DriftSignal] = []
        for signal in signals:
            if signal.detector in seen:
                continue
            seen.add(signal.detector)
            unique.append(signal)
        return MonitorResult(
            signals=tuple(unique),
            stats=stats,
            actionable_detectors=self.config.actionable_detectors,
        )

    def _distribution_signals(self, stats: dict[str, float | None]) -> list[DriftSignal]:
        """PSI must exceed the threshold for N consecutive batches.

        A single spike over a sparse vocabulary is noise; a sustained one is a
        shift. Requiring persistence is cheaper and more honest than tuning the
        threshold upward until the noise stops.
        """
        signals: list[DriftSignal] = []
        vocabulary = stats.get("psi_vocabulary")
        if vocabulary is not None and vocabulary > self.config.psi_threshold:
            self._psi_streak += 1
            if self._psi_streak >= self.config.psi_consecutive_batches:
                signals.append(
                    DriftSignal(
                        "psi_vocabulary", vocabulary, self.config.psi_threshold, self.in_grace
                    )
                )
                self._psi_streak = 0
        elif vocabulary is not None:
            self._psi_streak = 0

        predicted = stats.get("psi_predicted")
        if predicted is not None and predicted > self.config.psi_threshold:
            signals.append(
                DriftSignal("psi_predicted", predicted, self.config.psi_threshold, self.in_grace)
            )
        return signals

    def _distribution_stats(self, aggregate, *, record_window: bool) -> dict[str, float | None]:
        stats: dict[str, float | None] = {
            "psi_confidence": None,
            "psi_predicted": None,
            "psi_vocabulary": None,
            "js_vocabulary": None,
        }

        if any(aggregate.confidence_hist) and len(self.confidence_reference) == len(
            aggregate.confidence_hist
        ):
            stats["psi_confidence"] = population_stability_index(
                self.confidence_reference, aggregate.confidence_hist
            )
        if any(aggregate.predicted_hist):
            stats["psi_predicted"] = population_stability_index(
                [1.0 / N_CLASSES] * N_CLASSES, aggregate.predicted_hist
            )

        if self.reference_histogram and self.top_term_indices and aggregate.feature_sums:
            if record_window:
                self._feature_window.append(aggregate.feature_sums)
                if len(self._feature_window) > self.config.psi_window_batches:
                    self._feature_window.pop(0)
            if len(self._feature_window) == self.config.psi_window_batches:
                observed = self._fold()
                if observed is not None:
                    stats["psi_vocabulary"] = population_stability_index(
                        self.reference_histogram, observed
                    )
                    stats["js_vocabulary"] = jensen_shannon(self.reference_histogram, observed)
        return stats

    def _fold(self) -> list[float] | None:
        """Sum the window and fold to the model's top terms plus 'other'."""
        import numpy as np

        stacked = np.sum(np.asarray(self._feature_window, dtype=float), axis=0)
        total = float(stacked.sum())
        if total <= 0:
            return None
        if any(i >= stacked.size for i in self.top_term_indices):
            # A reference built against a different vocabulary means the bins do
            # not line up; silently truncating would produce a plausible-looking
            # number that means nothing.
            return None
        selected = stacked[self.top_term_indices]
        return [*selected.tolist(), max(0.0, total - float(selected.sum()))]
