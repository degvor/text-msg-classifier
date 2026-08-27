"""Concept-drift detectors over a stream of prediction errors.

Implemented here rather than imported from ``river``. Two reasons, one
practical and one that matters more:

* ``river`` pins ``>=3.11`` from 0.26, which constrains the runtime image; and
* the algorithms are the subject of the thesis. Writing DDM and ADWIN out shows
  the mechanism, makes every parameter inspectable, and -- because they are pure
  Python over a sequence of 0/1 errors -- lets them be unit-tested locally with
  no Spark, no JVM and no container.

Both operate on a stream of *errors*: ``update(1.0)`` for a misclassification,
``update(0.0)`` for a correct prediction.

References:
    Gama et al., "Learning with Drift Detection", SBIA 2004 (DDM).
    Bifet & Gavaldà, "Learning from Time-Changing Data with Adaptive
    Windowing", SDM 2007 (ADWIN2).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = ["ADWIN", "DDM", "DriftDetector"]


class DriftDetector(ABC):
    """Common interface, so the consumer can hold a heterogeneous list."""

    name: str = "detector"

    @abstractmethod
    def update(self, error: float) -> None:
        """Feed one observation. 1.0 = misclassified, 0.0 = correct."""

    @abstractmethod
    def reset(self) -> None:
        """Forget everything. Called after an adaptation."""

    @property
    @abstractmethod
    def drift_detected(self) -> bool: ...

    @property
    def warning_detected(self) -> bool:
        return False

    @property
    @abstractmethod
    def statistic(self) -> float:
        """A single number to store alongside a DriftEvent."""

    def extend(self, errors: list[float]) -> None:
        """Replay a sequence. Used to rebuild state after a restart."""
        for error in errors:
            self.update(error)


@dataclass(slots=True)
class DDM(DriftDetector):
    """Drift Detection Method (Gama et al., 2004).

    Models the error rate of a stream as a binomial. While the concept is
    stable the error rate should fall or hold, so the running minimum of
    ``p + s`` is a proxy for "as good as this learner has ever been". Drift is
    declared when the current ``p + s`` rises far enough above that minimum.

    Cheap -- O(1) per observation, three floats of state -- and it fires on the
    *error rate*, so it needs labels. That is exactly why it is reported next to
    the label-free statistics rather than instead of them.

    **Measured limitation, and it is the algorithm's, not this implementation's.**
    The trigger is ``p + s > p_min + 3 * s_min``, and ``s = sqrt(p(1-p)/n)``
    shrinks without bound as the run length grows. On a long *stationary*
    stream the band therefore becomes tighter than ordinary binomial
    fluctuation, and DDM eventually fires on noise. Over 40 seeded stationary
    streams of 3000 observations the false-alarm rate was 10 % at a 40 % error
    rate rising to 47.5 % at 5 %, while ADWIN's was 0 % throughout. Those
    figures match ``river``'s DDM to the decimal (see
    ``tests/unit/test_detectors.py::test_matches_river``), so this is a faithful
    implementation of a detector with a known weakness -- which is precisely why
    the adaptation loop is gated on a cooldown and a promotion guard rather than
    trusting a single detector.
    """

    #: Standard deviations above the running minimum that trigger a warning.
    warning_level: float = 2.0
    #: ...and a drift. 3.0 is the value from the paper.
    drift_level: float = 3.0
    #: Refuse to fire before this many observations. A handful of early errors
    #: would otherwise produce a confident-looking detection from nothing.
    min_instances: int = 30

    name: str = "ddm"

    _n: int = field(default=1, init=False)
    _p: float = field(default=1.0, init=False)
    _s: float = field(default=0.0, init=False)
    _p_min: float = field(default=float("inf"), init=False)
    _s_min: float = field(default=float("inf"), init=False)
    _drift: bool = field(default=False, init=False)
    _warning: bool = field(default=False, init=False)

    def update(self, error: float) -> None:
        if self._drift:
            # A detector that keeps firing on the same drift is not reporting
            # new information. The caller resets after acting.
            self.reset()

        # Incremental mean of the error indicator.
        self._p += (error - self._p) / self._n
        self._s = math.sqrt(self._p * (1.0 - self._p) / self._n)
        self._n += 1

        self._warning = False
        self._drift = False
        if self._n < self.min_instances:
            return

        if self._p + self._s <= self._p_min + self._s_min:
            self._p_min, self._s_min = self._p, self._s

        threshold = self._p_min + self.drift_level * self._s_min
        warning_threshold = self._p_min + self.warning_level * self._s_min
        if self._p + self._s > threshold:
            self._drift = True
        elif self._p + self._s > warning_threshold:
            self._warning = True

    def reset(self) -> None:
        self._n = 1
        self._p = 1.0
        self._s = 0.0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._drift = False
        self._warning = False

    @property
    def drift_detected(self) -> bool:
        return self._drift

    @property
    def warning_detected(self) -> bool:
        return self._warning

    @property
    def statistic(self) -> float:
        """Current error rate. What the threshold is applied to."""
        return self._p

    @property
    def error_rate(self) -> float:
        return self._p


@dataclass(slots=True)
class _Bucket:
    """One exponential-histogram bucket: a compressed run of observations."""

    total: float
    variance: float
    size: int


class ADWIN(DriftDetector):
    """ADaptive WINdowing, the ADWIN2 variant (Bifet & Gavaldà, 2007).

    Keeps a window of recent observations and, whenever any split of that
    window into an older and a newer half has means further apart than a
    statistically justified bound, declares drift and discards the older half.

    The appeal over DDM is that it carries no assumption about the error rate
    only ever improving, it adapts its own window size, and the bound comes with
    a false-positive guarantee parameterised by ``delta``. The cost is more
    state -- but the exponential histogram keeps that at O(log n) buckets rather
    than O(n) observations, which is what makes it usable per-message on a
    stream.
    """

    name = "adwin"

    def __init__(self, delta: float = 0.002, *, max_buckets: int = 5, min_window: int = 5) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window = min_window
        self._levels: list[list[_Bucket]] = []
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._drift = False

    # --- public interface ----------------------------------------------------
    def update(self, error: float) -> None:
        self._insert(error)
        self._drift = self._shrink()

    def reset(self) -> None:
        self._levels = []
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._drift = False

    @property
    def drift_detected(self) -> bool:
        return self._drift

    @property
    def statistic(self) -> float:
        return self.estimation

    @property
    def estimation(self) -> float:
        """Mean over the current window -- the detector's estimate of the rate."""
        return self._total / self._width if self._width else 0.0

    @property
    def width(self) -> int:
        """Current window size. Collapses sharply when drift is detected."""
        return self._width

    @property
    def variance(self) -> float:
        return self._variance / self._width if self._width else 0.0

    # --- exponential histogram ------------------------------------------------
    def _insert(self, value: float) -> None:
        if not self._levels:
            self._levels.append([])
        self._levels[0].append(_Bucket(total=value, variance=0.0, size=1))

        if self._width > 0:
            mean = self._total / self._width
            # Welford-style update of the window's total squared deviation.
            self._variance += self._width * (value - mean) ** 2 / (self._width + 1)
        self._total += value
        self._width += 1
        self._compress()

    def _compress(self) -> None:
        """Merge the two oldest buckets of any over-full level into the next.

        This is what bounds memory: each level holds at most ``max_buckets``
        buckets of size 2^level, so the window is represented in O(log n)
        buckets no matter how long the stream runs.
        """
        level = 0
        while level < len(self._levels):
            if len(self._levels[level]) <= self.max_buckets:
                break
            older = self._levels[level].pop(0)
            newer = self._levels[level].pop(0)
            merged = _merge(older, newer)
            if level + 1 == len(self._levels):
                self._levels.append([])
            self._levels[level + 1].append(merged)
            level += 1

    def _shrink(self) -> bool:
        """Drop the oldest bucket while some split shows a significant gap."""
        detected = False
        shrank = True
        while shrank and self._width >= 2 * self.min_window:
            shrank = False
            older_total = 0.0
            older_width = 0
            # Walk cut points from the oldest end: the oldest bucket first.
            for level in reversed(range(len(self._levels))):
                for bucket in self._levels[level]:
                    older_total += bucket.total
                    older_width += bucket.size
                    newer_width = self._width - older_width
                    if older_width < self.min_window or newer_width < self.min_window:
                        continue
                    newer_total = self._total - older_total
                    gap = abs(older_total / older_width - newer_total / newer_width)
                    if gap > self._epsilon(older_width, newer_width):
                        detected = True
                        shrank = True
                        self._drop_oldest()
                        break
                if shrank:
                    break
        return detected

    def _epsilon(self, n_older: int, n_newer: int) -> float:
        """The ADWIN2 cut threshold.

        Variance-aware (a low-variance window needs a smaller gap to be
        significant) and Bonferroni-corrected by ``ln(width)``, which is what
        keeps the false-positive rate near ``delta`` despite testing every
        possible cut on every observation.
        """
        n_older = max(n_older - self.min_window + 1, 1)
        n_newer = max(n_newer - self.min_window + 1, 1)
        harmonic = 1.0 / n_older + 1.0 / n_newer
        corrected = math.log(2.0 * math.log(max(self._width, 2)) / self.delta)
        return math.sqrt(2.0 * harmonic * self.variance * corrected) + 2.0 / 3.0 * harmonic * corrected

    def _drop_oldest(self) -> None:
        """Remove the oldest bucket, at the highest occupied level."""
        for level in reversed(range(len(self._levels))):
            if self._levels[level]:
                bucket = self._levels[level].pop(0)
                if self._width > bucket.size:
                    mean_rest = (self._total - bucket.total) / (self._width - bucket.size)
                    mean_bucket = bucket.total / bucket.size
                    self._variance -= (
                        bucket.variance
                        + bucket.size
                        * (self._width - bucket.size)
                        * (mean_bucket - mean_rest) ** 2
                        / self._width
                    )
                    self._variance = max(self._variance, 0.0)
                else:
                    self._variance = 0.0
                self._total -= bucket.total
                self._width -= bucket.size
                while self._levels and not self._levels[-1]:
                    self._levels.pop()
                return


def _merge(older: _Bucket, newer: _Bucket) -> _Bucket:
    """Combine two adjacent buckets, keeping the variance exact.

    The cross term is what makes this lossless: variance of a union is not the
    sum of variances, it also carries the spread between the two means.
    """
    size = older.size + newer.size
    total = older.total + newer.total
    cross = older.size * newer.size * (older.total / older.size - newer.total / newer.size) ** 2 / size
    return _Bucket(total=total, variance=older.variance + newer.variance + cross, size=size)
