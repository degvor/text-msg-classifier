"""Distribution-distance statistics for label-free drift detection.

These are the *unsupervised* half of the detection story. They matter because
in production the labels usually are not available, so a detector that needs
them is an upper bound rather than a deployable method. Comparing what these
catch against what the error-rate detectors catch is itself a result.

Pure Python and numpy on purpose: Spark is needed only to *count* a batch into a
histogram, and once that is done the statistics have no reason to sit behind a
JVM. Keeping them here means they are unit-tested locally, against
hand-computed values.

A statistical trap worth naming, because a reviewer will: PSI is unreliable
when bins hold fewer than roughly five expected counts. A 20 000-term
vocabulary against a 500-message batch is almost entirely empty bins, and the
statistic becomes noise. Three things guard against that -- the vocabulary is
folded to the top-N training terms plus one "other" bin
(:mod:`streamcls.spark.reference`), vocabulary PSI accumulates over a sliding
window of batches, and the well-populated confidence and predicted-class
histograms are reported alongside as per-batch signals.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "EPSILON",
    "PSI_MAJOR",
    "PSI_MINOR",
    "jensen_shannon",
    "normalise",
    "population_stability_index",
]

#: Additive smoothing. Without it an empty bin sends PSI to infinity, which is
#: not a detection -- it is a divide-by-zero wearing a hat.
EPSILON = 1e-6

#: Conventional PSI reading: below 0.1 is no meaningful shift, 0.1-0.25 is a
#: moderate shift worth watching, above 0.25 is a major shift.
PSI_MINOR = 0.1
PSI_MAJOR = 0.25


def normalise(counts: Sequence[float], *, epsilon: float = EPSILON) -> np.ndarray:
    """Turn counts into a smoothed probability vector summing to 1."""
    array = np.asarray(counts, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("counts must be a non-empty 1-D sequence")
    if np.any(array < 0):
        raise ValueError("counts must be non-negative")
    total = array.sum()
    if total <= 0:
        # An empty batch is not evidence of drift; report the uniform
        # distribution so the distance comes out as zero against itself.
        return np.full(array.size, 1.0 / array.size)
    smoothed = array / total + epsilon
    return smoothed / smoothed.sum()


def _aligned(
    expected: Sequence[float], observed: Sequence[float], epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(expected) != len(observed):
        raise ValueError(
            f"distributions must have the same number of bins: "
            f"{len(expected)} != {len(observed)}"
        )
    return normalise(expected, epsilon=epsilon), normalise(observed, epsilon=epsilon)


def population_stability_index(
    expected: Sequence[float], observed: Sequence[float], *, epsilon: float = EPSILON
) -> float:
    """PSI between a reference and an observed distribution.

    ``sum((o - e) * ln(o / e))``. Symmetric, unbounded above, zero when the two
    are identical.
    """
    reference, current = _aligned(expected, observed, epsilon)
    return float(np.sum((current - reference) * np.log(current / reference)))


def jensen_shannon(
    expected: Sequence[float], observed: Sequence[float], *, epsilon: float = EPSILON
) -> float:
    """Jensen-Shannon divergence in bits, bounded to [0, 1].

    Reported next to PSI because it is bounded: "0.62" is interpretable on its
    own, whereas a PSI of 0.62 only means something against the 0.1 / 0.25
    convention. Bounded also makes it safe to plot on a fixed axis.
    """
    reference, current = _aligned(expected, observed, epsilon)
    mixture = 0.5 * (reference + current)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log2(p / q)))

    value = 0.5 * kl(reference, mixture) + 0.5 * kl(current, mixture)
    # Float dust can push this a hair outside [0, 1].
    return float(min(1.0, max(0.0, value)))
