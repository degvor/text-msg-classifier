"""The drift reference distribution, computed at training time.

An unsupervised drift detector compares the current window against a reference.
That reference must be *this model's* training distribution, expressed in *this
model's* feature space -- which is why it is stored on ``ModelVersion`` and
swaps with the model rather than living in a global config.

Binning matters more than it looks. PSI is unreliable when bins hold fewer than
about five expected counts, and a 20 000-term vocabulary against a 500-message
batch is almost entirely empty bins -- the statistic would be noise. So the
vocabulary is folded to the top ``n_terms`` training terms plus a single
"other" bin, which keeps every bin well populated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from streamcls.spark.features import RAW_FEATURE_COLUMN

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

__all__ = ["DEFAULT_TOP_TERMS", "build_reference"]

DEFAULT_TOP_TERMS = 50


def build_reference(
    featurised: DataFrame, *, n_terms: int = DEFAULT_TOP_TERMS
) -> tuple[list[int], list[float], int]:
    """Return ``(top_term_indices, reference_histogram, vocab_size)``.

    ``reference_histogram`` has ``len(top_term_indices) + 1`` entries and sums
    to 1; the final entry is the "other" mass.
    """
    from pyspark.ml.stat import Summarizer
    from pyspark.sql import functions as F

    totals = featurised.select(
        Summarizer.sum(F.col(RAW_FEATURE_COLUMN)).alias("s")
    ).first()["s"]
    counts = list(totals.toArray())
    vocab_size = len(counts)

    ranked = sorted(range(vocab_size), key=lambda i: counts[i], reverse=True)
    # Sorted so the histogram's bin order is stable and inspectable.
    top = sorted(ranked[: min(n_terms, vocab_size)])

    grand_total = float(sum(counts))
    if grand_total <= 0:
        raise ValueError("training corpus produced an all-zero feature matrix")

    histogram = [counts[i] / grand_total for i in top]
    histogram.append(1.0 - sum(histogram))  # "other"
    # Guard against float dust turning the last bin very slightly negative.
    histogram[-1] = max(0.0, histogram[-1])

    return top, histogram, vocab_size
