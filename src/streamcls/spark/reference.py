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

__all__ = ["DEFAULT_TOP_TERMS", "build_confidence_reference", "build_reference"]

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


def build_confidence_reference(predictions: DataFrame, *, bins: int = 10) -> list[float]:
    """The model's own max-probability profile on held-out data.

    This is the reference a batch's confidence histogram is compared against.
    Using a uniform reference instead makes the statistic enormous and nearly
    constant -- a calibrated classifier concentrates mass in the top bins, so
    the distance to uniform says something about calibration, not about drift.
    """
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    scored = predictions.withColumn(
        "_confidence", F.array_max(vector_to_array(F.col("probability")))
    )
    aggregations = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        predicate = (F.col("_confidence") >= low) & (
            F.col("_confidence") < high if index < bins - 1 else F.col("_confidence") <= high
        )
        aggregations.append(F.sum(F.when(predicate, 1).otherwise(0)).alias(f"b{index}"))

    row = scored.agg(*aggregations).first()
    counts = [float(row[f"b{i}"] or 0) for i in range(bins)]
    total = sum(counts)
    if total <= 0:
        return [1.0 / bins] * bins
    return [c / total for c in counts]
