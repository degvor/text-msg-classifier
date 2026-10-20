"""Per-micro-batch computation: one aggregation, a handful of driver-side scalars.

The previous consumer did this instead::

    data = batch_df.select(to_json(...)).rdd.map(lambda r: r[0]).collect()

Every row of every batch, serialised to JSON and pulled to a 2 GB driver, with
no ``maxOffsetsPerTrigger`` bounding the batch -- so one Kafka backlog spike was
an OOM. It then did all downstream work single-threaded on the driver, which is
where the "distributed processing" claim actually collapsed: the executors
classified, and the driver did everything else serially.

Here a single ``.agg()`` returns one ``Row`` holding ~20 scalars, two bounded
per-row arrays and one vocabulary-sized vector -- a few KB regardless of batch
size. The per-row arrays are safe *because* ``maxOffsetsPerTrigger`` is set;
that setting is a correctness requirement here, not a tuning knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from streamcls.labels import N_CLASSES
from streamcls.spark.features import RAW_FEATURE_COLUMN

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

__all__ = ["CONFIDENCE_BINS", "BatchAggregate", "aggregate_batch", "pack_correct_bits", "score"]

#: Ten fixed bins over [0, 1]. Well populated at any realistic batch size, so
#: confidence PSI is a per-batch signal that needs no windowing -- unlike
#: vocabulary PSI over a 20 000-term space.
CONFIDENCE_BINS = 10

TRUE_LABEL_COLUMN = "true_label"
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "probability"
CONFIDENCE_COLUMN = "confidence"
CORRECT_COLUMN = "is_correct"


@dataclass(frozen=True, slots=True)
class BatchAggregate:
    """Everything the driver learns about a batch, in a few KB."""

    n_classified: int
    n_correct: int
    accuracy: float | None
    mean_confidence: float | None
    std_confidence: float | None
    confidence_hist: list[int]
    predicted_hist: list[int]
    true_hist: list[int]
    #: Per-row correctness, in offset order. Feeds the error-rate detectors and
    #: is packed to one bit per message for storage.
    correct_sequence: list[int]
    #: Summed raw term counts over the batch, in the model's vocabulary space.
    feature_sums: list[float]

    @property
    def is_empty(self) -> bool:
        return self.n_classified == 0


def score(predictions: DataFrame) -> DataFrame:
    """Add confidence and correctness columns to a scored DataFrame.

    ``probability`` was discarded entirely by the previous consumer, which took
    only ``prediction``. That left no confidence signal, no way to abstain, and
    nothing for a label-free detector to work with.
    """
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    return (
        predictions.withColumn(
            CONFIDENCE_COLUMN, F.array_max(vector_to_array(F.col(PROBABILITY_COLUMN)))
        )
        .withColumn(PREDICTION_COLUMN, F.col(PREDICTION_COLUMN).cast("int"))
        .withColumn(
            CORRECT_COLUMN,
            (F.col(PREDICTION_COLUMN) == F.col(TRUE_LABEL_COLUMN)).cast("int"),
        )
    )


def aggregate_batch(scored_df: DataFrame, *, features: bool = True) -> BatchAggregate:
    """Reduce a scored batch to driver-side scalars with one Spark job.

    ``features=False`` leaves out the vocabulary histogram -- the costliest
    term, a sum over 20 000-dimensional vectors -- for the run that measures
    classification without drift monitoring.
    """
    from pyspark.ml.stat import Summarizer
    from pyspark.sql import functions as F

    aggregations: list[Any] = [
        F.count(F.lit(1)).alias("n"),
        F.sum(CORRECT_COLUMN).alias("n_correct"),
        F.avg(CONFIDENCE_COLUMN).alias("conf_mean"),
        F.stddev_pop(CONFIDENCE_COLUMN).alias("conf_std"),
        # Ordered by Kafka offset so the error sequence fed to ADWIN/DDM is in
        # arrival order rather than partition order.
        F.array_sort(
            F.collect_list(F.struct(F.col("_offset"), F.col(CORRECT_COLUMN)))
        ).alias("correct_pairs"),
    ]
    if features:
        aggregations.append(Summarizer.sum(F.col(RAW_FEATURE_COLUMN)).alias("feature_sums"))
    for index in range(CONFIDENCE_BINS):
        low = index / CONFIDENCE_BINS
        high = (index + 1) / CONFIDENCE_BINS
        predicate = (F.col(CONFIDENCE_COLUMN) >= low) & (
            F.col(CONFIDENCE_COLUMN) < high if index < CONFIDENCE_BINS - 1
            else F.col(CONFIDENCE_COLUMN) <= high
        )
        aggregations.append(
            F.sum(F.when(predicate, 1).otherwise(0)).alias(f"conf_bin_{index}")
        )
    for label in range(N_CLASSES):
        aggregations.append(
            F.sum(F.when(F.col(PREDICTION_COLUMN) == label, 1).otherwise(0)).alias(f"pred_{label}")
        )
        aggregations.append(
            F.sum(F.when(F.col(TRUE_LABEL_COLUMN) == label, 1).otherwise(0)).alias(f"true_{label}")
        )

    row = scored_df.agg(*aggregations).first()
    n = int(row["n"] or 0)
    if n == 0:
        return BatchAggregate(
            n_classified=0, n_correct=0, accuracy=None,
            mean_confidence=None, std_confidence=None,
            confidence_hist=[0] * CONFIDENCE_BINS,
            predicted_hist=[0] * N_CLASSES, true_hist=[0] * N_CLASSES,
            correct_sequence=[], feature_sums=[],
        )

    n_correct = int(row["n_correct"] or 0)
    correct_sequence = [int(pair[CORRECT_COLUMN]) for pair in (row["correct_pairs"] or [])]
    feature_sums = (
        list(row["feature_sums"].toArray())
        if features and row["feature_sums"] is not None
        else []
    )

    return BatchAggregate(
        n_classified=n,
        n_correct=n_correct,
        accuracy=n_correct / n,
        mean_confidence=float(row["conf_mean"]) if row["conf_mean"] is not None else None,
        std_confidence=float(row["conf_std"]) if row["conf_std"] is not None else None,
        confidence_hist=[int(row[f"conf_bin_{i}"] or 0) for i in range(CONFIDENCE_BINS)],
        predicted_hist=[int(row[f"pred_{i}"] or 0) for i in range(N_CLASSES)],
        true_hist=[int(row[f"true_{i}"] or 0) for i in range(N_CLASSES)],
        correct_sequence=correct_sequence,
        feature_sums=feature_sums,
    )


def pack_correct_bits(sequence: list[int]) -> bytes:
    """Pack per-message correctness into one bit each.

    500 messages become 63 bytes, so a whole run's error stream costs a few KB.
    That is what makes detector state rebuildable *exactly* by replay after a
    restart, instead of pickling detector objects whose internals are not a
    stable contract across library versions.
    """
    if not sequence:
        return b""
    import numpy as np

    return np.packbits(np.asarray(sequence, dtype=np.uint8)).tobytes()


def unpack_correct_bits(blob: bytes | memoryview | None, count: int) -> list[int]:
    """Inverse of :func:`pack_correct_bits`, truncated to ``count`` messages."""
    if not blob or count <= 0:
        return []
    import numpy as np

    bits = np.unpackbits(np.frombuffer(bytes(blob), dtype=np.uint8))
    return [int(b) for b in bits[:count]]
