"""Turn a predictions DataFrame into a confusion matrix.

This is the only part of evaluation that needs Spark. Everything derived from
the matrix lives in :mod:`streamcls.metrics`, which is pure Python and testable
without a JVM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from streamcls.labels import N_CLASSES
from streamcls.metrics import Evaluation, evaluate_confusion
from streamcls.spark.features import LABEL_COLUMN

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

__all__ = ["confusion_matrix", "evaluate_predictions"]


def confusion_matrix(predictions: DataFrame, *, prediction_col: str = "prediction") -> list[list[int]]:
    """``matrix[true][predicted]`` counts, as plain Python ints.

    One shuffle and at most K^2 rows come back to the driver, so this is O(1) in
    driver memory regardless of dataset size.
    """
    from pyspark.sql import functions as F

    rows = (
        predictions.select(
            F.col(LABEL_COLUMN).cast("int").alias("t"),
            F.col(prediction_col).cast("int").alias("p"),
        )
        .groupBy("t", "p")
        .count()
        .collect()
    )

    matrix = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    for row in rows:
        true_label, predicted = row["t"], row["p"]
        if true_label is None or predicted is None:
            continue
        if 0 <= true_label < N_CLASSES and 0 <= predicted < N_CLASSES:
            matrix[true_label][predicted] += row["count"]
    return matrix


def evaluate_predictions(predictions: DataFrame, *, prediction_col: str = "prediction") -> Evaluation:
    return evaluate_confusion(confusion_matrix(predictions, prediction_col=prediction_col))
