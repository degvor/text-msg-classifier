"""Classification metrics, derived from a confusion matrix.

Deliberately pure Python. The only thing Spark is needed for is *counting* --
turning predictions into a KxK matrix -- and once that is done the arithmetic
has no reason to live behind a JVM. Keeping it here means the metric maths runs
in the local test suite with no container, which matters when the numbers it
produces are the thesis results.

The previous training script computed nothing at all: no split, no evaluator, no
cross-validation, no confusion matrix. Its entire output was
``print("train done")``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from streamcls.labels import LABELS, N_CLASSES, id_to_key

__all__ = ["ClassMetrics", "Evaluation", "evaluate_confusion"]


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: int
    key: str
    support: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "key": self.key,
            "support": self.support,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Everything a reader needs to judge a model, and nothing derived twice."""

    n: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: tuple[ClassMetrics, ...]
    #: ``confusion[true][predicted]``.
    confusion: tuple[tuple[int, ...], ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 6),
            "macro": {
                "precision": round(self.macro_precision, 6),
                "recall": round(self.macro_recall, 6),
                "f1": round(self.macro_f1, 6),
            },
            "weighted": {
                "precision": round(self.weighted_precision, 6),
                "recall": round(self.weighted_recall, 6),
                "f1": round(self.weighted_f1, 6),
            },
            "per_class": [c.as_dict() for c in self.per_class],
            "confusion": [list(row) for row in self.confusion],
            "labels": [label.key for label in LABELS],
        }

    def summary(self) -> str:
        """Human-readable block, for logs and the training console."""
        width = max(len(c.key) for c in self.per_class) + 1
        lines = [
            f"n={self.n}  accuracy={self.accuracy:.4f}  "
            f"macro-F1={self.macro_f1:.4f}  weighted-F1={self.weighted_f1:.4f}",
            f"  {'class':<{width}} {'prec':>7} {'rec':>7} {'f1':>7} {'support':>8}",
        ]
        for c in self.per_class:
            lines.append(
                f"  {c.key:<{width}} {c.precision:>7.4f} {c.recall:>7.4f} "
                f"{c.f1:>7.4f} {c.support:>8d}"
            )
        lines.append(f"  confusion (rows=true, cols=predicted), labels={[c.key for c in self.per_class]}")
        for key, row in zip((c.key for c in self.per_class), self.confusion, strict=False):
            lines.append(f"    {key:<{width}} " + " ".join(f"{v:>7d}" for v in row))
        return "\n".join(lines)


def _safe_divide(numerator: float, denominator: float) -> float:
    """A class with no predictions has undefined precision; report 0, not NaN.

    NaN would propagate silently into a macro average and quietly poison the
    headline number.
    """
    return numerator / denominator if denominator else 0.0


def evaluate_confusion(confusion: list[list[int]] | tuple[tuple[int, ...], ...]) -> Evaluation:
    """Compute every metric from a ``confusion[true][predicted]`` matrix."""
    matrix = [list(map(int, row)) for row in confusion]
    if len(matrix) != N_CLASSES or any(len(row) != N_CLASSES for row in matrix):
        raise ValueError(
            f"confusion matrix must be {N_CLASSES}x{N_CLASSES}, got "
            f"{len(matrix)}x{len(matrix[0]) if matrix else 0}"
        )

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(N_CLASSES))

    per_class: list[ClassMetrics] = []
    for i in range(N_CLASSES):
        true_positive = matrix[i][i]
        support = sum(matrix[i])  # actual instances of class i
        predicted = sum(matrix[r][i] for r in range(N_CLASSES))
        precision = _safe_divide(true_positive, predicted)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class.append(
            ClassMetrics(
                label=i, key=id_to_key(i), support=support,
                precision=precision, recall=recall, f1=f1,
            )
        )

    def macro(attribute: str) -> float:
        return sum(getattr(c, attribute) for c in per_class) / N_CLASSES

    def weighted(attribute: str) -> float:
        if not total:
            return 0.0
        return sum(getattr(c, attribute) * c.support for c in per_class) / total

    return Evaluation(
        n=total,
        accuracy=_safe_divide(correct, total),
        macro_precision=macro("precision"),
        macro_recall=macro("recall"),
        macro_f1=macro("f1"),
        weighted_precision=weighted("precision"),
        weighted_recall=weighted("recall"),
        weighted_f1=weighted("f1"),
        per_class=tuple(per_class),
        confusion=tuple(tuple(row) for row in matrix),
    )
