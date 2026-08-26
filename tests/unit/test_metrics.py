"""Metric arithmetic. These numbers become the thesis results, so the maths is
checked against hand-computed cases rather than against another implementation.
"""

import pytest

from streamcls.labels import N_CLASSES
from streamcls.metrics import evaluate_confusion

PERFECT = [[10 if i == j else 0 for j in range(N_CLASSES)] for i in range(N_CLASSES)]


def test_perfect_classifier():
    e = evaluate_confusion(PERFECT)
    assert e.accuracy == 1.0
    assert e.macro_f1 == 1.0
    assert e.n == 40
    assert all(c.precision == 1.0 and c.recall == 1.0 for c in e.per_class)


def test_hand_computed_case():
    # rows = true, cols = predicted
    matrix = [
        [8, 1, 1, 0],
        [0, 9, 0, 1],
        [2, 0, 7, 1],
        [0, 1, 0, 9],
    ]
    e = evaluate_confusion(matrix)
    assert e.n == 40
    assert e.accuracy == pytest.approx((8 + 9 + 7 + 9) / 40)

    world = e.per_class[0]
    assert world.support == 10                       # row sum
    assert world.recall == pytest.approx(8 / 10)     # TP / support
    assert world.precision == pytest.approx(8 / 10)  # TP / column sum (8+0+2+0)
    assert world.f1 == pytest.approx(0.8)

    business = e.per_class[2]
    assert business.precision == pytest.approx(7 / 8)   # column 2 = 1+0+7+0
    assert business.recall == pytest.approx(7 / 10)


def test_a_class_that_is_never_predicted_reports_zero_not_nan():
    """The baseline system predicted `politics` exactly zero times across 690
    messages. A NaN precision would have propagated into the macro average and
    quietly poisoned the headline number."""
    matrix = [
        [10, 0, 0, 0],
        [10, 0, 0, 0],   # class 1 exists but is never predicted
        [0, 0, 10, 0],
        [0, 0, 0, 10],
    ]
    e = evaluate_confusion(matrix)
    never_predicted = e.per_class[1]
    assert never_predicted.support == 10
    assert never_predicted.recall == 0.0
    assert never_predicted.precision == 0.0
    assert never_predicted.f1 == 0.0
    assert e.macro_f1 == e.macro_f1  # not NaN
    assert 0.0 < e.macro_f1 < 1.0


def test_macro_and_weighted_differ_under_imbalance():
    matrix = [
        [100, 0, 0, 0],
        [0, 1, 9, 0],     # tiny, badly classified class
        [0, 0, 100, 0],
        [0, 0, 0, 100],
    ]
    e = evaluate_confusion(matrix)
    # Macro treats the tiny class equally; weighted lets the big ones dominate.
    assert e.macro_f1 < e.weighted_f1


def test_weighted_recall_equals_accuracy():
    """A useful invariant: weighted recall is accuracy by definition."""
    matrix = [[8, 1, 1, 0], [0, 9, 0, 1], [2, 0, 7, 1], [0, 1, 0, 9]]
    e = evaluate_confusion(matrix)
    assert e.weighted_recall == pytest.approx(e.accuracy)


def test_empty_matrix_is_all_zeros_not_a_crash():
    e = evaluate_confusion([[0] * N_CLASSES for _ in range(N_CLASSES)])
    assert e.n == 0
    assert e.accuracy == 0.0
    assert e.macro_f1 == 0.0


@pytest.mark.parametrize("bad", [[[1, 2], [3, 4]], [[1] * N_CLASSES] * 2, []])
def test_wrong_shape_is_rejected(bad):
    with pytest.raises(ValueError, match="confusion matrix must be"):
        evaluate_confusion(bad)


def test_as_dict_is_json_serialisable_and_complete():
    import json

    payload = evaluate_confusion(PERFECT).as_dict()
    json.dumps(payload)  # must not raise: this goes into a JSONB column
    assert set(payload) == {"n", "accuracy", "macro", "weighted", "per_class", "confusion", "labels"}
    assert len(payload["confusion"]) == N_CLASSES
    assert len(payload["per_class"]) == N_CLASSES
    assert payload["labels"] == ["world", "sports", "business", "sci_tech"]


def test_summary_renders_every_class_and_the_matrix():
    text = evaluate_confusion(PERFECT).summary()
    for key in ("world", "sports", "business", "sci_tech"):
        assert key in text
    assert "confusion" in text
    assert "accuracy=1.0000" in text
