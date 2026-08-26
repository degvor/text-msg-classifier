"""The label map is a constant, and these tests are what keep it one.

The bug being guarded against: the previous pipeline derived label integers from
a StringIndexer ordered by class frequency, so retraining on differently
balanced data silently renumbered every class and retroactively mislabelled
every stored prediction.
"""

import pytest

from streamcls.labels import (
    LABELS,
    N_CLASSES,
    from_ag_news_index,
    id_to_key,
    key_to_id,
    label_by_id,
)

# Frozen on purpose. If this test fails, every category id already written to
# the database has changed meaning -- that is a migration, not a code change.
EXPECTED = {0: "world", 1: "sports", 2: "business", 3: "sci_tech"}


def test_label_ids_are_pinned():
    assert {lab.id: lab.key for lab in LABELS} == EXPECTED


def test_ids_are_dense_and_ordered():
    assert [lab.id for lab in LABELS] == list(range(N_CLASSES))


def test_keys_and_names_are_unique():
    assert len({lab.key for lab in LABELS}) == N_CLASSES
    assert len({lab.display_name for lab in LABELS}) == N_CLASSES


@pytest.mark.parametrize(("raw", "key"), [(1, "world"), (2, "sports"), (3, "business"), (4, "sci_tech")])
def test_ag_news_index_maps_to_our_ids(raw, key):
    assert id_to_key(from_ag_news_index(raw)) == key
    assert from_ag_news_index(str(raw)) == from_ag_news_index(raw)


@pytest.mark.parametrize("bad", [0, 5, -1])
def test_ag_news_index_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="out of range"):
        from_ag_news_index(bad)


def test_round_trip():
    for lab in LABELS:
        assert key_to_id(id_to_key(lab.id)) == lab.id
        assert label_by_id(lab.id) is lab


def test_unknown_lookups_raise():
    with pytest.raises(ValueError, match="unknown label id"):
        label_by_id(99)
    with pytest.raises(ValueError, match="unknown label key"):
        key_to_id("politics")  # a class from the old five-label set
