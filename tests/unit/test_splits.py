"""Splits must be deterministic, stratified and leak-free.

The previous pipeline had none of these: no split at all (``pipeline.fit(df)``
saw 100 % of the data), no seed, and a 24 % exact-duplicate rate that would have
put a byte-identical twin of ~35 % of any test set into training.
"""

import csv

import pytest

from streamcls.corpus.splits import build_splits, fingerprint, split_retrain_window
from streamcls.labels import N_CLASSES


@pytest.fixture
def corpus(tmp_path):
    """A miniature AG News: 400 train rows and 80 test rows, balanced."""

    def write(name, per_class, prefix):
        rows = [
            [str(label + 1), f"{prefix} title {label}-{i}", f"{prefix} body {label}-{i}"]
            for label in range(N_CLASSES)
            for i in range(per_class)
        ]
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)

    write("train.csv", 100, "tr")
    write("test.csv", 20, "te")
    return tmp_path


SIZES = {"val": 40, "stream_holdout": 80}


def test_sizes_and_remainder(corpus):
    splits = build_splits(corpus, sizes=SIZES)
    assert splits.counts() == {"train": 280, "val": 40, "stream_holdout": 80, "test": 80}


def test_every_split_keeps_the_class_balance(corpus):
    """All imbalance in the experiments must come from an injected scenario
    with known parameters, never from how the corpus happened to shuffle."""
    splits = build_splits(corpus, sizes=SIZES)
    for name in ("train", "val", "stream_holdout", "test"):
        counts = splits.label_distribution(name)
        assert set(counts) == set(range(N_CLASSES)), name
        assert len(set(counts.values())) == 1, f"{name} is not balanced: {counts}"


def test_splits_are_disjoint(corpus):
    splits = build_splits(corpus, sizes=SIZES)
    texts = {
        name: {doc.text for doc in getattr(splits, name)}
        for name in ("train", "val", "stream_holdout", "test")
    }
    names = list(texts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not texts[a] & texts[b], f"{a} leaks into {b}"


def test_same_seed_is_byte_identical(corpus):
    a = build_splits(corpus, seed=7, sizes=SIZES)
    b = build_splits(corpus, seed=7, sizes=SIZES)
    for name in ("train", "val", "stream_holdout", "test"):
        assert fingerprint(getattr(a, name)) == fingerprint(getattr(b, name)), name


def test_different_seed_reshuffles(corpus):
    a = build_splits(corpus, seed=7, sizes=SIZES)
    b = build_splits(corpus, seed=8, sizes=SIZES)
    assert fingerprint(a.stream_holdout) != fingerprint(b.stream_holdout)


def test_exact_duplicates_are_removed_before_splitting(corpus):
    """A duplicate straddling the train/test boundary is leakage that inflates
    the reported score."""
    with (corpus / "train.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for _ in range(20):
            writer.writerow(["1", "tr title 0-0", "tr body 0-0"])

    splits = build_splits(corpus, sizes=SIZES)
    everything = [
        doc.text
        for name in ("train", "val", "stream_holdout", "test")
        for doc in getattr(splits, name)
    ]
    assert len(everything) == len(set(everything))


def test_a_document_in_both_official_files_is_dropped_from_test(corpus):
    """Otherwise the benchmark set leaks into training."""
    with (corpus / "test.csv").open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["1", "tr title 0-5", "tr body 0-5"])

    splits = build_splits(corpus, sizes=SIZES)
    train_texts = {doc.text for doc in splits.train}
    assert not train_texts & {doc.text for doc in splits.test}


def test_requesting_more_than_the_corpus_holds_is_an_error(corpus):
    with pytest.raises(ValueError, match=r"only \d+ remain"):
        build_splits(corpus, sizes={"val": 10_000})


def test_fingerprint_is_order_sensitive(corpus):
    splits = build_splits(corpus, sizes=SIZES)
    reversed_docs = list(reversed(splits.val))
    assert fingerprint(splits.val) != fingerprint(reversed_docs)


# ---------------------------------------------------------------------------
# The retrain window: what an adaptive retrain learns from and is judged on
# ---------------------------------------------------------------------------


def test_the_retrain_holdout_leaves_most_of_the_new_concept_to_train_on():
    """thesis-v1 held out the newest block, which at retrain time *was* the
    new concept: every retrain trained on 0 of its 2000 post-drift rows."""
    # Newest first, as the stream window is read: 2000 new-concept rows, then 8000 old.
    window = [("new", i) for i in range(2_000)] + [("old", i) for i in range(8_000)]
    train, holdout = split_retrain_window(window, new_rows=2_000)

    assert len(holdout) == 400
    assert all(kind == "new" for kind, _ in holdout), "judged on the current concept only"
    assert sum(kind == "new" for kind, _ in train) == 1_600
    assert sorted(train + holdout) == sorted(window), "every row used exactly once"


def test_without_a_new_row_count_the_holdout_is_spread_over_the_window():
    train, holdout = split_retrain_window(list(range(10)), new_rows=0)
    assert holdout == [0, 5]
    assert train == [1, 2, 3, 4, 6, 7, 8, 9]


def test_the_retrain_split_copes_with_a_short_window():
    assert split_retrain_window([], new_rows=2_000) == ([], [])
    assert split_retrain_window([1, 2, 3], new_rows=2_000) == ([2, 3], [1])


def test_a_short_training_part_is_judged_on_the_same_holdout():
    """Composition control: 500 newest rows trained on, holdout over the newest 2000."""
    window = [("new", i) for i in range(2_000)] + [("old", i) for i in range(18_000)]
    train, holdout = split_retrain_window(window, new_rows=2_000, train_rows=500)
    _, full_holdout = split_retrain_window(window, new_rows=2_000)

    assert holdout == full_holdout, "same holdout whatever the training part"
    assert len(holdout) == 400
    assert len(train) == 400, "the newest 500 minus the 100 of them held out"
    assert all(kind == "new" for kind, _ in train)
    assert not set(train) & set(holdout)


def test_train_rows_beyond_the_window_change_nothing():
    window = list(range(10))
    assert split_retrain_window(window, new_rows=0, train_rows=50) == split_retrain_window(
        window, new_rows=0
    )

