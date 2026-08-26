"""Splits must be deterministic, stratified and leak-free.

The previous pipeline had none of these: no split at all (``pipeline.fit(df)``
saw 100 % of the data), no seed, and a 24 % exact-duplicate rate that would have
put a byte-identical twin of ~35 % of any test set into training.
"""

import csv

import pytest

from streamcls.corpus.splits import build_splits, fingerprint
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
