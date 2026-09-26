"""Deterministic, stratified, leak-free corpus splits.

Three properties matter, and the previous pipeline had none of them:

* **Deterministic.** ``gen_train_data.py`` never called ``random.seed``, so
  re-running it produced a different corpus every time.
* **Leak-free.** There was no split at all -- ``pipeline.fit(df)`` used 100 % of
  the data, and the label indexer was fitted on all of it too. A naive split
  would not have helped either: with a 24 % exact-duplicate rate, ~35 % of test
  rows had a byte-identical twin in train.
* **Genuinely held out for the stream.** The live stream must emit text the
  model has never seen, or measured "live accuracy" is just training accuracy.

Layout: the AG News *train* file is divided into ``train`` / ``val`` /
``stream_holdout``, and the AG News *test* file becomes ``test``. Keeping the
official test set exclusively for the reported benchmark number means the
headline metric stays comparable to published AG News results, while the stream
still draws on 20 000 documents the model never trained on.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from streamcls.corpus.agnews import Document, load_agnews

__all__ = [
    "DEFAULT_SIZES",
    "RETRAIN_HOLDOUT_EVERY",
    "Split",
    "SplitSet",
    "build_splits",
    "fingerprint",
    "split_retrain_window",
]

T = TypeVar("T")

#: Rows drawn from the 120 000-row AG News train file. The remainder goes to
#: ``train``.
DEFAULT_SIZES: dict[str, int] = {"val": 10_000, "stream_holdout": 20_000}


class Split(str):
    TRAIN = "train"
    VAL = "val"
    STREAM_HOLDOUT = "stream_holdout"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitSet:
    train: list[Document]
    val: list[Document]
    stream_holdout: list[Document]
    test: list[Document]

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "stream_holdout": len(self.stream_holdout),
            "test": len(self.test),
        }

    def label_distribution(self, split: str) -> dict[int, int]:
        return dict(Counter(doc.label for doc in getattr(self, split)))


def fingerprint(docs: Sequence[Document]) -> str:
    """Order-sensitive digest of a split, for asserting reproducibility.

    Used by the tests and recorded alongside an experiment run, so "same seed
    produced the same corpus" is checkable rather than assumed.
    """
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(str(doc.label).encode())
        digest.update(b"\x1f")
        digest.update(doc.text.encode())
        digest.update(b"\x1e")
    return digest.hexdigest()[:16]


def _dedupe(docs: list[Document]) -> list[Document]:
    """Drop exact-duplicate texts, keeping first occurrence.

    AG News contains a small number of repeated stories. Removing them before
    splitting is what makes the train/test boundary meaningful: a duplicate
    straddling the boundary is leakage that inflates the reported score.
    """
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        if doc.text in seen:
            continue
        seen.add(doc.text)
        out.append(doc)
    return out


def _stratified_take(
    docs: list[Document], sizes: dict[str, int], rng: random.Random
) -> tuple[dict[str, list[Document]], list[Document]]:
    """Carve ``sizes`` out of ``docs``, preserving the class balance of each.

    Stratifying matters because every class-imbalance effect in the experiments
    must come from an *injected* drift scenario with known parameters, not from
    an accident of how the corpus happened to be shuffled.
    """
    by_label: dict[int, list[Document]] = defaultdict(list)
    for doc in docs:
        by_label[doc.label].append(doc)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    labels = sorted(by_label)
    taken: dict[str, list[Document]] = {}
    for name, total in sizes.items():
        base, extra = divmod(total, len(labels))
        chunk: list[Document] = []
        for i, label in enumerate(labels):
            want = base + (1 if i < extra else 0)
            bucket = by_label[label]
            if len(bucket) < want:
                raise ValueError(
                    f"split {name!r} wants {want} docs of label {label} but only "
                    f"{len(bucket)} remain"
                )
            chunk.extend(bucket[:want])
            by_label[label] = bucket[want:]
        rng.shuffle(chunk)
        taken[name] = chunk

    remainder = [doc for label in labels for doc in by_label[label]]
    rng.shuffle(remainder)
    return taken, remainder


def build_splits(
    corpus_dir: Path, *, seed: int = 20260827, sizes: dict[str, int] | None = None
) -> SplitSet:
    """Build all four splits from the downloaded AG News files.

    Pure and deterministic: same ``seed`` and same input files give byte-identical
    splits, which is what makes two experiment runs comparable.
    """
    sizes = dict(sizes or DEFAULT_SIZES)
    rng = random.Random(seed)

    train_pool = _dedupe(list(load_agnews(corpus_dir / "train.csv")))
    test_docs = _dedupe(list(load_agnews(corpus_dir / "test.csv")))

    # A document appearing in both official files would otherwise leak the
    # benchmark set into training.
    train_texts = {doc.text for doc in train_pool}
    test_docs = [doc for doc in test_docs if doc.text not in train_texts]

    carved, remainder = _stratified_take(train_pool, sizes, rng)
    return SplitSet(
        train=remainder,
        val=carved["val"],
        stream_holdout=carved["stream_holdout"],
        test=test_docs,
    )


#: One row in this many of the current concept is held out to judge a retrain.
RETRAIN_HOLDOUT_EVERY = 5


def split_retrain_window(window: Sequence[T], *, new_rows: int) -> tuple[list[T], list[T]]:
    """Split a newest-first stream window into ``(train, holdout)`` for a retrain.

    The holdout decides whether the retrained model is promoted, so it must
    come from the current concept: the newest ``new_rows`` rows, which the
    consumer waited for before queueing the retrain. It takes every
    ``RETRAIN_HOLDOUT_EVERY``-th of them, not the newest block -- that block
    *is* the new concept, and holding all of it out trained every thesis-v1
    retrain on the old concept alone. ``new_rows = 0`` spreads the holdout over
    the whole window.
    """
    current = min(new_rows, len(window)) if new_rows > 0 else len(window)
    held = set(range(0, current, RETRAIN_HOLDOUT_EVERY))
    train = [row for i, row in enumerate(window) if i not in held]
    holdout = [row for i, row in enumerate(window) if i in held]
    return train, holdout
