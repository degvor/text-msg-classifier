"""AG News ingestion.

Replaces a generator that produced 500 rows from 50 sentence skeletons with a
24 % exact-duplicate rate, filled slots positionally regardless of which
template was drawn (``"cybersecurity launches upgraded version of its Apple."``),
and had no seed. Leave-one-out Naive Bayes scored 100 % on it, because 70-84 %
of each class's vocabulary was class-exclusive -- the task was a lookup table,
so any metric computed on it would have been meaningless.

AG News gives 120 000 training and 7 600 test rows over four balanced classes,
in a title+description form that is a reasonable analogue for short social
posts. It is also a standard benchmark, so the thesis can situate its numbers
against published baselines (~92 % for TF-IDF + logistic regression, ~94-95 %
for transformers).
"""

from __future__ import annotations

import csv
import hashlib
import html
import re
import sys
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from streamcls.labels import from_ag_news_index

__all__ = ["MIRROR", "SOURCES", "Document", "ensure_agnews", "load_agnews", "normalise_text"]

MIRROR = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv"

#: filename -> sha256. Verified on download; a truncated or substituted file is
#: an error, not a silently corrupt corpus.
SOURCES: dict[str, str] = {
    "train.csv": "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e",
    "test.csv": "521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699",
    "classes.txt": "612dbfa932f07a6ed837295a7d5746ce7412e7bb4bc32ba919fdb3cda3a1b9b5",
}

_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class Document:
    """One labelled document. ``label`` is our 0-based id, not AG News' 1-based."""

    text: str
    label: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_agnews(dest: Path, *, mirror: str = MIRROR) -> Path:
    """Download AG News into ``dest`` if absent, and verify every checksum.

    Idempotent: an already-correct file is left alone, so this is safe to call
    from a Makefile target on every run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCES.items():
        target = dest / name
        if target.exists() and _sha256(target) == expected:
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(f"{mirror}/{name}", timeout=300) as response:
            tmp.write_bytes(response.read())
        actual = _sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {name}: expected {expected}, got {actual}"
            )
        # Atomic: a reader never sees a partially written corpus file.
        tmp.replace(target)
    return dest


# AG News encodes line breaks inside fields as a literal backslash followed by
# a letter (``dwindling\band of ultra-cynics``). Left alone these fuse two words
# into one nonsense token.
_ESCAPE = re.compile(r"\\+[a-z]?")
_TAG = re.compile(r"<[^>]{0,400}>")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_WHITESPACE = re.compile(r"\s+")


def normalise_text(raw: str) -> str:
    """Strip markup artefacts that would otherwise act as free labels.

    1.5 % of AG News descriptions embed an HTML-escaped Reuters stock-quote
    link (``&lt;A HREF="…investor.reuters.com/FullQuote.aspx?ticker=MRK.N"&gt;``).
    Those appear almost exclusively in Business stories, so leaving them in
    hands the classifier a near-perfect shortcut feature that has nothing to do
    with the language of the post -- exactly the kind of artefact that inflates
    a benchmark number without any real generalisation behind it.

    URLs are removed here rather than kept as "realistic noise" because the
    perturbation engine injects them deliberately and at a known rate; noise we
    control is measurable, noise baked into the corpus is not.
    """
    text = _ESCAPE.sub(" ", raw)
    # Unescape twice: the corpus is escaped, and the payload it escapes is
    # itself HTML (``&amp;lt;`` occurs).
    text = html.unescape(html.unescape(text))
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def load_agnews(path: Path) -> Iterator[Document]:
    """Stream ``Document`` rows from an AG News CSV.

    The corpus is ``"label","title","description"`` with no header. Title and
    description are joined: the title alone is often too short to classify, and
    together they read like a forwarded news post.
    """
    # Some descriptions are long; the default field limit can trip on them.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 3:
                continue
            label_raw, title, description = row
            text = normalise_text(f"{title} {description}")
            if text:
                yield Document(text=text, label=from_ag_news_index(label_raw))
