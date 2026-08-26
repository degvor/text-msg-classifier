"""The label set, pinned in code.

The previous pipeline derived label integers with a Spark ``StringIndexer``
using the default ``stringOrderType="frequencyDesc"``. That made the integer for
each class a function of the *training class balance*: the mapping only looked
alphabetical because all five classes happened to tie at exactly 100 rows.
Deduplicating the corpus alone would have reshuffled every id, silently
remapping stored category colours and retroactively corrupting every archived
prediction, since rows store the raw integer and it is resolved at read time.

AG News has four fixed classes, so the mapping is a constant. A constant cannot
drift.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LABELS",
    "N_CLASSES",
    "Label",
    "from_ag_news_index",
    "id_to_key",
    "key_to_id",
    "label_by_id",
]


@dataclass(frozen=True, slots=True)
class Label:
    id: int
    key: str
    display_name: str
    description: str


#: Ordered by id. ``id`` is what every table, Kafka message and model output
#: stores; ``key`` is the stable machine-readable slug; ``display_name`` is for
#: the UI only and may be edited without consequence.
LABELS: tuple[Label, ...] = (
    Label(0, "world", "World", "International news, politics and current events"),
    Label(1, "sports", "Sports", "Sporting events, results and athletes"),
    Label(2, "business", "Business", "Markets, companies, economics and finance"),
    Label(3, "sci_tech", "Sci/Tech", "Science, technology and research"),
)

N_CLASSES: int = len(LABELS)

_BY_ID: dict[int, Label] = {label.id: label for label in LABELS}
_BY_KEY: dict[str, Label] = {label.key: label for label in LABELS}


def label_by_id(label_id: int) -> Label:
    try:
        return _BY_ID[label_id]
    except KeyError:
        raise ValueError(f"unknown label id {label_id!r}; expected 0..{N_CLASSES - 1}") from None


def id_to_key(label_id: int) -> str:
    return label_by_id(label_id).key


def key_to_id(key: str) -> int:
    try:
        return _BY_KEY[key].id
    except KeyError:
        raise ValueError(f"unknown label key {key!r}; expected one of {sorted(_BY_KEY)}") from None


def from_ag_news_index(raw: int | str) -> int:
    """Map an AG News CSV label (1..4) to our 0-based id.

    The published CSV encodes World=1, Sports=2, Business=3, Sci/Tech=4.
    """
    value = int(raw)
    if not 1 <= value <= N_CLASSES:
        raise ValueError(f"AG News label out of range: {raw!r}; expected 1..{N_CLASSES}")
    return value - 1
