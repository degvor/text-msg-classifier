"""The seeded message stream.

Given a run seed, a held-out document pool and a drift schedule, this produces
exactly the same sequence of messages every time -- which is what lets two
experiment runs be compared. The headline figure of the thesis is two runs with
an identical seed differing only in whether adaptation was enabled; that
comparison is only valid if the streams really are byte-identical up to the
first adaptation.

**One RNG per concern.** Ordering, perturbation, class sampling and timing each
draw from their own generator seeded from the run seed. With a single shared
generator, changing the perturbation intensity would shift every subsequent
draw, so an ablation would silently be comparing two different streams rather
than one stream under two settings.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from streamcls.corpus.agnews import Document
from streamcls.corpus.perturb import Perturber
from streamcls.corpus.schedule import DriftSchedule
from streamcls.labels import LABELS, N_CLASSES, label_by_id
from streamcls.schemas import PostMessage

__all__ = ["CHANNELS", "SeededStream"]

#: Telegram framing. One channel per class, so ``ChannelTitle`` stays a
#: plausible surface feature -- but note the consumer must never use it: it is
#: a perfect proxy for the label, which is exactly how the old pipeline's
#: "accuracy" was accidentally measurable.
CHANNELS: dict[int, tuple[str, str]] = {
    0: ("world_wire", "World Wire"),
    1: ("terrace_talk", "Terrace Talk"),
    2: ("market_desk", "Market Desk"),
    3: ("signal_lab", "Signal Lab"),
}


def _weighted_choice(rng: random.Random, weights: Sequence[float]) -> int:
    roll = rng.random()
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if roll < cumulative:
            return index
    return len(weights) - 1  # float dust


@dataclass(slots=True)
class SeededStream:
    """Deterministic iterator over :class:`~streamcls.schemas.PostMessage`."""

    documents: Sequence[Document]
    schedule: DriftSchedule
    seed: int
    run_id: int
    total_messages: int
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: Seconds between messages, used only to stamp ``produced_at``. In replay
    #: mode the producer does not actually sleep.
    interval_seconds: float = 0.02

    _pools: dict[int, list[Document]] = field(init=False, default_factory=dict)
    _cursors: dict[int, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_messages < 1:
            raise ValueError("total_messages must be positive")
        if not self.documents:
            raise ValueError("document pool is empty")

        # Distinct namespaces, so tweaking one knob cannot perturb another's
        # draw sequence.
        order_rng = random.Random(f"{self.seed}:order")

        pools: dict[int, list[Document]] = {label.id: [] for label in LABELS}
        for doc in self.documents:
            pools.setdefault(doc.label, []).append(doc)
        for label, bucket in pools.items():
            if not bucket:
                raise ValueError(f"no documents for label {label} in the pool")
            order_rng.shuffle(bucket)
        self._pools = pools
        self._cursors = dict.fromkeys(pools, 0)

    def _next_document(self, label: int) -> Document:
        """Take the next document for ``label``, wrapping around the pool.

        Wrapping is deterministic and reported: a run longer than the pool
        repeats text, which matters when reading the results and is why
        :meth:`wraps_at` exists.
        """
        pool = self._pools[label]
        cursor = self._cursors[label]
        self._cursors[label] = cursor + 1
        return pool[cursor % len(pool)]

    def wraps_at(self) -> dict[int, int]:
        """Messages of each class that can be emitted before text repeats."""
        return {label: len(pool) for label, pool in self._pools.items()}

    def __iter__(self) -> Iterator[PostMessage]:
        perturb_rng = random.Random(f"{self.seed}:perturb")
        prior_rng = random.Random(f"{self.seed}:prior")

        for index in range(self.total_messages):
            state = self.schedule.state_at(index)

            # The label the message genuinely carries.
            true_label = _weighted_choice(prior_rng, state.label_weights)
            # Under concept drift the text is drawn from a different class's
            # pool: same-looking text, different label. That is a change in
            # P(y|x), and it is what the error detectors must catch while
            # vocabulary PSI stays flat.
            source_label = state.concept_map.get(true_label, true_label)
            document = self._next_document(source_label)

            result = Perturber(intensity=state.intensity).apply(document.text, perturb_rng)

            handle, title = CHANNELS[true_label]
            yield PostMessage(
                experiment_run_id=self.run_id,
                post_id=f"{self.run_id}-{index:06d}",
                msg_index=index,
                text=result.text,
                channel_id=handle,
                channel_title=title,
                view_count=self._view_count(prior_rng),
                produced_at=self.start_time + timedelta(seconds=index * self.interval_seconds),
                true_label=true_label,
                perturbation_kind=state.kind,
                perturbation_intensity=round(state.intensity, 4),
                drift_phase=state.phase,
                is_drift_point=state.is_drift_point,
            )

    @staticmethod
    def _view_count(rng: random.Random) -> int:
        # Log-normal-ish, and never negative. The old generator computed
        # `base + randint(-100, 500)` where base could be 80, so it emitted
        # negative view counts into an unvalidated IntegerField.
        return round(rng.lognormvariate(6.5, 1.1))

    def describe(self) -> dict[str, object]:
        """Summary for logging at run start; also handy in the thesis appendix."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "total_messages": self.total_messages,
            "pool_sizes": self.wraps_at(),
            "change_points": list(self.schedule.change_points),
            "base_intensity": self.schedule.base_intensity,
            "classes": {label.id: label.key for label in LABELS},
            "n_classes": N_CLASSES,
        }


def channel_for(label_id: int) -> tuple[str, str]:
    return CHANNELS[label_by_id(label_id).id]
