"""The promotion rule, as one pure function with a record of how it decided.

A candidate replaces the incumbent when ``candidate >= incumbent - min_delta``.
What the two scores *are* is the part that matters, and it is recorded with
every decision so the results can be audited and the decisions of repeated
runs compared:

==================  ============================================================
``rule``            the incumbent's score is ...
==================  ============================================================
``matched_holdout`` its accuracy on the candidate's own holdout -- the same
                    messages, so the two numbers are directly comparable
``live``            its measured accuracy on recent batches of the stream --
                    the default, and a comparison across different messages
``recent``          its accuracy on *its own* training-time holdout (the
                    incumbent is itself a retrained model)
``test``            the fixed test set, for both -- no recent data at all
==================  ============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["RULES", "PromotionDecision", "decide"]

RULES = ("matched_holdout", "live", "recent", "test")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    rule: str
    candidate_score: float
    incumbent_score: float
    min_delta: float

    @property
    def margin(self) -> float:
        """How far the candidate cleared (positive) or missed (negative) the bar."""
        return self.candidate_score - (self.incumbent_score - self.min_delta)

    def as_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "rule": self.rule,
            "candidate_score": self.candidate_score,
            "incumbent_score": self.incumbent_score,
            "min_delta": self.min_delta,
            "margin": self.margin,
        }


def decide(
    candidate_score: float, incumbent_score: float, *, min_delta: float, rule: str
) -> PromotionDecision:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}; got {rule!r}")
    return PromotionDecision(
        promoted=candidate_score >= incumbent_score - min_delta,
        rule=rule,
        candidate_score=float(candidate_score),
        incumbent_score=float(incumbent_score),
        min_delta=float(min_delta),
    )
