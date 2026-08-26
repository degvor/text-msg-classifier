"""Turn clean news text into something that reads like a forwarded channel post.

This replaces 25 hardcoded headline strings. Those strings shared *zero*
sentences with the training corpus and 70.5 % of their vocabulary never appeared
in training at all, so the deployed model was asked to generalise out of
distribution on every single message -- which is why seven of the 25 produced an
all-zero feature vector and were classified purely by the model's intercepts.

Here the stream is drawn from a held-out split of the *same* corpus the model
trains on, and the domain gap is introduced deliberately, at a known and
tunable intensity. That is the difference between an accidental failure and a
controlled experimental variable: ``intensity`` is what a covariate-shift drift
scenario ramps.

Every operator is a pure function of ``(text, rng, intensity)``. Nothing reads
global state and nothing calls ``random`` at module level, so a run is
reproducible from its seed.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["OPERATORS", "PerturbationResult", "Perturber", "perturb"]

# --- vocabulary used by the operators ---------------------------------------
_EMOJI = ("🔥", "⚡", "🚨", "📈", "📉", "💥", "👀", "🇺🇸", "⚽", "🧠", "💰", "🛰", "‼️", "🤖")
_HASHTAGS = (
    "breaking", "news", "update", "urgent", "live", "markets", "tech",
    "world", "sports", "analysis", "must_read", "today",
)
_CHANNELS = (
    "@worldfeed", "@marketwatch_live", "@techdrop", "@sportsnow",
    "@thebriefing", "@wirefeed", "@dailypulse",
)
_SHORTLINKS = ("t.me/c/{n}", "bit.ly/{s}", "cutt.ly/{s}", "tgraph.io/{s}")
_FORWARD_PREFIXES = (
    "Forwarded from {channel}",
    "Forwarded from {channel} ({n} subscribers)",
    "via {channel}",
    "{channel}:",
)
_ABBREVIATIONS = {
    "government": "govt",
    "president": "pres",
    "international": "intl",
    "company": "co",
    "percent": "pct",
    "million": "mln",
    "billion": "bln",
    "market": "mkt",
    "quarter": "Q",
    "versus": "vs",
    "against": "vs",
    "approximately": "approx",
    "department": "dept",
    "management": "mgmt",
    "technology": "tech",
    "information": "info",
    "you": "u",
    "are": "r",
    "and": "&",
    "with": "w/",
    "without": "w/o",
    "because": "bc",
    "tonight": "2nite",
}
#: Zero-width joiner / non-joiner / space. Invisible, but they split tokens, so
#: they are a genuine tokeniser stressor rather than cosmetic noise.
_ZERO_WIDTH = ("​", "‌", "‍")

# QWERTY neighbours, for typos that look like real fat-finger errors rather
# than uniform random character noise.
_NEIGHBOURS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdx",
    "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm",
    "k": "jil", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

_WORD = re.compile(r"\b[A-Za-z]{3,}\b")


def _rand_slug(rng: random.Random, length: int = 7) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


# --- operators ---------------------------------------------------------------
# Each returns the transformed text. `intensity` in [0, 1] scales how strongly
# the operator applies; at 0 every operator is the identity.


def add_emoji(text: str, rng: random.Random, intensity: float) -> str:
    count = 1 + int(rng.random() * intensity * 3)
    picks = [rng.choice(_EMOJI) for _ in range(count)]
    if rng.random() < 0.5:
        return f"{''.join(picks)} {text}"
    return f"{text} {''.join(picks)}"


def add_hashtags(text: str, rng: random.Random, intensity: float) -> str:
    count = 1 + int(rng.random() * intensity * 3)
    tags = rng.sample(_HASHTAGS, min(count, len(_HASHTAGS)))
    return f"{text} " + " ".join(f"#{t}" for t in tags)


def add_mention(text: str, rng: random.Random, intensity: float) -> str:
    return f"{text} {rng.choice(_CHANNELS)}"


def add_shortlink(text: str, rng: random.Random, intensity: float) -> str:
    template = rng.choice(_SHORTLINKS)
    return f"{text} {template.format(n=rng.randint(1000, 9999), s=_rand_slug(rng))}"


def add_forward_header(text: str, rng: random.Random, intensity: float) -> str:
    prefix = rng.choice(_FORWARD_PREFIXES).format(
        channel=rng.choice(_CHANNELS), n=f"{rng.randint(2, 900)}K"
    )
    return f"{prefix}\n{text}"


def shout(text: str, rng: random.Random, intensity: float) -> str:
    """Upper-case a run of words, the way channels emphasise a lede."""
    words = text.split()
    if len(words) < 3:
        return text.upper()
    span = max(1, int(len(words) * min(0.5, 0.1 + intensity * 0.4)))
    start = rng.randrange(0, max(1, len(words) - span))
    for i in range(start, min(start + span, len(words))):
        words[i] = words[i].upper()
    return " ".join(words)


def typos(text: str, rng: random.Random, intensity: float) -> str:
    """Keyboard-adjacent substitutions, transpositions and drops."""
    chars = list(text)
    positions = [i for i, ch in enumerate(chars) if ch.lower() in _NEIGHBOURS]
    if not positions:
        return text
    n = max(1, int(len(positions) * intensity * 0.06))
    for index in rng.sample(positions, min(n, len(positions))):
        roll = rng.random()
        lower = chars[index].lower()
        if roll < 0.6:
            chars[index] = rng.choice(_NEIGHBOURS[lower])
        elif roll < 0.85 and index + 1 < len(chars):
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
        else:
            chars[index] = ""
    return "".join(chars)


def abbreviate(text: str, rng: random.Random, intensity: float) -> str:
    def swap(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = _ABBREVIATIONS.get(word.lower())
        if replacement is None or rng.random() > min(1.0, 0.3 + intensity):
            return word
        return replacement.upper() if word.isupper() else replacement

    return _WORD.sub(swap, text)


def inflate_punctuation(text: str, rng: random.Random, intensity: float) -> str:
    repeats = 2 + int(intensity * 3)
    mark = rng.choice(("!", "?", "!?"))
    return text.rstrip(".!? ") + mark * repeats


def truncate(text: str, rng: random.Random, intensity: float) -> str:
    """Clip the post, as a channel does when it exceeds the preview length."""
    words = text.split()
    if len(words) < 8:
        return text
    keep = max(5, int(len(words) * (1.0 - min(0.75, intensity * 0.75))))
    tail = rng.choice(("… read more", "…", "... continued", "[…]"))
    return " ".join(words[:keep]) + " " + tail


def zero_width(text: str, rng: random.Random, intensity: float) -> str:
    """Inject invisible characters. They split tokens without being visible."""
    chars = list(text)
    n = max(1, int(len(chars) * intensity * 0.02))
    for _ in range(n):
        chars.insert(rng.randrange(0, len(chars) + 1), rng.choice(_ZERO_WIDTH))
    return "".join(chars)


Operator = Callable[[str, random.Random, float], str]

#: name -> (operator, base probability at intensity 1.0). Probabilities are
#: scaled by intensity, so intensity 0 is a guaranteed no-op and the identity
#: case is exactly reproducible.
OPERATORS: dict[str, tuple[Operator, float]] = {
    "emoji": (add_emoji, 0.55),
    "hashtags": (add_hashtags, 0.45),
    "mention": (add_mention, 0.30),
    "shortlink": (add_shortlink, 0.25),
    "forward_header": (add_forward_header, 0.35),
    "shout": (shout, 0.20),
    "typos": (typos, 0.45),
    "abbreviate": (abbreviate, 0.40),
    "punctuation": (inflate_punctuation, 0.30),
    "truncate": (truncate, 0.25),
    "zero_width": (zero_width, 0.10),
}


@dataclass(frozen=True, slots=True)
class PerturbationResult:
    text: str
    #: Which operators actually fired. Recorded so a post's surface form can be
    #: explained after the fact, and so tests can assert on behaviour rather
    #: than on exact strings.
    applied: tuple[str, ...]


@dataclass(slots=True)
class Perturber:
    """Applies the operator set at a given intensity.

    ``intensity`` is the experimental knob: 0.0 leaves text untouched, 1.0 makes
    it heavily channel-styled. A ``gradual_covariate_shift`` scenario ramps it.
    """

    intensity: float = 0.5
    enabled: tuple[str, ...] = field(default_factory=lambda: tuple(OPERATORS))

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")
        unknown = set(self.enabled) - set(OPERATORS)
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")

    def apply(self, text: str, rng: random.Random) -> PerturbationResult:
        if self.intensity <= 0.0:
            return PerturbationResult(text=text, applied=())

        applied: list[str] = []
        out = text
        # Fixed order, so the same seed gives the same result regardless of
        # dict iteration details.
        for name in self.enabled:
            operator, base_probability = OPERATORS[name]
            if rng.random() < base_probability * self.intensity:
                out = operator(out, rng, self.intensity)
                applied.append(name)
        return PerturbationResult(text=out.strip(), applied=tuple(applied))


def perturb(text: str, rng: random.Random, intensity: float = 0.5) -> str:
    """Convenience wrapper for one-off use."""
    return Perturber(intensity=intensity).apply(text, rng).text
