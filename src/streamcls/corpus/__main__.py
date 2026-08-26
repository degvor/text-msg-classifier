"""Corpus CLI: ``python -m streamcls.corpus <command>``.

Used by ``make data``. Download is idempotent and checksum-verified, so running
it on every invocation costs one hash of a 30 MB file and nothing else.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from streamcls.config import get_settings
from streamcls.corpus.agnews import SOURCES, ensure_agnews
from streamcls.corpus.perturb import Perturber
from streamcls.corpus.splits import DEFAULT_SIZES, build_splits, fingerprint
from streamcls.labels import id_to_key


def _corpus_dir(args: argparse.Namespace) -> Path:
    if args.dir:
        return Path(args.dir)
    return Path(get_settings().paths.corpus_dir) / "agnews"


def cmd_download(args: argparse.Namespace) -> int:
    target = _corpus_dir(args)
    print(f"downloading AG News into {target}")
    ensure_agnews(target)
    for name in SOURCES:
        size = (target / name).stat().st_size
        print(f"  {name:<12} {size:>10,} bytes  checksum OK")
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    target = _corpus_dir(args)
    splits = build_splits(target, seed=args.seed)

    print(f"splits from {target} (seed={args.seed})")
    for name, count in splits.counts().items():
        distribution = splits.label_distribution(name)
        pretty = " ".join(f"{id_to_key(k)}={distribution[k]}" for k in sorted(distribution))
        print(f"  {name:<15} {count:>7,}  {pretty}")
        print(f"  {'':<15} {'':>7}  fingerprint={fingerprint(getattr(splits, name))}")

    # Leakage is the failure that silently inflates every reported number, so
    # assert it rather than trusting it.
    texts = {n: {d.text for d in getattr(splits, n)} for n in splits.counts()}
    names = list(texts)
    leaks = [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if texts[a] & texts[b]
    ]
    if leaks:
        print(f"  LEAKAGE between {leaks}", file=sys.stderr)
        return 1
    print("  no overlap between any pair of splits")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Show what the stream actually looks like at a given intensity."""
    splits = build_splits(_corpus_dir(args), seed=args.seed)
    rng = random.Random(args.seed)
    perturber = Perturber(intensity=args.intensity)
    for doc in splits.stream_holdout[: args.count]:
        result = perturber.apply(doc.text, rng)
        print(f"\n[{id_to_key(doc.label)}] applied={','.join(result.applied) or 'none'}")
        print(f"  {result.text[:300]}")
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    """Print a ready-made drift schedule to paste into an ExperimentRun."""
    total = args.total
    schedules = {
        "stationary": [],
        "prior_shift": [
            {"at": total // 3, "kind": "prior_shift", "params": {"weights": [0.7, 0.1, 0.1, 0.1]}}
        ],
        "covariate_gradual": [
            {
                "at": total // 3,
                "kind": "covariate_gradual",
                "params": {"to": 0.9, "over": total // 3},
            }
        ],
        "concept_abrupt": [
            {"at": total // 2, "kind": "concept_abrupt", "params": {"map": {"2": 3}}}
        ],
        "recurring": [
            {"at": total // 4, "kind": "recurring", "params": {"period": total // 8, "to": 0.85}}
        ],
    }
    print(json.dumps(schedules[args.scenario], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamcls.corpus")
    parser.add_argument("--dir", help="corpus directory (default: from settings)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch AG News and verify checksums").set_defaults(
        func=cmd_download
    )

    splits = sub.add_parser("splits", help="build splits and report the distribution")
    splits.add_argument("--seed", type=int, default=20260827)
    splits.set_defaults(func=cmd_splits)

    sample = sub.add_parser("sample", help="preview perturbed stream messages")
    sample.add_argument("--seed", type=int, default=20260827)
    sample.add_argument("--intensity", type=float, default=0.5)
    sample.add_argument("--count", type=int, default=5)
    sample.set_defaults(func=cmd_sample)

    schedule = sub.add_parser("schedule", help="emit a drift schedule as JSON")
    schedule.add_argument("scenario", choices=[
        "stationary", "prior_shift", "covariate_gradual", "concept_abrupt", "recurring"
    ])
    schedule.add_argument("--total", type=int, default=20_000)
    schedule.set_defaults(func=cmd_schedule)

    args = parser.parse_args(argv)
    if args.command == "splits" and args.seed is None:
        args.seed = 20260827
    _ = DEFAULT_SIZES  # documented default, referenced for discoverability
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
