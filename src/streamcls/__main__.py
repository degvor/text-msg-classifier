"""CLI: ``python -m streamcls <command>``.

``score`` turns finished runs into the results table the thesis reports.
``compare`` prints the A/B figure for a seed-matched pair.
"""

from __future__ import annotations

import argparse
import json
import sys

from streamcls.config import get_settings
from streamcls.db.queries import fetch_run_bundle, list_runs
from streamcls.db.repo import Repository
from streamcls.experiments import compare, format_table, score_run


def _score(repo, run_id: int, batch_size: int):
    return score_run(**fetch_run_bundle(repo, run_id), batch_size=batch_size)


def cmd_runs(args, repo) -> int:
    rows = list_runs(repo)
    if not rows:
        print("No experiment runs yet.")
        return 0
    print(f"{'id':>4}  {'name':<34} {'seed':>10} {'adapt':>6} {'status':>10}")
    for r in rows:
        adapt = "on" if r["adaptation_enabled"] else "off"
        print(f"{r['id']:>4}  {r['name'][:34]:<34} {r['seed']:>10} {adapt:>6} {r['status']:>10}")
    return 0


def cmd_score(args, repo) -> int:
    batch_size = args.batch_size or repo.read_config()["max_offsets_per_trigger"]
    ids = args.runs or [r["id"] for r in list_runs(repo, limit=args.limit)]
    scores = []
    for run_id in ids:
        try:
            scores.append(_score(repo, run_id, batch_size))
        except LookupError as error:
            print(f"skipping: {error}", file=sys.stderr)
    if not scores:
        print("Nothing to score.")
        return 1
    scores.sort(key=lambda s: s.run_id)
    if args.json:
        print(json.dumps([s.as_dict() for s in scores], indent=2, default=str))
    else:
        print(format_table(scores))
        print("\nFA = false alarms, miss = missed detections, "
              "delay/recov in batches.")
    return 0


def cmd_compare(args, repo) -> int:
    batch_size = args.batch_size or repo.read_config()["max_offsets_per_trigger"]
    baseline = _score(repo, args.baseline, batch_size)
    adaptive = _score(repo, args.adaptive, batch_size)
    result = compare(baseline, adaptive)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(format_table([baseline, adaptive]))
    if not result["seed_matches"]:
        # Without a shared seed the two streams differ, so any accuracy gap is
        # partly the stream and not the adaptation.
        print("\nWARNING: seeds differ -- this is not a controlled comparison.")
    print("\nA/B (adaptive - baseline):")
    for key in ("accuracy_after_drift", "accuracy_final"):
        block = result[key]
        delta = block["delta"]
        arrow = "" if delta is None else f"  ({delta * 100:+.1f} pp)"
        print(f"  {key:<22} baseline={_pct(block['baseline'])}  "
              f"adaptive={_pct(block['adaptive'])}{arrow}")
    print(f"  {'recovery_batches':<22} baseline={result['recovery_batches']['baseline']}  "
          f"adaptive={result['recovery_batches']['adaptive']}")
    print(f"  {'retrains':<22} baseline={result['retrains']['baseline']}  "
          f"adaptive={result['retrains']['adaptive']}")
    return 0


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamcls")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="messages per batch (default: from StreamConfig)")
    sub = parser.add_subparsers(dest="command", required=True)

    runs = sub.add_parser("runs", help="list experiment runs")
    runs.set_defaults(func=cmd_runs)

    score = sub.add_parser("score", help="score runs and print the results table")
    score.add_argument("runs", nargs="*", type=int, help="run ids (default: most recent)")
    score.add_argument("--limit", type=int, default=10)
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=cmd_score)

    cmp_parser = sub.add_parser("compare", help="A/B two seed-matched runs")
    cmp_parser.add_argument("baseline", type=int)
    cmp_parser.add_argument("adaptive", type=int)
    cmp_parser.add_argument("--json", action="store_true")
    cmp_parser.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    with Repository(get_settings().postgres) as repo:
        return int(args.func(args, repo))


if __name__ == "__main__":
    sys.exit(main())
