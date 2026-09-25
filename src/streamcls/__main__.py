"""CLI: ``python -m streamcls <command>``.

``score`` turns finished runs into the results table the thesis reports.
``compare`` prints the A/B figure for a seed-matched pair.
``matrix`` runs a declared experiment matrix unattended; ``report``
aggregates its ledger into the thesis tables.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import subprocess
import sys
from pathlib import Path

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


# --- matrix -----------------------------------------------------------------


def cmd_matrix(args) -> int:
    from streamcls.matrix import (
        MatrixRunner,
        RunnerError,
        SpecError,
        completed_keys,
        estimate_seconds,
        expand,
        load_spec,
        select_cells,
    )

    try:
        spec = load_spec(args.spec)
        if args.timeout is not None:
            spec = dataclasses.replace(spec, timeout_minutes=args.timeout)
        if args.base_model is not None:
            spec = dataclasses.replace(spec, base_model=args.base_model)
        cells = select_cells(expand(spec), args.only)
    except SpecError as error:
        print(f"{args.spec}: {error}", file=sys.stderr)
        return 2
    if not cells:
        print(f"{args.spec}: no cell matches {args.only}", file=sys.stderr)
        return 2

    ledger = Path(args.results) / spec.name / "runs.jsonl"
    done = completed_keys(ledger)
    pending = [c for c in cells if c.key not in done]

    print(f"matrix {spec.name}: {len(cells)} cell(s), {len(pending)} pending, ledger {ledger}")
    print(f"{'#':>3}  {'cell':<52} {'msgs':>6} {'batches':>7} {'est':>6}  status")
    for index, cell in enumerate(cells, start=1):
        state = "done" if cell.key in done else "pending"
        print(f"{index:>3}  {cell.key:<52} {cell.total_messages:>6} {cell.batches:>7} "
              f"{_hms(estimate_seconds(cell)):>6}  {state}")
    total = sum(estimate_seconds(c) for c in pending)
    print(f"\nestimated wall clock for the pending cells: ~{_hms(total)} "
          f"(trigger {spec.config['trigger_seconds']} s, "
          f"batch {spec.config['max_offsets_per_trigger']} msgs)")
    if args.dry_run:
        return 0

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # kafka-python logs every connection state change at INFO; one topic
    # creation per cell is nine lines of it.
    logging.getLogger("kafka").setLevel(logging.WARNING)
    settings = get_settings()
    if not settings.paths.data_root.is_dir():
        print(
            f"cannot start: STREAMCLS_DATA_ROOT={settings.paths.data_root} is not a directory "
            "on this host, so leftover checkpoints could not be seen; `make matrix` points it "
            "at ./data",
            file=sys.stderr,
        )
        return 2
    from streamcls.kafka_admin import ensure_topic

    commit, dirty = _git_state(Path(args.results))
    if dirty:
        logging.getLogger("streamcls.matrix").warning(
            "working tree is dirty; the ledger records commit %s with git_dirty=true", commit[:12]
        )
    with Repository(settings.postgres) as repo:
        runner = MatrixRunner(
            repo,
            topic_for_run=settings.kafka.topic_for_run,
            ensure_topic=functools.partial(
                ensure_topic, settings.kafka.bootstrap_servers,
                partitions=settings.kafka.num_partitions,
            ),
            checkpoint_for_run=settings.paths.checkpoint_for_run,
            score=functools.partial(_score, repo),
            git_commit=commit,
            git_dirty=dirty,
            poll_s=args.poll,
        )
        try:
            return runner.run(spec, cells, ledger=ledger, stop_on_timeout=args.stop_on_timeout)
        except (RunnerError, SpecError, LookupError) as error:
            print(f"cannot start: {error}", file=sys.stderr)
            return 2


def cmd_report(args) -> int:
    from streamcls.experiments import (
        Result,
        aggregate,
        detector_table,
        pair,
        render_markdown,
        write_csv,
        write_detectors_csv,
        write_pairs_csv,
    )
    from streamcls.matrix import read_ledger

    results_dir = Path(args.results_dir)
    ledger = results_dir / "runs.jsonl"
    rows = read_ledger(ledger)
    if not rows:
        print(f"{ledger}: no results", file=sys.stderr)
        return 1

    # The last row per cell is the one that counts: an earlier timed-out
    # attempt is superseded by the completed re-run appended after it.
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row.get("key", "")] = row
    completed = [r for r in latest.values() if r.get("status") == "completed" and r.get("score")]
    unfinished = sorted(k for k, r in latest.items() if r.get("status") != "completed")
    if not completed:
        print(f"{ledger}: no completed cells", file=sys.stderr)
        return 1

    results = [Result.from_row(r) for r in completed]
    name = str(completed[0].get("matrix") or results_dir.name)
    summaries = aggregate(results)
    pairs = pair(results)
    detectors = detector_table(results)

    write_csv(results_dir / "summary.csv", summaries)
    write_pairs_csv(results_dir / "pairs.csv", pairs)
    write_detectors_csv(results_dir / "detectors.csv", detectors)
    markdown = render_markdown(summaries, pairs, detectors, name=name)
    if unfinished:
        markdown += (
            f"\n\n_{len(unfinished)} cell(s) not completed: "
            + ", ".join(f"`{k}`" for k in unfinished) + "_\n"
        )
    markdown = markdown.rstrip("\n") + f"\n\n{_provenance(completed)}\n"
    (results_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nwrote summary.md, summary.csv, pairs.csv, detectors.csv to {results_dir}/",
          file=sys.stderr)
    return 0


def _provenance(rows: list[dict]) -> str:
    """Which code produced these cells, as one line the thesis can cite."""
    states = {True: "dirty working tree", False: "clean"}
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = ((row.get("git_commit") or "unknown")[:8],
               states.get(row.get("git_dirty"), "tree state not recorded"))
        counts[key] = counts.get(key, 0) + 1
    return "_Code: " + ", ".join(
        f"`{commit}` {state} ({n} cell{'' if n == 1 else 's'})"
        for (commit, state), n in sorted(counts.items())
    ) + "._"


def _git_state(results: Path) -> tuple[str, bool]:
    """``(HEAD, dirty)`` for the ledger; empty and clean when git is unavailable.

    ``results`` is left out of the dirty check: it is the matrix's own
    output, and a resumed matrix always finds its ledger there.
    """
    try:
        run = functools.partial(
            subprocess.run, capture_output=True, text=True, timeout=5, check=True
        )
        commit = run(["git", "rev-parse", "HEAD"]).stdout.strip()
        top = Path(run(["git", "rev-parse", "--show-toplevel"]).stdout.strip()).resolve()
        pathspec = [":/"]
        try:
            inside = results.resolve().relative_to(top).as_posix()
        except ValueError:  # outside the repository: nothing to leave out
            inside = "."
        if inside != ".":
            pathspec.append(f":(top,exclude){inside}")
        dirty = bool(run(["git", "status", "--porcelain", "--", *pathspec]).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return "", False
    return commit, dirty


def _hms(seconds: float) -> str:
    minutes, sec = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes}:{sec:02d}"


# --- entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streamcls")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="messages per batch (default: from StreamConfig)")
    sub = parser.add_subparsers(dest="command", required=True)

    runs = sub.add_parser("runs", help="list experiment runs")
    runs.set_defaults(func=cmd_runs, needs_db=True)

    score = sub.add_parser("score", help="score runs and print the results table")
    score.add_argument("runs", nargs="*", type=int, help="run ids (default: most recent)")
    score.add_argument("--limit", type=int, default=10)
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=cmd_score, needs_db=True)

    cmp_parser = sub.add_parser("compare", help="A/B two seed-matched runs")
    cmp_parser.add_argument("baseline", type=int)
    cmp_parser.add_argument("adaptive", type=int)
    cmp_parser.add_argument("--json", action="store_true")
    cmp_parser.set_defaults(func=cmd_compare, needs_db=True)

    matrix = sub.add_parser(
        "matrix",
        help="run an experiment matrix unattended",
        description="Runs every cell of a TOML matrix spec against the live stack, one at a "
                    "time, appending one line per cell to results/<name>/runs.jsonl. "
                    "Re-running skips cells already completed there.",
    )
    matrix.add_argument("spec", help="path to the matrix spec (TOML)")
    matrix.add_argument("--dry-run", action="store_true",
                        help="print the cell table and estimate; touch nothing")
    matrix.add_argument("--only", metavar="PATTERN", action="append", default=[],
                        help="run only cells whose key matches (shell glob; repeatable)")
    matrix.add_argument("--results", default="results", metavar="DIR",
                        help="ledger root (default: results/)")
    matrix.add_argument("--stop-on-timeout", action="store_true",
                        help="end the matrix at the first cell that times out")
    matrix.add_argument("--base-model", type=int, default=None, metavar="ID",
                        help="model version every cell starts from (default: from the spec)")
    matrix.add_argument("--timeout", type=float, default=None, metavar="MIN",
                        help="per-cell timeout in minutes (default: from the spec)")
    matrix.add_argument("--poll", type=float, default=5.0, metavar="SEC",
                        help="control-plane poll interval (default: 5)")
    matrix.set_defaults(func=cmd_matrix, needs_db=False)

    report = sub.add_parser(
        "report",
        help="aggregate a matrix ledger into summary tables",
        description="Reads results/<name>/runs.jsonl and writes summary.md, summary.csv, "
                    "pairs.csv and detectors.csv next to it. No database access.",
    )
    report.add_argument("results_dir", help="the matrix's results directory")
    report.set_defaults(func=cmd_report, needs_db=False)

    args = parser.parse_args(argv)
    if not args.needs_db:
        return int(args.func(args))
    with Repository(get_settings().postgres) as repo:
        return int(args.func(args, repo))


if __name__ == "__main__":
    sys.exit(main())
