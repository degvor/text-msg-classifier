"""Read-only queries used by the experiment scorer.

Kept apart from :mod:`streamcls.db.repo`, which is the services' read/write
surface. Scoring only ever reads, and it runs long after the stream is done.
"""

from __future__ import annotations

from typing import Any

from streamcls.db.tables import Table

__all__ = ["fetch_run_bundle", "list_runs"]


def list_runs(repo, *, limit: int = 50) -> list[dict[str, Any]]:
    with repo.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, seed, adaptation_enabled, drift_schedule, status, total_messages
            FROM {Table.EXPERIMENT_RUN} ORDER BY id DESC LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_run_bundle(repo, run_id: int) -> dict[str, Any]:
    """Everything the scorer needs for one run, in three queries."""
    with repo.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, seed, adaptation_enabled, drift_schedule, status, total_messages,
                   scenario, consumer_config, base_model_version_id
            FROM {Table.EXPERIMENT_RUN} WHERE id = %s
            """,
            (run_id,),
        )
        run = cur.fetchone()
        if run is None:
            raise LookupError(f"experiment run {run_id} not found")

        cur.execute(
            f"""
            SELECT batch_id, n_received, n_classified, n_correct, accuracy,
                   mean_confidence, psi_vocabulary, psi_confidence, psi_predicted,
                   adwin_estimate, adwin_width, model_version_id
            FROM {Table.BATCH_METRIC} WHERE experiment_run_id = %s ORDER BY batch_id
            """,
            (run_id,),
        )
        batches = [dict(r) for r in cur.fetchall()]

        cur.execute(
            f"""
            SELECT batch_id, detector, statistic, suppressed, action_taken,
                   detection_delay_batches, ground_truth_drift_index
            FROM {Table.DRIFT_EVENT} WHERE experiment_run_id = %s ORDER BY batch_id, id
            """,
            (run_id,),
        )
        events = [dict(r) for r in cur.fetchall()]

        # A retrain "promoted" if the model it produced was ever activated:
        # register_model leaves activated_at NULL and only promote_model sets
        # it, after the guard passes.
        cur.execute(
            f"""
            SELECT j.id, j.status, j.model_version_id, j.requested_at, j.finished_at,
                   (m.activated_at IS NOT NULL) AS promoted
            FROM {Table.TRAINING_JOB} j
            LEFT JOIN {Table.MODEL_VERSION} m ON m.id = j.model_version_id
            WHERE j.experiment_run_id = %s ORDER BY j.id
            """,
            (run_id,),
        )
        jobs = [dict(r) for r in cur.fetchall()]

    return {"run": dict(run), "batches": batches, "events": events, "jobs": jobs}
