"""Query layer. Views stay thin; nothing here knows about HTTP.

The previous view layer had no service layer at all -- a single free function
was the entire abstraction, and each of the twelve views did its own
``os.path.join``, its own ``open()``, its own ``json.load`` and its own ad-hoc
error mapping. Four of them full-scanned an append-only log on a five-second
polling path; ``latest_messages_json`` parsed the whole corpus into memory to
return ten rows.

Every query here is bounded and hits an index declared on the model.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Avg, Count, F, Q, Sum

from dashboard.models import (
    BatchMetric,
    Category,
    ClassifiedMessage,
    DriftEvent,
    ExperimentRun,
    ModelVersion,
    StreamConfig,
    StreamStatus,
    TrainingJob,
)

#: Hard ceiling on any list endpoint, whatever the caller asks for.
MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 50


def clamp_limit(raw: Any, default: int = DEFAULT_PAGE_SIZE) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, MAX_PAGE_SIZE))


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------
def stream_overview() -> dict[str, Any]:
    """Everything the header needs, in a handful of indexed queries.

    Replaces a dashboard that issued four separate polls every five seconds,
    two of which re-read the entire message history from disk.
    """
    status = StreamStatus.load()
    config = StreamConfig.load()
    run = status.active_experiment_run
    active_model = status.active_model_version

    latest = None
    totals: dict[str, Any] = {}
    if run is not None:
        latest = (
            BatchMetric.objects.filter(experiment_run=run)
            .order_by("-batch_id")
            .select_related("model_version")
            .first()
        )
        totals = BatchMetric.objects.filter(experiment_run=run).aggregate(
            batches=Count("id"),
            received=Sum("n_received"),
            classified=Sum("n_classified"),
            correct=Sum("n_correct"),
            mean_throughput=Avg("throughput_msgs_s"),
        )

    classified = totals.get("classified") or 0
    correct = totals.get("correct") or 0
    return {
        "is_active": status.is_active,
        "producer_active": status.producer_active,
        "consumer_heartbeat_at": status.consumer_heartbeat_at,
        "producer_heartbeat_at": status.producer_heartbeat_at,
        "last_batch_id": status.last_batch_id,
        "last_error": status.last_error,
        "run": _run_summary(run) if run else None,
        "active_model": _model_summary(active_model) if active_model else None,
        "desired_model": (
            _model_summary(status.desired_model_version)
            if status.desired_model_version_id
            else None
        ),
        "config": {
            "trigger_seconds": config.trigger_seconds,
            "max_offsets_per_trigger": config.max_offsets_per_trigger,
            "adaptation_enabled": config.adaptation_enabled,
            "auto_promote": config.auto_promote,
        },
        "totals": {
            "batches": totals.get("batches") or 0,
            # Counted separately, unlike the old metrics where both numbers were
            # incremented by the same value so drops were unobservable.
            "received": totals.get("received") or 0,
            "classified": classified,
            "dropped": (totals.get("received") or 0) - classified,
            "accuracy": (correct / classified) if classified else None,
            "mean_throughput": totals.get("mean_throughput"),
        },
        "latest_batch": _batch_summary(latest) if latest else None,
    }


def _run_summary(run: ExperimentRun) -> dict[str, Any]:
    return {
        "id": run.pk,
        "name": run.name,
        "scenario": run.scenario,
        "seed": run.seed,
        "mode": run.mode,
        "status": run.status,
        "adaptation_enabled": run.adaptation_enabled,
        "total_messages": run.total_messages,
        "change_points": [int(e["at"]) for e in (run.drift_schedule or []) if "at" in e],
        "started_at": run.started_at,
        "ended_at": run.ended_at,
    }


def _model_summary(model: ModelVersion) -> dict[str, Any]:
    return {
        "id": model.pk,
        "version": model.version,
        "status": model.status,
        "algorithm": model.algorithm,
        "test_accuracy": model.test_accuracy,
        "recent_accuracy": model.recent_accuracy,
        "trained_at": model.trained_at,
    }


def _batch_summary(batch: BatchMetric) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "accuracy": batch.accuracy,
        "n_received": batch.n_received,
        "n_classified": batch.n_classified,
        "mean_confidence": batch.mean_confidence,
        "psi_vocabulary": batch.psi_vocabulary,
        "psi_confidence": batch.psi_confidence,
        "psi_predicted": batch.psi_predicted,
        "duration_ms": batch.duration_ms,
        "model_version": batch.model_version.version,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def batch_series(run_id: int, *, limit: int = MAX_PAGE_SIZE) -> dict[str, Any]:
    """Per-batch accuracy and drift statistics, plus the annotations.

    The model-version band and the injected change points are what make the
    chart readable: without them a reader cannot tell an adaptation from noise.
    """
    rows = list(
        BatchMetric.objects.filter(experiment_run_id=run_id)
        .order_by("-batch_id")
        .select_related("model_version")[:limit]
    )
    rows.reverse()

    events = list(
        DriftEvent.objects.filter(experiment_run_id=run_id, suppressed=False)
        .order_by("batch_id")
        .values("batch_id", "detector", "action_taken", "statistic")
    )
    run = ExperimentRun.objects.filter(pk=run_id).first()
    per_batch = StreamConfig.load().max_offsets_per_trigger
    change_points = (
        [int(e["at"]) // per_batch for e in (run.drift_schedule or []) if "at" in e]
        if run
        else []
    )

    return {
        "batches": [b.batch_id for b in rows],
        "accuracy": [b.accuracy for b in rows],
        "cumulative_accuracy": _cumulative(rows),
        "mean_confidence": [b.mean_confidence for b in rows],
        "psi_vocabulary": [b.psi_vocabulary for b in rows],
        "psi_confidence": [b.psi_confidence for b in rows],
        "psi_predicted": [b.psi_predicted for b in rows],
        "adwin_width": [b.adwin_width for b in rows],
        "throughput": [b.throughput_msgs_s for b in rows],
        "model_versions": [b.model_version.version for b in rows],
        "drift_events": events,
        "change_point_batches": change_points,
    }


def _cumulative(rows: list[BatchMetric]) -> list[float | None]:
    correct = total = 0
    out: list[float | None] = []
    for row in rows:
        correct += row.n_correct
        total += row.n_classified
        out.append(correct / total if total else None)
    return out


def category_distribution(run_id: int) -> list[dict[str, Any]]:
    """Predicted-vs-true counts per class.

    One indexed GROUP BY. The previous version counted the whole message log in
    Python on every five-second poll.
    """
    categories = {c.pk: c for c in Category.objects.all()}
    predicted = dict(
        ClassifiedMessage.objects.filter(experiment_run_id=run_id)
        .values_list("predicted_label")
        .annotate(n=Count("id"))
    )
    actual = dict(
        ClassifiedMessage.objects.filter(experiment_run_id=run_id)
        .values_list("true_label")
        .annotate(n=Count("id"))
    )
    correct = dict(
        ClassifiedMessage.objects.filter(experiment_run_id=run_id, is_correct=True)
        .values_list("true_label")
        .annotate(n=Count("id"))
    )
    return [
        {
            "id": pk,
            "key": category.key,
            "label": category.display_name,
            "color": category.color,
            "predicted": predicted.get(pk, 0),
            "actual": actual.get(pk, 0),
            "correct": correct.get(pk, 0),
            "recall": (correct.get(pk, 0) / actual[pk]) if actual.get(pk) else None,
        }
        for pk, category in sorted(categories.items())
    ]


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------
def search_messages(
    *,
    run_id: int | None = None,
    query: str = "",
    category_id: int | None = None,
    only_errors: bool = False,
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated, indexed message search with a real total count.

    The old endpoint scanned the whole corpus in Python, hard-truncated to 100
    rows and reported neither the total nor that truncation had happened.
    """
    queryset = ClassifiedMessage.objects.select_related(
        "true_label", "predicted_label", "model_version"
    )
    if run_id is not None:
        queryset = queryset.filter(experiment_run_id=run_id)
    if category_id is not None:
        queryset = queryset.filter(
            Q(predicted_label_id=category_id) | Q(true_label_id=category_id)
        )
    if only_errors:
        queryset = queryset.filter(is_correct=False)
    if query:
        queryset = queryset.filter(post_text__icontains=query)

    total = queryset.count()
    rows = queryset.order_by("-msg_index")[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": [
            {
                "post_id": m.post_id,
                "msg_index": m.msg_index,
                "text": m.post_text,
                "channel_title": m.channel_title,
                "true_label": m.true_label.display_name,
                "predicted_label": m.predicted_label.display_name,
                "predicted_color": m.predicted_label.color,
                "is_correct": m.is_correct,
                "confidence": m.confidence,
                "model_version": m.model_version.version,
                "perturbation_kind": m.perturbation_kind,
                "produced_at": m.produced_at,
            }
            for m in rows
        ],
    }


def drift_events(run_id: int | None = None, *, limit: int = DEFAULT_PAGE_SIZE) -> list[dict]:
    queryset = DriftEvent.objects.select_related("training_job")
    if run_id is not None:
        queryset = queryset.filter(experiment_run_id=run_id)
    return [
        {
            "id": e.pk,
            "batch_id": e.batch_id,
            "detector": e.detector,
            "statistic": e.statistic,
            "threshold": e.threshold,
            "accuracy_before": e.accuracy_before,
            "suppressed": e.suppressed,
            "ground_truth_drift_index": e.ground_truth_drift_index,
            "detection_delay_batches": e.detection_delay_batches,
            "action_taken": e.action_taken,
            "training_job_id": e.training_job_id,
            "detected_at": e.detected_at,
        }
        for e in queryset.order_by("-batch_id", "-id")[:limit]
    ]


def model_registry() -> list[dict[str, Any]]:
    return [
        {
            **_model_summary(m),
            "train_rows": m.train_rows,
            "vocab_size": m.vocab_size,
            "test_macro_f1": m.test_macro_f1,
            "activated_at": m.activated_at,
            "created_by_run_id": m.created_by_run_id,
            "metrics": m.metrics,
        }
        for m in ModelVersion.objects.order_by("-trained_at")[:MAX_PAGE_SIZE]
    ]


def training_jobs(run_id: int | None = None, *, limit: int = DEFAULT_PAGE_SIZE) -> list[dict]:
    queryset = TrainingJob.objects.select_related("model_version", "trigger_drift_event")
    if run_id is not None:
        queryset = queryset.filter(experiment_run_id=run_id)
    return [
        {
            "id": j.pk,
            "status": j.status,
            "params": j.params,
            "model_version": j.model_version.version if j.model_version_id else None,
            "trigger_batch": (
                j.trigger_drift_event.batch_id if j.trigger_drift_event_id else None
            ),
            "error": j.error[:500],
            "requested_at": j.requested_at,
            "started_at": j.started_at,
            "finished_at": j.finished_at,
        }
        for j in queryset.order_by("-id")[:limit]
    ]


def experiment_runs(*, limit: int = DEFAULT_PAGE_SIZE) -> list[dict[str, Any]]:
    """Run list with the A/B comparison figures attached."""
    runs = ExperimentRun.objects.order_by("-created_at")[:limit]
    summaries = []
    for run in runs:
        totals = BatchMetric.objects.filter(experiment_run=run).aggregate(
            batches=Count("id"), classified=Sum("n_classified"), correct=Sum("n_correct")
        )
        classified = totals["classified"] or 0
        change_points = [int(e["at"]) for e in (run.drift_schedule or []) if "at" in e]
        summaries.append(
            {
                **_run_summary(run),
                "batches": totals["batches"] or 0,
                "accuracy": (totals["correct"] / classified) if classified else None,
                "accuracy_after_drift": _accuracy_after(run, change_points),
                "drift_events": DriftEvent.objects.filter(
                    experiment_run=run, suppressed=False
                ).count(),
                "retrains": TrainingJob.objects.filter(experiment_run=run).count(),
            }
        )
    return summaries


def _accuracy_after(run: ExperimentRun, change_points: list[int]) -> float | None:
    """Accuracy over messages produced after the first injected change point.

    This is the number the adaptation A/B actually compares: overall accuracy
    is dominated by the stable prefix both arms share.
    """
    if not change_points:
        return None
    aggregate = ClassifiedMessage.objects.filter(
        experiment_run=run, msg_index__gte=min(change_points)
    ).aggregate(n=Count("id"), correct=Count("id", filter=Q(is_correct=True)))
    return (aggregate["correct"] / aggregate["n"]) if aggregate["n"] else None


def confusion_for_run(run_id: int) -> dict[str, Any]:
    """Confusion matrix computed in the database, not in Python."""
    counts = (
        ClassifiedMessage.objects.filter(experiment_run_id=run_id)
        .values("true_label_id", "predicted_label_id")
        .annotate(n=Count("id"))
    )
    categories = list(Category.objects.order_by("id"))
    index = {c.pk: i for i, c in enumerate(categories)}
    matrix = [[0] * len(categories) for _ in categories]
    for row in counts:
        true_i = index.get(row["true_label_id"])
        pred_i = index.get(row["predicted_label_id"])
        if true_i is not None and pred_i is not None:
            matrix[true_i][pred_i] = row["n"]
    return {"labels": [c.display_name for c in categories], "matrix": matrix}


def toggle_stream() -> bool:
    """Flip the control flag atomically, and report the new value."""
    StreamStatus.objects.filter(pk=1).update(is_active=~F("is_active"))
    return StreamStatus.load().is_active
