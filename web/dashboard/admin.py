"""Admin for the control plane and the experiment registry.

This is the operator surface until the dashboard is rebuilt. The stream output
tables (``ClassifiedMessage``, ``BatchMetric``) are registered read-only: they
are written by Spark, and hand-editing them would corrupt the results a run is
supposed to produce.
"""

from __future__ import annotations

from django.contrib import admin, messages
from django.db.models import QuerySet
from django.http import HttpRequest
from django.utils import timezone

from dashboard.models import (
    BatchMetric,
    Category,
    ClassifiedMessage,
    DriftEvent,
    ExperimentRun,
    ModelStatus,
    ModelVersion,
    RunStatus,
    StreamConfig,
    StreamStatus,
    TrainingJob,
)


class ReadOnlyAdmin(admin.ModelAdmin):
    """Rows produced by the pipeline. Viewable, never editable."""

    def has_add_permission(self, request: HttpRequest, obj=None) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj=None) -> bool:
        return False


class SingletonAdmin(admin.ModelAdmin):
    """Exactly one row, which always exists and cannot be removed."""

    def has_add_permission(self, request: HttpRequest, obj=None) -> bool:
        return not self.model.objects.exists()

    def has_delete_permission(self, request: HttpRequest, obj=None) -> bool:
        return False

    def changelist_view(self, request: HttpRequest, extra_context=None):
        self.model.load()  # make sure the row is there before rendering
        return super().changelist_view(request, extra_context)


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("id", "key", "display_name", "color")
    # id and key come from streamcls.labels and must match what the model
    # predicts; only presentation is editable here.
    readonly_fields = ("id", "key")
    list_editable = ("color",)


@admin.register(StreamConfig)
class StreamConfigAdmin(SingletonAdmin):
    fieldsets = (
        ("Ingestion", {"fields": ("max_message_length", "max_offsets_per_trigger", "trigger_seconds")}),
        (
            "Drift detection",
            {"fields": ("adwin_delta", "psi_threshold", "psi_consecutive_batches", "grace_batches")},
        ),
        (
            "Adaptation",
            {
                "fields": (
                    "adaptation_enabled",
                    "auto_promote",
                    "promote_min_delta",
                    "min_batches_between_retrains",
                    "max_retrains_per_run",
                    "retrain_window_rows",
                    "retrain_base_rows",
                    "retrain_min_new_rows",
                    "actionable_detectors",
                    "label_delay_batches",
                )
            },
        ),
    )
    readonly_fields = ("updated_at",)


@admin.register(StreamStatus)
class StreamStatusAdmin(SingletonAdmin):
    list_display = ("__str__", "producer_active", "active_experiment_run", "active_model_version")
    readonly_fields = (
        "consumer_heartbeat_at",
        "producer_heartbeat_at",
        "last_batch_id",
        "consumer_query_run_id",
        "last_error",
        "updated_at",
    )


@admin.register(ExperimentRun)
class ExperimentRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "scenario",
        "seed",
        "mode",
        "adaptation_enabled",
        "status",
        "total_messages",
    )
    list_filter = ("status", "scenario", "mode", "adaptation_enabled")
    search_fields = ("name", "notes", "scenario")
    readonly_fields = ("created_at", "started_at", "ended_at", "git_commit")
    actions = ("clone_with_adaptation_disabled",)

    @admin.action(description="Clone as an A/B counterpart (adaptation disabled)")
    def clone_with_adaptation_disabled(self, request: HttpRequest, queryset: QuerySet) -> None:
        """The headline comparison: identical seed and schedule, adaptation off.

        Same seed means a byte-identical stream, so any difference in accuracy
        is attributable to adaptation alone.
        """
        made = 0
        for run in queryset:
            if not run.adaptation_enabled:
                continue
            ExperimentRun.objects.create(
                name=f"{run.name} (no adaptation)",
                notes=f"A/B counterpart of run #{run.pk}. Same seed and drift schedule.",
                scenario=run.scenario,
                seed=run.seed,
                mode=run.mode,
                status=RunStatus.DRAFT,
                adaptation_enabled=False,
                total_messages=run.total_messages,
                rate_msgs_per_sec=run.rate_msgs_per_sec,
                drift_schedule=run.drift_schedule,
                dataset_split=run.dataset_split,
                base_model_version=run.base_model_version,
                producer_config=run.producer_config,
                consumer_config=run.consumer_config,
            )
            made += 1
        self.message_user(request, f"Created {made} counterpart run(s).", messages.SUCCESS)


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("version", "status", "algorithm", "test_accuracy", "test_macro_f1", "trained_at")
    list_filter = ("status", "algorithm")
    search_fields = ("version",)
    readonly_fields = (
        "trained_at",
        "activated_at",
        "metrics",
        "reference_histogram",
        "top_term_indices",
        "vocab_size",
        "train_rows",
        "git_commit",
    )
    actions = ("promote",)

    @admin.action(description="Promote to active (manual override)")
    def promote(self, request: HttpRequest, queryset: QuerySet) -> None:
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one version.", messages.ERROR)
            return
        version = queryset.first()
        if version.status not in {ModelStatus.READY, ModelStatus.RETIRED}:
            self.message_user(
                request, f"Cannot promote a version with status {version.status}.", messages.ERROR
            )
            return

        # Retire first: `only_one_active_model_version` is a real constraint.
        ModelVersion.objects.filter(status=ModelStatus.ACTIVE).update(status=ModelStatus.RETIRED)
        version.status = ModelStatus.ACTIVE
        version.activated_at = timezone.now()
        version.save()

        # The consumer picks this up on its next control poll and swaps.
        status = StreamStatus.load()
        status.desired_model_version = version
        status.save()
        self.message_user(request, f"{version.version} promoted; consumer will swap.", messages.SUCCESS)


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "experiment_run", "model_version", "requested_at", "finished_at")
    list_filter = ("status",)
    readonly_fields = ("requested_at", "started_at", "finished_at", "error", "params")


@admin.register(DriftEvent)
class DriftEventAdmin(ReadOnlyAdmin):
    list_display = (
        "id",
        "experiment_run",
        "batch_id",
        "detector",
        "statistic",
        "suppressed",
        "detection_delay_batches",
        "action_taken",
    )
    list_filter = ("detector", "suppressed", "action_taken", "experiment_run")


@admin.register(BatchMetric)
class BatchMetricAdmin(ReadOnlyAdmin):
    list_display = (
        "experiment_run",
        "batch_id",
        "n_received",
        "n_classified",
        "accuracy",
        "mean_confidence",
        "psi_vocabulary",
        "duration_ms",
    )
    list_filter = ("experiment_run", "model_version")


@admin.register(ClassifiedMessage)
class ClassifiedMessageAdmin(ReadOnlyAdmin):
    list_display = (
        "post_id",
        "experiment_run",
        "batch_id",
        "true_label",
        "predicted_label",
        "is_correct",
        "confidence",
    )
    list_filter = ("experiment_run", "is_correct", "predicted_label", "perturbation_kind")
    search_fields = ("post_id", "post_text")


admin.site.site_header = "Streaming classification · control"
admin.site.site_title = "streamcls"
admin.site.index_title = "Pipeline control and experiment registry"
