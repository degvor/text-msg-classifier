"""Validated input for every state-changing endpoint.

The previous version had no forms, no serializers and no ``clean()`` anywhere:
both write endpoints hand-parsed ``json.loads(request.body)`` and coerced
directly, so ``throughput_limit=0`` was storable and a non-numeric value became
a 500 with the exception text -- including absolute server paths -- returned to
the caller.
"""

from __future__ import annotations

from django import forms

from dashboard.models import Category, ExperimentRun, StreamConfig


class StreamConfigForm(forms.ModelForm):
    class Meta:
        model = StreamConfig
        fields = [
            "max_message_length",
            "max_offsets_per_trigger",
            "trigger_seconds",
            "adwin_delta",
            "psi_threshold",
            "psi_consecutive_batches",
            "grace_batches",
            "adaptation_enabled",
            "auto_promote",
            "promote_min_delta",
            "min_batches_between_retrains",
            "max_retrains_per_run",
            "retrain_window_rows",
            "retrain_base_rows",
            "retrain_min_new_rows",
        ]

    def clean_adwin_delta(self) -> float:
        value = self.cleaned_data["adwin_delta"]
        if not 0.0 < value < 1.0:
            raise forms.ValidationError("delta must be strictly between 0 and 1.")
        return value


class CategoryForm(forms.ModelForm):
    """Colour is validated because it is interpolated into the page's CSS.

    The old endpoint stored it verbatim from an unauthenticated request and the
    template dropped it straight into a ``style=`` attribute.
    """

    class Meta:
        model = Category
        fields = ["display_name", "description", "color"]


class ExperimentRunForm(forms.ModelForm):
    class Meta:
        model = ExperimentRun
        fields = [
            "name",
            "notes",
            "seed",
            "mode",
            "adaptation_enabled",
            "total_messages",
            "rate_msgs_per_sec",
            "drift_schedule",
            "dataset_split",
        ]

    def clean_drift_schedule(self):
        """Reject a malformed schedule here rather than at stream time.

        An invalid schedule that reaches the producer aborts the run after the
        containers have already started.
        """
        from streamcls.corpus.schedule import DriftSchedule

        events = self.cleaned_data["drift_schedule"] or []
        if not isinstance(events, list):
            raise forms.ValidationError("Schedule must be a list of drift events.")
        try:
            DriftSchedule(events=list(events))
        except (ValueError, KeyError, TypeError) as error:
            raise forms.ValidationError(f"Invalid drift schedule: {error}") from error
        return events
