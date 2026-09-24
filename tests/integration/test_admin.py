"""The admin is the operator surface until the dashboard is rebuilt.

Django validates most admin configuration lazily, so a wrong field name in
``list_display`` only blows up when someone opens the page. These tests open
every page.
"""

from __future__ import annotations

import pytest
from django.contrib import admin
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

from dashboard.models import (
    ExperimentRun,
    ModelStatus,
    ModelVersion,
    RunStatus,
    StreamConfig,
    StreamStatus,
)

pytestmark = [pytest.mark.integration, pytest.mark.django_db]


@pytest.fixture
def staff_client(client):
    user = User.objects.create_superuser("op", "op@example.com", "pw-for-tests-only")
    client.force_login(user)
    return client


@pytest.mark.parametrize(
    "model", list(admin.site._registry), ids=lambda m: m.__name__
)
def test_every_changelist_renders(staff_client, model):
    url = reverse(f"admin:{model._meta.app_label}_{model._meta.model_name}_changelist")
    assert staff_client.get(url).status_code == 200


def test_singleton_changelists_create_their_row_on_first_view(staff_client):
    StreamStatus.objects.all().delete()
    url = reverse("admin:dashboard_streamstatus_changelist")
    assert staff_client.get(url).status_code == 200
    assert StreamStatus.objects.count() == 1


def test_pipeline_output_is_read_only(staff_client):
    """Hand-editing Spark's output would corrupt the results of a run."""
    for name in ("classifiedmessage", "batchmetric", "driftevent"):
        url = reverse(f"admin:dashboard_{name}_add")
        assert staff_client.get(url).status_code == 403


def test_promote_action_retires_the_incumbent_and_signals_the_consumer(staff_client):
    old = ModelVersion.objects.create(
        version="v1", artifact_path="/p/v1", status=ModelStatus.ACTIVE, test_accuracy=0.90
    )
    new = ModelVersion.objects.create(
        version="v2", artifact_path="/p/v2", status=ModelStatus.READY, test_accuracy=0.93
    )

    url = reverse("admin:dashboard_modelversion_changelist")
    staff_client.post(url, {"action": "promote", "_selected_action": [new.pk]}, follow=True)

    old.refresh_from_db()
    new.refresh_from_db()
    assert old.status == ModelStatus.RETIRED
    assert new.status == ModelStatus.ACTIVE
    assert new.activated_at is not None
    # This is the entire hot-reload signal: the consumer's control poll sees
    # desired != active and swaps.
    assert StreamStatus.load().desired_model_version_id == new.pk


def test_promote_rejects_a_version_that_is_still_training(staff_client):
    training = ModelVersion.objects.create(
        version="v9", artifact_path="/p/v9", status=ModelStatus.TRAINING
    )
    url = reverse("admin:dashboard_modelversion_changelist")
    staff_client.post(url, {"action": "promote", "_selected_action": [training.pk]}, follow=True)
    training.refresh_from_db()
    assert training.status == ModelStatus.TRAINING


def test_ab_clone_copies_the_seed_and_schedule_but_flips_adaptation(staff_client):
    """Same seed means a byte-identical stream, so any accuracy difference is
    attributable to adaptation alone. That is the headline figure."""
    run = ExperimentRun.objects.create(
        name="drift A",
        scenario="concept_abrupt",
        seed=1234,
        adaptation_enabled=True,
        drift_schedule=[{"at": 4000, "kind": "concept_abrupt"}],
        total_messages=8000,
        started_at=timezone.now(),
        status=RunStatus.COMPLETED,
    )
    url = reverse("admin:dashboard_experimentrun_changelist")
    staff_client.post(
        url,
        {"action": "clone_with_adaptation_disabled", "_selected_action": [run.pk]},
        follow=True,
    )

    clone = ExperimentRun.objects.exclude(pk=run.pk).get()
    assert clone.seed == run.seed
    assert clone.scenario == run.scenario
    assert clone.drift_schedule == run.drift_schedule
    assert clone.total_messages == run.total_messages
    assert clone.adaptation_enabled is False
    assert clone.status == RunStatus.DRAFT


def test_experiment_run_changelist_filters_by_scenario(staff_client):
    ExperimentRun.objects.create(name="matrix cell", seed=7, scenario="prior_shift")
    ExperimentRun.objects.create(name="other cell", seed=8, scenario="stationary")
    url = reverse("admin:dashboard_experimentrun_changelist")
    response = staff_client.get(url, {"scenario": "prior_shift"})
    assert response.status_code == 200
    assert b"matrix cell" in response.content
    assert b"other cell" not in response.content


def test_stream_config_form_offers_the_detector_choices(staff_client):
    """A wrong field name in a fieldset only fails when the change form renders."""
    config = StreamConfig.load()
    url = reverse("admin:dashboard_streamconfig_change", args=[config.pk])
    response = staff_client.get(url)
    assert response.status_code == 200
    assert b"actionable_detectors" in response.content
    assert b"retrain_min_new_rows" in response.content
