"""The web layer.

Most of these correspond to a specific defect in the previous dashboard, and
several are security tests: the old build could be started or stopped by an
``<img>`` tag on any page on the internet.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from dashboard.models import (
    BatchMetric,
    Category,
    ClassifiedMessage,
    DriftEvent,
    ExperimentRun,
    ModelVersion,
    StreamConfig,
    StreamStatus,
)
from streamcls.labels import LABELS

pytestmark = [pytest.mark.integration, pytest.mark.django_db]

PAGES = ["overview", "drift", "messages", "models", "runs", "settings"]
READ_API = [
    "api_overview", "api_series", "api_categories", "api_confusion",
    "api_messages", "api_drift", "api_models", "api_jobs", "api_runs",
]
WRITE_API = [
    ("api_toggle", {}),
    ("api_update_config", {}),
    ("api_create_run", {}),
]


@pytest.fixture(autouse=True)
def _reference_data():
    for label in LABELS:
        Category.objects.update_or_create(
            id=label.id, defaults={"key": label.key, "display_name": label.display_name}
        )
    StreamConfig.load()
    StreamStatus.load()


@pytest.fixture
def staff(client):
    user = User.objects.create_user("op", "op@example.com", "pw-for-tests", is_staff=True)
    client.force_login(user)
    return client


@pytest.fixture
def run():
    run = ExperimentRun.objects.create(
        name="web test", seed=1, total_messages=1000,
        drift_schedule=[{"at": 500, "kind": "concept_abrupt", "params": {"map": {"2": 3}}}],
    )
    model = ModelVersion.objects.create(version="m1", artifact_path="/p", test_accuracy=0.9)
    status = StreamStatus.load()
    status.active_experiment_run = run
    status.active_model_version = model
    status.save()

    now = datetime.now(UTC)
    for batch_id in range(4):
        BatchMetric.objects.create(
            experiment_run=run, batch_id=batch_id, model_version=model,
            started_at=now, n_received=100, n_classified=100,
            n_correct=90 if batch_id < 2 else 60,
            accuracy=0.9 if batch_id < 2 else 0.6,
        )
    for i in range(20):
        ClassifiedMessage.objects.create(
            experiment_run=run, post_id=f"1-{i:06d}", msg_index=i,
            post_text=f"story number {i} about markets",
            true_label_id=2, predicted_label_id=2 if i % 4 else 3,
            confidence=0.8, is_correct=bool(i % 4), model_version=model,
            batch_id=i // 5, produced_at=now, classified_at=now,
        )
    DriftEvent.objects.create(
        experiment_run=run, batch_id=2, detector="adwin_error", statistic=0.4,
        suppressed=False, action_taken="retrain_queued", detection_delay_batches=0,
    )
    return run


# --- pages -------------------------------------------------------------------
@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_anonymously(client, name):
    assert client.get(reverse(f"dashboard:{name}")).status_code == 200


@pytest.mark.parametrize("name", PAGES)
def test_every_page_extends_the_base_template(client, name):
    """Four standalone documents with ~150 lines of copy-pasted CSS, already
    drifted in four places, are what this replaces."""
    response = client.get(reverse(f"dashboard:{name}"))
    body = response.content.decode()
    assert "dashboard/css/app.css" in body
    assert "dashboard/vendor/chart.umd.min.js" in body
    assert "<nav" in body


@pytest.mark.parametrize("name", PAGES)
def test_no_inline_scripts_or_styles(client, name):
    """Every style and script is an external, vendored file.

    Also catches a specific template bug: ``{# #}`` is single-line in Django,
    so a multi-line comment written that way leaks its tail into the rendered
    page -- which is how an explanatory comment ended up inside a stray
    ``<script>`` block here.
    """
    import re

    body = client.get(reverse(f"dashboard:{name}")).content.decode()
    assert not re.search(r"<script(?![^>]*\bsrc=)", body), "inline <script> in the page"
    assert "<style" not in body
    assert "strict CSP" not in body, "a template comment leaked into the output"


def test_no_external_scripts_are_loaded(client):
    """Chart.js was an unpinned CDN <script> with no SRI."""
    body = client.get(reverse("dashboard:overview")).content.decode()
    assert "cdn.jsdelivr.net" not in body
    assert "https://" not in body.split("<body")[0].replace("http-equiv", "")


# --- security ----------------------------------------------------------------
def test_toggle_rejects_GET(client):
    """Django does not CSRF-check GET. The old /stream/toggle/ had no method
    guard, so `<img src=...>` on any page started or stopped the pipeline."""
    assert client.get(reverse("dashboard:api_toggle")).status_code == 405


@pytest.mark.parametrize(("name", "body"), WRITE_API)
def test_write_endpoints_reject_GET(client, name, body):
    assert client.get(reverse(f"dashboard:{name}")).status_code == 405


@pytest.mark.parametrize(("name", "body"), WRITE_API)
def test_write_endpoints_require_staff(client, name, body):
    """Three write endpoints were @csrf_exempt with no authentication at all."""
    response = client.post(
        reverse(f"dashboard:{name}"), data=body, content_type="application/json"
    )
    assert response.status_code in (302, 403), f"{name} was reachable anonymously"


def test_write_endpoints_enforce_csrf(staff):
    from django.test import Client

    csrf_client = Client(enforce_csrf_checks=True)
    user = User.objects.get(username="op")
    csrf_client.force_login(user)
    response = csrf_client.post(
        reverse("dashboard:api_toggle"), data={}, content_type="application/json"
    )
    assert response.status_code == 403


def test_toggle_flips_the_flag_for_staff(staff):
    before = StreamStatus.load().is_active
    payload = staff.post(
        reverse("dashboard:api_toggle"), data={}, content_type="application/json"
    ).json()
    assert payload["ok"] is True
    assert payload["data"]["is_active"] is not before
    assert StreamStatus.load().is_active is not before


def test_hostile_message_text_is_served_as_json_data(client, run):
    """Markup in a message must round-trip as *data*, never as markup.

    A literal ``<script>`` inside a JSON string is harmless -- the browser
    parses the body as JSON, not HTML. What made the old dashboard vulnerable
    was the client interpolating that value into ``innerHTML``; see
    :func:`test_client_never_assigns_innerHTML`.
    """
    ClassifiedMessage.objects.filter(msg_index=0).update(post_text="<script>alert(1)</script>")
    response = client.get(reverse("dashboard:api_messages"))
    assert response["Content-Type"].startswith("application/json")
    texts = [m["text"] for m in response.json()["data"]["results"]]
    assert "<script>alert(1)</script>" in texts, "the text must survive intact as data"


def test_client_never_assigns_innerHTML():
    """This is the actual XSS fix.

    The previous dashboard built every table with ``tbody.innerHTML += row``
    inside a loop -- both an O(n^2) DOM reparse and the injection vector for
    ``PostText`` and a user-supplied colour. Everything is built with
    ``createElement`` and ``textContent`` now, and a regression here would be
    invisible in any response-body assertion.
    """
    import re

    from django.conf import settings

    source = (settings.BASE_DIR / "dashboard/static/dashboard/js/app.js").read_text()
    # Strip comments first: the file *discusses* innerHTML at length, and a
    # substring check would only ever be testing the prose.
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)

    assert not re.search(r"\.innerHTML\s*(\+?=)", code), "innerHTML is assigned somewhere"
    assert not re.search(r"insertAdjacentHTML|outerHTML\s*=|document\.write", code)
    assert "textContent" in code


def test_an_invalid_colour_is_rejected(staff):
    """The colour is interpolated into the page's CSS; the old endpoint stored
    it verbatim from an unauthenticated request."""
    response = staff.post(
        reverse("dashboard:api_update_category", args=[0]),
        data={"display_name": "World", "color": "red; }body{display:none", "description": ""},
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.json()["ok"] is False
    assert Category.objects.get(pk=0).color != "red; }body{display:none"


# --- API shape ---------------------------------------------------------------
@pytest.mark.parametrize("name", READ_API)
def test_read_endpoints_use_one_envelope(client, run, name):
    """The old layer returned seven different response shapes."""
    payload = client.get(reverse(f"dashboard:{name}")).json()
    assert set(payload) >= {"ok", "data"}
    assert payload["ok"] is True


def test_errors_never_leak_exception_text(staff):
    """Five handlers returned str(e) to the caller, including absolute paths."""
    response = staff.post(
        reverse("dashboard:api_update_config"),
        data={"adwin_delta": 5.0},
        content_type="application/json",
    )
    assert response.status_code == 400
    body = response.json()
    assert body["ok"] is False
    assert "/" not in body["error"], "error text looks like it contains a path"
    assert "adwin_delta" in body["details"]


def test_malformed_json_is_a_400_not_a_500(staff):
    response = staff.post(
        reverse("dashboard:api_update_config"), data="not json", content_type="application/json"
    )
    assert response.status_code == 400


CONFIG_PAYLOAD = {
    "max_message_length": 1234, "max_offsets_per_trigger": 250, "trigger_seconds": 7,
    "adwin_delta": 0.005, "psi_threshold": 0.3, "psi_consecutive_batches": 2,
    "grace_batches": 4, "promote_min_delta": 0.02, "min_batches_between_retrains": 6,
    "max_retrains_per_run": 5, "retrain_window_rows": 100, "retrain_base_rows": 50,
    "retrain_min_new_rows": 10, "adaptation_enabled": True, "auto_promote": False,
    "actionable_detectors": ["ddm", "adwin_error"],
}


def test_config_round_trips_through_validation(staff):
    assert staff.post(
        reverse("dashboard:api_update_config"), data=CONFIG_PAYLOAD,
        content_type="application/json",
    ).json()["ok"]
    config = StreamConfig.load()
    assert config.max_message_length == 1234
    assert config.auto_promote is False
    assert config.actionable_detectors == ["ddm", "adwin_error"]


def test_actionable_detectors_are_stored_in_canonical_order(staff):
    payload = {**CONFIG_PAYLOAD, "actionable_detectors": ["psi_vocabulary", "ddm"]}
    assert staff.post(
        reverse("dashboard:api_update_config"), data=payload, content_type="application/json"
    ).json()["ok"]
    assert StreamConfig.load().actionable_detectors == ["ddm", "psi_vocabulary"]


def test_an_unknown_detector_is_rejected(staff):
    payload = {**CONFIG_PAYLOAD, "actionable_detectors": ["adwin_error", "oracle"]}
    response = staff.post(
        reverse("dashboard:api_update_config"), data=payload, content_type="application/json"
    )
    assert response.status_code == 400
    body = response.json()
    assert body["ok"] is False
    assert "actionable_detectors" in body["details"]
    assert StreamConfig.load().actionable_detectors == ["adwin_error"]


def test_settings_page_reflects_the_saved_detector_set(staff):
    config = StreamConfig.load()
    config.actionable_detectors = ["ddm"]
    config.save()
    html = staff.get(reverse("dashboard:settings")).content.decode()
    assert 'name="actionable_detectors" data-multi value="ddm" checked' in html
    assert 'name="actionable_detectors" data-multi value="adwin_error" >' in html


def test_runs_carry_their_scenario(client, run):
    run.scenario = "concept_abrupt"
    run.save()
    rows = client.get(reverse("dashboard:api_runs")).json()["data"]
    assert rows[0]["scenario"] == "concept_abrupt"


def test_an_invalid_drift_schedule_is_rejected_before_the_run_starts(staff):
    response = staff.post(
        reverse("dashboard:api_create_run"),
        data={"name": "bad", "seed": 1, "mode": "replay", "total_messages": 10,
              "rate_msgs_per_sec": 1, "dataset_split": "x",
              "drift_schedule": [{"at": 1, "kind": "not_a_kind"}]},
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "drift_schedule" in response.json()["details"]


# --- pagination and correctness ----------------------------------------------
def test_messages_are_paginated_with_a_real_total(client, run):
    """The old endpoint hard-truncated to 100 rows and reported neither the
    total nor that truncation had happened."""
    page = client.get(reverse("dashboard:api_messages") + "?limit=5").json()["data"]
    assert page["total"] == 20
    assert len(page["results"]) == 5
    assert page["limit"] == 5

    second = client.get(reverse("dashboard:api_messages") + "?limit=5&offset=5").json()["data"]
    assert {m["post_id"] for m in page["results"]}.isdisjoint(
        {m["post_id"] for m in second["results"]}
    )


def test_page_size_is_capped(client, run):
    page = client.get(reverse("dashboard:api_messages") + "?limit=100000").json()["data"]
    assert page["limit"] <= 200


def test_a_bad_limit_falls_back_instead_of_erroring(client, run):
    assert client.get(reverse("dashboard:api_messages") + "?limit=abc").status_code == 200


def test_errors_only_filter(client, run):
    page = client.get(reverse("dashboard:api_messages") + "?errors=1").json()["data"]
    assert page["total"] == 5
    assert all(m["is_correct"] is False for m in page["results"])


def test_series_carries_the_annotations_the_chart_needs(client, run):
    data = client.get(reverse("dashboard:api_series")).json()["data"]
    assert data["batches"] == [0, 1, 2, 3]
    assert data["accuracy"][0] == pytest.approx(0.9)
    assert data["cumulative_accuracy"][-1] == pytest.approx(0.75)
    # Injected change point at message 500 with 500 per batch -> batch 1.
    assert data["change_point_batches"] == [1]
    assert data["model_versions"] == ["m1"] * 4
    assert data["drift_events"][0]["detector"] == "adwin_error"


def test_overview_counts_received_and_classified_separately(client, run):
    """The old metrics incremented both by the same value, so drops were
    structurally unobservable."""
    totals = client.get(reverse("dashboard:api_overview")).json()["data"]["totals"]
    assert totals["received"] == 400
    assert totals["classified"] == 400
    assert totals["dropped"] == 0
    assert totals["accuracy"] == pytest.approx(0.75)


def test_confusion_matrix_is_computed_in_the_database(client, run):
    data = client.get(reverse("dashboard:api_confusion")).json()["data"]
    assert len(data["matrix"]) == len(LABELS)
    assert sum(sum(row) for row in data["matrix"]) == 20


def test_runs_report_accuracy_after_the_change_point(client, run):
    """Overall accuracy is dominated by the stable prefix both A/B arms share,
    so the after-drift column is the one that carries the comparison."""
    rows = client.get(reverse("dashboard:api_runs")).json()["data"]
    assert rows[0]["id"] == run.pk
    assert rows[0]["drift_events"] == 1
    assert "accuracy_after_drift" in rows[0]


def test_endpoints_work_with_no_active_run(client):
    StreamStatus.objects.filter(pk=1).update(active_experiment_run=None)
    for name in READ_API:
        assert client.get(reverse(f"dashboard:{name}")).status_code == 200
