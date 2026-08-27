"""JSON API.

Every response uses one envelope. The previous version returned seven different
shapes (``{"messages":…}``, ``{"categories":…}``, ``{"logs":…}``, ``{"data":…}``,
``{"error":…}``, ``{"status":…}``, ``{"message":…}``) with no versioning and no
consistency, and leaked ``str(e)`` -- including absolute server paths -- on any
failure.

Security, all of which the old layer got wrong:

* ``toggle`` is POST-only. Django does not CSRF-check GET, so the old
  ``/stream/toggle/`` could be fired by an ``<img>`` tag on any page on the
  internet and would start or stop the pipeline.
* Nothing is ``@csrf_exempt``. Three write endpoints were, with no auth at all.
* Mutations require a logged-in staff user.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from typing import Any

from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from dashboard import services
from dashboard.forms import CategoryForm, ExperimentRunForm, StreamConfigForm
from dashboard.models import Category, ExperimentRun, StreamConfig, StreamStatus


def ok(data: Any, **extra: Any) -> JsonResponse:
    return JsonResponse({"ok": True, "data": data, **extra})


def fail(message: str, *, status: int = 400, details: Any = None) -> JsonResponse:
    """A caller-safe message. Never an exception string, never a server path."""
    payload: dict[str, Any] = {"ok": False, "error": message}
    if details is not None:
        payload["details"] = details
    return JsonResponse(payload, status=status)


def json_body(view: Callable) -> Callable:
    @wraps(view)
    def wrapper(request: HttpRequest, *args, **kwargs):
        try:
            request.json = json.loads(request.body or b"{}")
        except json.JSONDecodeError:
            return fail("Request body must be valid JSON.")
        if not isinstance(request.json, dict):
            return fail("Request body must be a JSON object.")
        return view(request, *args, **kwargs)

    return wrapper


def _run_id(request: HttpRequest) -> int | None:
    raw = request.GET.get("run")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    status = StreamStatus.load()
    return status.active_experiment_run_id


# --- reads -------------------------------------------------------------------
@require_GET
def overview(request: HttpRequest) -> JsonResponse:
    return ok(services.stream_overview())


@require_GET
def series(request: HttpRequest) -> JsonResponse:
    run_id = _run_id(request)
    if run_id is None:
        return ok(None)
    limit = services.clamp_limit(request.GET.get("limit"), default=services.MAX_PAGE_SIZE)
    return ok(services.batch_series(run_id, limit=limit))


@require_GET
def categories(request: HttpRequest) -> JsonResponse:
    run_id = _run_id(request)
    return ok(services.category_distribution(run_id) if run_id else [])


@require_GET
def confusion(request: HttpRequest) -> JsonResponse:
    run_id = _run_id(request)
    return ok(services.confusion_for_run(run_id) if run_id else None)


@require_GET
def messages(request: HttpRequest) -> JsonResponse:
    category_raw = request.GET.get("category")
    try:
        category_id = int(category_raw) if category_raw else None
    except ValueError:
        return fail("category must be an integer id.")
    return ok(
        services.search_messages(
            run_id=_run_id(request),
            query=request.GET.get("q", "")[:200],
            category_id=category_id,
            only_errors=request.GET.get("errors") == "1",
            limit=services.clamp_limit(request.GET.get("limit")),
            offset=max(0, int(request.GET.get("offset") or 0)),
        )
    )


@require_GET
def drift(request: HttpRequest) -> JsonResponse:
    return ok(services.drift_events(_run_id(request), limit=services.clamp_limit(
        request.GET.get("limit"))))


@require_GET
def models(request: HttpRequest) -> JsonResponse:
    return ok(services.model_registry())


@require_GET
def jobs(request: HttpRequest) -> JsonResponse:
    return ok(services.training_jobs(_run_id(request)))


@require_GET
def runs(request: HttpRequest) -> JsonResponse:
    return ok(services.experiment_runs())


# --- writes ------------------------------------------------------------------
@require_POST
@staff_member_required
def toggle(request: HttpRequest) -> JsonResponse:
    """POST only. A GET here was a working CSRF that stopped the pipeline."""
    return ok({"is_active": services.toggle_stream()})


@require_POST
@staff_member_required
@json_body
def update_config(request: HttpRequest) -> JsonResponse:
    form = StreamConfigForm(request.json, instance=StreamConfig.load())
    if not form.is_valid():
        return fail("Invalid configuration.", details=form.errors)
    form.save()
    return ok({"saved": True})


@require_POST
@staff_member_required
@json_body
def update_category(request: HttpRequest, category_id: int) -> JsonResponse:
    category = Category.objects.filter(pk=category_id).first()
    if category is None:
        return fail("Unknown category.", status=404)
    form = CategoryForm(request.json, instance=category)
    if not form.is_valid():
        return fail("Invalid category.", details=form.errors)
    form.save()
    return ok({"saved": True})


@require_POST
@staff_member_required
@json_body
def create_run(request: HttpRequest) -> JsonResponse:
    form = ExperimentRunForm(request.json)
    if not form.is_valid():
        return fail("Invalid experiment run.", details=form.errors)
    run = form.save()
    return ok({"id": run.pk, "name": run.name})


@require_POST
@staff_member_required
@json_body
def activate_run(request: HttpRequest, run_id: int) -> JsonResponse:
    run = ExperimentRun.objects.filter(pk=run_id).first()
    if run is None:
        return fail("Unknown experiment run.", status=404)
    status = StreamStatus.load()
    status.active_experiment_run = run
    status.producer_active = bool(request.json.get("start_producer", True))
    status.is_active = bool(request.json.get("start_consumer", True))
    status.save()
    return ok({"activated": run.pk})
