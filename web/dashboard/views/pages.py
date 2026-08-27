"""Template views. No context building beyond what the page needs to render.

All data arrives via the JSON API, so these are thin on purpose -- the previous
version had three identical zero-context ``render()`` wrappers that existed only
because there was no base template.
"""

from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from dashboard.models import Category, ExperimentRun, StreamConfig, StreamStatus

NAV = (
    ("dashboard:overview", "Overview"),
    ("dashboard:drift", "Drift"),
    ("dashboard:messages", "Messages"),
    ("dashboard:models", "Models"),
    ("dashboard:runs", "Runs"),
    ("dashboard:settings", "Settings"),
)


def _base(request: HttpRequest, active: str) -> dict:
    return {"nav": NAV, "active_page": active, "status": StreamStatus.load()}


def overview(request: HttpRequest) -> HttpResponse:
    return render(request, "dashboard/overview.html", _base(request, "dashboard:overview"))


def drift(request: HttpRequest) -> HttpResponse:
    return render(request, "dashboard/drift.html", _base(request, "dashboard:drift"))


def messages(request: HttpRequest) -> HttpResponse:
    context = _base(request, "dashboard:messages")
    context["categories"] = Category.objects.all()
    return render(request, "dashboard/messages.html", context)


def models(request: HttpRequest) -> HttpResponse:
    return render(request, "dashboard/models.html", _base(request, "dashboard:models"))


def runs(request: HttpRequest) -> HttpResponse:
    context = _base(request, "dashboard:runs")
    context["runs"] = ExperimentRun.objects.order_by("-created_at")[:50]
    return render(request, "dashboard/runs.html", context)


def settings(request: HttpRequest) -> HttpResponse:
    context = _base(request, "dashboard:settings")
    context["config"] = StreamConfig.load()
    context["categories"] = Category.objects.all()
    return render(request, "dashboard/settings.html", context)
