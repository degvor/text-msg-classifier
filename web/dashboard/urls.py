"""URL map, namespaced and consistently prefixed.

The old urlconf had no namespace and split routes across three unrelated
prefixes -- ``stream/``, a single ``api/`` route, and bare ``chart-data/``,
``categories/``, ``search-messages/``, ``performance-logs/``.
"""

from django.urls import path

from dashboard.views import api, pages

app_name = "dashboard"

urlpatterns = [
    # Pages
    path("", pages.overview, name="overview"),
    path("drift/", pages.drift, name="drift"),
    path("messages/", pages.messages, name="messages"),
    path("models/", pages.models, name="models"),
    path("runs/", pages.runs, name="runs"),
    path("settings/", pages.settings, name="settings"),
    # Read API
    path("api/overview/", api.overview, name="api_overview"),
    path("api/series/", api.series, name="api_series"),
    path("api/categories/", api.categories, name="api_categories"),
    path("api/confusion/", api.confusion, name="api_confusion"),
    path("api/messages/", api.messages, name="api_messages"),
    path("api/drift/", api.drift, name="api_drift"),
    path("api/models/", api.models, name="api_models"),
    path("api/jobs/", api.jobs, name="api_jobs"),
    path("api/runs/", api.runs, name="api_runs"),
    # Write API -- POST only, CSRF enforced, staff only
    path("api/stream/toggle/", api.toggle, name="api_toggle"),
    path("api/config/", api.update_config, name="api_update_config"),
    path("api/categories/<int:category_id>/", api.update_category, name="api_update_category"),
    path("api/runs/create/", api.create_run, name="api_create_run"),
    path("api/runs/<int:run_id>/activate/", api.activate_run, name="api_activate_run"),
]
