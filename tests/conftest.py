import importlib.util
import os

# Set before anything imports Django settings. These are test-only values, and
# deliberately not defaults on the settings themselves: a real run must supply
# both, which is what stops a SECRET_KEY or a password being committed again.
os.environ.setdefault("DJANGO_SECRET_KEY", "pytest-only-not-a-real-secret")
os.environ.setdefault("POSTGRES_PASSWORD", "pytest-only")

import pytest

from streamcls.config import get_settings


def pytest_collection_modifyitems(config, items):
    """Skip `spark`-marked tests where there is no JVM/pyspark.

    They run inside the Spark image (`make test-spark`). Locally they should be
    reported as skipped, not failed -- the dev machine deliberately has no JRE.
    """
    if importlib.util.find_spec("pyspark") is not None:
        return
    skip = pytest.mark.skip(reason="pyspark not installed; run inside the Spark image")
    for item in items:
        if "spark" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Keep tests off any real .env and off each other's cached Settings."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-password")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
