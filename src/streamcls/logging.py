"""Structured JSON logging.

``python-json-logger`` has been a declared dependency since the original
project but was never imported; every service used ``logging.basicConfig`` and
emitted unparseable free text.
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["configure", "get_logger"]

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(service)s %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def configure(service: str, level: str | None = None) -> None:
    """Install a JSON handler on the root logger. Idempotent."""
    root = logging.getLogger()
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root.setLevel(resolved)

    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stdout)
    try:
        from pythonjsonlogger.json import JsonFormatter
    except ImportError:  # pragma: no cover - older python-json-logger layout
        from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[no-redef]

    handler.setFormatter(
        JsonFormatter(
            _FORMAT,
            datefmt=_DATEFMT,
            rename_fields={"asctime": "ts", "levelname": "level"},
        )
    )
    # The filter belongs on the handler, not the logger: filters attached to a
    # logger are skipped for records that merely propagate up from children,
    # so `service` would be missing from almost every line.
    handler.addFilter(_ServiceFilter(service))
    root.addHandler(handler)

    # These are chatty at INFO and drown out anything useful.
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("kafka").setLevel(logging.WARNING)


class _ServiceFilter(logging.Filter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self._service
        return True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
