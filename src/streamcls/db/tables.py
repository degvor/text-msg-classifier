"""Physical table names, imported by both Django and the raw SQL writers.

Django models set ``Meta.db_table`` from these constants, and the Spark/psycopg2
paths read them. Previously the consumer hardcoded ``django_app_streamstatus``,
so renaming the Django app would have silently broken the control plane.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Table"]


class Table(StrEnum):
    CATEGORY = "category"
    EXPERIMENT_RUN = "experiment_run"
    STREAM_CONFIG = "stream_config"
    STREAM_STATUS = "stream_status"
    MODEL_VERSION = "model_version"
    TRAINING_JOB = "training_job"
    CLASSIFIED_MESSAGE = "classified_message"
    BATCH_METRIC = "batch_metric"
    DRIFT_EVENT = "drift_event"
