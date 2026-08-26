"""The Kafka message contract, defined once.

The producer validates against the Pydantic model; the Spark consumer parses
with a ``StructType`` *derived from that same model*. Previously the schema
existed twice -- as a dict literal in the producer and a hand-written
``StructType`` in the consumer -- with nothing keeping them in step.

``pyspark`` is imported inside :func:`spark_schema`, never at module level, so
the producer and the web app can import this module without a JVM or a 300 MB
wheel.
"""

from __future__ import annotations

import types
import typing
from datetime import datetime

from pydantic import BaseModel, ConfigDict

__all__ = ["PostMessage", "spark_schema"]


class PostMessage(BaseModel):
    """One post on the wire.

    ``true_label`` and the ``drift_*`` fields are **evaluation ground truth**.
    The model never sees them: they exist so the consumer can measure live
    accuracy and score detection delay against known change points. Keep them
    out of any function that computes features or feeds a detector.
    """

    model_config = ConfigDict(extra="forbid")

    # --- identity ------------------------------------------------------------
    experiment_run_id: int
    #: ``f"{run_id}-{msg_index:06d}"``. Run-scoped because the same held-out
    #: split is replayed for every experiment; a bare post_id would collide
    #: across runs and silently destroy earlier results.
    post_id: str
    msg_index: int

    # --- payload -------------------------------------------------------------
    text: str
    channel_id: str
    channel_title: str
    view_count: int
    produced_at: datetime

    # --- ground truth (evaluation only) --------------------------------------
    true_label: int
    perturbation_kind: str = "none"
    perturbation_intensity: float = 0.0
    drift_phase: int = 0
    is_drift_point: bool = False

    def kafka_key(self) -> bytes:
        return self.post_id.encode()


_SPARK_TYPES: dict[type, str] = {
    str: "StringType",
    bool: "BooleanType",
    int: "IntegerType",
    float: "DoubleType",
    datetime: "TimestampType",
}


def _unwrap_optional(annotation: object) -> tuple[object, bool]:
    """Return ``(inner_type, nullable)`` for ``T`` or ``T | None``."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def spark_schema():
    """Build the ``StructType`` for :class:`PostMessage` by introspecting it."""
    from pyspark.sql import types as T

    fields = []
    for name, info in PostMessage.model_fields.items():
        annotation, nullable = _unwrap_optional(info.annotation)
        try:
            spark_type_name = _SPARK_TYPES[annotation]  # type: ignore[index]
        except KeyError:
            raise TypeError(
                f"PostMessage.{name}: no Spark mapping for {annotation!r}. "
                f"Add it to streamcls.schemas._SPARK_TYPES."
            ) from None
        # A field with a default may legitimately be absent from the payload.
        optional = nullable or not info.is_required()
        fields.append(T.StructField(name, getattr(T, spark_type_name)(), optional))
    return T.StructType(fields)
