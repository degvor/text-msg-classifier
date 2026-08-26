from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from streamcls.schemas import PostMessage, _unwrap_optional


def _msg(**over):
    base = {
        "experiment_run_id": 7,
        "post_id": "7-000123",
        "msg_index": 123,
        "text": "Oil prices climb on supply concerns",
        "channel_id": "c2",
        "channel_title": "Business Daily",
        "view_count": 812,
        "produced_at": datetime(2026, 8, 27, 12, 0, tzinfo=UTC),
        "true_label": 2,
    }
    return PostMessage(**{**base, **over})


def test_unknown_fields_are_rejected():
    """extra='forbid' is what stops the producer and consumer drifting apart."""
    with pytest.raises(ValidationError):
        _msg(PostText="wrong field name from the old schema")


def test_ground_truth_fields_default_to_no_drift():
    m = _msg()
    assert m.perturbation_kind == "none"
    assert m.perturbation_intensity == 0.0
    assert m.drift_phase == 0
    assert m.is_drift_point is False


def test_post_id_is_run_scoped():
    """A bare post_id would collide when a held-out split is replayed per run."""
    m = _msg()
    assert m.post_id.startswith(f"{m.experiment_run_id}-")


def test_json_round_trip():
    m = _msg()
    assert PostMessage.model_validate_json(m.model_dump_json()) == m


def test_unwrap_optional():
    assert _unwrap_optional(int) == (int, False)
    assert _unwrap_optional(int | None) == (int, True)
    assert _unwrap_optional(str) == (str, False)


@pytest.mark.spark
def test_spark_schema_matches_pydantic_fields():
    from streamcls.schemas import spark_schema

    schema = spark_schema()
    assert [f.name for f in schema.fields] == list(PostMessage.model_fields)
