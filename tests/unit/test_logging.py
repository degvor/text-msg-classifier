import json
import logging

import streamcls.logging as slog


def test_emits_json_with_service_and_extras(capsys):
    slog.configure("consumer", level="INFO")
    # A child logger, to prove the service field survives propagation -- a
    # logger-level filter would not have been applied here.
    logging.getLogger("streamcls.drift.ddm").warning("drift", extra={"batch_id": 12})

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["service"] == "consumer"
    assert payload["level"] == "WARNING"
    assert payload["name"] == "streamcls.drift.ddm"
    assert payload["message"] == "drift"
    assert payload["batch_id"] == 12
    assert "ts" in payload


def test_configure_is_idempotent(capsys):
    slog.configure("a")
    slog.configure("b")
    slog.get_logger("x").info("once")
    lines = [ln for ln in capsys.readouterr().out.strip().splitlines() if ln]
    assert len(lines) == 1, "duplicate handlers would double every log line"
    assert json.loads(lines[0])["service"] == "b"
