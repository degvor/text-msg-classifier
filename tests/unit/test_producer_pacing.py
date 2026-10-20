"""Live-mode pacing of the producer, without Kafka or a database.

In live mode a message is stamped with the moment it is due (start + index /
rate) and sent no earlier than that, so ``classified_at - produced_at`` is the
end-to-end latency. Replay mode keeps the nominal stamps and never waits.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

producer_main = pytest.importorskip("services.producer.main")

from streamcls.corpus.agnews import Document  # noqa: E402
from streamcls.db.repo import RunSpec  # noqa: E402
from streamcls.labels import N_CLASSES  # noqa: E402

T0 = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


class FakeClock:
    """datetime.now replacement; ``wait`` advances it instead of sleeping."""

    def __init__(self) -> None:
        self.now_value = T0
        self.waits: list[float] = []

    def now(self, tz=None):
        return self.now_value

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        self.now_value += timedelta(seconds=seconds)
        return False

    def is_set(self) -> bool:
        return False


class FakeKafka:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.sent: list[tuple[datetime, dict]] = []

    def send(self, topic, key, value):
        import json

        self.sent.append((self.clock.now_value, json.loads(value)))

    def flush(self, timeout=None):
        return None


def _producer(monkeypatch, clock: FakeClock):
    monkeypatch.setattr(producer_main, "ensure_topic", lambda *a, **k: None)
    monkeypatch.setattr(producer_main, "datetime", SimpleNamespace(now=clock.now))
    p = object.__new__(producer_main.Producer)
    p.settings = SimpleNamespace(
        kafka=SimpleNamespace(topic_for_run=lambda run_id: f"posts.run-{run_id}",
                              bootstrap_servers="unused", num_partitions=1))
    p.repo = SimpleNamespace(mark_run_started=lambda run_id: None,
                             mark_run_finished=lambda run_id, status: None,
                             producer_heartbeat=lambda: None)
    p.stop_event = clock
    p._producer = FakeKafka(clock)
    pool = [Document(f"label{c} story {i}", c) for c in range(N_CLASSES) for i in range(10)]
    p._stream_pool = lambda split: pool
    return p


def _run(mode: str, rate: float, total: int = 6) -> RunSpec:
    return RunSpec(id=7, name="pacing", seed=1, mode=mode, status="draft", adaptation_enabled=False,
                   total_messages=total, rate_msgs_per_sec=rate, drift_schedule=[],
                   dataset_split="agnews-stream-holdout-v1")


def test_live_messages_are_stamped_and_sent_on_schedule(monkeypatch):
    clock = FakeClock()
    p = _producer(monkeypatch, clock)
    p._emit_run(_run("live", rate=4.0))

    sent = p._producer.sent
    assert len(sent) == 6
    for index, (sent_at, message) in enumerate(sent):
        due = T0 + timedelta(seconds=index / 4.0)
        assert datetime.fromisoformat(message["produced_at"]) == due
        assert sent_at >= due  # never early
        assert sent_at - due < timedelta(milliseconds=1)  # and not late on a fast send


def test_replay_mode_does_not_wait(monkeypatch):
    clock = FakeClock()
    p = _producer(monkeypatch, clock)
    p._emit_run(_run("replay", rate=4.0))

    assert len(p._producer.sent) == 6
    assert clock.waits == []
