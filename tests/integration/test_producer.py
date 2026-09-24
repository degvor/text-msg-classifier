"""The producer, against a real Kafka and a real Postgres.

Two of these tests exist specifically because the previous producer failed
silently in ways no unit test would have caught: it stopped emitting after 100
messages while still appearing healthy, and it shipped no ground truth, so
accuracy was unmeasurable by construction.

Requires the dev stack: ``docker compose up -d db kafka``.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter

import pytest
from django.utils import timezone

from dashboard.models import ExperimentRun, RunStatus, StreamMode, StreamStatus
from streamcls.corpus.agnews import Document
from streamcls.corpus.schedule import DriftSchedule
from streamcls.corpus.stream import SeededStream
from streamcls.labels import N_CLASSES

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]

kafka = pytest.importorskip("kafka")
BOOTSTRAP = "localhost:9094"


@pytest.fixture
def topic():
    """A unique topic per test, so runs never see each other's messages."""
    name = f"test-posts-{uuid.uuid4().hex[:8]}"
    yield name
    try:
        admin = kafka.KafkaAdminClient(bootstrap_servers=BOOTSTRAP, request_timeout_ms=10_000)
        admin.delete_topics([name])
        admin.close()
    except Exception:
        pass


@pytest.fixture
def pool():
    return [
        Document(f"label{label} story {i} reporting on events of the day", label)
        for label in range(N_CLASSES)
        for i in range(40)
    ]


@pytest.fixture
def repo():
    """A Repository pointed at the same database Django is using.

    Deliberately built from ``django.conf.settings`` rather than the
    environment: the root conftest overrides ``POSTGRES_PASSWORD`` to isolate
    unit tests, so reading it here would pick up the fake value.
    """
    from django.conf import settings as django_settings

    from streamcls.config import PostgresSettings
    from streamcls.db.repo import Repository

    db = django_settings.DATABASES["default"]
    pg = PostgresSettings(
        _env_file=None,
        host=db["HOST"],
        port=int(db["PORT"]),
        db=db["NAME"],
        user=db["USER"],
        password=db["PASSWORD"],
    )
    with Repository(pg) as repository:
        yield repository


def _publish(topic: str, messages) -> int:
    producer = kafka.KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: v.encode(),
        key_serializer=lambda k: k,
        acks="all",
        retries=5,
        enable_idempotence=True,
        max_in_flight_requests_per_connection=1,
    )
    try:
        count = 0
        for message in messages:
            producer.send(topic, key=message.kafka_key(), value=message.model_dump_json())
            count += 1
        producer.flush(timeout=60)
        return count
    finally:
        producer.close(timeout=10)


def _consume(topic: str, expected: int) -> list[dict]:
    consumer = kafka.KafkaConsumer(
        topic,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="earliest",
        consumer_timeout_ms=20_000,
        value_deserializer=lambda v: json.loads(v.decode()),
    )
    try:
        out = []
        for record in consumer:
            out.append(record.value)
            if len(out) >= expected:
                break
        return out
    finally:
        consumer.close()


def _stream(pool, *, total=600, schedule=None, seed=4242, run_id=1):
    return SeededStream(
        documents=pool,
        schedule=schedule or DriftSchedule(base_intensity=0.2),
        seed=seed,
        run_id=run_id,
        total_messages=total,
    )


def test_round_trips_through_kafka_intact(topic, pool):
    stream = _stream(pool, total=600)
    sent = list(stream)
    assert _publish(topic, sent) == 600

    received = _consume(topic, 600)
    assert len(received) == 600
    assert [r["msg_index"] for r in received] == list(range(600))
    assert [r["post_id"] for r in received] == [m.post_id for m in sent]
    assert [r["text"] for r in received] == [m.text for m in sent]


def test_keeps_emitting_well_past_one_hundred_messages(topic, pool):
    """The old producer closed its connection in a `finally` after every batch,
    then failed the `if not self.producer` reconnect guard -- a closed producer
    is not None -- so it emitted exactly 100 messages per container start and
    then blocked and failed forever, without exiting."""
    assert _publish(topic, _stream(pool, total=1000)) == 1000
    assert len(_consume(topic, 1000)) == 1000


def test_every_message_carries_evaluation_ground_truth(topic, pool):
    """Without this the model's live accuracy cannot be computed at all, which
    is why the old system could only be scored by abusing ChannelTitle."""
    _publish(topic, _stream(pool, total=200))
    for row in _consume(topic, 200):
        assert row["true_label"] in range(N_CLASSES)
        assert isinstance(row["drift_phase"], int)
        assert isinstance(row["is_drift_point"], bool)
        assert 0.0 <= row["perturbation_intensity"] <= 1.0


def test_drift_schedule_survives_the_wire(topic, pool):
    schedule = DriftSchedule(
        events=[
            {"at": 200, "kind": "prior_shift", "params": {"weights": [0.7, 0.1, 0.1, 0.1]}},
            {"at": 400, "kind": "concept_abrupt", "params": {"map": {"2": 3}}},
        ],
        base_intensity=0.0,
    )
    _publish(topic, _stream(pool, total=600, schedule=schedule))
    rows = _consume(topic, 600)

    assert [r["msg_index"] for r in rows if r["is_drift_point"]] == [200, 400]
    assert sorted({r["drift_phase"] for r in rows}) == [0, 1, 2]

    before = Counter(r["true_label"] for r in rows[:200])
    after = Counter(r["true_label"] for r in rows[200:400])
    assert before[0] / 200 < 0.4
    assert after[0] / 200 > 0.6

    # Concept drift: label 2 keeps its label but carries label-3 text.
    drifted = [r for r in rows[400:] if r["true_label"] == 2]
    assert drifted
    assert all(r["text"].startswith("label3") for r in drifted)


def test_two_runs_with_the_same_seed_produce_identical_streams(topic, pool):
    """The headline A/B figure compares two runs that differ only in whether
    adaptation was enabled. That is only valid if the streams are identical."""
    a = [(m.msg_index, m.text, m.true_label) for m in _stream(pool, total=400, run_id=1)]
    b = [(m.msg_index, m.text, m.true_label) for m in _stream(pool, total=400, run_id=2)]
    assert a == b


def test_post_ids_from_different_runs_can_coexist(topic, pool):
    """The same held-out split is replayed for every run, so run-scoping the
    post_id is what stops run 2 overwriting run 1 in the sink."""
    ids_one = {m.post_id for m in _stream(pool, total=100, run_id=1)}
    ids_two = {m.post_id for m in _stream(pool, total=100, run_id=2)}
    assert ids_one.isdisjoint(ids_two)


def test_ensure_topic_creates_once_and_is_then_a_no_op(topic):
    """Whoever starts a run first creates its topic; the second caller (or a
    restart) must find it already there rather than fail."""
    from streamcls.kafka_admin import ensure_topic

    assert ensure_topic(BOOTSTRAP, topic, partitions=1) is True
    assert ensure_topic(BOOTSTRAP, topic, partitions=1) is False

    admin = kafka.KafkaAdminClient(bootstrap_servers=BOOTSTRAP, request_timeout_ms=10_000)
    try:
        assert topic in admin.list_topics()
    finally:
        admin.close()


def test_run_lifecycle_is_recorded_in_postgres(repo):
    """The control plane the producer polls: activate a run, and the producer's
    own bookkeeping marks it started and then completed."""
    run = ExperimentRun.objects.create(
        name="lifecycle", seed=1, mode=StreamMode.REPLAY, total_messages=10
    )
    status = StreamStatus.load()
    status.active_experiment_run = run
    status.producer_active = True
    status.save()

    control = repo.read_control()
    assert control.producer_active is True
    assert control.active_experiment_run_id == run.pk

    spec = repo.load_run(run.pk)
    assert spec.seed == 1
    assert spec.total_messages == 10
    assert spec.drift_schedule == []

    repo.mark_run_started(run.pk)
    repo.producer_heartbeat()
    repo.mark_run_finished(run.pk)
    repo.set_producer_active(False)

    run.refresh_from_db()
    status.refresh_from_db()
    assert run.status == RunStatus.COMPLETED
    assert run.started_at is not None and run.ended_at is not None
    assert status.producer_active is False
    assert status.producer_heartbeat_at is not None
    assert (timezone.now() - status.producer_heartbeat_at).total_seconds() < 60


def test_missing_run_raises_rather_than_idling_silently(repo):
    """The old consumer caught everything and returned False, so "Postgres is
    down" and "the user turned it off" were indistinguishable."""
    with pytest.raises(LookupError, match="not found"):
        repo.load_run(9_999_999)
