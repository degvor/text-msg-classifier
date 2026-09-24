"""Topic administration, for the one place topics are created on purpose.

Every experiment run has its own topic (:meth:`KafkaSettings.topic_for_run`).
The broker's auto-create would make it on the producer's first send, but Spark
3.5's admin-based offset reader fails outright when it subscribes to a topic
that does not exist yet, so the topic must exist *before* the consumer starts.
Whoever starts a run first -- the matrix runner or the producer -- calls
:func:`ensure_topic`; the consumer image has no kafka-python and never creates
topics.

``kafka`` is imported inside the function: this module lives in the shared
package, and only the ``[producer]`` extra installs kafka-python.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

__all__ = ["ensure_topic"]


def ensure_topic(
    bootstrap_servers: str,
    name: str,
    *,
    partitions: int = 1,
    replication_factor: int = 1,
    timeout_ms: int = 10_000,
) -> bool:
    """Create ``name`` if it is missing. Returns True if this call created it.

    Idempotent: a topic that already exists is left alone, whatever its
    partition count. The partition count only matters for the first creation;
    a run's topic is never resized.
    """
    from kafka import KafkaAdminClient
    from kafka.admin import NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers.split(","), request_timeout_ms=timeout_ms
    )
    try:
        admin.create_topics(
            [NewTopic(name=name, num_partitions=partitions, replication_factor=replication_factor)],
            timeout_ms=timeout_ms,
        )
    except TopicAlreadyExistsError:
        log.debug("topic exists", extra={"topic": name})
        return False
    finally:
        admin.close()
    log.info("created topic", extra={"topic": name, "partitions": partitions})
    return True
