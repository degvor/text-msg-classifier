"""SparkSession construction, in one place with pinned settings.

Two problems this fixes.

**Reproducibility.** No ``.master()`` was ever set and the consumer ran under
plain ``python3`` rather than ``spark-submit``, so Spark silently used
``local[*]`` -- a single JVM whose parallelism depended on the host's core
count. Since partitioning affects the order in which L-BFGS sees data, that
made model fitting non-reproducible across machines. The core count and shuffle
partitions are pinned here instead.

**Startup.** ``spark.jars.packages`` resolved the Kafka connector and JDBC
driver through Ivy on *every* container start, so a cold start needed internet
and had unbounded latency. Those jars are baked into the image now
(``docker/jars.txt``), so nothing is declared here at all.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from streamcls.config import SparkSettings, get_settings

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession

__all__ = ["build_session"]


def build_session(
    app_name: str, *, settings: SparkSettings | None = None, **extra: str
) -> SparkSession:
    from pyspark.sql import SparkSession

    cfg = settings or get_settings().spark
    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER", cfg.master))
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.memory", cfg.executor_memory)
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.default.parallelism", str(cfg.shuffle_partitions))
        # Timestamps cross Kafka JSON, Spark and Postgres. Pinning the session
        # zone to UTC removes an entire class of off-by-hours bug; the old code
        # mixed naive-UTC in one service with naive-local in another.
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
    )
    for key, value in extra.items():
        builder = builder.config(key.replace("__", "."), value)

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    return session
