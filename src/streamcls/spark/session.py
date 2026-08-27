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

**Cluster mode.** Setting ``SPARK_MASTER`` to a standalone master is not enough
on its own: Spark runs the driver in *client* mode, so every executor has to
connect back to it. In Docker that fails silently-ish -- executors launch,
cannot register, and are killed and relaunched forever while the job makes no
progress. ``SPARK_DRIVER_HOST`` must therefore name something the workers can
resolve, which in Compose means running the driver as a named service rather
than as a throwaway ``docker run``.
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
    master = os.getenv("SPARK_MASTER", cfg.master)
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
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
    if not master.startswith("local"):
        # Client mode: executors dial the driver back. Without a resolvable
        # host they retry forever and the job never starts.
        driver_host = os.getenv("SPARK_DRIVER_HOST")
        if not driver_host:
            raise RuntimeError(
                f"SPARK_MASTER={master!r} is a cluster master, but SPARK_DRIVER_HOST is unset. "
                "Executors must be able to resolve the driver; set it to this "
                "container's hostname on the Spark network."
            )
        builder = (
            builder.config("spark.driver.host", driver_host)
            .config("spark.driver.bindAddress", "0.0.0.0")
            .config("spark.cores.max", os.getenv("SPARK_CORES_MAX", "4"))
        )

    for key, value in extra.items():
        builder = builder.config(key.replace("__", "."), value)

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    return session
