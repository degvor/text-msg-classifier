# Spark runtime shared by the consumer and the trainer.
#
# Replaces `bitnami/spark:3.4.1`, which has been REMOVED from Docker Hub
# (Bitnami moved its catalogue to the frozen `bitnamilegacy/` namespace), so the
# previous build simply fails on a clean machine.
#
# Building on python:3.11-slim + `pip install pyspark` rather than adopting
# another vendor image gives: control of the Python version, no vendored-
# namespace risk, one image for master/worker/consumer/trainer, and a single
# place where the Spark version is pinned. The wheel ships bin/ and jars/, so a
# standalone cluster runs via `spark-class org.apache.spark.deploy.master.Master`
# -- note it ships only a subset of sbin/, so the start-master.sh / start-worker.sh
# daemon wrappers are NOT available.
# Debian release pinned explicitly: `python:3.11-slim` now tracks trixie, which
# dropped openjdk-17. Spark 3.5.x officially supports Java 8/11/17, so bookworm
# is the right floor -- and pinning it keeps the build reproducible.
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless curl procps \
    && rm -rf /var/lib/apt/lists/*

# JAVA_HOME differs by architecture on Debian (…-amd64 vs …-arm64); resolve it
# from the java binary instead of hardcoding one and breaking the other.
RUN ln -sfn "$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" /usr/lib/jvm/current
ENV JAVA_HOME=/usr/lib/jvm/current

# ---- Spark + Python deps ----------------------------------------------------
COPY pyproject.toml /build/
COPY src /build/src
RUN pip install --no-cache-dir "/build[spark,db]" && rm -rf /build

# SPARK_HOME must point at the pip-installed distribution, and PYSPARK_PYTHON
# must be explicit: the original images left the interpreter to whatever PATH
# resolved first, which worked only by coincidence.
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark \
    PYSPARK_PYTHON=/usr/local/bin/python3 \
    PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

# ---- Pinned jars, checksum-verified ----------------------------------------
COPY docker/jars.txt docker/fetch-jars.sh /tmp/
RUN /tmp/fetch-jars.sh /tmp/jars.txt "${SPARK_HOME}/jars" && rm /tmp/jars.txt /tmp/fetch-jars.sh

# ---- Non-root ---------------------------------------------------------------
RUN useradd --create-home --uid 10001 spark \
    && mkdir -p /data /app \
    && chown -R spark:spark /data /app
WORKDIR /app
USER spark

# Services override this; a bare image should not silently do nothing useful.
CMD ["python3", "-c", "import pyspark, sys; print('pyspark', pyspark.__version__, 'python', sys.version)"]
