# Streaming consumer. Same Spark base as the trainer, so the Spark and jar
# versions are pinned in exactly one place. The base is built separately
# (`make spark-image`) because compose does not order builds by FROM; CI passes
# its own tag through the build arg.
ARG SPARK_BASE_IMAGE=streamcls/spark:dev
FROM ${SPARK_BASE_IMAGE}

COPY --chown=spark:spark services /app/services

ENV PYTHONPATH=/app
CMD ["python3", "-m", "services.consumer.main"]
