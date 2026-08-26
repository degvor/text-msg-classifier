# Streaming consumer. Same Spark base as the trainer, so the Spark and jar
# versions are pinned in exactly one place.
FROM streamcls/spark:dev

COPY --chown=spark:spark services /app/services

ENV PYTHONPATH=/app
CMD ["python3", "-m", "services.consumer.main"]
