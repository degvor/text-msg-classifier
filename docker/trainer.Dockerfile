# Trainer. Same Spark base as the consumer, so there is exactly one place the
# Spark and jar versions are pinned -- the previous setup had a 4.4.0 jar baked
# into the image while the trainer asked Ivy for 5.1.3, putting both on the
# classpath at once.
FROM streamcls/spark:dev

COPY --chown=spark:spark services /app/services

ENV PYTHONPATH=/app
CMD ["python3", "-m", "services.trainer.train", "--help"]
