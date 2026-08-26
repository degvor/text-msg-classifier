# Stream producer. Pure Python -- no JVM, so this stays small.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml /app/
COPY src /app/src
RUN pip install --no-cache-dir "/app[producer,db]"

COPY services/producer /app/services/producer

RUN useradd --create-home --uid 10002 producer \
    && mkdir -p /data && chown -R producer:producer /data /app
USER producer

CMD ["python", "-m", "services.producer.main"]
