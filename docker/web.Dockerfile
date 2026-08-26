# Django dashboard and control plane.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=config.settings.dev \
    DJANGO_STATIC_ROOT=/app/staticfiles

WORKDIR /app

COPY pyproject.toml /app/
COPY src /app/src
RUN pip install --no-cache-dir "/app[web]"

COPY web /app/web

RUN useradd --create-home --uid 10003 web \
    && mkdir -p /data /app/staticfiles \
    && chown -R web:web /data /app
USER web
WORKDIR /app/web

EXPOSE 8000
# gunicorn was already a declared dependency; the previous image installed it
# and then ran `manage.py runserver` -- the dev server -- as its entrypoint.
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 3 --access-logfile -"]
