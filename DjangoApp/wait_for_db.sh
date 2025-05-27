#!/bin/sh

echo "Waiting for PostgreSQL..."

while ! nc -z postgres_db 5432; do
  sleep 1
done

echo "PostgreSQL started"

echo "Start migrate"
python manage.py migrate

echo "Copy static"
python manage.py collectstatic --noinput


exec "$@"