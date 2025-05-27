#!/bin/bash

echo "Starting Spark consumer with dependency checks..."

check_kafka() {
  nc -z -w 1 kafka 9092 > /dev/null 2>&1
  return $?
}

echo "Waiting for Kafka to be ready..."
for i in {1..30}; do
  if check_kafka; then
    echo "Kafka is ready. Starting Spark consumer..."
    break
  fi
  echo "Kafka is not ready yet. Waiting..."
  sleep 5
done

if ! check_kafka; then
  echo "Timeout waiting for Kafka. Exiting..."
  exit 1
fi

sleep 5

python3 /app/spark_kafka_consumer.py