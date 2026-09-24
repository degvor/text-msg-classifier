# Development entrypoints. Spark work runs in Docker because there is no local JRE.
.DEFAULT_GOAL := help
.PHONY: help env install lint fmt typecheck test test-unit test-integration test-spark \
        data data-info train score compare matrix report cluster spark-image \
        spark-image-present build up down logs ps shell-spark clean-pyc

VENV    := .venv
PY      := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
COMPOSE := docker compose

# Consumer and trainer build FROM this tag. Compose neither orders builds by
# FROM nor builds profiled services, so it is built explicitly first.
SPARK_IMAGE := streamcls/spark:dev

# Host-side commands reach the stack through the ports docker-compose.override.yml
# publishes. Environment variables beat .env in pydantic-settings, so this
# overrides POSTGRES_HOST=db without editing the file.
HOST_ENV := POSTGRES_HOST=localhost POSTGRES_PORT=$(or $(POSTGRES_HOST_PORT),55432) \
            KAFKA_BOOTSTRAP_SERVERS=localhost:$(or $(KAFKA_HOST_PORT),9094)

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example (never overwrites)
	@test -f .env && echo ".env already exists, leaving it alone" \
	|| (cp .env.example .env && echo "wrote .env - review the password before any real deployment")

install: ## Create the venv and install streamcls with dev extras
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(VENV) -e ".[dev,db,producer]"

data: ## Download AG News (idempotent, checksum-verified)
	$(PY) -m streamcls.corpus download

data-info: ## Show split sizes, balance and leakage check
	$(PY) -m streamcls.corpus splits

train: ## Train on AG News and register the model (A=algorithm, PROMOTE=1)
	docker compose --profile manual run --rm trainer \
	  python3 -m services.trainer.train --algorithm $(or $(A),logreg) $(if $(PROMOTE),--promote,)

score: ## Score finished runs and print the results table
	$(HOST_ENV) $(PY) -m streamcls score

compare: ## A/B two seed-matched runs: make compare BASELINE=1 ADAPTIVE=2
	$(HOST_ENV) $(PY) -m streamcls compare $(BASELINE) $(ADAPTIVE)

matrix: ## Run an experiment matrix unattended (SPEC=experiments/matrix.toml, ARGS=--dry-run)
	$(HOST_ENV) $(PY) -m streamcls matrix $(or $(SPEC),experiments/matrix.toml) $(ARGS)

report: ## Aggregate results/<NAME>/runs.jsonl into summary tables (NAME=thesis-v1)
	$(PY) -m streamcls report results/$(or $(NAME),thesis-v1)

cluster: ## Start a real Spark standalone cluster (N=worker count)
	$(COMPOSE) --profile cluster up -d --scale spark-worker=$(or $(N),2)
	@echo "Spark master UI: http://localhost:$${SPARK_UI_PORT:-8080}"
	@echo "Point jobs at it with SPARK_MASTER=spark://spark-master:7077"

lint: ## ruff check
	$(RUFF) check src services web tests

fmt: ## ruff format + autofix
	$(RUFF) format src services web tests
	$(RUFF) check --fix src services web tests

typecheck: ## mypy over the shared package
	$(VENV)/bin/mypy

test-unit: ## Pure-Python tests. No JVM, no Docker.
	$(PYTEST) tests/unit -m "not integration and not spark"

test-integration: ## Tests needing Kafka + Postgres (starts them)
	$(COMPOSE) up -d db kafka
	$(PYTEST) tests/integration -m integration --ds=config.settings.dev

test-spark: spark-image-present ## Tests needing a JVM, run inside the Spark image
	docker run --rm -e POSTGRES_PASSWORD=test -e SPARK_LOCAL_IP=127.0.0.1 \
	  -v $(PWD)/tests:/app/tests:ro -v $(PWD)/pyproject.toml:/app/pyproject.toml:ro \
	  $(SPARK_IMAGE) sh -c "pip install -q pytest && python -m pytest tests/unit -m spark -q -p no:cacheprovider"

test: test-unit ## Alias for the fast suite

spark-image: ## Build the shared Spark base image (consumer and trainer build FROM it)
	docker build -t $(SPARK_IMAGE) -f docker/spark.Dockerfile .

# Builds the base image only when the tag is missing; `make spark-image` forces it.
spark-image-present:
	@docker image inspect $(SPARK_IMAGE) >/dev/null 2>&1 || $(MAKE) spark-image

build: spark-image ## Build all images
	$(COMPOSE) build

up: spark-image-present ## Start the stack
	$(COMPOSE) up -d

down: ## Stop the stack and drop volumes
	$(COMPOSE) down -v

logs: ## Tail logs (S=service to narrow)
	$(COMPOSE) logs -f $(S)

ps: ## Show container status
	$(COMPOSE) ps

shell-spark: spark-image-present ## Interactive shell in the Spark image
	docker run --rm -it --env-file .env -v $(PWD)/src:/app/src -v $(PWD)/data:/data \
	  $(SPARK_IMAGE) bash

clean-pyc: ## Remove bytecode caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

