# Development entrypoints. Spark work runs in Docker because there is no local JRE.
.DEFAULT_GOAL := help
.PHONY: help env install lint fmt typecheck test test-unit test-integration test-spark \
        data data-info train score compare cluster build up down logs ps shell-spark clean-pyc

VENV    := .venv
PY      := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
COMPOSE := docker compose

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
	$(PY) -m streamcls score

compare: ## A/B two seed-matched runs: make compare BASELINE=1 ADAPTIVE=2
	$(PY) -m streamcls compare $(BASELINE) $(ADAPTIVE)

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

test-spark: ## Tests needing a JVM, run inside the Spark image
	docker run --rm -e POSTGRES_PASSWORD=test -e SPARK_LOCAL_IP=127.0.0.1 \
	  -v $(PWD)/tests:/app/tests:ro -v $(PWD)/pyproject.toml:/app/pyproject.toml:ro \
	  streamcls/spark:dev sh -c "pip install -q pytest && python -m pytest tests/unit -m spark -q -p no:cacheprovider"

test: test-unit ## Alias for the fast suite

build: ## Build all images
	$(COMPOSE) build

up: ## Start the stack
	$(COMPOSE) up -d

down: ## Stop the stack and drop volumes
	$(COMPOSE) down -v

logs: ## Tail logs (S=service to narrow)
	$(COMPOSE) logs -f $(S)

ps: ## Show container status
	$(COMPOSE) ps

shell-spark: ## Interactive shell in the Spark image
	docker run --rm -it --env-file .env -v $(PWD)/src:/app/src -v $(PWD)/data:/data \
	  streamcls/spark:dev bash

clean-pyc: ## Remove bytecode caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

