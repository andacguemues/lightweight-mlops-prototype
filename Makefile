.PHONY: install format lint pipeline test validate verify api docker-build docker-up docker-down

install:
	uv sync --locked

format:
	uv run ruff format src tests scripts

lint:
	uv run ruff check src tests scripts

pipeline:
	uv run dvc repro

test:
	uv run pytest -v

validate:
	uv run python scripts/validate_artifact.py

verify: format lint pipeline test validate

api:
	uv run uvicorn mlops_prototype.api.main:app --app-dir src --reload --host 127.0.0.1 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down