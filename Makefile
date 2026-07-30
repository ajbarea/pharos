.PHONY: help setup lint test gate

help:                      ## Show available targets
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:                     ## Sync the dev environment
	uv sync --all-groups

lint:                      ## ruff format --check, ruff check, ty
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check

test:                      ## Run the test suite with coverage
	uv run pytest --cov=pharos --cov-report=term-missing --cov-fail-under=95

gate:                      ## Generate a corpus and run the shortcut gate on it
	uv run python -m pharos.cli gate

ci:                        ## Run every CI gate in order, exactly as the workflow does
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check
	uv run pytest --cov=pharos --cov-report=term-missing --cov-fail-under=95
	for seed in 1 7 11 23 101 202 303; do uv run python -m pharos.cli gate --seed $$seed --events 400; done
