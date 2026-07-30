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
	uv run pytest --cov=pharos --cov-report=term-missing

gate:                      ## Generate a corpus and run the shortcut gate on it
	uv run python -m pharos.cli gate
