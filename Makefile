.PHONY: help setup lint test gate results ci

help:                      ## Show available targets
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:                     ## Sync the dev environment
	uv sync --all-groups

lint:                      ## ruff format --check, ruff check, ty
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check

test:                      ## Run the test suite with coverage
	uv run pytest --cov=pharos --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=88

gate:                      ## Generate a corpus and run the shortcut gate on it
	uv run python -m pharos.cli gate

results:                   ## Regenerate every measurement artifact in results/ (needs Ollama)
	@mkdir -p results
	uv run python scripts/measure_label_fidelity.py --out results/label_fidelity.json
	uv run python scripts/measure_federation_eligibility.py --out results/federation_eligibility.json
	uv run python scripts/measure_triage_lift.py --out results/triage_lift.json
	uv run python scripts/measure_rule_learnability.py --out results/learnability.json

ci:                        ## Run every CI gate in order, exactly as the workflow does
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check
	uv run pytest --cov=pharos --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=88
	for seed in 1 7 11 23 101 202 303; do uv run python -m pharos.cli gate --seed $$seed --events 400; done
