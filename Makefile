.PHONY: help setup lint test gate results review ci

help:                      ## Show available targets
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:                     ## Sync the dev environment
	uv sync --all-groups

lint:                      ## ruff format --check, ruff check, ty
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check

test:                      ## Run the test suite with coverage
	uv run pytest --cov=pharos --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=92

gate:                      ## Generate a corpus and run the shortcut gate on it
	uv run python -m pharos.cli gate

results:                   ## Regenerate every measurement artifact in results/ (needs Ollama)
	@mkdir -p results
	uv run python scripts/measure_label_fidelity.py --tasks 24 --out results/label_fidelity.json
	uv run python scripts/measure_federation_eligibility.py --out results/federation_eligibility.json
	uv run python scripts/measure_triage_lift.py --out results/triage_lift.json
	uv run python scripts/measure_rule_learnability.py --out results/learnability.json

review:                    ## Replay the committed model verdicts past the analyst grid (no model)
	@mkdir -p results
	uv run python scripts/measure_analyst_review.py --out results/analyst_review.json

external-validation:       ## Re-run the gate claim against three public corpora (downloads ~20k rows)
	@mkdir -p results
	# 12000 rows and 40 null trials, not the script defaults. At the 4,000-row
	# default the HellaSwag null is wide enough that its leak reads z=+1.47 rather
	# than +3.65, and the published claim rests on the larger sample. Encoding the
	# invocation here is what makes that number reproducible.
	uv run --extra external python scripts/validate_gate_externally.py \
		--limit 12000 --null-trials 40 --out results/external_gate_validation.json

ci:                        ## Run every CI gate in order, exactly as the workflow does
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check
	uv run pytest --cov=pharos --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=92
	for seed in 1 7 11 23 101 202 303; do uv run python -m pharos.cli gate --seed $$seed --events 400; done
	uv run python scripts/measure_analyst_review.py
