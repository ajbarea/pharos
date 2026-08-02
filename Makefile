.PHONY: help setup lint test gate results review sweep power linkage consensus tagged edge budget correlated logcheck docs-tables explorer ci

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
	uv run python scripts/measure_decode_stability.py --out results/decode_stability.json

sweep:                     ## Target accuracy across the reviewer parameter grid (no model)
	@mkdir -p results
	uv run python scripts/measure_review_sweep.py --out results/review_sweep.json

power:                     ## What each evaluation size can resolve (no model)
	@mkdir -p results
	uv run python scripts/measure_power.py --out results/power.json

linkage:                   ## What the fleet's contribution stream leaks about analysts (no model)
	@mkdir -p results
	uv run python scripts/measure_fleet_linkage.py --out results/fleet_linkage.json

consensus:                 ## Whether reliability survives pooling contributors (no model)
	@mkdir -p results
	uv run python scripts/measure_consensus_reliability.py --out results/consensus_reliability.json

tagged:                    ## Whether a reliability tag can replace identity (no model)
	@mkdir -p results
	uv run python scripts/measure_tagged_aggregation.py --out results/tagged_aggregation.json

edge:                      ## What the agent costs on laptop-class hardware (needs Ollama)
	@mkdir -p results
	uv run python scripts/measure_edge_cost.py --out results/edge_cost.json

budget:                    ## What a privacy budget buys against the linkage channel (no model)
	@mkdir -p results
	uv run python scripts/measure_privacy_budget.py --out results/privacy_budget.json

correlated:                ## What finding 12's cliff costs when analysts correlate (no model)
	@mkdir -p results
	uv run python scripts/measure_correlated_fleets.py --out results/correlated_fleets.json

logcheck:                  ## Run the model-free measurements and summarise what they logged
	uv run python scripts/logcheck.py

docs-tables:               ## Refresh the docs tables that restate a committed artifact
	uv run python scripts/sync_docs_tables.py

explorer:                  ## Freeze the explorer into docs/explorer for static hosting
	uv run python scripts/build_static_explorer.py --out docs/explorer

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
	uv run python scripts/measure_review_sweep.py
	uv run python scripts/measure_power.py
	uv run python scripts/measure_fleet_linkage.py
	uv run python scripts/measure_consensus_reliability.py
	uv run python scripts/measure_tagged_aggregation.py
	uv run python scripts/measure_privacy_budget.py
	uv run python scripts/sync_docs_tables.py --check
