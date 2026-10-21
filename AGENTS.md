# pharos

Pharos generates a labeled corpus for measuring a governed disclosure boundary in federated personalization, and gates whether a generated corpus may be used at all. Humans start with [README.md](README.md).

## Commands

`make help` lists every target. Run these before a PR:

```bash
make setup   # uv sync --all-groups, installs the git hooks
make lint    # ruff format --check, ruff check, ty
make test    # the suite with the branch-coverage floor set in the Makefile
make gate    # generates a corpus; exits non-zero when it is unusable
uv run python scripts/measure_analyst_review.py   # replays the committed verdicts as CI does; needs no model
```

`make ci` runs every CI gate in order. `make results` and `make edge` (need Ollama), `scripts/sweep_models.sh`, `scripts/train_adapter.py` (CUDA GPU, a cluster job), `make external-validation` (downloads public corpora), `make explorer-check` (drives Chromium) and `uv run python -m pharos.cli serve` (a server) are long-running or interactive: run them only when the task says to.

## Rules

- Work in a git worktree and open a pull request; `main` is protected.
- Never commit `COMMITS.md` (a local scratchpad).
- `results/*.json` are measurement artifacts: regenerate one with the script that writes it (its `make` target where one exists), never patch a number by hand. Docs tables that restate an artifact come from `make docs-tables`.
- `scenarios/maritime-watch.toml` is frozen and reproduces the published corpus byte for byte (`tests/test_scenario.py`). For a different world, copy it to a new scenario and pass `--scenario`.
- `src/pharos/cases/disclosure.json` is an audit artifact: each case keeps its written pass criterion and `verified` flag, and every reason code needs a case.
- The long comments in `src/pharos/` (`gate.py`, `scenario.py`, `validity.py`, `analyst.py`, `disclosure.py`) each record a leak, an attack or a retracted measurement. Keep them.
- The pre-push hook refuses a push while the branch's last CI run is red; set `PHAROS_PUSH_OVER_RED=1` only when the push is the fix.
- Telemetry is a core dependency and may report a run but never change it.
- Never commit credentials: `.env`, tokens and `secrets/` are ignored on purpose.

## State

`docs/findings.md` holds what has been measured and the script that reproduces each finding; `RESEARCH.md` holds the external citations, each checked against arXiv or a DOI.
`.claude/skill-context.md` holds the facts the techne skills read for this repo.
