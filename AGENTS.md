# pharos

Pharos generates a labeled corpus for measuring a governed disclosure boundary in federated personalization, and gates whether a generated corpus may be used at all. Humans start with [README.md](README.md).

## Commands

`make help` lists every target. Run these before a PR:

```bash
make setup   # uv sync --all-groups, installs the git hooks
make lint    # ruff format --check, ruff check, ty
make test    # the suite with the branch-coverage floor set in the Makefile
make gate    # generates a corpus; exits non-zero when it is unusable
make review  # replays the committed verdicts; needs no model
```

`make ci` runs every CI gate in order. `make results`, `make edge` and any target marked "needs Ollama", `scripts/train_adapter.py` (CUDA GPU) and `scripts/validate_gate_externally.py` (downloads corpora) are long-running: run them only when the task says to.

## Rules

- Work in a git worktree and open a pull request; `main` is protected.
- Never commit `COMMITS.md` (a local scratchpad).
- `results/*.json` are measurement artifacts: regenerate them with their `make` target, never patch a number by hand. Docs tables that restate an artifact come from `make docs-tables`.
- `scenarios/maritime-watch.toml` is generated; its header says how.
- `src/pharos/cases/disclosure.json` is an audit artifact, not a fixture: each case keeps its written pass criterion and `verified` flag, and every reason code needs a case.
- The long comments in `src/pharos/` (`gate.py`, `scenario.py`, `validity.py`, `analyst.py`, `disclosure.py`) each record a leak, an attack or a retracted measurement. Keep them.
- Telemetry is a core dependency and may report a run but never change it.
- Never commit credentials: `.env`, tokens and `secrets/` are ignored on purpose.

## State

`.claude/skill-context.md` holds the facts the techne skills read for this repo.
