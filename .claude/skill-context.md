# skill-context — pharos

Repo-specific facts the techne skills read. Logic lives in the skills; only facts
belong here.

## repo

- name: pharos
- kind: research software (labeled corpus generator + acceptance gate) with a
  Zensical docs site. Public.
- default_branch: main
- description: generates a labeled corpus for measuring a governed disclosure
  boundary in federated personalization, and gates whether a generated corpus may
  be used at all
- package_root: `src/pharos/` (src-layout, single package)
- language: Python (>=3.12,<3.15); CI matrix covers 3.12, 3.13, 3.14
- toolchain: uv (canonical); ruff (format + lint), ty (types), pytest, pip-audit
- cli_entrypoint: `uv run python -m pharos.cli {gate,export,models,serve}`
- runner: none. Targets run directly via `make`; there is no `logs/dev-<ts>-*.log`
  archive convention here, so the audit's log-reconciliation phase is N/A.
- has: Docker (OTel Collector + Jaeger + Prometheus, `docker-compose.yml`), a
  minimal FastAPI explorer (`src/pharos/web.py`, `ui` dependency group), no Rust.
  Runtime deps: numpy, scikit-learn, opentelemetry (core, deliberately not optional).
  Extras (`[project.optional-dependencies]`): `train` (torch/peft/transformers, GPU
  only), `external` (datasets, for the external gate validation), `otel` (no-op,
  retained for compatibility). Dependency groups: `dev`, `croissant`, `ui`.
  `train` and `external` are extras rather than groups on purpose: CI runs
  `uv sync --all-groups`, which would otherwise pull a CUDA torch into every run.
- cluster: RIT Research Computing. Jobs in `cluster/`; sync with
  `scripts/sync_cluster.sh`, which fetches a commit rather than copying files so
  artifacts can name the code that produced them.

## audit

### Phase 1 — Setup

- `make setup` → `uv sync --all-groups`

### Phase 2 — Fix (one-way door)

- `uv run ruff format .` then `uv run ruff check . --fix`

### Phase 3 — Lint

- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run ty check`

### Phase 4 — Test

- `make test` → `pytest --cov=pharos --cov=scripts --cov-branch --cov-fail-under=92`
- Coverage spans `scripts/` as well as the library, because the scripts produce
  every published number. Orchestration bodies are excluded via
  `[tool.coverage.report] exclude_also`; the pure logic is not.
- Margin over the floor is thin, so an uncovered addition fails the build rather than
  merely lowering a number. The floor is in the Makefile and the current total is
  whatever `make test` last printed; neither is repeated here, because a count written
  down in two places is a count that will disagree with itself.

### Phase 5 — Gates

- `make gate` → generates a corpus and exits non-zero when it is unusable
- `make review` → replays the committed triage verdicts past the analyst grid.
  Needs no model and no network, and refuses to run when a `results/triage_lift-*.json`
  artifact disagrees with the corpus this commit generates. That refusal is the
  drift guard for finding 7, and it runs in the `shortcut-gate` CI job.
- `uv run pip-audit`

### do_not_run (interactive / long-running)

- `make results` — needs Ollama serving a model; the label-fidelity pass alone is
  hundreds of sequential model calls. `make review` is *not* in this list: it is model-free
  and safe to run.
- `scripts/sweep_models.sh` — same, across every installed model.
- `scripts/train_adapter.py` — requires a CUDA GPU; runs as a cluster job.
- `scripts/validate_gate_externally.py` — downloads public corpora from the HF Hub.
- `uv run python -m pharos.cli serve` — starts a web server and does not return.

## ci_audit

- workflows: `.github/workflows/ci.yml` (lint-and-test on 3.12/3.13/3.14,
  shortcut-gate), `.github/workflows/docs.yml` (Zensical build + Pages deploy),
  `.github/workflows/sibling-links.yml` (weekly, plus any push to
  `tests/test_docs_claims.py`: clones `kourai-khryseai` beside this repository and runs
  the one exemption test the per-PR gate can only skip). Not a required check, and it is
  the only workflow whose failure arrives by notification rather than on a pull request.
- required checks on `main`: all three `lint-and-test` legs plus `shortcut-gate`
  (which also replays the analyst grid over the committed verdicts). 3.14 was
  added late, and until 2026-08-15 it was the only leg not required — while also
  being the strictest coverage denominator, since a bare annotation is not an
  executable statement there. The Codecov upload moved to the same leg for the
  same reason: the number the project publishes should be its least flattering one.
- `cancel-in-progress` is scoped to pull requests. It was unconditional once, and
  6 of 15 runs were cancelled on main; one real failure went unobserved because the
  run that would have shown it fixed was itself cancelled.
- The shortcut-gate job is blocking and is a release condition, not a formality: a
  corpus whose surface baseline is unmeasured, insignificant against its own null,
  or above the ceiling cannot support a triage claim.
- expected noise: one `rdflib` DeprecationWarning via `mlcroissant`, upstream, once per
  matrix leg. `pip-audit` is currently clean.
- the coverage denominator is interpreter-dependent, so one floor is three bars. Under
  3.14 a bare annotation is not an executable statement (deferred annotations), so
  dataclass fields leave the denominator: the same commit measures 92.66% on 3.12 and
  3.13 and 92.26% on 3.14. The strictest leg is the one not required on `main`, and the
  leg that uploads to Codecov is the most generous. See the comment beside the pytest
  step in `ci.yml`.

## slop_ground_truth

- `src/pharos/gate.py`, `src/pharos/scenario.py`, `src/pharos/validity.py`,
  `src/pharos/analyst.py`, and `src/pharos/disclosure.py` carry long explanatory
  comments **on purpose**. Each documents a leak the gate caught, an
  attack the loader refuses, a measurement that had to be retracted, or the reason a
  simulated reviewer is a parameter grid rather than a prompted persona, or why a
  compartment shortfall escalates while a level shortfall refuses. They are the
  reason a constraint exists and must not be trimmed as verbosity.
- `src/pharos/cases/disclosure.json` is an **audit artifact**, not a fixture. Each
  case carries a written pass criterion and a `verified` flag, and tests assert both
  are present and that every reason code is exercised. Adding a reason code without
  a case fails the suite.
- `scenarios/maritime-watch.toml` is generated, not hand-written. Its header says so.

## scan_scope

- include: `src/`, `tests/`, `scripts/`, `cluster/`, `docs/`, `scenarios/`
- exclude: `site/` (Zensical output), `.venv/`, `export/`, `adapter-out/`
- `results/*.json` are tracked but are data, never edited by hand. Regenerate with
  `make results` rather than patching a number.

## docs_site

- generator: Zensical (`zensical.toml`), a dev-group dependency
- build: `uv run zensical build --clean` → `site/`
- preview: `python3 -m http.server -d site 8000`
- deploy: `.github/workflows/docs.yml` → <https://ajbarea.github.io/pharos/>
- overrides: `overrides/main.html` (Open Graph tags; no `og:image`, since there is
  no card artwork and a 404 renders worse than an absent tag)
- nav lives in `zensical.toml`; a page not listed there ships unreachable.
