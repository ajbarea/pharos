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
  minimal FastAPI explorer (`src/pharos/web.py`, optional `ui` extra), no Rust.
  Runtime deps: numpy, scikit-learn, opentelemetry (core, deliberately not optional).
  Extras: `ui` (fastapi), `train` (torch/peft/transformers, GPU only), `otel` (no-op,
  retained for compatibility). Dependency groups: `dev`, `croissant`.
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

- `make test` → `pytest --cov=pharos --cov=scripts --cov-fail-under=90`
- Coverage spans `scripts/` as well as the library, because the scripts produce
  every published number. Orchestration bodies are excluded via
  `[tool.coverage.report] exclude_also`; the pure logic is not.
- Margin is thin: the suite currently lands near 90%, so an uncovered addition
  fails the build rather than merely lowering a number.

### Phase 5 — Gates

- `make gate` → generates a corpus and exits non-zero when it is unusable
- `uv run pip-audit`

### do_not_run (interactive / long-running)

- `make results` — needs Ollama serving a model; the label-fidelity pass alone is
  216 sequential model calls.
- `scripts/sweep_models.sh` — same, across every installed model.
- `scripts/train_adapter.py` — requires a CUDA GPU; runs as a cluster job.
- `scripts/validate_gate_externally.py` — downloads public corpora from the HF Hub.
- `uv run python -m pharos.cli serve` — starts a web server and does not return.

## ci_audit

- workflows: `.github/workflows/ci.yml` (lint-and-test on 3.12/3.13/3.14,
  shortcut-gate), `.github/workflows/docs.yml` (Zensical build + Pages deploy)
- required checks on `main`: `lint-and-test (3.12)`, `lint-and-test (3.13)`,
  `shortcut-gate`. Note 3.14 is in the matrix but not yet required, since it was
  added after protection was configured.
- `cancel-in-progress` is scoped to pull requests. It was unconditional once, and
  6 of 15 runs were cancelled on main; one real failure went unobserved because the
  run that would have shown it fixed was itself cancelled.
- The shortcut-gate job is blocking and is a release condition, not a formality: a
  corpus whose surface baseline is unmeasured, insignificant against its own null,
  or above the ceiling cannot support a triage claim.
- expected noise: one `rdflib` DeprecationWarning via `mlcroissant`, upstream.
  `pip-audit` is currently clean.

## slop_ground_truth

- `src/pharos/gate.py`, `src/pharos/scenario.py`, and `src/pharos/validity.py` carry
  long explanatory comments **on purpose**. Each documents a leak the gate caught, an
  attack the loader refuses, or a measurement that had to be retracted. They are the
  reason a constraint exists and must not be trimmed as verbosity.
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
