# Getting Started

## Quick Installation

!!! tip "Requirements"
    Pharos supports **Python 3.12 to 3.14**. Generation and gating require `numpy`, `scikit-learn` and the OpenTelemetry API/SDK — the last is a runtime dependency rather than an extra, because a published number has to be traceable to the run that produced it, and a missing collector degrades to silence rather than to a different result. The trainer, the external validator and the explorer are optional.

```bash
make setup     # uv sync --all-groups
```

---

## Core Commands

| Command | Purpose | Notes |
| :--- | :--- | :--- |
| `make test` | Runs the test suite | 92% branch-coverage floor enforced |
| `make gate` | Generates a corpus & runs shortcut gating | Exits non-zero if corpus fails validation (blocks CI) |
| `make results` | Regenerates all measurement artifacts | Requires Ollama serving models locally |

---

## Understanding Gate Verdicts

Run the CLI gate command to inspect synthetic corpus quality:

```console
$ uv run python -m pharos.cli gate --seed 7 --events 400
surface baseline  0.6545  (ceiling 0.72)
permutation null  0.4979 +/- 0.0300  p95 0.5437  over 20 trials
leak vs null      z = +5.22  significant: True
strict band       0.45 to 0.55  (met: False)
VERDICT           USABLE
```

!!! info "Key Metrics Explained"

    * **`strict band ... met: False` is NOT a failure**: The strict band represents a theoretical ideal that content-defined ground truth cannot achieve. The `VERDICT` line is the ultimate authority.
    * **Surface Baseline**: The accuracy a model achieves without reading task content. Downstream models must be benchmarked against this baseline, not against random guess (0.50).
    * **`significant: True`**: Indicates the observed signal is statistically distinct from label shuffling. The gate is measuring genuine task structure rather than noise.

---

## Visual Explorer UI

!!! note "Offline Browser Explorer"
    You can open the static explorer directly in your browser: **[Open Explorer](explorer/index.html){ target=_blank }** (no installation or backend process required).

To run locally with the live model execution tab enabled:

```bash
uv sync --group ui
uv run python -m pharos.cli serve     # Available at http://127.0.0.1:8080
```

### Explorer Tabs Overview

1. 📜 **Corpus**: Generate datasets and inspect labelled maritime reports.
2. 📐 **Lattice**: Explore sensitivity levels, need-to-know compartments, and dominance rules.
3. 🤖 **Model**: Run live triage tasks against registered models (local server mode only).
4. 🛡️ **Gate**: Execute shortcut detection and inspect permutation null distributions.
5. 👥 **Review**: Test proposed verdicts against the simulated analyst grid.

!!! tip "Hosted vs. Live Explorer"
    * **Hosted Static UI**: Built as a frozen client with zero runtime backend dependencies. Computed deterministically at build time for seeds 1, 7, and 101.
    * **Live Local UI**: Connects to your local Python process and Ollama daemon, enabling dynamic model benchmarking.

---

## Observability & Telemetry

OpenTelemetry tracing is built into the core pipeline. A single `trace_id` links corpus generation, gate scoring, and permutation trials.

```bash
# Launch telemetry stack (Collector, Jaeger, Prometheus)
docker compose up -d

# Direct telemetry output
export PHAROS_OTLP_ENDPOINT=http://localhost:4318
uv run python -m pharos.cli gate
```

| Service | Endpoint | Role |
| :--- | :--- | :--- |
| **OpenTelemetry Collector** | `localhost:4318` | Central ingest point |
| **Jaeger v2** | [http://127.0.0.1:16686](http://127.0.0.1:16686) | Distributed tracing UI |
| **Prometheus** | [http://127.0.0.1:9090](http://127.0.0.1:9090) | Metrics collection & dashboarding |

!!! note "The collector is optional; the log is not"
    With no OTLP endpoint configured, every span becomes a no-op and measurements still
    go to `stdout` as structured JSON. A run is therefore analysable from its own output
    alone, which matters because a result usually outlives the collector that watched it.
    What degrades without a collector is the *export*, never the number.

```json
{"timestamp": "2026-08-03T14:53:48.121-04:00", "level": "INFO", "logger": "pharos", "message": "gate.surface_baseline", "metric": "gate.surface_baseline", "value": 0.6545, "n_reports": 1200, "n_folds": 4}
```

Timestamps are RFC 3339 with milliseconds because the gate emits both probe AUCs, the
baseline, and its duration inside the same second. `trace_id` and `span_id` are added to
every line when a span is active, so one identifier ties a generation, its gate, and its
permutation null together.

---

## Optional Extensions

```bash
# Install Croissant metadata support (includes pandas and validation tools)
uv sync --group croissant
uv run pytest tests/test_croissant_validation.py
```

