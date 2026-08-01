# Getting Started

## Install

```bash
make setup     # uv sync --all-groups
```

Python 3.12 to 3.14. Generation and gating need only `numpy` and `scikit-learn`;
everything else is optional or development-only.

## Everyday commands

```bash
make test      # the suite, with a 92% branch-coverage floor
make gate      # generate a corpus and decide whether it is usable
make results   # regenerate every measurement artifact (needs Ollama)
```

`make gate` exits non-zero when a corpus is not usable, so it can block CI.

## Reading a gate verdict

```console
$ uv run python -m pharos.cli gate --seed 7 --events 400
surface baseline  0.6588  (ceiling 0.72)
permutation null  0.5047 +/- 0.0361  p95 0.5557  over 20 trials
leak vs null      z = +4.27  significant: True
strict band       0.45 to 0.55  (met: False)
VERDICT           USABLE
```

Three parts of that output are easy to misread.

**`strict band ... met: False` is not a failure.** The band is an ideal, and it is
unreachable for content-defined ground truth. The verdict line decides.

**The surface baseline is not noise to be removed.** It is what a model scores
while reading nothing, and every downstream score has to be reported against it
rather than against 0.5. See [the gate](reference/gate.md) for why it cannot be
driven to chance.

**`significant: True` is good news.** The observed statistic exceeds what label
shuffling alone produces, so the gate is measuring something real rather than its
own sampling error.

## The explorer

```bash
uv sync --group ui
uv run python -m pharos.cli serve     # http://127.0.0.1:8080
```

Five tabs over one page: generate a corpus and read labelled reports, ask whether
one label dominates another, run a triage task against a model from the registry,
run the gate, and put a proposed verdict in front of the analyst grid. It exists so
the ideas can be understood without reading Python.

The review tab is the one worth a minute. Pick a task whose proposed release is
blocked at the aggregator ceiling and every reviewer objects, but only the one who
sheds compartments hands back a correction that can actually leave. That is
[finding 7](findings.md#7-review-is-abundant-what-it-costs-is-correctness) in a
single screen.

Every endpoint returns the objects the Python API produces, so the page is a client
rather than a second implementation -- a label shown in the UI came from
`pharos.labels`, not from a formatting routine written twice. It also loads nothing
from a CDN, because a testbed that promises to run offline should not have a front
door that needs internet.

Corpus size is capped at 400 events and the gate tab runs fewer null trials than the
published runs, so its numbers are noisier and the response says so. Reproduce
published numbers with the CLI, not here.

## Observability

OpenTelemetry is a **core dependency**, not an extra. One `trace_id` ties a
generation to its gate and its permutation null, so a surprising number leads back
to the run that produced it.

```bash
docker compose up -d
export PHAROS_OTLP_ENDPOINT=http://localhost:4318
uv run python -m pharos.cli gate
```

| Service | Where | Role |
| --- | --- | --- |
| OpenTelemetry Collector | `localhost:4318` | Ingest. The only endpoint Pharos knows about |
| Jaeger v2 | <http://127.0.0.1:16686> | Traces |
| Prometheus | <http://127.0.0.1:9090> | Metrics |

The collector is the ingest point rather than a backend, so switching to
ClickHouse, SigNoz, or a hosted vendor is an edit to `docker/otel-collector.yaml`
and changes no Python.

Telemetry may never change a measurement. With no endpoint configured, or an
unreachable collector, every span becomes a no-op and logging falls back to JSON on
stdout; the number is identical either way. CI asserts this on every push.

Every measurement goes out as a structured log line carrying typed fields, so a
surface baseline or a per-fold AUC can be queried rather than parsed back out of
a message:

```json
{"message": "gate.surface_baseline", "metric": "gate.surface_baseline",
 "value": 0.5867, "n_reports": 1200, "n_folds": 4}
```

## Optional: metadata validation

```bash
uv sync --group croissant
uv run pytest tests/test_croissant_validation.py
```

Kept in its own dependency group because it pulls pandas and roughly two dozen
packages. See [releasing a corpus](releasing.md).
