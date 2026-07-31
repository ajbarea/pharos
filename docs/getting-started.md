# Getting Started

## Install

```bash
make setup     # uv sync --all-groups
```

Python 3.12 or 3.13. The generation and gating path needs only `numpy` and
`scikit-learn`; everything else is optional or development-only.

## The three commands you will actually use

```bash
make test      # the suite, with a 95% coverage floor
make gate      # generate a corpus and decide whether it is usable
make results   # regenerate every measurement artifact (needs Ollama)
```

`make gate` exits non-zero when a corpus is not usable, so it can gate CI rather
than merely inform a human who may not be reading.

## Reading a gate verdict

```console
$ uv run python -m pharos.cli gate --seed 7 --events 400
surface baseline  0.6588  (ceiling 0.72)
permutation null  0.5047 +/- 0.0361  p95 0.5557  over 20 trials
leak vs null      z = +4.27  significant: True
strict band       0.45 to 0.55  (met: False)
VERDICT           USABLE
```

Three things are worth knowing about that output before it misleads you.

**`strict band ... met: False` is not a failure.** The band is retained as an
ideal, and for content-defined ground truth it is unreachable. The verdict line is
what decides.

**The surface baseline is not noise to be removed.** It is what a model scores
while reading nothing, and every downstream score has to be reported against it
rather than against 0.5. See [the gate](reference/gate.md) for why it cannot be
driven to chance.

**`significant: True` is the good news, not the bad news.** It says the observed
statistic exceeds what label shuffling alone produces, which is how you know the
gate is measuring something rather than reporting its own sampling error.

## Observability

OpenTelemetry is a **core dependency**, not an extra. Traceability is part of what
makes a result checkable: one `trace_id` ties a generation to its gate and its
permutation null, so a surprising number leads back to the run that produced it
rather than having to be re-derived.

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

What telemetry may **never** do is change a measurement. Absent the configuration
or an unreachable collector, every span becomes a no-op and logging falls back to
JSON on stdout; the number is identical either way. That is a correctness property
rather than an optionality one, and CI asserts it on every push.

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
