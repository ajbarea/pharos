"""Structured logging and optional OpenTelemetry, for analysing runs as they happen.

Two hard constraints shape this module.

**Telemetry can never change a measurement.** Pharos generates corpora reproducibly
from a seed, and an experiment must not depend on a collector being reachable. So
absent the packages, or absent configuration, every span becomes a no-op and logging
falls back to stdlib. Nothing here can change a measurement, only report it.

This is a correctness property, not optionality: OpenTelemetry is a core dependency,
because traceability is part of what makes a result checkable rather than a
debugging convenience. What degrades is the *export*, never the number.

**Measurements are structured, not printed.** A corpus's surface baseline or a
sweep's per-fold AUC is data, so it goes out as log attributes and metrics rather
than as an f-string a reader has to parse back. Console output belongs to the CLI;
this is for the record.

Logs carry `trace_id` and `span_id` when a span is active, which is the whole point
of the bridge pattern: one identifier ties a generation, its gate, and its
permutation null together, so a surprising number can be traced to the run that
produced it rather than re-derived.

Enable by setting `PHAROS_OTLP_ENDPOINT` (or calling `configure` directly). Without
it you still get JSON logs, which is the useful default for a research tool.
`docker compose up -d` brings up a collector, Jaeger, and Prometheus to point at.
"""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, override

_LOGGER_NAME = "pharos"
_CONFIGURED = False

#: Set when the OpenTelemetry SDK is importable AND configured. Kept separate from
#: mere importability so an installed-but-unconfigured environment stays silent.
_TRACING_ACTIVE = False
_tracer: Any = None
_meter: Any = None
_histograms: dict[str, Any] = {}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with trace correlation when a span is active.

    Structured output is what makes a run queryable after the fact. Extra fields
    passed via `logger.info(..., extra={...})` are merged into the object, so a
    measurement travels as typed values rather than inside a message string.
    """

    _RESERVED = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    @override
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            # RFC 3339 with milliseconds. `%z` alone gives `-0400`, which strict RFC 3339
            # parsers reject for the missing colon, and second resolution left the gate's
            # own lines unorderable: it emits both probe AUCs, the baseline, and the
            # duration inside one second, so a log that exists to explain a slow run
            # could not say which part of it was slow.
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in self._RESERVED and not key.startswith("_")
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        trace_id, span_id = _current_ids()
        if trace_id:
            payload["trace_id"] = trace_id
            payload["span_id"] = span_id
        return json.dumps(payload, default=str)


def _current_ids() -> tuple[str | None, str | None]:
    """Hex trace and span ids for the active span, or `(None, None)`."""
    if not _TRACING_ACTIVE:
        return None, None
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None, None
        return f"{context.trace_id:032x}", f"{context.span_id:016x}"
    except Exception:  # telemetry must never break a measurement
        return None, None


def configure(
    *,
    service_name: str = "pharos",
    otlp_endpoint: str | None = None,
    level: int = logging.INFO,
    json_logs: bool = True,
) -> bool:
    """Set up logging and, when an endpoint is available, tracing and metrics.

    Returns True when tracing is active. Idempotent: calling twice is harmless, so a
    library entry point and a script can both call it.
    """
    global _CONFIGURED, _TRACING_ACTIVE, _tracer, _meter
    logger = logging.getLogger(_LOGGER_NAME)

    # An environment override so routine DEBUG metrics can be turned on without
    # editing a call site. `record_routine` is invisible at the default level by
    # design, and a developer chasing a corpus-generation question needs a way to see
    # it that does not involve changing code.
    requested = os.environ.get("PHAROS_LOG_LEVEL")
    if requested:
        resolved = logging.getLevelNamesMapping().get(requested.upper())
        if resolved is None:
            logger.warning(
                "telemetry.bad_log_level",
                extra={"event": "telemetry.bad_log_level", "requested": requested},
            )
        else:
            level = resolved

    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONFIGURED = True

    endpoint = otlp_endpoint or os.environ.get("PHAROS_OTLP_ENDPOINT")
    if not endpoint or _TRACING_ACTIVE:
        return _TRACING_ACTIVE

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.info(
            "otel_unavailable",
            extra={"hint": "install the 'otel' extra to export traces", "endpoint": endpoint},
        )
        return False

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(_LOGGER_NAME)

    reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"))
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
    _meter = metrics.get_meter(_LOGGER_NAME)

    _TRACING_ACTIVE = True
    logger.info("otel_configured", extra={"endpoint": endpoint, "service_name": service_name})
    return True


def get_logger() -> logging.Logger:
    """The package logger, configuring JSON output on first use."""
    if not _CONFIGURED:
        configure()
    return logging.getLogger(_LOGGER_NAME)


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    """A span when tracing is active, a no-op otherwise.

    Deliberately swallows nothing: an exception inside the block propagates, and is
    recorded on the span when there is one. Telemetry that hides a failure is worse
    than no telemetry.
    """
    if not _TRACING_ACTIVE or _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as active:
        for key, value in attributes.items():
            active.set_attribute(
                key, value if isinstance(value, int | float | str | bool) else str(value)
            )
        yield


def record(metric: str, value: float, **attributes: object) -> None:
    """Record a measurement as a histogram point, and always as a structured log.

    The log happens whether or not a collector is configured, so a run is analysable
    from its own output alone. That matters for a research tool whose results often
    outlive the collector that watched them.

    Emitted at INFO. Use `record_routine` for an inner operation a caller repeats,
    which is the difference between a log a human reads and one they filter out.
    """
    _emit_metric(metric, value, logging.INFO, attributes)


def record_routine(metric: str, value: float, **attributes: object) -> None:
    """The same measurement, at DEBUG, for an operation a caller performs in bulk.

    The distinction is about volume rather than importance. Generating one corpus is
    worth an INFO line; generating thirty inside a sweep buries every decision the
    sweep made underneath them. This project's fleet-linkage script emitted thirty
    identical corpus lines and nothing at all about the attack it ran, which is how a
    log stops being read.
    """
    _emit_metric(metric, value, logging.DEBUG, attributes)


#: UCUM unit and description per metric, because an instrument without them is not
#: readable by the backend that receives it: a bare number cannot be converted, rendered
#: with an axis, or aggregated against anything else. OTel's metric conventions ask for
#: both, and ask that the unit live here rather than in the name -- which is why
#: `gate.duration_s` was renamed `gate.duration`, the one metric here that carried its
#: unit as a suffix. Durations are seconds; a dimensionless ratio is `1`; a count of
#: things is that thing in braces, singular.
_METRIC_META: dict[str, tuple[str, str]] = {
    "adapter.train_loss": ("1", "Final training loss for a personalization adapter"),
    "budget.participation": ("{contribution}", "Contributions a analyst made under the budget"),
    "consensus.composition": ("1", "Consensus agreement with the world at one fleet composition"),
    "correlated.composition": ("1", "Probability of a wrong majority at one correlation structure"),
    "correlated.understatement": ("1", "How far independence understates that probability"),
    "decode.cross_pass_bound": ("1", "Upper bound on the cross-pass disagreement rate"),
    "decode.unstable_share": ("1", "Share of tasks answering differently across repeats"),
    "difficulty.class_conditional_separation": (
        "1",
        "Routine-class ability gap in logits between wrong and correct reviewers",
    ),
    "difficulty.composition": ("1", "Spread of estimated item difficulty across overlap bands"),
    "fleet.linkage": ("1", "Share of contributors the stream identifies exactly"),
    "fleet_sensitivity.cliff": (
        "{contributor}",
        "Wrong-standard contributors at which consensus collapses, per fleet size",
    ),
    "gate.duration": ("s", "Wall-clock time to run the shortcut gate over a corpus"),
    "gate.null_mean": ("1", "Mean probe AUC under label permutation"),
    "gate.null_z": ("1", "Standard deviations separating the baseline from its null"),
    "gate.probe_auc": ("1", "AUC of one surface probe against the class label"),
    "gate.surface_baseline": ("1", "Best surface-feature AUC, the score to report against"),
    "generate.reports": ("{report}", "Reports rendered for a generated corpus"),
    "inference.cc_rasch": ("{iteration}", "EM iterations CC-Rasch ran before settling"),
    "inference.dawid_skene": ("1", "Estimated prevalence of the significant class"),
    "inference.glad": ("{iteration}", "EM iterations GLAD ran before settling"),
    "learnability.accuracy": ("1", "Share of evaluation tasks answered correctly"),
    "ledger.routing": ("{decision}", "Routing decisions recorded in the ledger"),
    "linkage.control": ("1", "Recovery rate for the control condition"),
    "power.claims_resolved": ("{claim}", "Headline claims their evaluation size can resolve"),
    "review.replayed": ("{task}", "Committed verdicts replayed past the analyst grid"),
    "review_sweep.cells_clearing_floor": (
        "{cell}",
        "Grid cells whose target accuracy clears the majority floor",
    ),
    "tagging.scheme": ("1", "Clearance inferable from a tagged aggregate"),
    "teacher_fleet.inheritance": (
        "1",
        "Share of decisions an adapter reproduces from the teacher that taught it",
    ),
}

#: Anything not in the registry above. `1` is UCUM's unity, which is the honest default
#: for an unannotated number and is wrong loudly rather than quietly if the metric turns
#: out to have a dimension.
_DEFAULT_META = ("1", "")


def _emit_metric(metric: str, value: float, level: int, attributes: dict[str, object]) -> None:
    get_logger().log(
        level,
        metric,
        extra={"metric": metric, "value": value, **{str(k): v for k, v in attributes.items()}},
    )
    if not _TRACING_ACTIVE or _meter is None:
        return
    histogram = _histograms.get(metric)
    if histogram is None:
        unit, description = _METRIC_META.get(metric, _DEFAULT_META)
        histogram = _meter.create_histogram(f"pharos.{metric}", unit=unit, description=description)
        _histograms[metric] = histogram
    histogram.record(
        value,
        {
            k: (v if isinstance(v, int | float | str | bool) else str(v))
            for k, v in attributes.items()
        },
    )


def progress(event: str, **attributes: object) -> None:
    """A structured checkpoint inside a long operation.

    Distinct from `record`, which reports a *result*. This reports that work is
    still moving, which is a different question and the one that matters when a run
    appears stuck. A gate that logs only its final AUC is indistinguishable from a
    gate that has hung, and that ambiguity cost a debugging session on a cluster:
    a job sat at the same byte count for twenty minutes with no way to tell whether
    it was crawling or dead.
    """
    get_logger().info(event, extra={"event": event, **{str(k): v for k, v in attributes.items()}})


def execution_context() -> dict[str, Any]:
    """What a slow run needs explained: how much parallelism is real.

    `os.cpu_count()` reports the machine's CPUs; `sched_getaffinity` reports the
    ones this process may actually use. Under a Slurm cgroup those differ, and
    numerical libraries size their thread pools from the former unless told
    otherwise. The result is a process spawning many times more threads than it has
    cores to run them on, which does not error, does not warn, and simply takes
    forever.

    So the two numbers are reported together, with the thread-limit variables that
    would reconcile them, and `oversubscription_risk` is set when they disagree and
    nothing has capped the pools.
    """
    machine_cpus = os.cpu_count() or 0
    try:
        usable_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # not available on every platform
        usable_cpus = machine_cpus

    thread_vars = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    capped = any(value for value in thread_vars.values())
    return {
        "machine_cpus": machine_cpus,
        "usable_cpus": usable_cpus,
        "thread_limits": thread_vars,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "oversubscription_risk": bool(machine_cpus > usable_cpus and not capped),
    }


def log_execution_context() -> dict[str, Any]:
    """Emit `execution_context`, warning when the numbers do not reconcile."""
    context = execution_context()
    progress("run.context", **context)
    if context["oversubscription_risk"]:
        get_logger().warning(
            "run.oversubscription_risk",
            extra={
                "event": "run.oversubscription_risk",
                "machine_cpus": context["machine_cpus"],
                "usable_cpus": context["usable_cpus"],
                "advice": (
                    "numerical libraries will size thread pools from machine_cpus while only "
                    "usable_cpus are schedulable. Set OMP_NUM_THREADS (and OPENBLAS/MKL) to "
                    "usable_cpus."
                ),
            },
        )
    return context
