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
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
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
    """
    get_logger().info(
        metric,
        extra={"metric": metric, "value": value, **{str(k): v for k, v in attributes.items()}},
    )
    if not _TRACING_ACTIVE or _meter is None:
        return
    histogram = _histograms.get(metric)
    if histogram is None:
        histogram = _meter.create_histogram(f"pharos.{metric}")
        _histograms[metric] = histogram
    histogram.record(
        value,
        {
            k: (v if isinstance(v, int | float | str | bool) else str(v))
            for k, v in attributes.items()
        },
    )
