"""Telemetry must be informative when configured and invisible when not.

The invariant worth protecting: nothing in this module may change a measurement.
Pharos runs offline and deterministically, so a missing collector, a missing
optional dependency, or a broken exporter has to degrade to silence rather than to
an exception or a different number.
"""

import json
import logging

from pharos import telemetry
from pharos.gate import run_gate
from pharos.generate import GeneratorConfig, generate


def test_spans_are_no_ops_without_configuration():
    # Absent tracing, a span must still be a usable context manager.
    with telemetry.span("unconfigured", attribute=1):
        pass


def test_a_span_does_not_swallow_exceptions():
    """Telemetry that hides a failure is worse than no telemetry."""
    try:
        with telemetry.span("raises"):
            raise ValueError("must propagate")
    except ValueError as exc:
        assert str(exc) == "must propagate"
    else:
        raise AssertionError("span swallowed the exception")


def test_logs_are_one_json_object_per_line(caplog):
    formatter = telemetry.JsonFormatter()
    record = logging.LogRecord(
        name="pharos",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="gate.surface_baseline",
        args=(),
        exc_info=None,
    )
    record.metric = "gate.surface_baseline"
    record.value = 0.5867
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "gate.surface_baseline"
    assert payload["metric"] == "gate.surface_baseline"
    assert payload["value"] == 0.5867
    assert payload["level"] == "INFO"


def test_extra_fields_travel_as_typed_values_not_inside_the_message():
    """A measurement must be queryable without parsing a string back."""
    formatter = telemetry.JsonFormatter()
    record = logging.LogRecord(
        name="pharos",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="generate.reports",
        args=(),
        exc_info=None,
    )
    record.value = 1200
    record.seed = 7
    payload = json.loads(formatter.format(record))
    assert payload["value"] == 1200 and isinstance(payload["value"], int)
    assert payload["seed"] == 7


def test_no_trace_ids_when_tracing_is_inactive():
    assert telemetry._current_ids() == (None, None)


def test_configure_is_idempotent_and_adds_one_handler():
    telemetry.configure()
    before = len(logging.getLogger("pharos").handlers)
    telemetry.configure()
    assert len(logging.getLogger("pharos").handlers) == before


def test_configure_reports_false_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("PHAROS_OTLP_ENDPOINT", raising=False)
    assert telemetry.configure() is False


def test_record_emits_a_log_even_with_no_collector(caplog):
    with caplog.at_level(logging.INFO, logger="pharos"):
        telemetry.record("test.metric", 1.5, probe="unit")
    assert any(r.msg == "test.metric" for r in caplog.records)


def test_instrumentation_does_not_change_the_measurement():
    """The same seed must give the same gate result whatever telemetry is doing."""
    reports = generate(GeneratorConfig(seed=3, n_events=120))
    first = run_gate(reports)
    telemetry.configure()
    second = run_gate(generate(GeneratorConfig(seed=3, n_events=120)))
    assert first.auc == second.auc
    assert first.per_probe_auc == second.per_probe_auc


# --- the configured path, exercised with an in-memory exporter ----------------


def _reset() -> None:
    telemetry._TRACING_ACTIVE = False
    telemetry._tracer = None
    telemetry._meter = None
    telemetry._histograms.clear()


def test_spans_export_and_carry_attributes_when_tracing_is_active():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        telemetry._tracer = provider.get_tracer("test")
        telemetry._TRACING_ACTIVE = True
        with telemetry.span("gate.sweep", n_reports=360, note="unit"):
            trace_id, span_id = telemetry._current_ids()
            assert trace_id and len(trace_id) == 32
            assert span_id and len(span_id) == 16
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["gate.sweep"]
        attributes = spans[0].attributes
        assert attributes is not None
        assert attributes["n_reports"] == 360
        assert attributes["note"] == "unit"
    finally:
        _reset()


def test_a_non_primitive_span_attribute_is_stringified():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        telemetry._tracer = provider.get_tracer("test")
        telemetry._TRACING_ACTIVE = True
        with telemetry.span("gate.sweep", labels=frozenset({"SENSOR"})):
            pass
        attributes = exporter.get_finished_spans()[0].attributes
        assert attributes is not None
        assert isinstance(attributes["labels"], str)
    finally:
        _reset()


def test_logs_gain_trace_correlation_inside_an_active_span():
    import json as _json

    from opentelemetry.sdk.trace import TracerProvider

    try:
        telemetry._tracer = TracerProvider().get_tracer("test")
        telemetry._TRACING_ACTIVE = True
        formatter = telemetry.JsonFormatter()
        with telemetry.span("gate.sweep"):
            rec = logging.LogRecord("pharos", logging.INFO, __file__, 1, "m", (), None)
            payload = _json.loads(formatter.format(rec))
        assert len(payload["trace_id"]) == 32
        assert len(payload["span_id"]) == 16
    finally:
        _reset()


def test_metrics_record_into_a_histogram_when_a_meter_exists():
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    try:
        telemetry._meter = MeterProvider(metric_readers=[reader]).get_meter("test")
        telemetry._TRACING_ACTIVE = True
        telemetry.record("gate.surface_baseline", 0.5867, probe="logistic")
        telemetry.record("gate.surface_baseline", 0.61, probe="logistic")
        data = reader.get_metrics_data()
        assert data is not None
        names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "pharos.gate.surface_baseline" in names
        # The histogram is created once and reused across calls.
        assert len(telemetry._histograms) == 1
    finally:
        _reset()


def test_configure_with_an_endpoint_activates_tracing(monkeypatch):
    try:
        assert telemetry.configure(otlp_endpoint="http://localhost:4318") is True
        assert telemetry._TRACING_ACTIVE is True
        # Idempotent: a second call reports the already-active state.
        assert telemetry.configure(otlp_endpoint="http://localhost:4318") is True
    finally:
        _reset()


def test_an_exception_is_formatted_into_the_json_payload():
    import json as _json

    formatter = telemetry.JsonFormatter()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        rec = logging.LogRecord("pharos", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    payload = _json.loads(formatter.format(rec))
    assert "RuntimeError: boom" in payload["exception"]


def test_a_broken_tracer_yields_no_trace_ids_rather_than_raising():
    """Telemetry must never break a measurement, including when it is itself broken."""

    import sys
    from types import ModuleType

    def explode() -> None:
        raise RuntimeError("tracer is broken")

    broken = ModuleType("opentelemetry.trace")
    # setattr rather than attribute assignment: a bare ModuleType has no declared
    # attributes, so a direct assignment reads as an unresolved-attribute error.
    setattr(broken, "get_current_span", explode)  # noqa: B010

    telemetry._TRACING_ACTIVE = True
    saved = sys.modules.get("opentelemetry.trace")
    try:
        sys.modules["opentelemetry.trace"] = broken
        assert telemetry._current_ids() == (None, None)
    finally:
        if saved is not None:
            sys.modules["opentelemetry.trace"] = saved
        else:
            sys.modules.pop("opentelemetry.trace", None)
        telemetry._TRACING_ACTIVE = False


def test_configure_degrades_to_false_when_the_sdk_is_absent(monkeypatch):
    """The normal state for an offline install: endpoint set, packages missing."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    telemetry._TRACING_ACTIVE = False
    try:
        assert telemetry.configure(otlp_endpoint="http://localhost:4318") is False
    finally:
        telemetry._TRACING_ACTIVE = False
