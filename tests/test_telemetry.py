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
