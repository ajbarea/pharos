"""Telemetry must be informative when configured and invisible when not.

The invariant worth protecting: nothing in this module may change a measurement.
Pharos runs offline and deterministically, so a missing collector, a missing
optional dependency, or a broken exporter has to degrade to silence rather than to
an exception or a different number.
"""

import json
import logging
import re

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


def test_every_emitted_metric_declares_a_unit_and_description():
    """The unit registry has to stay level with the call sites that need it.

    A hand-maintained registry drifts in one direction: a new `record(...)` lands, nobody
    adds the entry, and the instrument ships with the unity default -- which is wrong and
    silent for anything that is not a ratio. Deriving the expected set from the source
    makes the drift fail here instead of in a dashboard.
    """
    import re
    from pathlib import Path

    from pharos.telemetry import _METRIC_META

    root = Path(__file__).resolve().parent.parent
    emitted: set[str] = set()
    for path in [*(root / "src").rglob("*.py"), *(root / "scripts").rglob("*.py")]:
        # `\s*` because a call wrapped across lines by the formatter still emits the
        # metric, and a guard that misses it reports the registry entry as unused --
        # which is the safe direction to fail in, and still a false alarm.
        emitted |= set(re.findall(r'record(?:_routine)?\(\s*"([a-z_.]+)"', path.read_text()))

    assert emitted, "no record() call sites found; the pattern above has gone stale"
    assert emitted <= set(_METRIC_META), (
        f"metrics with no unit declared: {emitted - set(_METRIC_META)}"
    )
    assert set(_METRIC_META) <= emitted, (
        f"registry entries nothing emits: {set(_METRIC_META) - emitted}"
    )


def test_no_metric_name_carries_its_unit_as_a_suffix():
    """OTel: units live in the instrument's metadata, not in its name.

    `gate.duration_s` did, and a reader had to know that `_s` meant seconds while the
    exported instrument said nothing at all.
    """
    from pharos.telemetry import _METRIC_META

    offenders = [m for m in _METRIC_META if re.search(r"_(s|ms|us|ns|b|kb|mb|gb|mib|pct)$", m)]
    assert offenders == [], f"unit encoded in metric name: {offenders}"


def test_timestamps_are_rfc3339_with_subsecond_precision():
    """Second resolution left same-second lines unorderable, which the gate produces."""
    import json
    import logging
    import re as _re

    from pharos.telemetry import JsonFormatter

    record = logging.LogRecord("pharos", logging.INFO, __file__, 1, "x", None, None)
    stamp = json.loads(JsonFormatter().format(record))["timestamp"]
    assert _re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}", stamp), stamp


def _fresh_tally():
    """A tally isolated from whatever the rest of the suite has already logged."""
    from pharos.telemetry import _WarningTally

    return _WarningTally()


def test_the_tally_counts_only_warnings_and_above():
    """INFO is the volume the summary exists to see past, so it must not be counted."""
    import logging

    tally = _fresh_tally()
    logger = logging.getLogger("pharos.test.tally")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(tally)

    for _ in range(2000):
        logger.info("tick", extra={"event": "adapter.eval_progress"})
    for _ in range(24):
        logger.warning("unauth", extra={"event": "hf.unauthenticated"})
    logger.error("boom", extra={"event": "adapter.failed"})

    assert tally.counts == {
        ("WARNING", "hf.unauthenticated"): 24,
        ("ERROR", "adapter.failed"): 1,
    }


def test_the_tally_keys_on_the_event_not_the_message():
    """Records carry a stable `event`; the message is prose and varies per call site."""
    import logging

    tally = _fresh_tally()
    logger = logging.getLogger("pharos.test.tally.event")
    logger.propagate = False
    logger.addHandler(tally)

    logger.warning("accuracy 0.47 below floor", extra={"event": "validity.below_majority"})
    logger.warning("accuracy 0.33 below floor", extra={"event": "validity.below_majority"})
    assert tally.counts == {("WARNING", "validity.below_majority"): 2}

    # No `event` at all: fall back to the message, so a stdlib warning still tallies.
    logger.warning("a bare warning")
    assert tally.counts[("WARNING", "a bare warning")] == 1


def test_the_summary_is_empty_when_nothing_warned():
    """A clean run must print nothing. A summary that always fires is noise itself."""
    import pharos.telemetry as t

    saved = dict(t._TALLY.counts)
    t._TALLY.counts.clear()
    try:
        assert t.format_warning_summary() == ""
    finally:
        t._TALLY.counts.update(saved)


def test_the_summary_orders_errors_first_then_by_count():
    """What a human scanning the tail of a 1.1 MB log needs at the top."""
    import pharos.telemetry as t

    saved = dict(t._TALLY.counts)
    t._TALLY.counts.clear()
    t._TALLY.counts.update(
        {
            ("WARNING", "hf.unauthenticated"): 24,
            ("WARNING", "validity.below_majority"): 12,
            ("ERROR", "adapter.failed"): 1,
        }
    )
    try:
        lines = t.format_warning_summary().splitlines()
        assert "37 warning(s)" in lines[0]
        assert "adapter.failed" in lines[1], "an ERROR must outrank a 24x WARNING"
        assert "hf.unauthenticated" in lines[2]
        assert "validity.below_majority" in lines[3]
    finally:
        t._TALLY.counts.clear()
        t._TALLY.counts.update(saved)


def test_usable_cpus_reports_the_allocation_not_the_machine(monkeypatch):
    """The number a pool may be sized from, which is not what the hardware has.

    `cluster/README.md` records what the difference costs: a 96-core node with eight CPUs
    granted, numerical libraries sizing pools from 96, and a job that sat on one byte of
    output for twenty minutes. Anything choosing a worker count has to ask this, not
    `os.cpu_count`, so all three of its answers are pinned here.
    """
    monkeypatch.setattr(telemetry.os, "cpu_count", lambda: 96)

    # 3.13+: the cross-platform answer, which respects an affinity mask.
    monkeypatch.setattr(telemetry.os, "process_cpu_count", lambda: 8, raising=False)
    assert telemetry.usable_cpus() == 8

    # 3.12 on Linux: no `process_cpu_count`, so fall back to the affinity mask itself.
    monkeypatch.delattr(telemetry.os, "process_cpu_count", raising=False)
    monkeypatch.setattr(telemetry.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}, raising=False)
    assert telemetry.usable_cpus() == 4

    # Neither available: the machine count is the last resort, not the first answer.
    monkeypatch.delattr(telemetry.os, "sched_getaffinity", raising=False)
    assert telemetry.usable_cpus() == 96


def test_usable_cpus_prefers_the_mask_when_the_count_is_undeterminable(monkeypatch):
    """`process_cpu_count` returning None must not collapse the answer to 1.

    It is documented to return None when it cannot tell. Treating that as "one CPU" would
    serialize a job that has eight, which is the same class of wrong answer as claiming 96
    -- just in the direction that looks safe.
    """
    monkeypatch.setattr(telemetry.os, "cpu_count", lambda: 96)
    monkeypatch.setattr(telemetry.os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(telemetry.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}, raising=False)
    assert telemetry.usable_cpus() == 4


def test_execution_context_flags_the_mismatch_it_exists_to_report(monkeypatch):
    """`oversubscription_risk` is the whole point: it fires when the two disagree."""
    monkeypatch.setattr(telemetry.os, "cpu_count", lambda: 96)
    monkeypatch.setattr(telemetry.os, "process_cpu_count", lambda: 8, raising=False)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    context = telemetry.execution_context()
    assert (context["machine_cpus"], context["usable_cpus"]) == (96, 8)
    assert context["oversubscription_risk"] is True

    # Capping the pools is what reconciles them, so the risk clears.
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert telemetry.execution_context()["oversubscription_risk"] is False


def test_execution_context_never_reports_a_machine_smaller_than_the_process(monkeypatch):
    """`os.cpu_count()` returning None must not print a 96-core node as having fewer than one.

    `usable_cpus` floors at 1 and `machine_cpus` used to floor at 0, so the one record
    written to explain a slow run described a machine smaller than the process on it.
    """
    monkeypatch.setattr(telemetry.os, "cpu_count", lambda: None)
    monkeypatch.setattr(telemetry.os, "process_cpu_count", lambda: 4, raising=False)

    context = telemetry.execution_context()
    assert context["machine_cpus"] >= context["usable_cpus"]
    assert (context["machine_cpus"], context["usable_cpus"]) == (4, 4)
    assert context["oversubscription_risk"] is False
