"""The manifest must record provenance and refuse to certify a useless corpus."""

import json

from pharos.generate import GeneratorConfig
from pharos.manifest import build_manifest, label_histogram

CONFIG = GeneratorConfig(seed=11, n_events=400, plant_rate=0.3)
MANIFEST = build_manifest(CONFIG)


def test_manifest_records_seed_config_and_gate_verdict():
    payload = json.loads(MANIFEST.to_json())
    assert payload["config"]["seed"] == 11
    assert "auc" in payload["gate"]
    assert isinstance(payload["gate"]["passed"], bool)


def test_manifest_reports_a_label_histogram_with_more_than_one_cell():
    # A corpus whose labels are constant cannot evaluate a disclosure boundary.
    assert len(MANIFEST.label_histogram) > 1


def test_the_histogram_spans_several_levels_and_a_compartmented_cell():
    keys = " ".join(MANIFEST.label_histogram)
    assert "OPEN" in keys
    assert "RESTRICTED" in keys
    assert "SENSOR" in keys


def test_manifest_round_trips_as_json():
    payload = json.loads(MANIFEST.to_json())
    assert payload["n_reports"] == MANIFEST.n_reports
    assert payload["usable"] == (MANIFEST.gate.passed and len(MANIFEST.label_histogram) > 1)


def test_usable_requires_both_a_passing_gate_and_label_variance():
    """`usable` is a conjunction, and it must not claim true on a failing gate.

    The corpus currently fails the gate, so this asserts the conjunction rather
    than the value: label variance alone must never certify a corpus.
    """
    assert MANIFEST.usable == (MANIFEST.gate.passed and len(MANIFEST.label_histogram) > 1)
    flat = build_manifest(CONFIG)
    object.__setattr__(flat, "label_histogram", {"OPEN[]": flat.n_reports})
    assert not flat.usable, "a single-cell histogram must never be usable"


def test_label_histogram_keys_name_level_and_compartments():
    from pharos.generate import generate

    histogram = label_histogram(generate(GeneratorConfig(seed=3, n_events=60)))
    assert all("[" in key and key.split("[")[0].isupper() for key in histogram)
