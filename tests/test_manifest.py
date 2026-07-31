"""The manifest must record provenance and refuse to certify a useless corpus."""

import json

from pharos.generate import GeneratorConfig
from pharos.manifest import build_manifest, label_histogram

CONFIG = GeneratorConfig(seed=11, n_events=400, plant_rate=0.3)
MANIFEST = build_manifest(CONFIG, null_trials=12)


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
    assert payload["usable"] is True


def test_usable_requires_variance_a_measured_null_and_a_bounded_baseline():
    assert MANIFEST.usable

    flat = build_manifest(CONFIG, null_trials=12)
    object.__setattr__(flat, "label_histogram", {"OPEN[]": flat.n_reports})
    assert not flat.usable, "a single-cell histogram must never be usable"


def test_a_manifest_without_a_permutation_null_cannot_certify():
    """An unmeasured baseline is not a small baseline."""
    unmeasured = build_manifest(CONFIG, null_trials=0)
    assert unmeasured.gate.null_trials == 0
    assert unmeasured.gate.leak_is_significant is None
    assert not unmeasured.usable


def test_a_baseline_over_the_ceiling_cannot_certify():
    tampered = build_manifest(CONFIG, null_trials=12)
    object.__setattr__(tampered, "max_surface_baseline", 0.50)
    assert not tampered.usable, "shape explaining too much must never certify"


def test_label_histogram_keys_name_level_and_compartments():
    from pharos.generate import generate

    histogram = label_histogram(generate(GeneratorConfig(seed=3, n_events=60)))
    assert all("[" in key and key.split("[")[0].isupper() for key in histogram)


def test_manifest_names_the_code_that_generated_it():
    """A corpus figure is worth what its provenance is worth."""
    payload = json.loads(build_manifest(CONFIG, null_trials=4).to_json())
    assert set(payload["code_provenance"]) == {"pharos_version", "git_commit", "git_dirty"}


def test_manifest_provenance_carries_no_clock():
    """Two manifests from one seed must compare equal, so no timestamp may leak in."""
    first = build_manifest(CONFIG, null_trials=4).to_json()
    second = build_manifest(CONFIG, null_trials=4).to_json()
    assert first == second
