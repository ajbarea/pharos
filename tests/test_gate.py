"""The gate must pass a clean corpus and catch a planted surface tell."""

import pytest

from pharos.gate import SURFACE_FEATURES, run_gate, surface_features
from pharos.generate import GeneratorConfig, generate

REPORTS = generate(GeneratorConfig(seed=7, n_events=400, plant_rate=0.3))


def test_surface_features_are_shape_only():
    features = surface_features(REPORTS[0])
    assert set(features) == set(SURFACE_FEATURES)
    assert all(isinstance(value, float) for value in features.values())


def test_a_clean_corpus_passes_the_gate_at_chance():
    result = run_gate(REPORTS)
    assert result.passed, f"AUC {result.auc:.3f} outside {result.band}: {result.per_probe_auc}"


def test_the_gate_holds_out_whole_centers_rather_than_sampling_at_random():
    result = run_gate(REPORTS)
    assert result.held_out_centers
    assert result.n_train > 0
    assert result.n_test > 0
    train_centers = {r.center.center_id for r in REPORTS} - set(result.held_out_centers)
    assert train_centers, "nothing left to train on"


def test_the_gate_runs_more_than_one_probe_and_reports_each():
    result = run_gate(REPORTS)
    assert set(result.per_probe_auc) == {"logistic", "gradient_boosting"}
    # The verdict is the probe furthest from chance, never an average.
    assert result.auc in set(result.per_probe_auc.values())


def test_an_injected_length_tell_fails_the_gate():
    """Length confounds are the canonical benchmark shortcut, so the gate must catch one."""
    tampered = [
        r.with_text(r.text + " Additional corroborating detail was appended." * 6)
        if r.is_plant
        else r
        for r in REPORTS
    ]
    result = run_gate(tampered)
    assert not result.passed
    assert result.auc > 0.55


def test_an_injected_punctuation_tell_fails_the_gate():
    """A subtler tell than length: the same content, different surface texture."""
    tampered = [r.with_text(r.text.replace(".", " ;")) if r.is_plant else r for r in REPORTS]
    result = run_gate(tampered)
    assert not result.passed


def test_a_split_without_both_classes_raises_rather_than_passing():
    background_only = [r for r in REPORTS if not r.is_plant]
    with pytest.raises(ValueError, match="both classes"):
        run_gate(background_only)


def test_holding_out_every_center_raises():
    with pytest.raises(ValueError, match="cannot hold out"):
        run_gate(REPORTS, holdout_centers=4)
