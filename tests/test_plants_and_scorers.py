"""Tests for PlantRegistry, SpecialistScorer, and Adversarial perturbation engines."""

from __future__ import annotations

import pytest

from pharos.adversarial import apply_decoy_injection, apply_lexical_substitution
from pharos.generate import GeneratorConfig, generate
from pharos.plants import PlantRegistry, default_maritime_registry
from pharos.scorers import SpecialistScorer
from pharos.tasks import build_triage_tasks


@pytest.fixture
def sample_tasks():
    reports = generate(GeneratorConfig(seed=7, n_events=20))
    return build_triage_tasks(reports)


def test_plant_registry_default():
    reg = default_maritime_registry()
    assert "maritime_watch_v1" in reg.signatures
    sig = reg.get_signature("maritime_watch_v1")
    assert sig is not None
    assert sig.is_triggered(("course_deviation", "draft_mismatch", "unlit_contact"))
    assert not sig.is_triggered(("course_deviation", "draft_mismatch"))


def test_custom_plant_registry():
    reg = PlantRegistry()
    reg.register_fact("fact_a", "Fact A description", ("term_a",))
    reg.register_fact("fact_b", "Fact B description", ("term_b",))
    sig = reg.register_signature("sig_ab", ("fact_a", "fact_b"))

    assert sig.is_triggered(["fact_a", "fact_b"])
    assert not sig.is_triggered(["fact_a"])


def test_specialist_scorer_triage(sample_tasks):
    scorer = SpecialistScorer()
    task = sample_tasks[0]

    # Test correct prediction
    res_correct = scorer.score_triage(task, predicted_verdict=task.significant)
    assert res_correct.correct
    assert not res_correct.over_escalation
    assert not res_correct.under_escalation

    # Test over escalation
    if not task.significant:
        res_over = scorer.score_triage(task, predicted_verdict=True)
        assert not res_over.correct
        assert res_over.over_escalation


def test_specialist_scorer_provenance():
    scorer = SpecialistScorer()
    res = scorer.score_provenance(
        true_sources=["R-001", "R-002", "R-003"],
        attributed_sources=["R-001", "R-002"],
    )
    assert res.total_sources == 3
    assert res.attributed_sources == 2
    assert pytest.approx(res.recall, 0.01) == 0.67
    assert not res.exact_match


def test_adversarial_perturbations(sample_tasks):
    task = sample_tasks[0]

    lexical_task = apply_lexical_substitution(task, seed=42)
    assert lexical_task.original_task_id == task.task_id
    assert len(lexical_task.perturbed_reports) == len(task.sources)

    decoy_task = apply_decoy_injection(task, seed=42)
    assert decoy_task.original_task_id == task.task_id
    assert len(decoy_task.perturbed_reports) == len(task.sources)
