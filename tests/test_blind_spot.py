"""Finding 21: the corpus finding 20's policy cannot handle, built on purpose."""

import random

import pytest
from measure_blind_spot import BLIND, BUDGETS, ORTHOGONALITY_SLACK, SHARES, blind_fleet

from pharos.analyst import AnalystPolicy, evidence_shown
from pharos.disclosure import DROP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.tasks import build_triage_tasks

EVENTS = 200


@pytest.fixture(scope="module")
def tasks():
    return build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=EVENTS)))


def test_a_sighted_reviewer_is_unchanged_by_the_blind_spot_field():
    """The default must be exactly the old behaviour, or every prior finding moves."""
    plain = AnalystPolicy("plain")
    assert plain.blind_compartment is None
    corpus = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=40)))
    for task in corpus:
        assert plain.evidence_visible_to(task) == evidence_shown(task)


def test_the_blind_reviewer_credits_less_never_more(tasks):
    """A first version returned every fact id and made the blind reviewer see MORE."""
    blind = AnalystPolicy("blind", blind_compartment=BLIND)
    for task in tasks:
        visible = blind.evidence_visible_to(task)
        assert visible <= evidence_shown(task)


def test_the_blind_spot_actually_corrupts_something(tasks):
    """A blind spot that changes no verdict would make finding 21 measure nothing."""
    reference, blind = AnalystPolicy("r"), AnalystPolicy("b", blind_compartment=BLIND)
    rng = random.Random(0)
    changed = [t for t in tasks if reference.verdict_for(t, rng) != blind.verdict_for(t, rng)]
    assert changed, "the blind spot changes no verdict; the experiment is void"
    # Every change is an error, never a lucky correction: crediting less evidence can
    # only move a verdict toward ROUTINE, and the affected tasks are all significant.
    assert all(t.significant for t in changed)


def test_the_corrupted_slice_is_not_the_boundary_slice(tasks):
    """The whole point: corruption picked by provenance, not by difficulty.

    Finding 20's policy works because a threshold error hits the boundary items. If
    this blind spot hit them too, finding 21 would be measuring the same confound it
    was built to escape.
    """
    reference, blind = AnalystPolicy("r"), AnalystPolicy("b", blind_compartment=BLIND)
    rng = random.Random(0)
    changed = [t for t in tasks if reference.verdict_for(t, rng) != blind.verdict_for(t, rng)]
    affected_mean = sum(len(evidence_shown(t)) for t in changed) / len(changed)
    corpus_mean = sum(len(evidence_shown(t)) for t in tasks) / len(tasks)
    assert abs(affected_mean - corpus_mean) <= ORTHOGONALITY_SLACK
    # And it lands on the unambiguous end, which is where a threshold error never does.
    assert affected_mean > corpus_mean


def test_the_fleet_sheds_compartments_so_the_corruption_reaches_the_aggregator():
    """Under the fail-closed default every affected task is escalated and lost.

    This is the defect the first run of finding 21 had: the blind spot reached the
    aggregator on zero tasks and the experiment silently measured nothing.
    """
    fleet = blind_fleet(9, 9)
    assert all(p.release_policy is DROP_COMPARTMENTS for p in fleet)
    assert all(p.blind_compartment is BLIND for p in fleet)


def test_the_fleet_splits_at_the_requested_share():
    fleet = blind_fleet(7, 9)
    assert len(fleet) == 9
    assert sum(1 for p in fleet if p.blind_compartment is BLIND) == 7
    assert sum(1 for p in fleet if p.blind_compartment is None) == 2
    # Every reviewer holds the correct threshold: this is not a fleet of over-escalators.
    assert all(p.escalation_threshold == 3 for p in fleet)


def test_the_sweep_reaches_unanimity():
    """The interesting cell is the one where no dissent remains."""
    assert max(SHARES) == 9
    assert 0 in SHARES
    assert list(SHARES) == sorted(SHARES)
    assert max(BUDGETS) < EVENTS
