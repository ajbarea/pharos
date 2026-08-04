"""Findings 18 and 19: the estimator under a sum, and what an authority costs."""

import pytest
from measure_authority_anchors import ANCHOR_COUNTS, COMPOSITIONS, REPAIRED, choose_anchors
from measure_secure_reliability import (
    CLIFF_GAP,
    Readership,
    _cliff,
    contributions_for,
    fleet_of,
    measure_readership,
    sweep_equivalence,
)

from pharos.analyst import Proposal
from pharos.disclosure import DROP_COMPARTMENTS, KEEP_COMPARTMENTS
from pharos.fleet import assign_fleet
from pharos.generate import GeneratorConfig, generate
from pharos.labels import declassify
from pharos.tasks import build_triage_tasks

EVENTS = 40


@pytest.fixture(scope="module")
def corpus():
    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=EVENTS)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}
    return tasks, proposals, truth


def test_fleet_composition_splits_at_the_requested_count():
    fleet = fleet_of(4, 9)
    assert len(fleet) == 9
    assert sum(1 for p in fleet if p.name.startswith("wrong-")) == 4
    assert sum(1 for p in fleet if p.name.startswith("right-")) == 5


def test_a_rejection_contributes_no_verdict(corpus):
    """Finding 7's result, preserved: a rejection is an absence, not a vote."""
    tasks, proposals, _ = corpus
    rows = contributions_for(fleet_of(0, 9), tasks, proposals, seed=7)
    assert rows
    assert len(rows) <= len(tasks) * 9
    assert {who for _, who, _ in rows} <= {f"right-{i}" for i in range(9)}


def test_cliff_finds_the_first_real_drop():
    assert _cliff([1.0, 1.0, 1.0, 0.66, 0.66]) == 3
    assert _cliff([1.0, 1.0, 1.0]) is None
    # A drop smaller than the threshold is noise, not a cliff.
    assert _cliff([1.0, 1.0 - CLIFF_GAP / 2]) is None


def test_the_federated_estimator_matches_the_centralized_one_at_every_composition(corpus):
    """Finding 18's first leg, at a corpus small enough to run in the suite."""
    tasks, proposals, truth = corpus
    rows = sweep_equivalence(tasks, proposals, truth, 9)

    assert len(rows) == 10
    assert all(r.label_disagreements == 0 for r in rows)
    assert max(r.max_posterior_gap for r in rows) < 1e-6
    assert all(r.both_converged for r in rows)


def test_the_cliff_does_not_move_when_the_computation_does(corpus):
    """Finding 18's headline: the failure is identifiability, not pooling."""
    tasks, proposals, truth = corpus
    rows = sweep_equivalence(tasks, proposals, truth, 9)
    assert _cliff([r.central for r in rows]) == _cliff([r.federated for r in rows])


def test_the_aggregate_discloses_the_clearance_census_exactly(corpus):
    """Finding 18's residual: nobody is named and every headcount is still exact."""
    tasks, _, _ = corpus
    fleet = assign_fleet(60, seed=11)
    readership = measure_readership(tasks, fleet, policy=DROP_COMPARTMENTS)

    assert readership.labels_probed > 1
    # Exact, not estimated. If this ever falls short the channel has become noisy and
    # the finding's wording has to change with it.
    assert readership.exact_headcounts == readership.labels_probed
    assert readership.headcounts == readership.truth
    assert set(readership.prior_expectation) == set(readership.headcounts)


def test_readership_is_reported_per_join_not_per_compartment_set(corpus):
    """Two joins sharing compartments at different sensitivities are separate rows."""
    tasks, _, _ = corpus
    readership = measure_readership(tasks, assign_fleet(60, seed=11), policy=DROP_COMPARTMENTS)
    assert readership.labels_probed >= readership.compartment_sets_determined
    assert all(" | " in key for key in readership.headcounts)


def test_readership_serializes_every_field():
    readership = Readership(
        labels_probed=2,
        exact_headcounts=2,
        compartment_sets_determined=1,
        headcounts={"a": 1},
        truth={"a": 1},
        prior_expectation={"a": 1.0},
    )
    assert set(readership.as_dict()) == {
        "labels_probed",
        "exact_headcounts",
        "compartment_sets_determined",
        "headcounts",
        "truth",
        "prior_expectation",
    }


def test_anchor_draw_is_reproducible_and_sized():
    ids = [f"T-{i}" for i in range(50)]
    first = choose_anchors(ids, 8, seed=909)
    assert len(first) == 8
    assert first == choose_anchors(ids, 8, seed=909)
    assert choose_anchors(ids, 0, seed=909) == ()
    # A different seed draws a different set, so the sweep is not reading one draw.
    assert first != choose_anchors(ids, 8, seed=5)


def test_the_anchor_sweep_is_wide_enough_to_price_every_composition():
    """A sweep that never reaches the answer reports its own bound, not a price."""
    assert max(ANCHOR_COUNTS) >= 180
    assert ANCHOR_COUNTS[0] == 0
    assert list(ANCHOR_COUNTS) == sorted(ANCHOR_COUNTS)
    assert 0.5 < REPAIRED < 1.0
    assert max(COMPOSITIONS) == 9
