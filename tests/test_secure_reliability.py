"""Findings 18 and 19: the estimator under a sum, and what an authority costs."""

import pytest
from measure_authority_anchors import (
    ANCHOR_COUNTS,
    ANCHOR_SEED,
    ANCHOR_SEEDS,
    COMPOSITIONS,
    REPAIRED,
    choose_anchors,
    summarize_thresholds,
)
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


def test_the_anchor_draw_grows_rather_than_being_redrawn():
    """Budget must be the only thing moving across a column of the sweep."""
    ids = [f"T-{i}" for i in range(50)]
    previous: set[str] = set()
    for count in (0, 1, 8, 20, 50):
        current = set(choose_anchors(ids, count, seed=909))
        assert len(current) == count
        assert previous <= current, f"budget {count} redrew rather than extended"
        previous = current


def test_a_budget_past_the_pool_is_refused_rather_than_silently_clipped():
    """A slice would report a budget that was never spent."""
    with pytest.raises(ValueError, match="exceeds"):
        choose_anchors(["a", "b"], 3, seed=909)


def test_the_threshold_is_summarized_over_draws_not_read_off_one():
    spread = summarize_thresholds(5, [30, 5, 12, 20, 8])
    assert (spread.median, spread.lowest, spread.highest) == (12, 5, 30)
    assert (spread.reached, spread.seeds) == (5, 5)


def test_a_draw_that_never_repairs_is_censored_rather_than_counted_as_the_maximum():
    """Sorting unreached last is the whole convention, so it is pinned on both sides."""
    # Two of five never repair: the median is still an observed price, and it is the
    # third of five once the unreached draws are ordered last -- pulled upward by the
    # censoring rather than computed over the reached draws alone, which would report
    # 50 and quietly drop the evidence that two draws did worse than any number here.
    mostly = summarize_thresholds(6, [50, None, 20, None, 80])
    assert mostly.median == 80
    assert (mostly.lowest, mostly.highest) == (20, 80)
    assert mostly.reached == 3

    # Three of five never repair: "usually not reached" is the answer, not a number.
    mostly_not = summarize_thresholds(7, [50, None, None, None, 80])
    assert mostly_not.median is None
    # The draws that did repair are still reported; the range is not erased by the
    # median being censored, because "never, except twice at 50-80" is the finding.
    assert (mostly_not.lowest, mostly_not.highest) == (50, 80)
    assert mostly_not.reached == 2


def test_a_composition_no_draw_repairs_reports_nothing_rather_than_a_bound():
    spread = summarize_thresholds(9, [None, None, None])
    assert (spread.median, spread.lowest, spread.highest) == (None, None, None)
    assert spread.reached == 0
    assert spread.as_dict()["seeds"] == 3


def test_summarizing_no_draws_is_refused():
    """An empty sweep must not summarize to `not reached`, which looks like a result."""
    with pytest.raises(ValueError, match="no draws"):
        summarize_thresholds(5, [])


def test_the_seed_sweep_is_odd_and_the_published_grid_is_one_of_its_draws():
    """An even count would make the median an average of two censored-ordered draws."""
    assert len(ANCHOR_SEEDS) % 2 == 1
    assert len(set(ANCHOR_SEEDS)) == len(ANCHOR_SEEDS)
    assert ANCHOR_SEED in ANCHOR_SEEDS


def test_the_anchor_sweep_is_wide_enough_to_price_every_composition():
    """A sweep that never reaches the answer reports its own bound, not a price."""
    assert max(ANCHOR_COUNTS) >= 180
    assert ANCHOR_COUNTS[0] == 0
    assert list(ANCHOR_COUNTS) == sorted(ANCHOR_COUNTS)
    assert 0.5 < REPAIRED < 1.0
    assert max(COMPOSITIONS) == 9
