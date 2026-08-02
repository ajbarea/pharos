"""The stream-level disclosure the per-item gates do not see."""

import pytest

from pharos.disclosure import DROP_COMPARTMENTS, KEEP_COMPARTMENTS
from pharos.fleet import (
    FLEET_CEILING,
    Clearance,
    Contribution,
    Mitigation,
    _jaccard,
    apply_k_anonymity,
    apply_pooling,
    apply_rarity_suppression,
    apply_subsample,
    assign_fleet,
    candidate_clearances,
    contribute,
    link,
)
from pharos.generate import GeneratorConfig, generate
from pharos.labels import Compartment, Sensitivity
from pharos.tasks import build_triage_tasks

SENSOR = frozenset({Compartment.SENSOR})
ALL_COMPARTMENTS = frozenset(Compartment)


@pytest.fixture(scope="module")
def tasks():
    return build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=60)))


@pytest.fixture(scope="module")
def fleet():
    return assign_fleet(40, seed=11)


def test_candidate_space_is_the_full_product():
    space = candidate_clearances()
    assert len(space) == len(Sensitivity) * 2 ** len(Compartment)
    assert len({subset for _, subset in space}) == 2 ** len(Compartment)


def test_assign_fleet_is_deterministic_and_ids_are_unique():
    first = assign_fleet(25, seed=3)
    assert first == assign_fleet(25, seed=3)
    assert first != assign_fleet(25, seed=4)
    assert len({c.analyst_id for c in first}) == 25


def compartmented(tasks):
    """A task that is genuinely above the corpus floor on both axes.

    Selected rather than assumed: a test that silently falls back to an
    uncompartmented task would assert nothing while still passing.
    """
    task = next(
        (
            t
            for t in tasks
            if any(r.label.compartments for r in t.sources)
            and max(r.label.sensitivity for r in t.sources) > Sensitivity.OPEN
        ),
        None,
    )
    assert task is not None, "corpus produced no compartmented task above OPEN"
    return task


def test_can_read_requires_every_source(tasks):
    task = compartmented(tasks)
    assert Clearance("top", Sensitivity.RESTRICTED, ALL_COMPARTMENTS).can_read(task)

    # A holder at the top level with no compartments fails on need-to-know alone.
    assert not Clearance("uncleared", Sensitivity.RESTRICTED, frozenset()).can_read(task)

    # And a holder of every compartment still fails below the level.
    assert not Clearance("low", Sensitivity.OPEN, ALL_COMPARTMENTS).can_read(task)


def test_can_read_is_all_sources_not_any(tasks):
    """The conjunction rule means a partial view is a different task, not a smaller one."""
    mixed = next(
        (t for t in tasks if len({frozenset(r.label.compartments) for r in t.sources}) > 1),
        None,
    )
    assert mixed is not None, "corpus produced no task spanning two compartment cells"

    union = frozenset().union(*(r.label.compartments for r in mixed.sources))
    proper = next(
        fs for fs in (frozenset(r.label.compartments) for r in mixed.sources) if fs != union
    )
    # Holding a proper subset of what the task spans is not enough, by construction.
    assert proper < union
    assert not Clearance("partial", Sensitivity.RESTRICTED, proper).can_read(mixed)


def test_contribute_respects_both_gates(tasks, fleet):
    dropped = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    kept = contribute(fleet, tasks, policy=KEEP_COMPARTMENTS)
    # Finding 2's ruling governs volume here exactly as it governs eligibility.
    assert len(kept) < len(dropped)
    assert all(c.analyst_id == c.pseudonym for c in dropped)


def test_jaccard_scores_no_evidence_as_zero():
    assert _jaccard(frozenset(), frozenset()) == 0.0
    assert _jaccard(frozenset({"a"}), frozenset({"a"})) == 1.0
    assert _jaccard(frozenset({"a", "b"}), frozenset({"a"})) == pytest.approx(0.5)


def test_attack_beats_the_prior_and_is_monotone_in_clearance(tasks, fleet):
    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    linkages = link(stream, tasks, fleet, policy=DROP_COMPARTMENTS)
    assert len(linkages) == len(fleet)

    by_id = {c.analyst_id: c for c in fleet}
    top = [x for x in linkages if by_id[x.analyst_id].sensitivity == Sensitivity.RESTRICTED]
    bottom = [x for x in linkages if by_id[x.analyst_id].sensitivity == Sensitivity.OPEN]
    if top and bottom:
        top_rate = sum(x.exact for x in top) / len(top)
        bottom_rate = sum(x.exact for x in bottom) / len(bottom)
        # The finding: the fleet identifies the analysts with the most to protect.
        assert top_rate > bottom_rate


def test_a_tie_is_never_scored_as_an_identification(tasks, fleet):
    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    for linkage in link(stream, tasks, fleet, policy=DROP_COMPARTMENTS):
        if linkage.anonymity_set > 1:
            assert not linkage.exact


def test_keep_compartments_leaves_nobody_identifiable(tasks, fleet):
    """Finding 2's fail-closed ruling buys perfect unlinkability, and costs the fleet."""
    stream = contribute(fleet, tasks, policy=KEEP_COMPARTMENTS)
    linkages = link(stream, tasks, fleet, policy=KEEP_COMPARTMENTS)
    assert not any(x.exact for x in linkages)


def test_silent_analysts_are_reported_separately(tasks):
    """Unlinkable because they contributed nothing is not a control working.

    The corpus always carries a few uncompartmented tasks that anyone can read, so
    an analyst is made silent here by restricting the task pool to the compartmented
    ones rather than by hoping the draw produces one.
    """
    pool = [t for t in tasks if any(r.label.compartments for r in t.sources)]
    assert pool, "corpus produced no compartmented tasks"

    lonely = (Clearance("A-999", Sensitivity.OPEN, frozenset()),)
    stream = contribute(lonely, pool, policy=DROP_COMPARTMENTS)
    assert stream == (), "expected an uncleared analyst to reach nothing"

    linkage = link(stream, pool, lonely, policy=DROP_COMPARTMENTS)[0]
    assert linkage.silent
    assert not linkage.exact
    assert linkage.anonymity_set == 0


def test_k_anonymity_drops_only_rare_tasks():
    stream = (
        Contribution("A-000", "A-000", "T-1", True),
        Contribution("A-001", "A-001", "T-1", True),
        Contribution("A-002", "A-002", "T-2", False),
    )
    assert len(apply_k_anonymity(stream, 1)) == 3
    kept = apply_k_anonymity(stream, 2)
    assert {c.task_id for c in kept} == {"T-1"}
    assert apply_k_anonymity(stream, 3) == ()


def test_rarity_suppression_keeps_the_reachable_half():
    stream = (
        Contribution("A-000", "A-000", "T-1", True),
        Contribution("A-001", "A-001", "T-1", True),
        Contribution("A-002", "A-002", "T-2", False),
    )
    kept = apply_rarity_suppression(stream, 0.5)
    assert {c.task_id for c in kept} == {"T-1"}
    # Deterministic: the same input yields the same suppression every time.
    assert apply_rarity_suppression(stream, 0.5) == kept
    # Never suppresses everything, which would report protection by deletion.
    assert apply_rarity_suppression(stream, 0.0) != ()


def test_subsample_is_bounded_and_seeded(tasks, fleet):
    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    half = apply_subsample(stream, 0.5, seed=3)
    assert apply_subsample(stream, 0.5, seed=3) == half
    assert 0 < len(half) < len(stream)
    assert apply_subsample(stream, 0.0, seed=3) == ()


def test_pooling_costs_no_volume_and_hides_everyone(tasks, fleet):
    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    pooled = apply_pooling(stream)
    assert len(pooled) == len(stream)
    assert {c.pseudonym for c in pooled} == {"POOLED"}
    # Ground truth survives the control; only the adversary's view of it changes.
    assert [c.analyst_id for c in pooled] == [c.analyst_id for c in stream]

    linkages = link(pooled, tasks, fleet, policy=DROP_COMPARTMENTS)
    assert not any(x.exact for x in linkages)
    assert all(x.anonymity_set >= len(fleet) for x in linkages if not x.silent)


def test_mitigation_names_are_stable():
    assert {m.value for m in Mitigation} == {"none", "k_anonymity", "subsample", "pooled"}


def test_fleet_ceiling_is_releasable_to_everyone():
    assert FLEET_CEILING.compartments == frozenset()
    assert FLEET_CEILING.sensitivity == Sensitivity.OPEN


def test_identifiability_ceiling_bounds_any_attack(tasks):
    """The structural number the finding's prose rests on, computed rather than quoted."""
    from pharos.fleet import identifiability_ceiling

    ceiling = identifiability_ceiling(tasks, policy=DROP_COMPARTMENTS)
    assert ceiling["candidate_clearances"] == len(Sensitivity) * 2 ** len(Compartment)
    # Collapsing can only lose distinctions, never invent them.
    assert 1 <= ceiling["distinct_reachable_sets"] <= ceiling["candidate_clearances"]
    assert ceiling["uniquely_identifying_sets"] <= ceiling["distinct_reachable_sets"]
    # Some clearance is uniquely identifiable, which is the finding's premise.
    assert ceiling["uniquely_identifying_sets"] > 0
    # And some class hides more than one, or the anonymity-set column is meaningless.
    assert ceiling["largest_anonymity_class"] > 1


def test_keeping_compartments_raises_the_ceiling(tasks):
    """The fail-closed ruling protects structurally, not just on the drawn fleet."""
    from pharos.fleet import identifiability_ceiling

    dropped = identifiability_ceiling(tasks, policy=DROP_COMPARTMENTS)
    kept = identifiability_ceiling(tasks, policy=KEEP_COMPARTMENTS)
    assert kept["distinct_reachable_sets"] <= dropped["distinct_reachable_sets"]
    assert kept["uniquely_identifying_sets"] <= dropped["uniquely_identifying_sets"]


def test_the_attack_is_invariant_to_every_verdict(tasks, fleet):
    """The finding's headline claim, enforced rather than asserted in prose.

    "Reads no content" is the whole of what makes this result unanswerable by
    redaction, so flipping every verdict in the stream must change nothing. If a
    future change lets a verdict reach the scorer, this fails.
    """
    from dataclasses import replace as _replace

    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    flipped = tuple(_replace(c, verdict=not c.verdict) for c in stream)
    assert [c.verdict for c in flipped] != [c.verdict for c in stream]

    original = link(stream, tasks, fleet, policy=DROP_COMPARTMENTS)
    assert link(flipped, tasks, fleet, policy=DROP_COMPARTMENTS) == original
