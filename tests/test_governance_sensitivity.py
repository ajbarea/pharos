"""The fleet ladder, and the sweep of findings 19-23 built on top of it."""

import pytest
from conftest import artifact
from measure_audit_policy import AUDIT_RUNGS
from measure_authority_anchors import ANCHOR_RUNGS, RUNGS, ladder, majority
from measure_blind_spot import BLIND_RUNGS
from measure_channel_bias import ALPHA, CHANNEL_RUNGS
from measure_governance_sensitivity import SWEEP_PERMUTATIONS, _better

#: What each script's constant tuple was before the ladder replaced it. These are the
#: published compositions: every number in findings 19-23 was measured at one of them.
COMMITTED_AT_NINE = {
    "anchors": (ANCHOR_RUNGS, (4, 5, 6, 7, 9)),
    "audit": (AUDIT_RUNGS, (5, 6, 7)),
    "blind": (BLIND_RUNGS, (0, 3, 5, 7, 8, 9)),
    "channel": (CHANNEL_RUNGS, (0, 1, 2, 3, 4, 5, 7, 9)),
}


@pytest.mark.parametrize("name", sorted(COMMITTED_AT_NINE))
def test_the_ladder_reproduces_every_committed_constant_at_nine(name):
    """The one property the refactor rests on, asserted per script.

    Rewriting four absolute tuples as fleet positions is only safe if it is a no-op at
    the fleet every published number came from. If it is not, every artifact in
    `results/` silently describes a different experiment than the prose does.
    """
    rungs, expected = COMMITTED_AT_NINE[name]
    assert ladder(9, rungs) == expected


@pytest.mark.parametrize("fleet", [5, 9, 15, 25, 51])
def test_the_ladder_lands_on_the_majority_crossing_at_every_fleet(fleet):
    """The crossing is the finding, so a sweep that misses it measures the wrong thing.

    Scaling the old constants by a fixed ratio would not have this property: 5/9 of 25
    rounds to 14 while the majority of 25 is 13, so the composition finding 19 prices
    as "a bare majority" would have been a supermajority at that fleet, under the same
    label.
    """
    assert majority(fleet) in ladder(fleet, ANCHOR_RUNGS)
    assert majority(fleet) in ladder(fleet, AUDIT_RUNGS)


@pytest.mark.parametrize("fleet", [5, 9, 15, 25, 51])
def test_no_rung_falls_outside_the_fleet(fleet):
    """`below` at a fleet of one, `all-but-one` at zero: clamped, not negative."""
    every = ladder(fleet, tuple(RUNGS))
    assert every == tuple(sorted(set(every))), "rungs must be ordered and deduplicated"
    assert all(0 <= rung <= fleet for rung in every)


def test_distinct_rungs_collide_at_small_fleets_and_are_not_double_counted():
    """At five, the majority and two-thirds are both 3; the sweep must see one cell."""
    assert RUNGS["majority"](5) == RUNGS["two-thirds"](5) == 3
    assert ladder(5, ("majority", "two-thirds")) == (3,)


def test_the_sweep_cannot_disarm_the_detector_it_is_checking():
    """A permutation p-value floors at 1/(m+1), so too few permutations detect nothing.

    A 300-permutation smoke run of this sweep reported finding 22 as moved. Nothing had
    moved: the floor was 0.0033 against an alpha of 0.001, so no cell could be detected
    at any effect size, and the sweep manufactured a failure to replicate out of its own
    argument. The guard is a hard stop for that reason -- the artifact would otherwise be
    indistinguishable from a real negative.
    """
    from measure_governance_sensitivity import channel_row

    with pytest.raises(SystemExit, match="floor the p-value"):
        channel_row(9, 300)

    assert 1.0 / (SWEEP_PERMUTATIONS + 1) < ALPHA, (
        "the sweep's own default must clear the floor it refuses others for"
    )


def test_an_unrepaired_threshold_loses_to_every_finite_one():
    """`None` means the budget ran out, not that the policy repaired for free.

    Ordering censored values by coercing them to a number is how a policy that never
    repairs gets credited with the best possible score.
    """
    assert _better(20, 45) is True
    assert _better(45, 20) is False
    assert _better(20, None) is True, "finite beats never-repaired"
    assert _better(None, 20) is False
    assert _better(None, None) is False, "two failures are not a win"
    assert _better(20, 20) is False, "a tie is not a win"


def test_every_governance_finding_survives_the_fleet_it_was_measured_at():
    """The claims findings 19-23 make, checked at every fleet in the committed sweep.

    Each of these was measured at nine analysts and only nine. This is the guard that
    they are properties of the mechanism rather than of that number.
    """
    payload = artifact("governance_sensitivity")
    invariants = payload["invariants"]

    assert len(payload["fleets"]) >= 3, "a sweep of two points is not a multiverse"
    assert invariants["selection_beats_uniform_wherever_a_repair_is_needed"], (
        "finding 20's headline no longer holds at every fleet swept"
    )
    assert invariants["selection_ties_the_oracle_bound"]
    assert invariants["disagreement_collapses_at_unanimity"], (
        "finding 21's premise has changed; findings 22 and 23 answer a question nobody has"
    )
    assert invariants["provenance_recovers_every_corrupted_item"], (
        "finding 23's selection half no longer holds at every fleet swept"
    )
    assert invariants["no_policy_repairs_an_unanchored_label"], (
        "the open half of LAS item 7 has closed somewhere, which is a result and not a "
        "test failure -- but it must be published rather than absorbed silently"
    )


def test_the_cliff_is_a_share_of_the_fleet_and_not_a_bare_majority():
    """Finding 24, and it retires a phrase every earlier finding used.

    Findings 19 to 23 all describe the failure as "a majority holds the wrong standard".
    At nine analysts the estimator recovers at 4 and collapses at 5, so the majority and
    the crossing are the same cell and the phrase was never wrong -- it was never tested
    either. Scanning every composition puts the crossing at a share the bare majority
    exceeds only in small fleets: at fifteen and twenty-five the estimator holds at a
    bare majority where plain consensus, by definition, cannot.

    Asserted as an inequality rather than a value because four fleet sizes bracket the
    share without resolving it, and pinning a number the sweep cannot see would be the
    same overclaim in the other direction.
    """
    bracket = artifact("governance_sensitivity")["cliff_bracket"]

    assert bracket["highest_safe_share"] < bracket["lowest_broken_share"], (
        "a fleet recovers above a share where another breaks; the cliff is not a share"
    )
    assert bracket["fleets_surviving_a_bare_majority"], (
        "no fleet survives a bare majority, which would restore the retired phrasing -- "
        "a result worth publishing, not a test to relax"
    )
    assert bracket["fleets_where_the_cliff_is_at_the_majority"], (
        "the crossing coincides with the majority nowhere, including at nine, where "
        "every committed number in findings 19-23 was measured"
    )


def test_the_cliff_deepens_nowhere_even_where_it_moves():
    """Location and depth come apart, and only the location is fleet-dependent.

    Post-cliff agreement is the same value at every fleet and every composition past the
    crossing. That is what makes this a relabelling of the latent class rather than a
    gradual loss of signal: there is no partial failure to be found between the two
    levels, so a fleet is either identified or it is not.
    """
    bracket = artifact("governance_sensitivity")["cliff_bracket"]
    assert len(bracket["post_cliff_agreement"]) == 1, (
        f"post-cliff agreement now takes several values {bracket['post_cliff_agreement']}; "
        "the failure has a gradient it did not have, which changes the mechanism"
    )


def test_a_composition_that_was_never_broken_is_not_scored_as_unrepaired():
    """The ambiguity the sweep hit at fifteen analysts, kept separable.

    `threshold` returns None both when no budget repaired the fleet and when there was
    nothing to repair. The first run read the second as the first and reported finding
    20 as holding while the crossing row had quietly stopped being a failure at all.
    """
    payload = artifact("governance_sensitivity")
    unbroken = [
        row for entry in payload["audit"] for row in entry["rows"] if row["nothing_to_repair"]
    ]
    assert unbroken, (
        "no swept composition is unbroken at baseline, so the flag is untested here; "
        "at fifteen and twenty-five analysts the bare majority was"
    )
    for row in unbroken:
        assert row["margin"] is None and row["oracle"] is None, (
            "a composition with nothing to repair reports a repair threshold"
        )


def test_the_sweep_visits_the_crossing_and_unanimity_at_every_fleet():
    """A sweep that skipped either end would report an invariant it never tested."""
    payload = artifact("governance_sensitivity")
    for entry in payload["audit"]:
        assert entry["majority"] in entry["compositions"]
        assert any(row["is_crossing"] for row in entry["rows"])
    for entry in payload["blind"]:
        assert max(entry["shares"]) == entry["fleet"], "unanimity is the load-bearing cell"
