"""The fleet ladder, and the sweep of findings 19-23 built on top of it."""

import pytest
from conftest import artifact
from measure_audit_policy import AUDIT_RUNGS
from measure_authority_anchors import ANCHOR_RUNGS
from measure_channel_bias import CHANNEL_RUNGS
from measure_governance_sensitivity import SWEEP_PERMUTATIONS, _better

from pharos.governance import ALPHA, BLIND_RUNGS, RUNGS, ladder, majority

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
    payload = artifact("governance_sensitivity")
    bracket = payload["cliff_bracket"]

    assert bracket["fleets_where_the_cliff_is_at_the_majority"], (
        "the crossing coincides with the majority nowhere, including at nine, where "
        "every committed number in findings 19-23 was measured"
    )
    assert not payload["invariants"]["the_cliff_is_at_the_majority_at_every_fleet"], (
        "the crossing is at the majority everywhere, which would restore the retired "
        "phrasing -- a result worth publishing, not a test to relax"
    )

    low, high = bracket["breaking_share_range"]
    assert low < high, (
        "the crossing takes one value across every fleet and draw swept, which would "
        "make it the constant finding 24 first claimed it was"
    )
    assert 0.5 <= low <= high <= 1.0, "a crossing outside (0.5, 1] is not a wrong majority"


def test_the_cliff_has_no_gradient_within_a_draw():
    """The mechanism claim, at the only scope where it survived the draw sweep.

    Past the crossing, agreement is one value at every composition above it -- so the
    failure is a relabelling of the latent class rather than a gradual loss of signal,
    and a fleet is either identified or it is not.

    Asserted per draw rather than across them, because that is where it holds. The
    committed artifact once reported a single global depth of 0.6598; over eight draws
    the level takes eight values, and reading the one-draw constant as a property was
    half of finding 24's retraction.
    """
    payload = artifact("governance_sensitivity")
    checked = 0
    for entry in payload["cliff"]:
        for draw in entry["per_draw"]:
            broken = draw["lowest_broken"]
            if not broken:
                continue
            above = {
                row["agreement"] for row in draw["grid"] if row["n_wrong"] >= broken["n_wrong"]
            }
            assert len(above) == 1, (
                f"fleet {entry['fleet']} draw {draw['draw']}: agreement past the crossing "
                f"takes several values {sorted(above)}; the failure has a gradient it did "
                "not have, which changes the mechanism"
            )
            checked += 1
    assert checked >= 8, "too few draws carried a crossing for this to have tested anything"

    assert len(payload["cliff_bracket"]["post_cliff_agreement"]) > 1, (
        "the depth is one value across every draw, which would restore the constant "
        "finding 24 first published -- publish it rather than relaxing the retraction"
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
