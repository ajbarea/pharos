"""Findings 20 to 23 against the corpus draw, and the denominators that keeps honest."""

from conftest import artifact


def test_the_seed_is_a_sweepable_dimension_of_every_governance_script():
    """The defect finding 26 exists because of, asserted so it cannot come back.

    All four scripts hard-coded `SEED = 7` and took no `--seed`, so the corpus was not a
    parameter of this project's headline governance results. It is the same shape as the
    `--fleet` defect finding 24 found -- a sweepable dimension held by a constant sized
    for one draw of it -- and this is the guard that a new measurement script does not
    reintroduce it.
    """
    import importlib
    from pathlib import Path

    for name in (
        "measure_audit_policy",
        "measure_blind_spot",
        "measure_channel_bias",
        "measure_authority_anchors",
    ):
        # Read the script's source rather than calling main(): each runs for minutes.
        module = importlib.import_module(name)
        source = module.__file__ or ""
        assert source
        text = Path(source).read_text(encoding="utf-8")
        assert '"--seed"' in text, f"{name} takes no --seed; the corpus is not sweepable"
        assert "seed=SEED" not in text.split("args = parser.parse_args()")[-1], (
            f"{name} still reads the module constant after parsing, so --seed is accepted "
            "and then ignored -- the exact shape of the --fleet defect"
        )


def test_selection_ties_the_oracle_bound_in_every_draw():
    """Finding 20's deep claim, and the most robust result in the governance set.

    A policy that ties the bound means no better selection rule exists on this signal.
    That is a stronger statement than beating a uniform draw, and unlike the headline it
    survives every corpus swept.
    """
    payload = artifact("corpus_sensitivity")
    assert payload["invariants"]["selection_ties_the_oracle_bound_in_every_draw"], (
        "selection no longer ties the oracle bound somewhere; finding 20 stops being a "
        "bound and becomes a comparison, which is a different and weaker claim"
    )
    for entry in payload["by_composition"]:
        assert entry["ties"] == entry["draws"], (
            f"{entry['n_wrong']} of 9 ties the bound in {entry['ties']}/{entry['draws']}"
        )


def test_the_headline_that_moved_is_recorded_as_moved():
    """Finding 23's 1.00, and finding 20's beats-uniform at seven of nine.

    Both were published unconditionally and neither holds in every draw. Asserted false
    rather than deleted, because an invariant that quietly stops being checked is how the
    number gets requoted.
    """
    payload = artifact("corpus_sensitivity")
    invariants = payload["invariants"]
    assert not invariants["provenance_finds_every_corrupted_item_in_every_draw"], (
        "provenance now finds every corrupted item in every draw, which would restore "
        "finding 23's original headline -- publish it rather than relaxing the retraction"
    )
    assert not invariants["selection_beats_uniform_in_every_draw"], (
        "selection now beats a uniform draw everywhere, a stronger result than published"
    )
    assert not invariants["the_auditable_pool_is_a_constant"], (
        "the auditable pool is one value across every draw, which would restore the 97 "
        "the script documented as a property of the design"
    )


def test_the_advantage_survives_even_where_the_absolute_score_does_not():
    """What finding 23 should be quoted on, and the reason the retraction is partial.

    The comparison is the finding: provenance recovers most corrupted items while every
    policy reading disagreement is at chance. That holds in every draw. Only the 1.00
    moved.
    """
    payload = artifact("corpus_sensitivity")
    blind = payload["blind"]
    assert blind, "no draw hosted the blind-spot experiment; nothing here is tested"
    for row in blind:
        assert row["disagreement_policies_at_chance"], (
            f"draw {row['seed']}: a disagreement policy is above chance at unanimity, so "
            "finding 21's premise moved and finding 23 answers a question nobody has"
        )
        assert row["channel_hit_rate"] > 0.5, (
            f"draw {row['seed']}: provenance recovers {row['channel_hit_rate']}, which is "
            "no longer an advantage worth quoting over a policy at 0.25"
        )


def test_the_negative_half_holds_in_every_draw():
    """The open question finding 23 leaves, and the twelve-month deliverable behind it."""
    payload = artifact("corpus_sensitivity")
    assert payload["invariants"]["no_policy_repairs_an_unanchored_label"], (
        "a policy now repairs an unanchored label somewhere; the residual obstacle would "
        "be one of selection after all, which changes what is left to build"
    )


def test_a_draw_that_cannot_host_the_control_is_excluded_not_counted_against():
    """Finding 21's precondition, kept separable from a result.

    The blind-spot corpus needs a channel orthogonal to item difficulty. Where it is not,
    the script refuses, and folding that refusal into the denominator would score the
    finding down for a corpus on which it was never tested.
    """
    payload = artifact("corpus_sensitivity")
    hosted = payload["draws_hosting_the_blind_spot"]
    attempted = payload["draws_attempted"]
    assert hosted == len(payload["blind"]), (
        "the hosted count and the rows disagree, so every rate scored against it is wrong"
    )
    assert hosted < attempted, (
        "every draw hosts the experiment, so the exclusion path is untested here; at seed "
        "23 the PARTNER compartment was entangled with difficulty and the script refused"
    )
    assert hosted >= 3, "too few draws host the experiment for any rate to mean anything"


def test_the_budget_ladder_is_truncated_to_the_pool_it_draws_from():
    """The defect that made this script exit non-zero on half the draws.

    `select` refuses a budget past the auditable pool rather than clipping it, which is
    right -- a threshold reported at an untested budget is worse than a missing row. So
    the ladder has to shrink instead, and the artifact has to say when it did.
    """
    payload = artifact("corpus_sensitivity")
    truncated = [row for row in payload["audit"] if row["ladder_truncated"]]
    assert truncated, (
        "no draw truncated the ladder, so the fix is untested here; four of eight draws "
        "have a pool below the ladder's top rung of 95"
    )
    for row in payload["audit"]:
        assert max(row["budgets"]) <= row["auditable_pool"], (
            f"draw {row['seed']}: a budget exceeds the {row['auditable_pool']} auditable "
            "tasks, which `select` refuses outright"
        )
        assert row["budgets"] == [
            b for b in row["budgets_requested"] if b <= row["auditable_pool"]
        ], (
            "the used ladder is not the requested one filtered by the pool, so it is not "
            "a truncation but a different ladder"
        )
