"""Finding 20: which items an authority should rule on, and a fallible one."""

import pytest
from measure_audit_policy import (
    AUTHORITY_ERROR,
    BUDGETS,
    DEPLOYABLE,
    POLICIES,
    ServerObservation,
    authority_ruling,
    observe,
    select,
    threshold,
)
from measure_audit_policy import Row as AuditRow


def _view(votes, seen, posterior):
    return ServerObservation(votes, seen, posterior)


def test_margin_is_zero_on_a_dead_heat_and_one_on_unanimity():
    view = _view({"a": 2.0, "b": 4.0, "c": 0.0}, {"a": 4.0, "b": 4.0, "c": 4.0}, {})
    assert view.margin("a") == 0.0
    assert view.margin("b") == 1.0
    assert view.margin("c") == 1.0
    # A task nobody reported on has no split to measure, and must not read as a dead
    # heat -- that would make it the *first* thing an uncertainty policy audits.
    assert view.margin("absent") == 1.0


def test_observe_reads_only_the_two_sums_the_protocol_reveals():
    partitioned = {
        "a": [("T-1", True), ("T-2", False)],
        "b": [("T-1", True)],
        "c": [("T-1", False), ("T-2", False)],
    }
    view = observe(partitioned)
    assert view.votes["T-1"] == 2.0
    assert view.seen["T-1"] == 3.0
    assert view.seen["T-2"] == 2.0
    # `observe` still returns only what the protocol reveals: the two per-task sums and
    # the estimator's own output. The two structural fields below are empty here and are
    # populated by the caller, never by `observe`.
    assert not view.carries
    assert not view.evidence

    # The deployability contract, and the reason this is asserted on `__slots__` rather
    # than on a happy path: a policy that reaches past these fields is reading something
    # the aggregator does not have, and the result would be an oracle wearing a method's
    # name. The set grew on 2026-08-06 and the growth was deliberate.
    #
    # `carries` and `evidence` are *public corpus structure* -- which tasks carry a
    # channel, and how many defining facts each shows. Finding 22's detector already
    # reads exactly these two and is deployable for that reason: it conditions the
    # verdict rate on evidence stratum and tests association with the channel. Finding
    # 23's policy reads the same two and nothing else.
    #
    # What must never appear here is a per-analyst field. That is the line this test
    # exists to hold, and widening the set for public structure does not move it.
    assert set(ServerObservation.__slots__) == {
        "votes",
        "seen",
        "posterior",
        "evidence",
        "carries",
    }
    forbidden = {"contributions", "contributors", "analysts", "partitioned", "truth", "by_analyst"}
    assert not forbidden & set(ServerObservation.__slots__), (
        "a per-analyst field reached the observation; finding 20's deployability rule "
        "is that a policy cannot distinguish contributors"
    )


def test_consensus_is_the_exact_inverse_of_margin():
    view = _view({"a": 2.0, "b": 4.0}, {"a": 4.0, "b": 4.0}, {"a": 0.5, "b": 0.9})
    margin = POLICIES["margin"](view, {})
    consensus = POLICIES["consensus"](view, {})
    assert all(consensus[t] == -margin[t] for t in margin)


def test_selection_is_nested_as_the_budget_grows():
    """A larger budget must audit a superset, or the sweep confounds two variables."""
    view = _view(
        {f"T-{i}": float(i % 5) for i in range(20)},
        {f"T-{i}": 4.0 for i in range(20)},
        {f"T-{i}": 0.1 * (i % 9) for i in range(20)},
    )
    # `uniform` is in this list now. It used to be excluded, because `choose_anchors`
    # called `random.sample` afresh per budget: every cell was a uniform subset of the
    # right size, but consecutive budgets audited different items, so the baseline
    # column moved selection and budget together while the targeted columns moved only
    # budget. The baseline now slices one shuffled order, and the sweep compares like
    # with like.
    for policy in ("uniform", "margin", "posterior", "consensus"):
        small = set(select(policy, view, {}, 3, seed=1))
        large = set(select(policy, view, {}, 7, seed=1))
        assert small <= large, policy
        assert len(small) == 3 and len(large) == 7


def test_a_budget_past_the_auditable_pool_is_refused():
    """Clipping would report a threshold reached at a budget never tested."""
    view = _view({"a": 1.0}, {"a": 2.0}, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="auditable"):
        select("margin", view, {}, 5, seed=1)
    assert select("margin", view, {}, 0, seed=1) == ()


def test_the_oracle_selects_only_tasks_the_estimate_gets_wrong():
    posterior = {"right": 0.9, "wrong": 0.9, "also-wrong": 0.2}
    truth = {"right": True, "wrong": False, "also-wrong": True}
    view = _view({}, {}, posterior)
    chosen = set(select("oracle", view, truth, 2, seed=1))
    assert chosen == {"wrong", "also-wrong"}


def test_the_oracle_is_not_offered_as_deployable():
    """It reads ground truth, so quoting it as a method would be an overclaim."""
    assert "oracle" not in DEPLOYABLE
    assert set(DEPLOYABLE) == {"uniform", "margin", "posterior", "consensus"}


def test_a_perfect_authority_asserts_the_truth():
    truth = {f"T-{i}": i % 2 == 0 for i in range(10)}
    ruling = authority_ruling(sorted(truth), truth, error=0.0, seed=1)
    assert ruling == truth


def test_a_fallible_authority_errs_at_roughly_its_rate_and_reproducibly():
    truth = {f"T-{i}": i % 2 == 0 for i in range(400)}
    ruling = authority_ruling(sorted(truth), truth, error=0.2, seed=7)
    wrong = sum(1 for t, v in ruling.items() if v != truth[t])
    assert 0.12 < wrong / len(truth) < 0.28
    assert ruling == authority_ruling(sorted(truth), truth, error=0.2, seed=7)


def test_threshold_is_the_smallest_repairing_budget():
    def row(budget, repaired):
        # Every field named, because `Row` gained four when the mechanical baseline
        # landed and this construction is the one place a new field goes unnoticed.
        # `measure_secure_reliability.Row` carries a docstring warning about exactly
        # this call site; the warning was right and this test still had to be caught
        # by CI rather than by reading it.
        return AuditRow(
            policy="margin",
            n_wrong=6,
            budget=budget,
            error=0.0,
            scored_tasks=10,
            agreement=1.0 if repaired else 0.5,
            repaired=repaired,
            remaining_errors=0 if repaired else 5,
            hits=0,
            mechanical=0.5,
            corrected=3 if repaired else 0,
        )

    assert threshold([row(20, True), row(5, False), row(12, True)]) == 12
    assert threshold([row(5, False), row(20, False)]) is None


def test_the_sweep_stays_inside_the_auditable_pool():
    """97 tasks are auditable at these compositions; a budget past that cannot run."""
    assert max(BUDGETS) < 97
    assert BUDGETS[0] == 0
    assert list(BUDGETS) == sorted(BUDGETS)
    assert AUTHORITY_ERROR[0] == 0.0, "the first rate must reproduce finding 19"


# ------------------------------------ the mechanical baseline (review correction) -----


def _tiny_fleet():
    """Three contributors, four tasks, one of which the majority gets wrong."""
    truth = {"a": True, "b": True, "c": False, "d": False}
    partitioned = {
        "x": [("a", True), ("b", True), ("c", True), ("d", False)],
        "y": [("a", True), ("b", True), ("c", True), ("d", False)],
        "z": [("a", True), ("b", False), ("c", False), ("d", False)],
    }
    return partitioned, truth


def test_baseline_errors_counts_the_pool_the_estimator_covers_not_the_corpus():
    from measure_audit_policy import baseline_errors

    partitioned, truth = _tiny_fleet()
    pool, errors = baseline_errors(partitioned, truth)
    assert pool == 4
    # Task "c" is reported True by a majority and is False in truth.
    assert errors >= 1


def test_an_outcome_reports_what_was_corrected_not_merely_what_was_removed():
    """The defect this class exists for: anchoring a wrong task flatters the score.

    Excluding an anchored task removes it from the denominator, so a policy that
    targets errors climbs toward 1.000 without changing a single label. `mechanical` is
    that climb; `corrected` is what is left over once it is subtracted.
    """
    from measure_audit_policy import baseline_errors, evaluate, observe

    partitioned, truth = _tiny_fleet()
    baseline = baseline_errors(partitioned, truth)
    view = observe(partitioned)

    zero = evaluate(
        partitioned, view, truth, policy="oracle", budget=0, error=0.0, baseline=baseline
    )
    assert zero.hits == 0
    assert zero.corrected == 0
    assert zero.agreement == pytest.approx(zero.mechanical, abs=1e-9)

    one = evaluate(
        partitioned, view, truth, policy="oracle", budget=1, error=0.0, baseline=baseline
    )
    # The oracle anchors a task the estimator gets wrong, so it leaves the pool.
    assert one.hits == 1
    assert one.scored == zero.scored - 1
    # `corrected` may be NEGATIVE, and on this fleet it is: clamping one task perturbs
    # the M step and flips another. That is the metric doing its job -- raw agreement
    # cannot distinguish "the audit helped", "the audit changed nothing and the score
    # rose by deletion", and "the audit actively hurt", and this separates all three.
    assert one.corrected == (baseline[1] - one.hits) - one.remaining_errors
    assert not one.genuine


def test_genuine_is_false_when_the_score_rose_only_by_deletion():
    from measure_audit_policy import Outcome

    deletion_only = Outcome(
        agreement=0.95, scored=20, remaining_errors=5, hits=5, mechanical=0.95, corrected=0
    )
    real = Outcome(
        agreement=1.0, scored=20, remaining_errors=0, hits=5, mechanical=0.75, corrected=5
    )
    assert not deletion_only.genuine
    assert real.genuine


def test_evaluate_computes_its_own_baseline_when_none_is_given():
    from measure_audit_policy import evaluate, observe

    partitioned, truth = _tiny_fleet()
    view = observe(partitioned)
    out = evaluate(partitioned, view, truth, policy="uniform", budget=0, error=0.0)
    assert out.scored == 4
    assert out.corrected == 0


def test_margin_picks_only_wrong_items_until_it_runs_out_of_them():
    """The published reason `margin` ties the oracle, asserted against the artifact.

    This was prose --- "a subset of the items the fleet gets wrong at every budget
    tested --- 20 of 20, 30 of 30" --- typed beside a run and true of no artifact field
    anyone checked. It was also overstated: the property cannot hold above the number of
    items the estimator gets wrong, because there is nothing left to pick. `hits` is the
    count of anchors landing on a task the zero-anchor estimate got wrong, so the claim
    is exactly `hits == budget` below the saturation point and `hits == errors` above it.
    """
    import json
    from pathlib import Path

    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "results" / "audit_policy.json").read_text(
            encoding="utf-8"
        )
    )
    grid = [r for r in payload["grid"] if r["policy"] == "margin"]
    assert grid, "no margin rows in the artifact"

    # The wrong-set size per composition: the largest `hits` any budget achieves, since
    # margin exhausts the set before the sweep ends.
    saturation = {}
    for row in grid:
        saturation[row["n_wrong"]] = max(saturation.get(row["n_wrong"], 0), row["hits"])

    for row in grid:
        errors = saturation[row["n_wrong"]]
        expected = min(row["budget"], errors)
        assert row["hits"] == expected, (
            f"margin at {row['n_wrong']} wrong, budget {row['budget']}: "
            f"{row['hits']} hits, expected {expected}"
        )

    # And the claim only says something where the budgets actually fit inside the wrong
    # set. If a corpus change pushed saturation below the budgets the docs cite, the
    # sentence would be vacuously true and this test would stop testing it.
    cited = (20, 30)
    for n_wrong, errors in saturation.items():
        assert errors >= max(cited), (
            f"composition {n_wrong} gets only {errors} wrong, below the cited budgets "
            f"{cited}; the published subset claim no longer has room to hold"
        )


def test_the_uniform_baseline_is_reported_as_a_spread_not_a_draw():
    """A point estimate here would credit the policy with whatever the draw supplied."""
    import json
    from pathlib import Path

    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "results" / "audit_policy.json").read_text(
            encoding="utf-8"
        )
    )
    spread = payload["uniform_spread"]
    assert spread, "the artifact carries no uniform spread"
    assert len(payload["uniform_seeds"]) >= 21

    for key, draw in spread.items():
        # The headline uniform threshold must be the median, not one draw.
        assert payload["thresholds"]["uniform"][key] == draw["median"]
        if draw["reached"]:
            assert draw["lowest"] <= draw["median"] <= draw["highest"]

    # The spread is load-bearing: at least one composition must have a baseline whose
    # best draw ties the winning policy, because that is the caveat the docs state. If
    # this stops being true the caveat is wrong and should be rewritten, not deleted.
    best = payload["best_deployable"]
    ties = [
        key
        for key, draw in spread.items()
        if draw["reached"] and draw["lowest"] <= payload["thresholds"][best][key]
    ]
    assert ties, "no composition where a uniform draw matches the winner; the caveat is stale"
