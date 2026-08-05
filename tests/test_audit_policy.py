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
    # No per-analyst field: a policy cannot reach a contributor through this object.
    assert set(ServerObservation.__slots__) == {"votes", "seen", "posterior"}


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
    for policy in ("margin", "posterior", "consensus"):
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
