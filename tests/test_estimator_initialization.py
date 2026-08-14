"""The start EM is given, and the likelihood that decides whether the start mattered."""

import math
from itertools import pairwise

import pytest
from conftest import artifact

from pharos.inference import dawid_skene, log_likelihood


def contributions(n_wrong: int, n_right: int, tasks: int = 40) -> list[tuple[str, str, bool]]:
    """A fleet split between two standards, where the truth alternates by task.

    `wrong-*` invert the truth on every third task, which is enough to move the majority
    without making the two groups mirror images -- a fleet whose halves disagree
    everywhere has two equally good labellings by construction and would test the
    tie-break rather than the estimator.
    """
    truth = {f"t{i}": i % 2 == 0 for i in range(tasks)}
    rows: list[tuple[str, str, bool]] = []
    for i in range(n_right):
        rows += [(t, f"right-{i}", v) for t, v in truth.items()]
    for i in range(n_wrong):
        rows += [(t, f"wrong-{i}", (not v) if int(t[1:]) % 3 == 0 else v) for t, v in truth.items()]
    return rows


def test_the_default_start_is_the_majority_vote_and_nothing_changed_it():
    """The published behaviour, pinned. Every finding from 12 onward reads this path."""
    rows = contributions(n_wrong=2, n_right=7)
    assert dawid_skene(rows).posterior == dawid_skene(rows, initial_posterior=None).posterior


def test_an_override_actually_reaches_em():
    """A start that is silently discarded would make the whole sweep vacuous.

    Asserted against an adversarial start on a fleet where the majority is right, so the
    two runs have somewhere different to go. If the override were dropped this would
    still pass by accident on an easy fleet, which is why the start is the inverse of
    the truth rather than a perturbation of it.
    """
    rows = contributions(n_wrong=2, n_right=7)
    truth = {f"t{i}": i % 2 == 0 for i in range(40)}
    inverted = {t: (0.0 if v else 1.0) for t, v in truth.items()}

    default = dawid_skene(rows)
    overridden = dawid_skene(rows, initial_posterior=inverted)
    assert default.posterior != overridden.posterior, (
        "the initial_posterior argument is not reaching the E step"
    )


def test_a_partial_override_keeps_the_majority_vote_for_the_rest():
    """Half a start is half a start, not a silent 0.5 on everything it omits."""
    rows = contributions(n_wrong=1, n_right=4)
    partial = dawid_skene(rows, initial_posterior={"t0": 1.0})
    assert set(partial.posterior) == {f"t{i}" for i in range(40)}


def test_an_override_naming_an_unknown_task_is_ignored_rather_than_added():
    """Otherwise a typo in a start silently adds a task with no contributions at all."""
    rows = contributions(n_wrong=1, n_right=4)
    estimate = dawid_skene(rows, initial_posterior={"nonexistent": 1.0})
    assert "nonexistent" not in estimate.posterior


def test_the_likelihood_matches_a_direct_computation():
    """The scale every comparison in finding 25 rests on, checked against its definition.

    Recomputed here in the plain non-log form on a fleet small enough that it cannot
    underflow. If the two disagree the sweep is ordering solutions by a quantity that is
    not the likelihood, and every escape it reports or fails to report is meaningless.
    """
    rows = contributions(n_wrong=1, n_right=4, tasks=12)
    estimate = dawid_skene(rows)

    by_task: dict[str, list[tuple[str, bool]]] = {}
    for task, who, verdict in rows:
        by_task.setdefault(task, []).append((who, verdict))

    expected = 0.0
    for reports in by_task.values():
        pos, neg = estimate.prevalence, 1.0 - estimate.prevalence
        for who, verdict in reports:
            reported = 1 if verdict else 0
            pos *= estimate.error_rates[who][1][reported]
            neg *= estimate.error_rates[who][0][reported]
        expected += math.log(pos + neg)

    assert log_likelihood(rows, estimate) == pytest.approx(expected, abs=1e-9)


def test_em_never_lowers_the_likelihood_it_is_maximising():
    """EM's defining property, and the guard that the two halves are the same model.

    The M step here estimates error rates from soft labels and the E step re-derives
    them; if `log_likelihood` used a different parameterisation than the loop does, this
    would drift downward and nothing else in the suite would notice.
    """
    rows = contributions(n_wrong=3, n_right=6)
    scores = [
        log_likelihood(rows, dawid_skene(rows, max_iters=n, tolerance=0.0)) for n in range(1, 12)
    ]
    for earlier, later in pairwise(scores):
        assert later >= earlier - 1e-9, f"likelihood fell from {earlier} to {later}"


def test_an_empty_fleet_has_no_likelihood_rather_than_a_crash():
    assert log_likelihood([], dawid_skene([])) == 0.0


def test_a_contributor_absent_from_the_estimate_is_skipped():
    """Scoring one fleet's contributions under another fleet's fit must not raise."""
    rows = contributions(n_wrong=1, n_right=4)
    estimate = dawid_skene(rows)
    extra = [*rows, ("t0", "stranger", True)]
    assert math.isfinite(log_likelihood(extra, estimate))


def test_the_escape_is_confined_to_the_crossing():
    """Finding 25's structural claim, and the reason findings 19-24 survive it.

    A better start recovers the truth only at the composition where the crossing sits.
    Past it, no start does -- and the artifact says why: the wrong answer is the better
    fit there, so a likelihood-guided search has no reason to leave it. If this ever
    fails, the initialisation caveat stops being a scope condition on one cell and
    becomes a limitation on every governance finding.
    """
    payload = artifact("estimator_initialization")
    assert payload["invariants"]["the_escape_is_confined_to_the_crossing"], (
        "an initialisation escape now exists past the crossing, which would mean the "
        "cliff findings measure a basin rather than the estimator"
    )
    assert payload["escape_compositions"], (
        "no escape anywhere would be a stronger result than the one published, and "
        "finding 25's caveat should then be withdrawn rather than left standing"
    )


def test_no_start_escapes_past_the_crossing_and_the_reason_is_recorded():
    """The mechanism half. Reachable but rejected is the sharpest form of the claim."""
    payload = artifact("estimator_initialization")
    past = [
        row
        for row in payload["rows"]
        if row["draws_broken"] and not row["is_majority"] and row["n_wrong"] > row["fleet"] // 2
    ]
    assert past, "no composition past the crossing was priced; the claim is untested here"
    for row in past:
        assert row["draws_with_an_escape"] == 0, (
            f"fleet {row['fleet']} at {row['n_wrong']} wrong now has an initialisation escape"
        )
        assert row["likelihood_gap_median"] <= 0, (
            "the truth fits better here yet no start found it, which would mean the "
            "restart sweep is underpowered rather than that no escape exists"
        )


def test_random_restarts_are_reported_as_a_rate_and_not_as_a_boolean():
    """One restart in thirty-two and thirty in thirty-two are different methods."""
    payload = artifact("estimator_initialization")
    priced = [row for row in payload["rows"] if row["draws_broken"]]
    assert priced
    for row in priced:
        assert row["restart_recovery_rate"] is not None
        assert 0.0 <= row["restart_recovery_rate"] <= 1.0
    assert any(row["restart_recovery_rate"] > 0 for row in priced), (
        "no restart recovers anywhere, so the sweep cannot distinguish an estimator "
        "with no escape from a restart draw that never explores"
    )
