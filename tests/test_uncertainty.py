"""Intervals over a measurement that is itself noisy.

The properties here are the ones a wrong implementation would violate silently:
a bootstrap that resamples runs instead of tasks still returns a plausible-looking
interval, just far too narrow, and nothing about the number says so.
"""

import random
from typing import Any

import pytest

from pharos.uncertainty import (
    Interval,
    Trial,
    cluster_bootstrap,
    consensus_rate,
    resolves,
    scored,
    single_run_rate,
    summarize,
    variance_split,
)


def trials(pattern: dict[str, list[Any]]) -> list[Trial]:
    return [Trial(task_id, outcome) for task_id, runs in pattern.items() for outcome in runs]


# ------------------------------------------------------------- estimands -----


def test_single_run_weights_every_call_equally():
    """What one agent answering once experiences, which is the deployment here."""
    data = trials({"a": [True, True, True], "b": [False, False, False]})
    assert single_run_rate(data) == 0.5

    lopsided = trials({"a": [True, True, True], "b": [True, False, False]})
    assert single_run_rate(lopsided) == pytest.approx(4 / 6)


def test_consensus_takes_a_majority_per_task_and_a_tie_is_not_a_win():
    assert consensus_rate(trials({"a": [True, True, False]})) == 1.0
    assert consensus_rate(trials({"a": [True, False, False]})) == 0.0
    # A tie is not a verdict. Scoring it as half a success would credit a fleet
    # that could not decide.
    assert consensus_rate(trials({"a": [True, False]})) == 0.0
    assert consensus_rate([]) == 0.0


def test_voting_can_beat_or_lose_to_answering_once():
    """Consensus is not automatically better, and the direction is informative."""
    # Mostly-right tasks: voting cleans up the stray wrong run.
    helps = trials({"a": [True, True, False], "b": [True, True, False]})
    assert consensus_rate(helps) > single_run_rate(helps)
    # Mostly-wrong tasks: voting locks in the error the stray right run escaped.
    hurts = trials({"a": [False, False, True], "b": [False, False, True]})
    assert consensus_rate(hurts) < single_run_rate(hurts)


def test_unparsable_answers_are_dropped_rather_than_counted_wrong():
    assert scored([True, None, False]) == [True, False]
    data = trials({"a": [True, None], "b": [True, True]})
    # Three answers, all correct: an unparsable call is not a wrong call.
    assert single_run_rate(data) == 1.0
    assert summarize(data, label="x", resamples=200).unparsed == 1


# ---------------------------------------------------- variance attribution -----


def test_a_perfectly_stable_measurement_has_no_within_task_variance():
    """The whole point of the split: stable tasks contribute nothing to it."""
    split = variance_split(trials({"a": [True, True, True], "b": [False, False, False]}))
    assert split.within_task == 0.0
    assert split.between_task > 0.0
    assert split.within_share == 0.0


def test_a_measurement_that_only_flips_has_no_between_task_variance():
    """Every task equally unstable: adding tasks cannot help, and the split says so."""
    split = variance_split(trials({"a": [True, False], "b": [True, False], "c": [False, True]}))
    assert split.between_task == 0.0
    assert split.within_task > 0.0
    assert split.within_share == 1.0


def test_variance_split_survives_empty_and_all_unparsed_input():
    assert variance_split([]).within_share == 0.0
    assert variance_split(trials({"a": [None, None]})).total == 0.0


# ------------------------------------------------------------- bootstrap -----


def test_the_bootstrap_covers_the_point_estimate():
    data = trials({f"t{i}": [i % 3 != 0] for i in range(30)})
    interval = cluster_bootstrap(data, resamples=500)
    assert interval.low <= interval.point <= interval.high
    assert interval.covers(interval.point)


def test_clustering_widens_the_interval_against_treating_runs_as_independent():
    """The correction this module exists for.

    Runs of one task are correlated, so counting them as independent observations
    shrinks the interval for no reason. Giving each run its own task id simulates
    that mistake, and the correct interval must come out wider.
    """
    pattern = {f"t{i}": [i % 2 == 0] * 5 for i in range(20)}
    clustered = cluster_bootstrap(trials(pattern), resamples=800, seed=3)

    flattened = [
        Trial(f"{task}-{run}", outcome)
        for task, runs in pattern.items()
        for run, outcome in enumerate(runs)
    ]
    naive = cluster_bootstrap(flattened, resamples=800, seed=3)

    assert clustered.width > naive.width, "clustering must not narrow the interval"


def test_the_bootstrap_is_reproducible_from_its_seed():
    data = trials({f"t{i}": [i % 4 != 0, i % 3 != 0] for i in range(25)})
    first = cluster_bootstrap(data, resamples=400, seed=11)
    second = cluster_bootstrap(data, resamples=400, seed=11)
    assert first == second
    assert cluster_bootstrap(data, resamples=400, seed=12) != first


def test_more_tasks_narrow_the_interval():
    small = trials({f"t{i}": [i % 3 != 0] for i in range(10)})
    large = trials({f"t{i}": [i % 3 != 0] for i in range(120)})
    assert (
        cluster_bootstrap(large, resamples=600).width
        < cluster_bootstrap(small, resamples=600).width
    )


def test_a_single_task_yields_a_degenerate_interval_rather_than_a_fake_one():
    """One cluster cannot be resampled, and inventing a range would be a lie."""
    interval = cluster_bootstrap(trials({"a": [True, False]}), resamples=500)
    assert interval.low == interval.high == interval.point
    assert interval.resamples == 0


def test_an_invalid_confidence_level_is_refused():
    data = trials({"a": [True], "b": [False]})
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="level"):
            cluster_bootstrap(data, level=bad)


# ----------------------------------------------------------- the verdict -----


def test_resolves_is_false_when_intervals_overlap_the_other_point():
    wide_a = Interval(0.50, 0.40, 0.60, 0.95, 100)
    wide_b = Interval(0.55, 0.45, 0.65, 0.95, 100)
    assert not resolves(wide_a, wide_b), "overlapping intervals do not settle an ordering"

    tight_a = Interval(0.20, 0.18, 0.22, 0.95, 100)
    tight_b = Interval(0.80, 0.78, 0.82, 0.95, 100)
    assert resolves(tight_a, tight_b)


def test_summarize_reports_both_estimands_and_the_split():
    data = trials({f"t{i}": [i % 3 != 0, i % 3 != 0, i % 5 != 0] for i in range(24)})
    m = summarize(data, label="demo", resamples=400)

    assert m.n_tasks == 24
    assert m.repeats == 3
    payload: dict[str, Any] = m.as_dict()
    assert payload["label"] == "demo"
    assert payload["single_run"]["level"] == 0.95
    assert set(payload["variance"]) == {"between_task", "within_task", "within_share"}
    assert payload["consensus_gain"] == pytest.approx(m.consensus - m.single_run.point, abs=1e-4)


def test_difference_test_is_stricter_than_point_coverage():
    """The pair that taught us the two criteria disagree, kept as a regression.

    Finding 5's 0-shot versus 2-shot at n=600: neither interval covers the other's
    point, so `resolves` says yes, while an interval on the difference says no. A
    draft claimed the difference on the strength of the weaker test.
    """
    from pharos.uncertainty import Interval, resolves, resolves_difference

    zero = Interval(0.5234, 0.4841, 0.5654, 0.95, 2000)
    two = Interval(0.4683, 0.4269, 0.5085, 0.95, 2000)
    assert resolves(zero, two)
    assert not resolves_difference(zero, two)


def test_difference_test_accepts_a_genuinely_large_gap():
    from pharos.uncertainty import Interval, resolves_difference

    low = Interval(0.469, 0.400, 0.538, 0.95, 2000)
    high = Interval(1.000, 1.000, 1.000, 0.95, 2000)
    assert resolves_difference(low, high)
    # Symmetric: which one is passed first cannot change the answer.
    assert resolves_difference(high, low)


def test_identical_conditions_never_resolve():
    from pharos.uncertainty import Interval, resolves_difference

    same = Interval(0.5, 0.45, 0.55, 0.95, 2000)
    assert not resolves_difference(same, same)


# ------------------------------------------------------ interval validity -----


def _simulate(rng, *, n_tasks: int, repeats: int, p: float, clustered: bool):
    """Trials from a known true rate, optionally correlated within a task."""
    out: list[Trial] = []
    for i in range(n_tasks):
        # Clustered: the task is wholly easy or wholly hard, drawn with probability p.
        # That is the structure a real corpus has and the reason clusters are tasks.
        task_p = (1.0 if rng.random() < p else 0.0) if clustered else p
        out.extend(Trial(f"T-{i}", rng.random() < task_p) for _ in range(repeats))
    return out


def _naive_over_trials(trials, *, resamples=400, level=0.95, seed=7):
    """Resampling individual (task, run) pairs -- what the docstring says not to do."""
    rng = random.Random(seed)
    n = len(trials)
    draws = sorted(
        single_run_rate([trials[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    tail = (1 - level) / 2
    return draws[int(tail * resamples)], draws[int((1 - tail) * resamples)]


def test_cluster_bootstrap_achieves_its_nominal_coverage():
    """The definitive test for an interval: does 95% of the time mean 95% of the time?

    Simulated from a known rate, so coverage is checkable rather than assumed. This
    matters beyond the function: finding 5b argues that these intervals understate
    total uncertainty because they do not price run-to-run variation. That argument
    only holds if they correctly price the variation they DO claim, which is over
    which tasks were drawn.
    """
    true_p = 0.5
    for clustered in (False, True):
        hits = 0
        reps = 120
        for r in range(reps):
            trials = _simulate(
                random.Random(1000 + r), n_tasks=60, repeats=5, p=true_p, clustered=clustered
            )
            interval = cluster_bootstrap(trials, resamples=400, seed=r)
            hits += interval.low <= true_p <= interval.high
        coverage = hits / reps
        assert 0.88 <= coverage <= 1.0, (
            f"clustered={clustered}: nominal 95% interval covered {coverage:.0%} of the time"
        )


def test_resampling_runs_instead_of_tasks_undercovers_badly():
    """Why clusters are tasks, demonstrated rather than asserted in a docstring.

    Correlated repeats resampled as independent observations shrink the interval by
    roughly sqrt(k), and the resulting interval misses the truth most of the time.
    """
    true_p = 0.5
    cluster_hits = naive_hits = 0
    reps = 120
    for r in range(reps):
        trials = _simulate(random.Random(2000 + r), n_tasks=60, repeats=5, p=true_p, clustered=True)
        interval = cluster_bootstrap(trials, resamples=400, seed=r)
        cluster_hits += interval.low <= true_p <= interval.high
        low, high = _naive_over_trials(trials, seed=r)
        naive_hits += low <= true_p <= high

    assert cluster_hits / reps >= 0.88, "the correct estimator must hold its coverage"
    assert naive_hits / reps < 0.80, (
        "resampling runs should visibly undercover on clustered data; if this passes, "
        "the simulation stopped exercising the correlation it is meant to"
    )
    assert cluster_hits > naive_hits
