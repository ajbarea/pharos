"""Intervals over a measurement that is itself noisy.

The properties here are the ones a wrong implementation would violate silently:
a bootstrap that resamples runs instead of tasks still returns a plausible-looking
interval, just far too narrow, and nothing about the number says so.
"""

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
