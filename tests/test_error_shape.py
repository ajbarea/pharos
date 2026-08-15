"""Finding 29: whether the shape of the error is visible from the aggregate.

The statistic is the load-bearing part and it is three lines of arithmetic, so the tests
here are mostly about the arithmetic being the arithmetic claimed: a binomial fleet has to
read as 1, a fleet split into two deterministic groups has to read as more, and a fleet
with no variance at all has to read as *undiagnosable* rather than as either.
"""

import pytest
from conftest import artifact
from measure_error_shape import _monotone

from pharos.governance import ALPHA, MIN_STRATUM, REFUSED_EXIT, ServerObservation, dispersion

# A stratum large enough to be scored, with every task seen by the same nine analysts.
FLEET = 9
TASKS = tuple(f"t{i}" for i in range(40))


def _view(votes: dict[str, float]) -> ServerObservation:
    return ServerObservation(
        votes=dict(votes),
        seen=dict.fromkeys(votes, float(FLEET)),
        posterior=dict.fromkeys(votes, 0.5),
    )


def _one_stratum() -> dict[str, int]:
    return dict.fromkeys(TASKS, 2)


def test_a_binomial_fleet_reads_as_one():
    """The calibration the whole statistic rests on.

    Votes drawn as binomial at a single rate must give an index near 1. If this drifts,
    every "overdispersed" verdict downstream is measuring the drift instead.
    """
    import random

    rng = random.Random(11)
    rate = 0.4
    votes = {task: float(sum(1 for _ in range(FLEET) if rng.random() < rate)) for task in TASKS}
    spread = dispersion(_view(votes), _one_stratum(), draws=200, seed=3)
    assert spread.index is not None
    assert 0.6 < spread.index < 1.6, spread.index
    assert spread.p_value is not None and spread.p_value > ALPHA


def test_two_deterministic_groups_read_as_overdispersed():
    """A shared standard splits a stratum in two, which is the signal being detected.

    No randomness at all here: half the tasks are unanimous one way and half the other.
    A binomial at the pooled rate would spread its counts around the middle; this does
    not, and the index has to say so.
    """
    draws = 200
    votes = {task: (0.0 if i % 2 else float(FLEET)) for i, task in enumerate(TASKS)}
    spread = dispersion(_view(votes), _one_stratum(), draws=draws, seed=3)
    # Derived rather than guessed: every task deviates by n/2 from the pooled rate of
    # 0.5, so the observed sum is 40*(4.5**2) and the binomial sum is 40*9*0.25 -- an
    # index of exactly 9.
    assert spread.index == pytest.approx(9.0), spread.index
    # And the null reports it as extreme as it can. Asserting `<= ALPHA` here would fail
    # for a reason that has nothing to do with the fleet: 200 draws floor the p-value at
    # 1/201, above the alpha the committed run uses 2000 draws to clear. That is the same
    # arithmetic the script now refuses on, met in a test that assumed its way past it.
    # Rounded to six places by `dispersion`, so the comparison is too. Exact equality
    # against the unrounded floor fails on the seventh digit and says nothing.
    assert spread.p_value is not None
    assert spread.p_value == round(1.0 / (draws + 1), 6)
    assert spread.p_value > ALPHA, "this floor is above alpha, which is why the run uses more"


def test_a_fleet_with_no_variance_is_undiagnosable_rather_than_clean():
    """The distinction this project keeps having to make.

    Every analyst agreeing everywhere leaves no binomial variance to compare against. The
    honest answer is that nothing can be said, and reporting a low index instead would
    read as evidence that the error is independent -- which is the same defect as reading
    a guard that never fired as a guard that passed.
    """
    votes = dict.fromkeys(TASKS, float(FLEET))
    spread = dispersion(_view(votes), _one_stratum(), draws=50, seed=3)
    assert spread.index is None
    assert spread.p_value is None
    assert spread.degenerate_strata == 1
    assert spread.strata == 0


def test_a_stratum_too_small_to_estimate_is_not_estimated_from():
    """`MIN_STRATUM` is a floor on the evidence, and a floor nobody has seen refuse is not
    a floor. Two tasks produce a number; they do not produce an estimate."""
    votes = {"a": 9.0, "b": 0.0}
    spread = dispersion(
        ServerObservation(votes=votes, seen={"a": 9.0, "b": 9.0}, posterior={"a": 1.0, "b": 0.0}),
        {"a": 2, "b": 2},
        draws=50,
        seed=3,
    )
    assert spread.index is None, "a stratum of two was scored"
    assert MIN_STRATUM > 2


def test_strata_are_pooled_rather_than_averaged():
    """A stratum of forty and a stratum of twelve are not two equal observations.

    Averaging their indices would let the small one move the answer as much as the large
    one; pooling weights each by the variance it actually contributes. The check is that
    a large clean stratum is not overwhelmed by a small split one.
    """
    import random

    rng = random.Random(5)
    votes = {task: float(sum(1 for _ in range(FLEET) if rng.random() < 0.5)) for task in TASKS}
    evidence = _one_stratum()
    for i in range(MIN_STRATUM):
        task = f"small{i}"
        votes[task] = 0.0 if i % 2 else float(FLEET)
        evidence[task] = 3
    spread = dispersion(_view(votes), evidence, draws=200, seed=3)
    assert spread.strata == 2
    # The two strata have indices near 1 and exactly 9. Averaging them would give ~5;
    # pooling weights each by the binomial variance it contributes, and the split stratum
    # contributes a quarter of the tasks, so the pooled value lands near 2.6. What the
    # test pins is the gap between the two ways of combining them.
    assert spread.index is not None
    assert 1.5 < spread.index < 3.5, spread.index
    assert spread.index < 5.0 - 1.0, "pooled looks like an average of the per-stratum indices"


def test_monotone_rejects_a_flat_sequence():
    """ "Rises with the shared share" has to mean rises. A constant sequence is
    non-decreasing and is not a rise, and reading it as one would let a statistic that
    responds to nothing pass its own prediction."""
    assert _monotone([1.0, 2.0, 3.0])
    assert not _monotone([1.0, 1.0, 1.0])
    assert not _monotone([3.0, 2.0, 1.0])
    assert not _monotone([1.0])


def test_a_null_too_coarse_to_fire_is_refused():
    """The defect this script shipped with, pinned so it cannot come back.

    A simulated p-value floors at 1/(m+1). With the floor at or above alpha the test
    reports "not overdispersed" for every input, including a fleet with an index of 9.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(  # noqa: S603  # fixed argv, this repository's own script
        [sys.executable, str(root / "scripts" / "measure_error_shape.py"), "--draws", "10"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == REFUSED_EXIT, result.stderr[-400:]
    assert "floor" in result.stderr


@pytest.fixture(scope="module")
def report():
    return artifact("error_shape.json")


def test_the_committed_null_can_actually_fire(report):
    """The same arithmetic, asserted against the artifact rather than the script."""
    floor = 1.0 / (report["null_draws"] + 1)
    assert floor < report["alpha"], (
        f"the committed run floors its p-value at {floor}, at or above alpha "
        f"{report['alpha']}; every cell would read as not overdispersed"
    )


def test_the_index_separates_the_regimes_it_claims_to(report):
    """The finding, read off the artifact: calibrated where nothing is shared, and high
    where everything is."""
    healthy = [c for c in report["cells"] if c["n_blind"] == 0 and c["dispersion"]["index"]]
    assert healthy
    for cell in healthy:
        assert cell["dispersion"]["p_value"] > report["alpha"]
    unanimous = next(
        c
        for c in report["cells"]
        if c["n_blind"] == max(report["shares"]) and c["slip_rate"] == min(report["slip_rates"])
    )
    assert unanimous["dispersion"]["index"] > 3.0
    assert unanimous["dispersion"]["p_value"] <= report["alpha"]


def test_the_misses_are_priced_and_not_merely_counted(report):
    """A rate of correct calls is not a result about what the wrong ones cost.

    This project has already published one accuracy that read as worse than it was. Where
    the predictor is wrong, the artifact has to carry the gap in published error rate.
    """
    missed = [c for c in report["cells"] if c["decidable"] and not c["correct"]]
    for cell in missed:
        assert cell["cost_of_following_the_prediction"] is not None
        assert cell["risk"]["channel"] is not None
        assert cell["risk"]["confidence"] is not None
    if missed:
        assert report["worst_cost_of_a_wrong_call"] is not None


def test_an_undecidable_cell_is_not_counted_as_a_correct_call(report):
    """Cells where both rules work, or neither does, have no single right answer.

    Counting them either way would manufacture an accuracy out of cells the question does
    not apply to.
    """
    for cell in report["cells"]:
        if cell["actual_winner"] not in ("channel", "confidence"):
            assert not cell["decidable"]
            assert not cell["correct"]
    assert report["decidable_cells"] == sum(1 for c in report["cells"] if c["decidable"])


def test_the_cost_of_a_wrong_call_is_none_where_it_was_never_tested():
    """Zero and "not applicable" are different, and only one of them means it was free.

    An undecidable cell, an undiagnosable fleet, and a rule whose risk was never computed
    all have no cost to report. Returning 0.0 for those would read as "following the
    prediction cost nothing" in exactly the cells where the prediction was never scored.
    """
    from measure_error_shape import Outcome, _cost

    priced = Outcome(winner="channel", risk={"channel": 0.10, "confidence": 0.15})
    assert _cost(priced, "channel", "confidence") == pytest.approx(0.05)
    assert _cost(priced, "channel", "channel") == 0.0
    assert _cost(priced, "either", "channel") is None, "an undecidable cell was priced"
    assert _cost(priced, "channel", None) is None, "an undiagnosable fleet was priced"
    assert _cost(Outcome(winner="channel", risk={}), "channel", "confidence") is None


def test_the_winner_is_decided_against_the_best_untargeted_draw():
    """A rule wins only by beating every uniform draw, and both winning is not a win.

    Constructed so the answer is known: the wrong labels are exactly the tasks carrying
    the channel, and they are the ones the fleet is most unanimous about, so provenance
    selects them and confidence cannot.
    """
    from measure_error_shape import winning_rule

    from pharos.governance import ServerObservation

    # Sixty tasks and fifteen wrong, against a twenty-label withhold. Fewer wrong labels
    # than that and a lucky uniform draw sweeps them all, which ties rather than losing --
    # the same reason the measurement compares against the best of twenty-one draws.
    pool = [f"t{i}" for i in range(60)]
    wrong = set(pool[:15])
    # Unanimous on the corrupted tasks, split everywhere else: confidence looks away from
    # exactly the items that are wrong.
    votes = {t: (9.0 if t in wrong else 5.0) for t in pool}
    labels = dict.fromkeys(pool, True)
    truth = {t: (t not in wrong) for t in pool}
    view = ServerObservation(
        votes=votes,
        seen=dict.fromkeys(pool, 9.0),
        posterior={t: (0.99 if t in wrong else 0.55) for t in pool},
        evidence={t: (3 if t in wrong else 1) for t in pool},
        carries={t: (t in wrong) for t in pool},
    )
    outcome = winning_rule(view, labels, truth)
    assert outcome.winner == "channel", outcome
    channel, confidence, floor = (
        outcome.risk["channel"],
        outcome.risk["confidence"],
        outcome.risk["uniform_best"],
    )
    assert channel is not None and confidence is not None and floor is not None
    assert channel < floor
    assert confidence >= floor


def test_a_fleet_with_nothing_wrong_prices_no_rule():
    """Scoring a fleet the estimator already gets right would mark both rules down for
    failing to remove errors that do not exist."""
    from measure_error_shape import winning_rule

    from pharos.governance import ServerObservation

    pool = [f"t{i}" for i in range(40)]
    view = ServerObservation(
        votes=dict.fromkeys(pool, 9.0),
        seen=dict.fromkeys(pool, 9.0),
        posterior=dict.fromkeys(pool, 0.99),
    )
    outcome = winning_rule(view, dict.fromkeys(pool, True), dict.fromkeys(pool, True))
    assert outcome.winner is None
    assert outcome.risk == {}
