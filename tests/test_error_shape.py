"""Finding 29: whether the shape of the error is visible from the aggregate.

The statistic is the load-bearing part and it is three lines of arithmetic, so the tests
here are mostly about the arithmetic being the arithmetic claimed: a binomial fleet has to
read as 1, a fleet split into two deterministic groups has to read as more, and a fleet
with no variance at all has to read as *undiagnosable* rather than as either.
"""

import pytest
from conftest import artifact
from measure_audit_policy import ServerObservation
from measure_blind_spot import REFUSED_EXIT
from measure_error_shape import ALPHA, MIN_STRATUM, _monotone, dispersion

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
