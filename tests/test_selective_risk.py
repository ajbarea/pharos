"""Finding 28: abstention as the action once the audit budget buys nothing.

Every test here pins a property that would fail silently. A metric that rewards
withholding, a threshold read off a rate instead of a count, and a comparison against the
median of a variable baseline all produce well-formed artifacts carrying claims that were
never measured -- which is the shape of three retractions already on this project's
record.
"""

import pytest
from conftest import artifact
from measure_blind_spot import REFUSED_EXIT, assert_channel_usable
from measure_selective_risk import (
    HALVED,
    Cell,
    beats_every_draw,
    first_budget_halving,
    score,
)

from pharos.generate import GeneratorConfig, generate
from pharos.labels import Compartment
from pharos.tasks import build_triage_tasks

POOL = tuple(f"t{i}" for i in range(10))
WRONG = frozenset({"t0", "t1"})


def test_withholding_a_correct_label_raises_risk():
    """The asymmetry the whole measurement rests on.

    If abstention could lower risk by removing anything at all, every policy would look
    like it worked and the table would rank noise. Withholding two correct labels has to
    move the rate the wrong way.
    """
    base = score(n_blind=9, slip_rate=0.0, policy="none", withheld=(), pool=POOL, wrong=WRONG)
    correct_only = score(
        n_blind=9, slip_rate=0.0, policy="p", withheld=("t8", "t9"), pool=POOL, wrong=WRONG
    )
    assert correct_only.risk > base.risk
    assert correct_only.caught == 0
    assert correct_only.precision == 0.0


def test_withholding_a_wrong_label_lowers_risk_and_is_priced_in_coverage():
    held = score(n_blind=9, slip_rate=0.0, policy="p", withheld=("t0",), pool=POOL, wrong=WRONG)
    assert held.risk < 0.2
    assert held.coverage == 0.9
    assert held.caught == 1
    assert held.precision == 1.0


def test_a_perfect_withhold_reaches_zero_risk_without_claiming_a_repair():
    """Coverage is what a zero in the risk column costs, and it must be readable beside it."""
    perfect = score(
        n_blind=9, slip_rate=0.0, policy="oracle", withheld=("t0", "t1"), pool=POOL, wrong=WRONG
    )
    assert perfect.risk == 0.0
    assert perfect.coverage == 0.8


def _cell(withheld: int, errors: int, published: int) -> Cell:
    return Cell(
        n_blind=9,
        slip_rate=0.0,
        policy="p",
        withheld=withheld,
        coverage=published / 100,
        published=published,
        risk=round(errors / published, 4),
        errors_published=errors,
        caught=0,
        precision=0.0,
    )


def test_the_threshold_counts_errors_not_the_rate():
    """A policy that shrinks the corpus must not reach the bar by shrinking it.

    Both cells below have a *lower* risk rate than the fleet started with, and neither has
    removed an error: the denominator moved. Counting errors is what makes the threshold
    mean a label was withheld rather than a fraction re-expressed.
    """
    deletion_only = [_cell(0, 10, 100), _cell(20, 10, 90), _cell(45, 10, 70)]
    assert first_budget_halving(deletion_only, 10) is None

    genuine = [_cell(0, 10, 100), _cell(20, 5, 90)]
    assert first_budget_halving(genuine, 10) == 20


def test_the_threshold_bar_is_the_published_constant():
    """Half, and read from the constant rather than restated, so moving it moves the test."""
    at_the_bar = [_cell(0, 10, 100), _cell(12, int(10 * HALVED), 95)]
    assert first_budget_halving(at_the_bar, 10) == 12


def test_beating_the_median_of_a_variable_baseline_is_not_beating_it():
    """The correction finding 20 made, in the one place this script could repeat it."""
    draws = [0.089, 0.100, 0.111]
    assert not beats_every_draw(0.094, draws), "0.094 loses to the best of these draws"
    assert beats_every_draw(0.088, draws)
    assert not beats_every_draw(0.089, draws), "a tie with the best draw carries no information"
    assert not beats_every_draw(None, draws)
    assert not beats_every_draw(0.0, []), "no draws is not a win"


def test_the_channel_guard_refuses_a_channel_entangled_with_difficulty():
    """The precondition this script shares with finding 21, exercised on the channel the
    documentation names as unusable.

    SENSOR carries mean evidence 2.00 on the tasks that have it against 0.48 on the tasks
    that do not, so blinding it would select the corpus's hard items and smuggle back the
    difficulty confound the blind-spot design exists to escape.
    """
    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=200)))
    with pytest.raises(SystemExit) as excinfo:
        assert_channel_usable(tasks, compartment=Compartment.SENSOR)
    assert excinfo.value.code == REFUSED_EXIT


def test_the_channel_guard_accepts_the_channel_the_finding_uses():
    """The control that keeps the test above from passing for the wrong reason.

    A guard that refused every channel would satisfy the assertion above and make the
    experiment unbuildable, which is a different failure with the same signature.
    """
    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=200)))
    check = assert_channel_usable(tasks)
    assert check.affected > 0
    assert abs(check.mean_with - check.mean_without) < 0.5


@pytest.fixture(scope="module")
def report():
    return artifact("selective_risk.json")


def test_the_artifact_isolates_a_regime_where_the_shared_error_is_the_whole_error(report):
    """Every claim at unanimity is quantified over these rates, so they have to exist."""
    assert report["shared_only_slip_rates"], "no regime isolates the shared blind spot"
    for rate in report["shared_only_slip_rates"]:
        healthy = next(f for f in report["fleets"] if f["n_blind"] == 0 and f["slip_rate"] == rate)
        assert healthy["base_errors"] == 0


def test_confidence_abstention_is_at_chance_at_unanimity(report):
    """The failure the finding is about, read off the artifact rather than the prose.

    At unanimity a corrupted item is one the whole fleet agrees on, so a rule ranking by
    disagreement cannot see it. `precision` is the share of a 20-label withhold that lands
    on a wrong label, and chance is the fleet's own error rate.
    """
    unanimous = max(report["shares"])
    for rate in report["shared_only_slip_rates"]:
        fleet = next(
            f for f in report["fleets"] if f["n_blind"] == unanimous and f["slip_rate"] == rate
        )
        chance = fleet["base_errors"] / fleet["pool"]
        for policy in ("margin", "posterior", "consensus"):
            # Twice chance, which is the loosest bar that still separates these policies
            # from `channel`'s 1.00. The artifact is deterministic, so this reads the
            # committed numbers rather than tolerating a spread; the spread across corpus
            # draws is measure_corpus_sensitivity's to report.
            assert fleet["policies"][policy]["precision_at_20"] <= 2 * chance
        assert fleet["policies"]["channel"]["precision_at_20"] == 1.0


def test_provenance_abstention_removes_the_errors_and_the_artifact_prices_it(report):
    """Zero risk is not free, and the coverage it costs must be in the same artifact."""
    unanimous = max(report["shares"])
    for rate in report["shared_only_slip_rates"]:
        halved_at = next(
            f for f in report["fleets"] if f["n_blind"] == unanimous and f["slip_rate"] == rate
        )["policies"]["channel"]["halved_at"]
        assert halved_at is not None
        cell = next(
            c
            for c in report["grid"]
            if c["n_blind"] == unanimous
            and c["slip_rate"] == rate
            and c["policy"] == "channel"
            and c["withheld"] == halved_at
        )
        assert 0.0 < cell["coverage"] < 1.0


def test_the_false_detection_price_is_published_as_a_number(report):
    """Prediction 5 is a cost, not a claim.

    A boolean saying a false detection "removes nothing" would be true by construction on
    a fleet with no blind spot, and this project does not publish checks that cannot fail.
    """
    priced = [e for e in report["false_detection"] if e["coverage_at_20"] is not None]
    assert priced
    for entry in priced:
        assert 0.0 < entry["coverage_at_20"] < 1.0
        if entry["base_errors"] == 0:
            assert entry["caught_at_20"] == 0


def test_the_artifact_flags_the_cells_whose_estimator_did_not_converge(report):
    """Convergence travels with the number, in a project that publishes flags rather than
    filtering on them. The cells that do not converge must not be the ones a claim rests on.
    """
    unconverged = {(c["n_blind"], c["slip_rate"]) for c in report["unconverged_cells"]}
    for rate in report["shared_only_slip_rates"]:
        assert not any(slip == rate for _, slip in unconverged)


def test_the_measurement_is_quotable_at_the_thinnest_cell(report):
    assert report["validity"]["quotable"], report["validity"]["concerns"]
    assert report["published_min"] >= 30


def test_the_channel_policy_is_only_proposed_where_the_detector_licenses_it(report):
    """Finding 28's rule is deployable *because* finding 22 names the channel.

    That licence is not uniform across this grid -- detection reaches one blind analyst
    in nine on a noiseless fleet and four of nine at a realistic slip rate -- so a claim
    resting on a cell where the detector never fired would be proposing a policy no
    deployment could have known to run.
    """
    assert report["unlicensed_claim_cells"] == [], (
        "a claim is quantified over a cell where finding 22's detector did not fire: "
        f"{report['unlicensed_claim_cells']}"
    )
    unanimous = max(report["shares"])
    for rate in report["shared_only_slip_rates"]:
        fleet = next(
            f for f in report["fleets"] if f["n_blind"] == unanimous and f["slip_rate"] == rate
        )
        assert fleet["detector_fired"] is True


def test_the_licence_field_is_not_vacuous(report):
    """The control. A field that is true everywhere records nothing.

    At a realistic slip rate the detector needs four of nine, so the low-share rows are
    exactly the cells where the `channel` column is scored without a licence. If this
    ever stops being so, the field has become decoration and the test above stops
    meaning anything.
    """
    fired = [f["detector_fired"] for f in report["fleets"]]
    assert any(value is not True for value in fired), (
        "every cell reports the detector firing, so the licence field cannot distinguish "
        "a proposal from a policy nobody could have run"
    )
    assert any(value is True for value in fired), "no cell reports a detection at all"


def test_a_licence_read_from_another_corpus_is_not_a_licence():
    """The cross-corpus guard, exercised rather than asserted.

    `channel_bias.json` is one corpus draw and this measurement is swept over eight. A
    licence read without checking the seed would hand every draw the committed corpus's
    detections -- the same cross-corpus confound that cost this project a reproducibility
    claim -- so a mismatched seed must read as "cannot say" rather than as a detection.
    """
    from measure_selective_risk import detector_fired

    committed = artifact("channel_bias.json")["provenance"]["seed"]
    assert detector_fired(0.0, 9, seed=committed) is True
    assert detector_fired(0.0, 9, seed=committed + 1) is None, (
        "a detection measured on one corpus was reported for another"
    )
    # A share the detector's own sweep does not carry is also "cannot say", not False.
    assert detector_fired(0.0, 8, seed=committed) is None
