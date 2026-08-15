"""Fleet composition under correlation, where the probability is exact."""

from random import Random

import pytest
from measure_correlated_fleets import FLEET, compose, draw_fleet

from pharos.governance import WRONG_THRESHOLD, exact_wrong_majority


def test_probability_is_exact_not_estimated():
    """Closed form, so these are equalities rather than approximations."""
    # One culture: the fleet is wrong exactly when the single school is.
    for rate in (0.0, 0.1, 0.37, 1.0):
        assert exact_wrong_majority(rate, schools=1, fleet=FLEET) == pytest.approx(rate)

    # Independent, rate 0: never. Rate 1: always.
    assert exact_wrong_majority(0.0, schools=FLEET, fleet=FLEET) == 0.0
    assert exact_wrong_majority(1.0, schools=FLEET, fleet=FLEET) == 1.0

    # Three schools of three: majority needs 2 of 3 schools wrong.
    p = 0.5
    assert exact_wrong_majority(p, schools=3, fleet=FLEET) == pytest.approx(
        3 * p**2 * (1 - p) + p**3
    )


def test_clustering_never_reduces_the_risk():
    """The ordering the mathematics forbids breaking, and that sampling broke."""
    for rate in (0.05, 0.1, 0.2, 0.3, 0.4):
        indep = exact_wrong_majority(rate, schools=FLEET, fleet=FLEET)
        three = exact_wrong_majority(rate, schools=3, fleet=FLEET)
        one = exact_wrong_majority(rate, schools=1, fleet=FLEET)
        assert indep <= three <= one, f"non-monotone at rate {rate}"


def test_the_structures_agree_at_a_coin_flip():
    """At 0.5 clustering cannot move the mean, which is the sanity check on the model."""
    for schools in (1, 3, FLEET):
        assert exact_wrong_majority(0.5, schools=schools, fleet=FLEET) == pytest.approx(0.5)


def test_understatement_is_worst_where_it_is_most_reassuring():
    """The finding: independence flatters most in the regime a designer would cite."""
    low = exact_wrong_majority(0.1, schools=1, fleet=FLEET) / exact_wrong_majority(
        0.1, schools=FLEET, fleet=FLEET
    )
    high = exact_wrong_majority(0.4, schools=1, fleet=FLEET) / exact_wrong_majority(
        0.4, schools=FLEET, fleet=FLEET
    )
    assert low > 50
    assert high < 2
    assert low > high


def test_draws_have_the_requested_structure():
    rng = Random(1)
    everyone_wrong = draw_fleet(1.0, schools=3, rng=rng)
    assert len(everyone_wrong) == FLEET
    assert all(p.escalation_threshold == WRONG_THRESHOLD for p in everyone_wrong)

    nobody = draw_fleet(0.0, schools=3, rng=rng)
    assert all(p.escalation_threshold != WRONG_THRESHOLD for p in nobody)

    # Names are unique, or contributions would collide in the aggregation.
    assert len({p.name for p in draw_fleet(0.5, schools=3, rng=rng)}) == FLEET


def test_compose_handles_a_regime_that_was_never_drawn():
    """A missing conditional must not be read as zero agreement."""
    assert compose(0.2, [], [0.9]) == pytest.approx(0.9)
    assert compose(0.2, [0.7], []) == pytest.approx(0.7)
    assert compose(0.0, [], []) == 0.0
    assert compose(0.5, [0.7], [0.9]) == pytest.approx(0.8)


# ------------------------------------------- the understatement ratio (finding 16) -----


def test_understatement_uses_the_exact_binomial_not_the_rounded_rate():
    """111.1 came from dividing by a four-decimal rounded 0.0009; the exact is 112.2.

    The prose had already been corrected to 112 while the artifact still said 111.1,
    so tracing the published number to its evidence found the retracted value.
    """
    from measure_fleet_sensitivity import understatement

    assert understatement(0.1, 9) == 112.2


def test_understatement_is_finite_where_the_docs_said_unbounded():
    """The independent rate prints as 0.0000 past fleet 9; it is not zero.

    These three were published as "unbounded" on the stated grounds that the ratio
    "stops being finite". It does not, and the true values are a stronger claim.
    """
    from measure_fleet_sensitivity import understatement

    assert understatement(0.1, 15) == 2974.0
    assert understatement(0.1, 25) == 616966.3
    assert understatement(0.1, 51) is not None


def test_understatement_grows_monotonically_with_fleet_size():
    """The finding's actual claim, which the rounding had obscured."""
    from measure_fleet_sensitivity import understatement

    values = [understatement(0.1, n) for n in (5, 9, 15, 25, 51)]
    # Narrowed by comprehension rather than by an `all(... is not None)` assert, which
    # reads as a guard to a person and not to a type checker.
    finite = [v for v in values if v is not None]
    assert len(finite) == len(values)
    assert finite == sorted(finite)


def test_understatement_is_none_only_when_the_exact_probability_is_zero():
    """A rate of 0 makes a wrong majority genuinely impossible, not merely unlikely."""
    from measure_fleet_sensitivity import understatement

    assert understatement(0.0, 9) is None
    assert understatement(0.5, 9) is not None
