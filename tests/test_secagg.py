"""Masked aggregation: what cancels, what refuses, and what the server is left with."""

import random

import pytest

from pharos.secagg import (
    MIN_PARTICIPANTS,
    MODULUS,
    SCALE,
    CohortTooSmallError,
    IncompleteCohortError,
    SecureAggregator,
    ServerView,
    decode,
    encode,
    secure_sum,
)


def test_masks_cancel_exactly():
    """The property the whole module rests on. Exact, not approximate."""
    vectors = [[-1.5, 2.25, 0.0], [0.5, -0.25, 3.0], [1.0, 1.0, -3.0]]
    view = secure_sum(vectors, seed=1)
    assert view.total == (0.0, 3.0, 0.0)
    assert view.participants == 3


def test_round_trip_is_faithful_within_one_quantum():
    for value in (0.0, 1.0, -1.0, -39.75, 0.1, -1e-6):
        assert decode(encode(value)) == pytest.approx(value, abs=1.0 / SCALE)


def test_negatives_survive_the_ring():
    """Two's complement, so a negative total does not read as an enormous positive."""
    assert decode(encode(-12.5)) == pytest.approx(-12.5, abs=1.0 / SCALE)
    assert encode(-1.0) > MODULUS // 2


def test_quantization_error_stays_below_the_measurement_it_serves():
    """Nine clients of log-likelihoods, the shape finding 18 actually aggregates."""
    rng = random.Random(0)
    vectors = [[rng.uniform(-40.0, 0.0) for _ in range(64)] for _ in range(9)]
    view = secure_sum(vectors, seed=7)
    exact = [sum(column) for column in zip(*vectors, strict=True)]
    worst = max(abs(a - b) for a, b in zip(view.total, exact, strict=True))
    # Nine terms, each rounded to a 2^-24 grid, so the bound is 9/2^25 and the
    # tolerance below is that bound rather than a number chosen to pass.
    assert worst < 9.0 / (2 * SCALE)


def test_a_cohort_below_the_floor_is_refused():
    """A sum over two is one participant's input plus the other's, not an aggregate."""
    with pytest.raises(CohortTooSmallError):
        secure_sum([[1.0], [2.0]], seed=1)
    with pytest.raises(CohortTooSmallError):
        secure_sum([], seed=1)


def test_the_floor_is_enforced_on_the_aggregator_too():
    aggregator = SecureAggregator(length=2, seed=3)
    aggregator.submit(0, [1.0, 1.0], cohort=3)
    aggregator.submit(1, [1.0, 1.0], cohort=3)
    with pytest.raises(CohortTooSmallError):
        aggregator.reveal()
    aggregator.submit(2, [1.0, 1.0], cohort=3)
    assert aggregator.reveal().total == (3.0, 3.0)


def test_a_client_cannot_submit_twice():
    """Otherwise one client could inflate its own weight in the total unobserved."""
    aggregator = SecureAggregator(length=1, seed=3)
    aggregator.submit(0, [1.0], cohort=3)
    with pytest.raises(ValueError, match="twice"):
        aggregator.submit(0, [1.0], cohort=3)


def test_mismatched_lengths_are_refused():
    aggregator = SecureAggregator(length=2, seed=3)
    with pytest.raises(ValueError, match="coordinates"):
        aggregator.submit(0, [1.0], cohort=3)
    with pytest.raises(ValueError, match="same length"):
        secure_sum([[1.0, 2.0], [1.0], [1.0, 1.0]], seed=1)


def test_the_server_view_holds_no_per_client_field():
    """The privacy claim, asserted against the type rather than against prose.

    If a future edit adds a per-client attribute, this fails and the module's whole
    reason for existing has to be re-argued rather than quietly lost.
    """
    view = secure_sum([[1.0], [2.0], [3.0]], seed=1)
    assert set(ServerView.__slots__) == {"total", "participants"}
    assert view.total == (6.0,)
    assert view.participants == 3


def test_the_total_is_reproducible_across_seeds_and_orders():
    """Different masks, same sum. The masks must not survive into the result."""
    vectors = [[1.5, -2.5], [0.25, 4.0], [-1.0, 1.0], [3.0, 0.5]]
    first = secure_sum(vectors, seed=1)
    second = secure_sum(vectors, seed=99999)
    assert first.total == second.total

    reordered = secure_sum(list(reversed(vectors)), seed=1)
    assert reordered.total == pytest.approx(first.total)


def test_min_participants_default_is_three():
    """Two is not enough and the constant says so, so a change here is deliberate."""
    assert MIN_PARTICIPANTS == 3


def test_min_participants_cannot_be_argued_below_the_floor():
    """A floor that is a settable keyword is not a floor.

    `secure_sum(..., min_participants=1)` used to return one client's vector verbatim,
    inside the type whose entire job is to not be one client's vector.
    """
    with pytest.raises(ValueError, match="below the 3 floor"):
        secure_sum([[7.5]], seed=1, min_participants=1)
    with pytest.raises(ValueError, match="below the 3 floor"):
        SecureAggregator(length=1, seed=1, min_participants=2)


def test_a_short_cohort_is_refused_rather_than_silently_wrong():
    """Masks are derived per cohort, so a dropout leaves them uncancelled.

    Three clients submitting under `cohort=5` used to return a confident
    `(-267720907167.4, ...)` for a true `(3.0, 3.0)` and pass the participant floor.
    """
    aggregator = SecureAggregator(length=2, seed=3)
    for index in range(3):
        aggregator.submit(index, [1.0, 1.0], cohort=5)
    with pytest.raises(IncompleteCohortError, match="3 of 5"):
        aggregator.reveal()


def test_a_client_outside_the_cohort_is_refused():
    aggregator = SecureAggregator(length=1, seed=3)
    with pytest.raises(ValueError, match="outside a cohort"):
        aggregator.submit(99, [1.0], cohort=3)


def test_the_cohort_cannot_change_mid_round():
    aggregator = SecureAggregator(length=1, seed=3)
    aggregator.submit(0, [1.0], cohort=3)
    with pytest.raises(ValueError, match="cohort changed"):
        aggregator.submit(1, [1.0], cohort=4)


def test_encode_refuses_what_the_ring_cannot_hold():
    """2^40 used to encode to 0.0, and a sum of large terms to a wrong negative."""
    assert encode(1.0) > 0
    for value in (2.0**40, -(2.0**40), 1e30):
        with pytest.raises(ValueError, match="fixed-point ring"):
            encode(value)
