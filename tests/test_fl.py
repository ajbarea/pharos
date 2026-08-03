"""Tests for FL Strategies, Differential Privacy, and Byzantine Attacks."""

import random

import pytest

from pharos.fl import (
    Bulyan,
    FedAvg,
    FedMedian,
    FedProx,
    GeometricMedian,
    Krum,
    MultiKrum,
    PrivacyBudget,
    TrimmedMean,
    add_gaussian_dp_noise,
    apply_gaussian_noise,
    apply_sign_flip,
)

#: A fleet where the honest answer is unambiguous: 12 clients at 1.0 and 4 sign-flipped
#: to -1.5. Any rule claiming Byzantine robustness has to land near 1.0; the mean does
#: not, which is what makes this fleet discriminating rather than decorative.
HONEST = 1.0
POISONED = -1.5


def _fleet(dim: int = 6, n_honest: int = 12, n_byz: int = 4) -> list[list[float]]:
    return [[HONEST] * dim for _ in range(n_honest)] + [[POISONED] * dim for _ in range(n_byz)]


ROBUST = [
    FedMedian(),
    TrimmedMean(k=4),
    Krum(f=4),
    MultiKrum(f=4, m=8),
    Bulyan(f=4),
    GeometricMedian(),
]


@pytest.mark.parametrize("strategy", ROBUST, ids=lambda s: type(s).__name__)
def test_robust_rules_reject_a_sign_flipped_minority(strategy):
    """Each robust rule recovers the honest value; FedAvg is the contrast, below."""
    out = strategy.aggregate(_fleet())
    assert len(out) == 6
    assert all(abs(v - HONEST) < 0.1 for v in out), f"{type(strategy).__name__} -> {out}"


def test_fedavg_is_dragged_by_the_same_minority():
    """The control. Without it the test above only shows the fleet is easy."""
    out = FedAvg().aggregate(_fleet())
    assert all(abs(v - HONEST) > 0.4 for v in out)


def test_bulyan_trims_after_selecting():
    """Bulyan's second stage is what distinguishes it from Multi-Krum.

    The fleet is built so the stage is observable: two extreme updates lose the Krum
    selection outright, and one mild outlier survives it. Bulyan then trims that
    survivor when it keeps the beta = theta-2f values closest to each coordinate's
    median; Multi-Krum averages it in. Both rules agree on almost any other fleet, so
    a version of Bulyan that returned its selection directly passes every other test
    here -- including the obvious one this replaced.
    """
    fleet = [[1.0] * 4 for _ in range(9)] + [[2.0] * 4] + [[50.0] * 4 for _ in range(2)]
    assert Bulyan(f=1).aggregate(fleet) == [1.0] * 4
    assert MultiKrum(f=1, m=10).aggregate(fleet) == [1.1] * 4


@pytest.mark.parametrize("strategy", [FedAvg(), *ROBUST], ids=lambda s: type(s).__name__)
def test_empty_update_list_is_not_an_error(strategy):
    assert strategy.aggregate([]) == []


def test_fedprox_server_step_equals_fedavg():
    """Documented behaviour: FedProx's proximal term is client-side, so mu cannot move
    the server aggregation. A mu that changed this result would mean the class is doing
    something its paper does not describe."""
    fleet = _fleet()
    assert FedProx(mu=0.5).aggregate(fleet) == FedAvg().aggregate(fleet)


def test_strategy_instantiation():
    fedprox = FedProx(mu=0.05)
    trimmed = TrimmedMean(k=2)
    krum = Krum(f=1)
    geo = GeometricMedian(max_iter=10)

    assert fedprox.mu == 0.05
    assert trimmed.k == 2
    assert krum.f == 1
    assert geo.max_iter == 10


def test_differential_privacy_noise():
    rng = random.Random(42)
    weights = [1.0, 2.0, -3.0, 0.5]
    budget = PrivacyBudget(epsilon=1.0, delta=1e-5, clip_norm=2.0)

    noisy = add_gaussian_dp_noise(weights, budget, rng=rng)
    assert len(noisy) == len(weights)
    assert noisy != weights  # DP noise added


def test_byzantine_attacks():
    rng = random.Random(42)
    weights = [1.0, -2.0, 3.0]

    flipped = apply_sign_flip(weights, severity=1.0)
    assert flipped == [-1.0, 2.0, -3.0]

    noisy = apply_gaussian_noise(weights, scale=0.1, rng=rng)
    assert len(noisy) == len(weights)
    assert noisy != weights
