"""The privacy budget, and the proposition that value noise cannot spend it usefully."""

import math

import pytest

from pharos.budget import (
    Budget,
    distinguishing_tasks,
    label_noise,
    randomized_participation,
    value_noise,
    widest_separation,
)
from pharos.disclosure import DROP_COMPARTMENTS
from pharos.fleet import FLEET_CEILING, Clearance, assign_fleet, contribute, link
from pharos.generate import GeneratorConfig, generate
from pharos.labels import Compartment, Sensitivity
from pharos.tasks import build_triage_tasks


@pytest.fixture(scope="module")
def tasks():
    return build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=60)))


@pytest.fixture(scope="module")
def fleet():
    return assign_fleet(40, seed=11)


def test_no_fabrication_means_no_finite_guarantee():
    """Subsampling drops contributions without making any present one deniable."""
    assert math.isinf(Budget(keep=0.9, fabricate=0.0).epsilon)
    assert math.isinf(Budget(keep=1.0, fabricate=0.5).epsilon)
    assert math.isinf(Budget(keep=0.9, fabricate=0.0).basic_composition(10))


def test_epsilon_falls_as_the_two_rates_converge():
    """Deniability is exactly how hard it is to tell reachable from fabricated."""
    wide = Budget(keep=0.95, fabricate=0.05).epsilon
    narrow = Budget(keep=0.6, fabricate=0.4).epsilon
    assert wide > narrow > 0
    # At keep == fabricate the stream carries no participation signal at all.
    assert Budget(keep=0.5, fabricate=0.5).epsilon == pytest.approx(0.0, abs=1e-9)


def test_composition_dominates_the_per_indicator_figure():
    """The number a mechanism advertises is not the number an adversary faces."""
    budget = Budget(keep=0.6, fabricate=0.4)
    per = budget.epsilon
    composed = budget.effective_epsilon(195)
    assert composed > 100 * per, "composing over 195 indicators must dwarf one of them"
    # Advanced composition is chosen when it is the tighter of the two.
    assert budget.effective_epsilon(195) <= budget.basic_composition(195)
    # And with a single indicator the two agree on the per-indicator bound.
    assert budget.effective_epsilon(1) <= budget.basic_composition(1) + 1e-9


def test_budget_serializes_infinities_as_null():
    row = Budget(keep=0.9, fabricate=0.0).as_dict(10)
    assert row["epsilon_per_indicator"] is None
    assert row["epsilon_effective"] is None
    finite = Budget(keep=0.6, fabricate=0.4).as_dict(10)
    assert isinstance(finite["epsilon_per_indicator"], float)


def test_value_noise_cannot_change_the_attack(tasks, fleet):
    """The proposition, enforced end to end rather than argued in a docstring."""
    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    reference = link(stream, tasks, fleet, policy=DROP_COMPARTMENTS)
    for flip in (0.1, 0.5, 1.0):
        noised = value_noise(stream, flip=flip, seed=3)
        assert link(noised, tasks, fleet, policy=DROP_COMPARTMENTS) == reference
    # And the mechanism is doing something, so the invariance is not vacuous.
    flipped = value_noise(stream, flip=1.0, seed=3)
    assert [c.verdict for c in flipped] != [c.verdict for c in stream]


def test_participation_noise_fabricates_and_drops(tasks, fleet):
    honest = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    noised = randomized_participation(
        fleet,
        tasks,
        policy=DROP_COMPARTMENTS,
        ceiling=FLEET_CEILING,
        budget=Budget(keep=0.6, fabricate=0.4),
        seed=7,
    )
    # Fabrication is what distinguishes this from subsampling.
    assert label_noise(noised, fleet, tasks) > 0.0
    assert label_noise(honest, fleet, tasks) == 0.0
    # Deterministic given the seed.
    again = randomized_participation(
        fleet,
        tasks,
        policy=DROP_COMPARTMENTS,
        ceiling=FLEET_CEILING,
        budget=Budget(keep=0.6, fabricate=0.4),
        seed=7,
    )
    assert again == noised


def test_fabrication_never_federates_an_ineligible_task(tasks, fleet):
    """Deniability about who saw what is not licence to release what the gate refused."""
    from pharos.labels import shared_eligible

    noised = randomized_participation(
        fleet,
        tasks,
        policy=DROP_COMPARTMENTS,
        ceiling=FLEET_CEILING,
        budget=Budget(keep=1.0, fabricate=1.0),
        seed=7,
    )
    by_id = {t.task_id: t for t in tasks}
    assert noised
    for c in noised:
        assert shared_eligible(by_id[c.task_id].label, FLEET_CEILING, DROP_COMPARTMENTS)


def test_distinguishing_tasks_counts_only_disagreements(tasks):
    top = Clearance("a", Sensitivity.RESTRICTED, frozenset(Compartment))
    bottom = Clearance("b", Sensitivity.OPEN, frozenset())
    assert distinguishing_tasks(top, top, tasks) == 0
    apart = distinguishing_tasks(top, bottom, tasks)
    assert apart > 0
    assert distinguishing_tasks(bottom, top, tasks) == apart, "symmetric"


def test_widest_separation_is_the_worst_pair(tasks, fleet):
    widest = widest_separation(fleet, tasks)
    assert widest > 0
    assert widest <= len(tasks)
    assert widest_separation([], tasks) == 0
    assert widest_separation(fleet[:1], tasks) == 0
