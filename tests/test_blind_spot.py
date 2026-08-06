"""Finding 21: the corpus finding 20's policy cannot handle, built on purpose."""

import random
from typing import Any

import pytest
from conftest import artifact
from measure_blind_spot import (
    BLIND,
    BUDGETS,
    CHANNEL_ENTANGLEMENT_SLACK,
    SHARES,
    blind_fleet,
)

from pharos.analyst import AnalystPolicy, evidence_shown
from pharos.disclosure import DROP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.labels import Compartment
from pharos.tasks import build_triage_tasks

EVENTS = 200


@pytest.fixture(scope="module")
def tasks():
    return build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=EVENTS)))


def test_a_sighted_reviewer_is_unchanged_by_the_blind_spot_field():
    """The default must be exactly the old behaviour, or every prior finding moves."""
    plain = AnalystPolicy("plain")
    assert plain.blind_compartment is None
    corpus = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=40)))
    for task in corpus:
        assert plain.evidence_visible_to(task) == evidence_shown(task)


def test_the_blind_reviewer_credits_less_never_more(tasks):
    """A first version returned every fact id and made the blind reviewer see MORE."""
    blind = AnalystPolicy("blind", blind_compartment=BLIND)
    for task in tasks:
        visible = blind.evidence_visible_to(task)
        assert visible <= evidence_shown(task)


def test_the_blind_spot_actually_corrupts_something(tasks):
    """A blind spot that changes no verdict would make finding 21 measure nothing."""
    reference, blind = AnalystPolicy("r"), AnalystPolicy("b", blind_compartment=BLIND)
    rng = random.Random(0)
    changed = [t for t in tasks if reference.verdict_for(t, rng) != blind.verdict_for(t, rng)]
    assert changed, "the blind spot changes no verdict; the experiment is void"
    # Every change is an error, never a lucky correction: crediting less evidence can
    # only move a verdict toward ROUTINE, and the affected tasks are all significant.
    assert all(t.significant for t in changed)


def _channel_means(tasks, compartment):
    carrying = [t for t in tasks if any(compartment in r.label.compartments for r in t.sources)]
    rest = [t for t in tasks if t not in set(carrying)]
    with_it = sum(len(evidence_shown(t)) for t in carrying) / len(carrying)
    without = sum(len(evidence_shown(t)) for t in rest) / len(rest)
    return with_it, without


def test_the_affected_slice_is_always_max_difficulty_for_every_channel(tasks):
    """The property the old guard was written against, shown to be vacuous.

    Blinding only removes evidence and the threshold is 3 of 3, so a verdict can flip
    only on a task whose visible evidence was exactly 3. The affected mean is therefore
    3.00 for EVERY compartment, and a guard comparing it to the corpus mean passes
    unconditionally. This test exists so that fact is recorded rather than rediscovered.
    """
    rng = random.Random(0)
    reference = AnalystPolicy("r")
    for compartment in Compartment:
        blind = AnalystPolicy("b", blind_compartment=compartment)
        changed = [t for t in tasks if reference.verdict_for(t, rng) != blind.verdict_for(t, rng)]
        if not changed:
            continue
        assert {len(evidence_shown(t)) for t in changed} == {3}


def test_the_guard_rejects_the_channel_the_docs_say_is_unusable(tasks):
    """SENSOR must fail and PARTNER must pass, or the guard discriminates nothing.

    The previous guard passed for every compartment including SENSOR, while the prose
    claimed the choice was "asserted in the script rather than trusted".
    """
    with_p, without_p = _channel_means(tasks, Compartment.PARTNER)
    with_s, without_s = _channel_means(tasks, Compartment.SENSOR)
    assert abs(with_p - without_p) <= CHANNEL_ENTANGLEMENT_SLACK
    assert abs(with_s - without_s) > CHANNEL_ENTANGLEMENT_SLACK


def test_the_corrupted_slice_sits_at_the_opposite_extreme_from_the_boundary(tasks):
    """Anti-correlation, which is the real mechanism -- not orthogonality.

    A threshold error hits boundary items; this blind spot hits the unambiguous end.
    That opposition is why the fleet is otherwise unanimous on the corrupted slice, and
    it is what makes the disagreement signal vanish at unanimity.
    """
    reference, blind = AnalystPolicy("r"), AnalystPolicy("b", blind_compartment=BLIND)
    rng = random.Random(0)
    changed = [t for t in tasks if reference.verdict_for(t, rng) != blind.verdict_for(t, rng)]
    affected_mean = sum(len(evidence_shown(t)) for t in changed) / len(changed)
    corpus_mean = sum(len(evidence_shown(t)) for t in tasks) / len(tasks)
    assert affected_mean > corpus_mean


def test_the_fleet_sheds_compartments_so_the_corruption_reaches_the_aggregator():
    """Under the fail-closed default every affected task is escalated and lost.

    This is the defect the first run of finding 21 had: the blind spot reached the
    aggregator on zero tasks and the experiment silently measured nothing.
    """
    fleet = blind_fleet(9, 9)
    assert all(p.release_policy is DROP_COMPARTMENTS for p in fleet)
    assert all(p.blind_compartment is BLIND for p in fleet)


def test_the_fleet_splits_at_the_requested_share():
    fleet = blind_fleet(7, 9)
    assert len(fleet) == 9
    assert sum(1 for p in fleet if p.blind_compartment is BLIND) == 7
    assert sum(1 for p in fleet if p.blind_compartment is None) == 2
    # Every reviewer holds the correct threshold: this is not a fleet of over-escalators.
    assert all(p.escalation_threshold == 3 for p in fleet)


def test_the_sweep_reaches_unanimity():
    """The interesting cell is the one where no dissent remains."""
    assert max(SHARES) == 9
    assert 0 in SHARES
    assert list(SHARES) == sorted(SHARES)
    assert max(BUDGETS) < EVENTS


def test_the_channel_policy_finds_what_disagreement_cannot():
    """Finding 23: the detector names a channel, and that is enough to select on.

    Every policy in `DEPLOYABLE` reads disagreement, and a unanimously blind fleet has
    none -- which is the whole of finding 21. Finding 22 supplies the missing input: it
    names *which* channel is being discounted, from data the aggregator already holds.
    Selecting tasks that carry that channel, deepest evidence first, then finds every
    corrupted item at the unanimity where uncertainty sampling is at chance.

    Both halves are asserted, because the second is what makes the first meaningful:
    the channel policy matches the oracle, and the disagreement-based policies do not.
    """
    payload = artifact("blind_spot")
    unanimous = str(payload["fleet"])
    rates = payload["audit_hit_rate"][unanimous]

    assert rates["channel"] == 1.0, (
        "the channel policy no longer finds every corrupted item at unanimity; "
        "finding 23's whole claim is that provenance succeeds where disagreement fails"
    )
    assert rates["channel"] == rates["oracle"], "the channel policy no longer ties the bound"
    for name in ("uniform", "margin", "posterior", "consensus"):
        assert rates[name] <= 0.25, (
            f"{name} is no longer at chance at unanimity, which would mean finding 21's "
            "premise has changed and finding 23 is answering a question nobody has"
        )


def test_finding_all_of_them_is_not_repairing_any_of_them():
    """The limit finding 23 must be quoted with, and the reason it is not a solution.

    At a 20-item budget the channel policy drives `remaining_errors` to zero: every
    corrupted label in the corpus is correct afterwards. And `corrected` stays at zero,
    because not one *unanchored* label changed. The authority overrode the twenty items
    it ruled on and the estimator learned nothing from any of them.

    The oracle does the same thing, which is the load-bearing part: an obstacle that
    defeats a policy handed the ground truth is not a selection problem, so no better
    selection rule closes it. That is what keeps item 7 of the LAS alignment open.
    """
    payload = artifact("blind_spot")
    unanimous = payload["fleet"]

    def row(policy: str, budget: int) -> dict[str, Any]:
        return next(
            r
            for r in payload["grid"]
            if r["n_blind"] == unanimous and r["policy"] == policy and r["budget"] == budget
        )

    for policy in ("channel", "oracle"):
        full = row(policy, 20)
        assert full["remaining_errors"] == 0, f"{policy} left corrupted labels at budget 20"
        assert full["hits"] == 20, f"{policy} did not audit all twenty corrupted items"
        assert full["corrected"] == 0, (
            f"{policy} corrected an unanchored label, which would be a real repair and "
            "would mean this test's finding has changed"
        )
        assert not full["repaired"], "repaired must stay false while corrected is zero"
