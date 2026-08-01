"""The graded release decision, and the audited cases it has to reproduce.

The case table is the point of this file. Property tests say the decision is
self-consistent; the cases say it is *right*, one worked example per gate, each
with a written pass criterion a reader can check without running anything.
"""

from typing import Any

import pytest

from pharos.disclosure import (
    CARRYING_CAPACITIES,
    Disposition,
    ProhibitedUse,
    Purpose,
    Reason,
    ReleasePolicy,
    decide,
    load_cases,
    most_restrictive,
)
from pharos.labels import (
    Capacity,
    Compartment,
    DeclassificationPolicy,
    Label,
    Sensitivity,
    shared_eligible,
)

CASES = load_cases()


def _label(spec: dict[str, Any]) -> Label:
    return Label(
        Sensitivity[spec["sensitivity"]],
        frozenset(Compartment[c] for c in spec["compartments"]),
        Capacity[spec["capacity"]],
    )


def _policy(name: str) -> ReleasePolicy:
    spec = CASES["policies"][name]
    return ReleasePolicy(
        declassification=DeclassificationPolicy(
            declassifiable=frozenset(Capacity[c] for c in spec["declassifiable"]),
            release_floor=Sensitivity[spec["release_floor"]],
            drop_compartments=spec["drop_compartments"],
        ),
        prohibited=frozenset(
            ProhibitedUse(Compartment[p["compartment"]], Purpose(p["purpose"]))
            for p in spec["prohibited"]
        ),
    )


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["id"])
def test_ground_truth_case(case):
    decision = decide(
        _label(case["label"]),
        _label(CASES["ceilings"][case["ceiling"]]),
        _policy(case["policy"]),
        purpose=Purpose(case["purpose"]),
    )
    assert decision.disposition == Disposition(case["expected_disposition"]), case["pass_criteria"]
    assert decision.reason == Reason(case["expected_reason"]), case["pass_criteria"]


def test_every_case_is_marked_verified():
    """An unverified row is a guess, and a guess in a case table is worse than a gap."""
    unverified = [c["id"] for c in CASES["cases"] if not c.get("verified")]
    assert unverified == []


def test_every_gate_has_a_case():
    """Each reason code must be exercised, or the table is documenting a subset."""
    covered = {c["expected_reason"] for c in CASES["cases"]}
    assert covered == {r.value for r in Reason}


def test_every_case_carries_its_own_criterion():
    for case in CASES["cases"]:
        assert case["pass_criteria"].strip(), f"{case['id']} has no stated criterion"
        assert case["gate_under_test"].strip(), f"{case['id']} names no gate"


# ------------------------------------------------------------- properties -----

FLEET = Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)


def test_release_agrees_with_the_unattended_boolean():
    """The graded decision and `shared_eligible` must never split on the same entry.

    They answer different questions -- may this leave unattended, and what may
    happen to it -- but RELEASE *is* the boolean's true case. A disagreement would
    mean a training pass and a review reach opposite conclusions about one entry.
    """
    policy = ReleasePolicy()
    for sensitivity in Sensitivity:
        for capacity in Capacity:
            for compartments in (
                frozenset(),
                frozenset({Compartment.SENSOR}),
                frozenset({Compartment.LEGAL}),
                frozenset({Compartment.SENSOR, Compartment.PARTNER}),
            ):
                label = Label(sensitivity, compartments, capacity)
                decision = decide(label, FLEET, policy)
                assert decision.may_release == shared_eligible(
                    label, FLEET, policy.declassification
                ), f"{label} splits the two answers"


def test_only_a_compartment_shortfall_is_authorizable():
    """Escalation is offered for exactly one cause, and that is a design claim.

    A level shortfall and a carrying capacity are not rulings anybody is entitled
    to make, so offering approval on them would invite a request that must be
    refused.
    """
    policy = ReleasePolicy()
    for sensitivity in Sensitivity:
        for capacity in Capacity:
            for compartments in (frozenset(), frozenset({Compartment.LEGAL})):
                decision = decide(Label(sensitivity, compartments, capacity), FLEET, policy)
                assert decision.is_authorizable == (decision.reason is Reason.COMPARTMENT_NOT_HELD)


def test_a_level_shortfall_outranks_a_compartment_one():
    """Both objections hold; the unauthorizable one has to be the one reported."""
    label = Label(Sensitivity.RESTRICTED, frozenset({Compartment.LEGAL}), Capacity.ENUM)
    ceiling = Label(Sensitivity.OPEN, frozenset(), Capacity.FREETEXT)
    # A floor high enough that declassification cannot reach the ceiling.
    policy = ReleasePolicy(
        declassification=DeclassificationPolicy(release_floor=Sensitivity.PROTECTED)
    )
    decision = decide(label, ceiling, policy)
    assert decision.disposition is Disposition.WITHHOLD
    assert decision.reason is Reason.LEVEL_ABOVE_CEILING


def test_a_prohibited_purpose_beats_a_policy_that_would_release():
    label = Label(Sensitivity.PROTECTED, frozenset({Compartment.LEGAL}), Capacity.ENUM)
    permissive = DeclassificationPolicy(drop_compartments=True)
    veto = ProhibitedUse(Compartment.LEGAL, Purpose.FLEET_TRAINING, "not for training")

    assert decide(label, FLEET, ReleasePolicy(permissive)).may_release

    guarded = ReleasePolicy(permissive, frozenset({veto}))
    blocked = decide(label, FLEET, guarded, purpose=Purpose.FLEET_TRAINING)
    assert blocked.disposition is Disposition.WITHHOLD
    assert blocked.reason is Reason.PURPOSE_PROHIBITED
    assert blocked.prohibition == veto
    assert not blocked.is_authorizable, "a purpose veto is not a clearance question"

    # Same label, same policy, a purpose the declaration does not name.
    assert decide(label, FLEET, guarded, purpose=Purpose.INCIDENT_REVIEW).may_release


def test_a_prohibition_on_an_absent_compartment_does_nothing():
    label = Label(Sensitivity.PROTECTED, frozenset({Compartment.SENSOR}), Capacity.ENUM)
    guarded = ReleasePolicy(
        prohibited=frozenset({ProhibitedUse(Compartment.LEGAL, Purpose.FLEET_TRAINING)})
    )
    assert decide(label, FLEET, guarded).may_release


def test_prohibition_selection_does_not_depend_on_set_iteration_order():
    """Two applicable vetoes must always report the same one, or a message flaps."""
    label = Label(
        Sensitivity.PROTECTED, frozenset({Compartment.LEGAL, Compartment.PARTNER}), Capacity.ENUM
    )
    uses = [
        ProhibitedUse(Compartment.PARTNER, Purpose.FLEET_TRAINING),
        ProhibitedUse(Compartment.LEGAL, Purpose.FLEET_TRAINING),
    ]
    first = decide(label, FLEET, ReleasePolicy(prohibited=frozenset(uses))).prohibition
    second = decide(label, FLEET, ReleasePolicy(prohibited=frozenset(reversed(uses)))).prohibition
    assert first == second


def test_a_batch_is_only_as_releasable_as_its_worst_member():
    policy = ReleasePolicy()
    releasable = decide(
        Label(Sensitivity.PROTECTED, frozenset({Compartment.SENSOR}), Capacity.ENUM), FLEET, policy
    )
    escalatable = decide(
        Label(Sensitivity.PROTECTED, frozenset({Compartment.LEGAL}), Capacity.ENUM), FLEET, policy
    )
    withheld = decide(
        Label(Sensitivity.PROTECTED, frozenset({Compartment.SENSOR}), Capacity.FREETEXT),
        FLEET,
        policy,
    )

    assert most_restrictive([releasable]) is Disposition.RELEASE
    assert most_restrictive([releasable, escalatable]) is Disposition.APPROVAL
    assert most_restrictive([releasable, escalatable, withheld]) is Disposition.WITHHOLD
    assert most_restrictive([withheld, releasable]) is Disposition.WITHHOLD
    assert most_restrictive([]) is Disposition.RELEASE


def test_carrying_capacities_are_the_ones_no_default_policy_declassifies():
    default = DeclassificationPolicy()
    assert frozenset(set(Capacity) - default.declassifiable) == CARRYING_CAPACITIES


def test_decision_serialises_its_reason_and_prohibition():
    veto = ProhibitedUse(Compartment.LEGAL, Purpose.FLEET_TRAINING)
    decision = decide(
        Label(Sensitivity.PROTECTED, frozenset({Compartment.LEGAL}), Capacity.ENUM),
        FLEET,
        ReleasePolicy(prohibited=frozenset({veto})),
    )
    payload = decision.as_dict()
    assert payload["disposition"] == "withhold"
    assert payload["reason"] == "PURPOSE_PROHIBITED"
    assert payload["prohibition"] == "LEGAL/fleet_training"

    clean = decide(
        Label(Sensitivity.PROTECTED, frozenset({Compartment.SENSOR}), Capacity.ENUM),
        FLEET,
        ReleasePolicy(),
    )
    assert clean.as_dict()["prohibition"] is None
    assert clean.as_dict()["released"] == "OPEN[SENSOR]@ENUM"
