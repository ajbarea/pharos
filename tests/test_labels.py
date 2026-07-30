"""The lattice algebra: joins, dominance, and what may be declassified."""

from pharos.labels import (
    Capacity,
    Compartment,
    DeclassificationPolicy,
    Label,
    Sensitivity,
    declassify,
    join,
    shared_eligible,
)

DEFAULT = DeclassificationPolicy()


def test_sensitivity_is_totally_ordered():
    assert Sensitivity.OPEN < Sensitivity.INTERNAL < Sensitivity.PROTECTED < Sensitivity.RESTRICTED


def test_join_takes_max_sensitivity_and_union_of_compartments():
    a = Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    b = Label(Sensitivity.RESTRICTED, frozenset({Compartment.LEGAL}), Capacity.FREETEXT)
    result = join([a, b], capacity=Capacity.ENUM)
    assert result.sensitivity is Sensitivity.RESTRICTED
    assert result.compartments == frozenset({Compartment.SENSOR, Compartment.LEGAL})


def test_join_takes_capacity_from_the_output_not_the_inputs():
    # Capacity is a property of the form of the derived output, never of what fed it.
    src = Label(Sensitivity.RESTRICTED, frozenset(), Capacity.FREETEXT)
    assert join([src], capacity=Capacity.ENUM).capacity is Capacity.ENUM


def test_join_of_nothing_is_the_bottom_label():
    assert join([], capacity=Capacity.ENUM) == Label(Sensitivity.OPEN, frozenset(), Capacity.ENUM)


def test_equal_sensitivity_incomparable_compartments_do_not_dominate():
    holder = Label(Sensitivity.PROTECTED, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    other = Label(Sensitivity.PROTECTED, frozenset({Compartment.LIAISON}), Capacity.FREETEXT)
    assert not holder.dominates(other)
    assert not other.dominates(holder)


def test_dominates_requires_both_level_and_compartments():
    holder = Label(
        Sensitivity.RESTRICTED,
        frozenset({Compartment.SENSOR, Compartment.LEGAL}),
        Capacity.FREETEXT,
    )
    readable = Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    unreadable = Label(Sensitivity.INTERNAL, frozenset({Compartment.PARTNER}), Capacity.FREETEXT)
    assert holder.dominates(readable)
    assert not holder.dominates(unreadable)


def test_join_is_associative_and_commutative_over_three_labels():
    a = Label(Sensitivity.OPEN, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    b = Label(Sensitivity.PROTECTED, frozenset({Compartment.LEGAL}), Capacity.FREETEXT)
    c = Label(Sensitivity.INTERNAL, frozenset({Compartment.PARTNER}), Capacity.FREETEXT)
    forward = join([a, b, c], capacity=Capacity.SPAN)
    reverse = join([c, b, a], capacity=Capacity.SPAN)
    assert forward == reverse


def test_freetext_never_declassifies():
    label = Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    assert declassify(label, DEFAULT) == label


def test_span_never_declassifies():
    label = Label(Sensitivity.PROTECTED, frozenset(), Capacity.SPAN)
    assert declassify(label, DEFAULT) == label


def test_enum_drops_to_the_release_floor():
    label = Label(Sensitivity.RESTRICTED, frozenset(), Capacity.ENUM)
    assert declassify(label, DEFAULT).sensitivity is Sensitivity.OPEN


def test_compartments_survive_declassification_by_default():
    # Fail closed: dropping a compartment is a policy act, not a side effect of low capacity.
    label = Label(Sensitivity.RESTRICTED, frozenset({Compartment.LEGAL}), Capacity.ENUM)
    assert declassify(label, DEFAULT).compartments == frozenset({Compartment.LEGAL})


def test_compartments_drop_only_when_policy_says_so():
    policy = DeclassificationPolicy(drop_compartments=True)
    label = Label(Sensitivity.RESTRICTED, frozenset({Compartment.LEGAL}), Capacity.ENUM)
    assert declassify(label, policy).compartments == frozenset()


def test_shared_eligible_is_dominance_after_declassification():
    ceiling = Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    verdict = Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.ENUM)
    prose = Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    assert shared_eligible(verdict, ceiling, DEFAULT)
    assert not shared_eligible(prose, ceiling, DEFAULT)


def test_unknown_capacity_is_not_declassifiable():
    # Fail closed against a capacity the policy does not name.
    policy = DeclassificationPolicy(declassifiable=frozenset())
    label = Label(Sensitivity.RESTRICTED, frozenset(), Capacity.ENUM)
    assert declassify(label, policy) == label


def test_label_creep_is_what_declassification_prevents():
    """A verdict over many restricted sources is releasable; the prose is not.

    This is the whole point of carrying capacity in the lattice. Without it the
    join drives every derived entry to RESTRICTED and nothing federates.
    """
    sources = [
        Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.FREETEXT),
        Label(Sensitivity.PROTECTED, frozenset({Compartment.LEGAL}), Capacity.FREETEXT),
        Label(Sensitivity.INTERNAL, frozenset({Compartment.PARTNER}), Capacity.FREETEXT),
    ]
    crept = join(sources, capacity=Capacity.FREETEXT)
    assert crept.sensitivity is Sensitivity.RESTRICTED
    assert declassify(crept, DEFAULT) == crept

    verdict = join(sources, capacity=Capacity.ENUM)
    permissive = DeclassificationPolicy(drop_compartments=True)
    assert declassify(verdict, permissive).sensitivity is Sensitivity.OPEN
