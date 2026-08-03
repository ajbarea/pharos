"""The lattice algebra: joins, dominance, and what may be declassified."""

import itertools

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


# ------------------------------------------------ exhaustive lattice laws -----
#
# The lattice is 4 sensitivities x 16 compartment sets x 4 capacities = 256 labels,
# which is small enough to check every law on every element and every pair rather
# than on chosen examples. That matters because this is the governance core: a
# `join` that is not a least upper bound, or a `declassify` that moves a label the
# wrong way, is wrong for every finding built on it and wrong silently.
#
# This is how the `declassify` raising bug was found. Nothing in the case table
# happened to apply the PROTECTED-floor policy to a label already below PROTECTED,
# so no example test could have caught it; the exhaustive sweep did on the first run.

_POINTS = [
    (sensitivity, frozenset(combo))
    for sensitivity in Sensitivity
    for size in range(len(Compartment) + 1)
    for combo in itertools.combinations(list(Compartment), size)
]
_ALL_LABELS = [Label(s, c, k) for (s, c) in _POINTS for k in Capacity]


def test_dominates_is_a_partial_order():
    for a in _ALL_LABELS:
        assert a.dominates(a), f"not reflexive at {a}"
    for a, b in itertools.product(_ALL_LABELS, repeat=2):
        if a.dominates(b) and b.dominates(a):
            assert a.sensitivity == b.sensitivity and a.compartments == b.compartments, (
                f"mutual domination between distinct points: {a} {b}"
            )
    at_enum = [Label(s, c, Capacity.ENUM) for (s, c) in _POINTS]
    for a, b, c in itertools.product(at_enum, repeat=3):
        if a.dominates(b) and b.dominates(c):
            assert a.dominates(c), f"not transitive: {a} {b} {c}"


def test_incomparable_labels_exist_which_is_the_whole_point():
    """Equal levels with disjoint compartments order in neither direction.

    If this ever came out zero the lattice would have collapsed to a chain, and
    every finding about incomparability would be measuring nothing.
    """
    at_enum = [Label(s, c, Capacity.ENUM) for (s, c) in _POINTS]
    incomparable = sum(
        1
        for a, b in itertools.product(at_enum, repeat=2)
        if not a.dominates(b) and not b.dominates(a)
    )
    assert incomparable > 0
    # Roughly 62% of ordered pairs at the time of writing; assert the order of
    # magnitude rather than the exact count, which is a property of the enum sizes.
    assert incomparable > len(at_enum) ** 2 // 4


def test_join_is_the_least_upper_bound_of_every_pair():
    at_free = [Label(s, c, Capacity.FREETEXT) for (s, c) in _POINTS]
    for a, b in itertools.product(at_free, repeat=2):
        joined = join([a, b], capacity=Capacity.FREETEXT)
        assert joined.dominates(a) and joined.dominates(b), f"not an upper bound: {a} {b}"
        for candidate in at_free:
            if candidate.dominates(a) and candidate.dominates(b):
                assert candidate.dominates(joined), (
                    f"{candidate} bounds {a} and {b} but does not dominate the join {joined}"
                )


def test_join_is_commutative_associative_and_idempotent():
    for a, b in itertools.product(_ALL_LABELS, repeat=2):
        assert join([a, b], capacity=Capacity.ENUM) == join([b, a], capacity=Capacity.ENUM)
    for a in _ALL_LABELS:
        assert join([a, a], capacity=a.capacity) == a
        assert join([a], capacity=a.capacity) == a
    sample = _ALL_LABELS[:32]
    for a, b, c in itertools.product(sample, repeat=3):
        left = join([join([a, b], capacity=Capacity.ENUM), c], capacity=Capacity.ENUM)
        right = join([a, join([b, c], capacity=Capacity.ENUM)], capacity=Capacity.ENUM)
        assert left == right, f"not associative: {a} {b} {c}"
    assert join([], capacity=Capacity.ENUM) == Label(Sensitivity.OPEN, frozenset(), Capacity.ENUM)


def test_declassify_never_raises_a_label_under_any_policy():
    """A release operation that increases restriction is wrong by its own name.

    `declassify` assigned `policy.release_floor` unconditionally, so a label already
    below the floor came back higher. The shipped case table configures a
    PROTECTED floor, so this was reachable rather than hypothetical -- it simply
    errs in the safe direction, which is why it survived a full test suite.
    """
    policies = [
        DeclassificationPolicy(),
        DeclassificationPolicy(drop_compartments=True),
        DeclassificationPolicy(declassifiable=frozenset()),
        DeclassificationPolicy(release_floor=Sensitivity.INTERNAL),
        DeclassificationPolicy(release_floor=Sensitivity.PROTECTED),
        DeclassificationPolicy(release_floor=Sensitivity.RESTRICTED, drop_compartments=True),
    ]
    for policy in policies:
        for label in _ALL_LABELS:
            released = declassify(label, policy)
            assert released.capacity == label.capacity, "release must not change the form"
            assert label.dominates(released), f"{label} was raised to {released} under {policy}"
            assert declassify(released, policy) == released, "release must be idempotent"


def test_declassify_still_stops_at_a_high_floor():
    """The floor's actual job, which the `min` fix must not undo."""
    policy = DeclassificationPolicy(release_floor=Sensitivity.PROTECTED)
    high = Label(Sensitivity.RESTRICTED, frozenset({Compartment.LEGAL}), Capacity.ENUM)
    assert declassify(high, policy).sensitivity is Sensitivity.PROTECTED

    # And a label already below the floor is left alone rather than pushed up to it.
    low = Label(Sensitivity.OPEN, frozenset(), Capacity.ENUM)
    assert declassify(low, policy).sensitivity is Sensitivity.OPEN
