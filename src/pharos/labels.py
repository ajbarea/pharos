"""The label lattice: what an object carries, who may read it, and what may be released.

Three axes, following the label-plus-type construction that information-flow
control for agents uses. Sensitivity is a total order, so joining is a maximum.
Compartments form a subset lattice, so joining is a union and two holders at the
same level can still be unable to read each other. Capacity is the form of a
derived output, and it is what makes declassification decidable at all.

The reason capacity belongs in the lattice is label creep. An entry's label is
the join over every object that fed the turn, and a conservative join drives
every derived entry to the top of the lattice, at which point nothing is ever
releasable and the whole federation degrades to local-only learning. Capacity
breaks that: an output whose form cannot carry an instance does not need to
inherit the level of the instances it was derived from.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Sensitivity(IntEnum):
    """Classification level. A total order, so a join is a maximum."""

    OPEN = 0
    INTERNAL = 1
    PROTECTED = 2
    RESTRICTED = 3


class Compartment(StrEnum):
    """Need-to-know compartment. A subset lattice, so a join is a union.

    Cross-cutting by construction: an officer cleared for SENSOR and LEGAL and
    one cleared for LIAISON and PARTNER share a level and still cannot pool what
    they know. That incomparability is what makes this a lattice test rather
    than a ladder test.
    """

    SENSOR = "SENSOR"
    LIAISON = "LIAISON"
    LEGAL = "LEGAL"
    PARTNER = "PARTNER"


class Capacity(IntEnum):
    """How much an output's form can carry, which is what permits declassification.

    Ordered by carrying capacity, not by sensitivity. An ENUM is a bounded
    choice and cannot smuggle a record; FREETEXT can reproduce one verbatim.
    """

    ENUM = 0
    SCALAR = 1
    SPAN = 2
    FREETEXT = 3


@dataclass(frozen=True, slots=True)
class Label:
    """One object's classification, plus the form of the object carrying it."""

    sensitivity: Sensitivity
    compartments: frozenset[Compartment]
    capacity: Capacity

    def dominates(self, other: "Label") -> bool:
        """Whether a holder of this label may read `other`.

        Both conditions bind: the level must be at least as high, and the
        holder's compartment set must contain the object's. Equal levels with
        incomparable compartments dominate in neither direction.
        """
        return self.sensitivity >= other.sensitivity and other.compartments <= self.compartments


def join(labels: Iterable[Label], *, capacity: Capacity) -> Label:
    """The least upper bound of `labels`, at the form `capacity` of the derived output.

    `capacity` is required rather than joined. An enum verdict is an enum verdict
    however sensitive its inputs were, and conflating the two is precisely what
    makes a derived label creep to the top of the lattice.

    An empty input joins to the bottom of the lattice at the given capacity,
    which is the identity a fold needs.
    """
    sensitivity = Sensitivity.OPEN
    compartments: frozenset[Compartment] = frozenset()
    for label in labels:
        sensitivity = max(sensitivity, label.sensitivity)
        compartments |= label.compartments
    return Label(Sensitivity(sensitivity), compartments, capacity)


@dataclass(frozen=True, slots=True)
class DeclassificationPolicy:
    """When a low-capacity output may be released below the level of its inputs.

    `declassifiable` names the capacities eligible at all; a capacity absent from
    it is never released, so an unrecognized form fails closed.
    `release_floor` is the level an eligible output drops to.
    `drop_compartments` defaults to False, so compartments survive: shedding a
    compartment discloses that the compartment had something to say, which is a
    policy act in its own right rather than an inference from low capacity.
    """

    declassifiable: frozenset[Capacity] = frozenset({Capacity.ENUM, Capacity.SCALAR})
    release_floor: Sensitivity = Sensitivity.OPEN
    drop_compartments: bool = False


def declassify(label: Label, policy: DeclassificationPolicy) -> Label:
    """`label` as it may be released, or unchanged when it may not be.

    Never raises the level. A label already below the release floor stays where it
    is: `min`, not assignment. This read `Label(policy.release_floor, ...)`
    unconditionally, so under the PROTECTED-floor policy in `cases/disclosure.json`
    an OPEN input came back PROTECTED -- a function named `declassify` classifying
    something up. The error was in the safe direction, which is why nothing caught
    it, and it was found by checking the lattice laws exhaustively rather than by
    any case in the table happening to exercise it.
    """
    if label.capacity not in policy.declassifiable:
        return label
    compartments = frozenset() if policy.drop_compartments else label.compartments
    released = min(label.sensitivity, policy.release_floor)
    return Label(Sensitivity(released), compartments, label.capacity)


def shared_eligible(label: Label, release_ceiling: Label, policy: DeclassificationPolicy) -> bool:
    """Whether an entry carrying `label` may train an adapter released at `release_ceiling`.

    This is the gradient boundary: the question is not whether the analyst could
    read the object, but whether what the object contributes may leave at the
    ceiling the shared adapter is released under.

    Kept boolean, and kept here, because it is the *unattended* question: a
    training pass has nobody to ask. `pharos.disclosure.decide` answers the richer
    one, distinguishing an entry a cleared human could authorise from one that can
    never leave, and this is exactly its `RELEASE` case. Anything that can consult
    an authority should call that instead.
    """
    return release_ceiling.dominates(declassify(label, policy))
