"""Whether a derived output may leave, and when the answer is "ask someone".

This replaces a boolean. `labels.shared_eligible` answered *may this release* with
yes or no, and that binary quietly decided two findings. Finding 2 reported that
eligibility is bimodal on the compartment ruling; finding 7 reported that no
fail-closed reviewer's correction could clear the ceiling. Both are true of a
system with two doors, and both were measured on one that had exactly two because
that is how the function was written, not because that is how disclosure works.

Governed systems do not stop at allow and deny. The Metro City Meteors governance
suite built during SCADS distinguishes five outcomes, and the makesense policy
engine underneath it three: `auto_ok`, `needs_approval`, `blocked`. The middle one
is the interesting one, and Pharos had no counterpart to it.

**Where the middle door belongs, and why.** A compartment shortfall becomes
`APPROVAL` rather than `WITHHOLD`. That is not a softening; it follows from finding
2's own conclusion. Shedding a compartment was found there to be *a policy act
rather than an engineering problem*, and a policy act is precisely the thing a human
with authority can perform and a rule cannot. A level above the ceiling is not like
that: no one authorises a downgrade by being asked, so it stays `WITHHOLD`. Neither
is a capacity that can carry an instance verbatim, because the objection is to the
form of the output rather than to who is reading it.

**A reason travels with every decision.** A caller that learns only "blocked" cannot
tell a compartment it might get cleared for from a prohibition it will never get
past. Finding 7 measured what unattributed objections cost a learner; emitting the
locus without the reason would have reproduced that defect one layer down.

**Purpose is a separate axis from clearance.** The lattice answers who may read.
It does not answer what the reading may be *for*, and a data owner's declaration
that a compartment must not serve some purpose is not overridable by clearance --
so a prohibited purpose is checked first and denies outright. This is the same
distinction `makesense.types.ProhibitedUse` draws, narrowed to the vocabulary
Pharos has.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pharos.labels import (
    Capacity,
    Compartment,
    DeclassificationPolicy,
    Label,
    declassify,
    shared_eligible,
)

#: The fail-closed default from finding 2: an eligible output drops to OPEN but
#: keeps every compartment it inherited.
KEEP_COMPARTMENTS = DeclassificationPolicy()

#: The one ruling finding 2 showed the federation's viability turns on.
DROP_COMPARTMENTS = DeclassificationPolicy(drop_compartments=True)


class Disposition(StrEnum):
    """What may happen to a derived output.

    Ordered by how much it lets out, so a fold over several sources can take the
    most restrictive without a lookup table.
    """

    RELEASE = "release"
    APPROVAL = "approval"
    WITHHOLD = "withhold"


#: Most restrictive first, which is the order a conservative fold must respect.
_SEVERITY: dict[Disposition, int] = {
    Disposition.WITHHOLD: 2,
    Disposition.APPROVAL: 1,
    Disposition.RELEASE: 0,
}


class Reason(StrEnum):
    """Why a decision came out the way it did.

    One code per distinct cause, because a caller acts differently on each: a
    compartment shortfall is worth escalating, a prohibited purpose never is, and a
    capacity objection is fixed by changing the shape of the output rather than by
    asking anyone.
    """

    RELEASABLE = "RELEASABLE"
    COMPARTMENT_NOT_HELD = "COMPARTMENT_NOT_HELD"
    LEVEL_ABOVE_CEILING = "LEVEL_ABOVE_CEILING"
    CAPACITY_NOT_DECLASSIFIABLE = "CAPACITY_NOT_DECLASSIFIABLE"
    PURPOSE_PROHIBITED = "PURPOSE_PROHIBITED"


class Purpose(StrEnum):
    """What a release is for.

    Deliberately few. The point is that purpose is a dimension the lattice does not
    carry, not that this particular list is the right taxonomy for anything.
    """

    FLEET_TRAINING = "fleet_training"
    INCIDENT_REVIEW = "incident_review"
    PUBLIC_REPORTING = "public_reporting"


@dataclass(frozen=True, slots=True)
class ProhibitedUse:
    """A data owner's declaration that a compartment must not serve a purpose.

    Distinct from a clearance rule in who may lift it: a clearance shortfall can be
    authorised by someone cleared, and this cannot be authorised at all. It is the
    reason `PURPOSE_PROHIBITED` denies rather than escalates.
    """

    compartment: Compartment
    purpose: Purpose
    description: str = ""


@dataclass(frozen=True, slots=True)
class ReleasePolicy:
    """A declassification ruling, plus the uses its owners have ruled out.

    Wraps `DeclassificationPolicy` rather than extending it. That type is the
    lattice's own algebra and is used by `declassify` in contexts with no notion of
    purpose; bolting a purpose list onto it would push a policy concern into the
    label module.
    """

    declassification: DeclassificationPolicy = KEEP_COMPARTMENTS
    prohibited: frozenset[ProhibitedUse] = field(default_factory=frozenset)

    def prohibits(self, label: Label, purpose: Purpose) -> ProhibitedUse | None:
        """The first declaration this release would violate, or None."""
        for use in sorted(self.prohibited, key=lambda u: (str(u.compartment), str(u.purpose))):
            if use.purpose is purpose and use.compartment in label.compartments:
                return use
        return None


#: The fail-closed ruling with no owner vetoes, as a singleton so it can be an
#: argument default. Frozen, so sharing one instance is safe.
DEFAULT_RELEASE_POLICY = ReleasePolicy()


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    """What may happen, why, and the label it would carry if it happens."""

    disposition: Disposition
    reason: Reason
    released: Label
    prohibition: ProhibitedUse | None = None

    @property
    def may_release(self) -> bool:
        """Whether this leaves with no further authority. The old boolean."""
        return self.disposition is Disposition.RELEASE

    @property
    def is_authorizable(self) -> bool:
        """Whether a human with authority could let this out.

        False for a level shortfall, a capacity objection, and a prohibited
        purpose. True only where the block is a ruling somebody is entitled to make.
        """
        return self.disposition is Disposition.APPROVAL

    def as_dict(self) -> dict[str, object]:
        return {
            "disposition": str(self.disposition),
            "reason": str(self.reason),
            "released": (
                f"{self.released.sensitivity.name}"
                f"[{','.join(sorted(str(c) for c in self.released.compartments))}]"
                f"@{self.released.capacity.name}"
            ),
            "prohibition": (
                None
                if self.prohibition is None
                else f"{self.prohibition.compartment}/{self.prohibition.purpose}"
            ),
        }


def _classify(released: Label, ceiling: Label) -> tuple[Disposition, Reason]:
    """Why an already-derived label does or does not clear `ceiling`.

    Capacity is not consulted: `Label.dominates` compares levels and compartments,
    and capacity decided earlier whether a downgrade was available at all. Shared by
    `decide` and `admit` so the two cannot disagree about the same pair.
    """
    if ceiling.dominates(released):
        return Disposition.RELEASE, Reason.RELEASABLE
    if released.sensitivity > ceiling.sensitivity:
        return Disposition.WITHHOLD, Reason.LEVEL_ABOVE_CEILING
    return Disposition.APPROVAL, Reason.COMPARTMENT_NOT_HELD


def admit(
    released: Label,
    ceiling: Label,
    policy: ReleasePolicy = DEFAULT_RELEASE_POLICY,
    *,
    purpose: Purpose = Purpose.FLEET_TRAINING,
) -> ReleaseDecision:
    """Whether a label that has *already* been derived may leave at `ceiling`.

    Distinct from `decide`, which starts from a source label and applies a
    declassification ruling to it. The two are easy to confuse and the confusion is
    not harmless: running a ruling over an already-released label declassifies it
    twice, and a permissive reviewer would silently shed compartments the proposer
    deliberately kept. Use this to judge a proposal; use `decide` to derive one.
    """
    prohibition = policy.prohibits(released, purpose)
    if prohibition is not None:
        return ReleaseDecision(
            Disposition.WITHHOLD, Reason.PURPOSE_PROHIBITED, released, prohibition
        )
    disposition, reason = _classify(released, ceiling)
    return ReleaseDecision(disposition, reason, released)


def decide(
    label: Label,
    ceiling: Label,
    policy: ReleasePolicy,
    *,
    purpose: Purpose = Purpose.FLEET_TRAINING,
) -> ReleaseDecision:
    """Whether an entry carrying `label` may leave at `ceiling`, and on what grounds.

    Checks run most-binding first, so a label failing on several counts reports the
    one that cannot be lifted rather than the one that can. A label both above the
    ceiling and outside its compartments is `WITHHOLD`, not `APPROVAL`: reporting
    the authorizable objection would invite an escalation that must still be
    refused, which is a worse outcome than a clear no.
    """
    prohibition = policy.prohibits(label, purpose)
    if prohibition is not None:
        return ReleaseDecision(Disposition.WITHHOLD, Reason.PURPOSE_PROHIBITED, label, prohibition)

    released = declassify(label, policy.declassification)
    # Asked through `shared_eligible` rather than restated, so the unattended
    # boolean and the graded decision cannot answer differently. If they ever did,
    # a training pass and a review would disagree about the same entry.
    if shared_eligible(label, ceiling, policy.declassification):
        return ReleaseDecision(Disposition.RELEASE, Reason.RELEASABLE, released)

    if label.capacity not in policy.declassification.declassifiable:
        return ReleaseDecision(Disposition.WITHHOLD, Reason.CAPACITY_NOT_DECLASSIFIABLE, released)

    # Whatever is left is a level or a compartment, and `_classify` decides which.
    # A compartment shortfall lands on APPROVAL: finding 2 identified shedding one
    # as a policy act rather than an engineering problem, which is what makes it the
    # one objection an authority can answer.
    disposition, reason = _classify(released, ceiling)
    return ReleaseDecision(disposition, reason, released)


def most_restrictive(decisions: Iterable[ReleaseDecision]) -> Disposition:
    """The disposition a set of outputs must be handled under, together.

    A batch is only as releasable as its worst member, and folding by taking the
    maximum severity is the only combination that cannot leak: any rule that let a
    releasable majority carry a withheld minority would release it.
    """
    worst = Disposition.RELEASE
    for decision in decisions:
        if _SEVERITY[decision.disposition] > _SEVERITY[worst]:
            worst = decision.disposition
    return worst


#: The capacities that can carry an instance verbatim, kept here so a reader can see
#: which side of the capacity check a form falls on without opening the lattice.
CARRYING_CAPACITIES: frozenset[Capacity] = frozenset({Capacity.SPAN, Capacity.FREETEXT})

#: The audited case table. Ships inside the package so a consumer can read the
#: policy's own worked examples without a checkout, which is the point of writing
#: them as data rather than as assertions buried in a test.
CASES_PATH = Path(__file__).parent / "cases" / "disclosure.json"


def load_cases() -> dict[str, Any]:
    """The ground-truth cases, as written.

    Returned unparsed on purpose. The test that runs them builds the labels and
    policies from these strings, so a typo in the table fails the suite rather than
    being silently normalised into something that passes.
    """
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))
