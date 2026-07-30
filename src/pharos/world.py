"""The fictional maritime watch: entities, channels, and the fact vocabulary.

Everything here is invented. No real vessel, operator, port, or organization is
referenced, and nothing in this module is derived from a real corpus.

The one invariant this module exists to protect: a significant event differs
from background only in *which facts co-occur*. It does not differ in how many
facts it carries, which channels report it, how long those reports run, or which
officer voice renders them. Every one of those would be a surface tell, and the
shortcut gate exists to catch exactly that kind of leak.
"""

from dataclasses import dataclass
from enum import StrEnum

from pharos.labels import Compartment, Sensitivity


class ReportType(StrEnum):
    """A channel a watch officer receives reporting through."""

    VESSEL_REPORT = "VESSEL_REPORT"
    SENSOR_TRACK = "SENSOR_TRACK"
    PORT_NOTICE = "PORT_NOTICE"
    CREW_MANIFEST = "CREW_MANIFEST"
    PRESS_ITEM = "PRESS_ITEM"
    LIAISON_TIP = "LIAISON_TIP"
    DETENTION_RECORD = "DETENTION_RECORD"
    PARTNER_REPORT = "PARTNER_REPORT"


#: The classification each channel confers on what arrives through it.
#: Two channels sit at RESTRICTED with incomparable compartments (LIAISON_TIP and
#: PARTNER_REPORT), which is the pair that makes the lattice non-trivial: an
#: officer cleared for one cannot read the other despite equal level.
CHANNEL_LABELS: dict[ReportType, tuple[Sensitivity, frozenset[Compartment]]] = {
    ReportType.VESSEL_REPORT: (Sensitivity.INTERNAL, frozenset({Compartment.SENSOR})),
    ReportType.SENSOR_TRACK: (Sensitivity.PROTECTED, frozenset({Compartment.SENSOR})),
    ReportType.PORT_NOTICE: (Sensitivity.OPEN, frozenset()),
    ReportType.CREW_MANIFEST: (Sensitivity.PROTECTED, frozenset({Compartment.LEGAL})),
    ReportType.PRESS_ITEM: (Sensitivity.OPEN, frozenset()),
    ReportType.LIAISON_TIP: (Sensitivity.RESTRICTED, frozenset({Compartment.LIAISON})),
    ReportType.DETENTION_RECORD: (Sensitivity.PROTECTED, frozenset({Compartment.LEGAL})),
    ReportType.PARTNER_REPORT: (Sensitivity.RESTRICTED, frozenset({Compartment.PARTNER})),
}


class Voice(StrEnum):
    """A regional center's reporting register.

    Voice varies word choice and framing, deliberately not length, because a
    voice that ran systematically longer would become a surface feature. Voice is
    also assigned independently of whether an event is significant, so it cannot
    correlate with the label the gate probes for.
    """

    TERSE = "TERSE"
    FORMAL = "FORMAL"
    NARRATIVE = "NARRATIVE"
    CLIPPED = "CLIPPED"


@dataclass(frozen=True, slots=True)
class Center:
    """A regional maritime operations center, and the officer voice it reports in."""

    center_id: str
    name: str
    voice: Voice


CENTERS: tuple[Center, ...] = (
    Center("MOC-1", "Northern Approaches", Voice.TERSE),
    Center("MOC-2", "Eastern Shoals", Voice.FORMAL),
    Center("MOC-3", "Southern Gate", Voice.NARRATIVE),
    Center("MOC-4", "Western Reach", Voice.CLIPPED),
)


@dataclass(frozen=True, slots=True)
class Operator:
    operator_id: str
    name: str


@dataclass(frozen=True, slots=True)
class Vessel:
    vessel_id: str
    name: str
    hull_type: str
    flag: str
    operator: Operator


OPERATORS: tuple[Operator, ...] = (
    Operator("OP-01", "Halcyon Shipping Management"),
    Operator("OP-02", "Meridian Lines"),
    Operator("OP-03", "Calder Maritime Holdings"),
    Operator("OP-04", "Tessaly Bulk Transport"),
    Operator("OP-05", "Ardent Coastal Freight"),
)

VESSEL_NAMES: tuple[str, ...] = (
    "Kestrel Dawn",
    "Northern Aster",
    "Selene Bright",
    "Corvid Passage",
    "Amber Lantern",
    "Thalia Grace",
    "Iron Meridian",
    "Pale Harrier",
    "Cordera Star",
    "Vesper Tide",
    "Blackmoor Fen",
    "Sable Crossing",
)

HULL_TYPES: tuple[str, ...] = (
    "bulk carrier",
    "general cargo",
    "product tanker",
    "container feeder",
    "refrigerated cargo",
)

FLAGS: tuple[str, ...] = ("Panama", "Liberia", "Malta", "Marshall Islands", "Cyprus")


#: The three facts whose co-occurrence defines a significant event. Any one or
#: two of them is unremarkable and appears freely in background reporting, which
#: is what forces a triage agent to reason over co-occurrence rather than spot a
#: keyword.
SIGNIFICANT_PATTERN: frozenset[str] = frozenset(
    {"draft_mismatch", "course_deviation", "unlit_contact"}
)

#: Co-occurring fact triples that are *not* significant. Background events draw
#: one of these, so both classes are structurally "one fixed triple plus two
#: fillers" and differ only in which triple.
#:
#: This exists because balancing per-fact rendering lengths by hand does not
#: converge. If plants are the only events with a deterministic triple, then any
#: property of those three facts (length, digit count, sentence shape) becomes a
#: class signal, and tuning one property just moves the leak to another. Decoys
#: remove the whole class of leak: whatever the significant facts look like,
#: background events look like some triple too.
#:
#: Each decoy holds at most two of the significant facts, which is also what
#: keeps any single significant fact from being a giveaway on its own. The task
#: is therefore to recognise *which* pattern co-occurs, not whether one does.
DECOY_PATTERNS: tuple[frozenset[str], ...] = (
    frozenset({"draft_mismatch", "course_deviation", "routine_transit"}),
    frozenset({"course_deviation", "unlit_contact", "weather_note"}),
    frozenset({"draft_mismatch", "unlit_contact", "cargo_declared"}),
    frozenset({"unlit_contact", "shoal_activity", "berth_closure"}),
    frozenset({"draft_mismatch", "pilot_absent", "inspection_history"}),
    frozenset({"crew_substitution", "certificate_unresolved", "inspection_history"}),
    frozenset({"reflagging", "demurrage_dispute", "ship_transfer_claim"}),
    frozenset({"course_deviation", "routine_transit", "weather_note"}),
    frozenset({"shoal_activity", "ship_transfer_claim", "cargo_declared"}),
    frozenset({"pilot_absent", "berth_closure", "reflagging"}),
)


@dataclass(frozen=True, slots=True)
class Fact:
    """One observable detail, and the channels that could plausibly carry it.

    Renderings obey two density rules, both learned from the shortcut gate
    rejecting an earlier draft of this vocabulary at AUC 0.74.

    First, comparable length. A fact whose phrasings ran longer than its peers
    would give every report carrying it a length signature, and since the
    significant facts appear together, that signature identifies plants with no
    semantics at all.

    Second, and less obvious: **every rendering carries exactly one timestamp and
    exactly two other numerals.** The first draft gave the significant facts
    numeric slots (draft, bearing, range, time) while most filler facts had none,
    so `digit_ratio` alone reached 0.65 and `has_timestamp` 0.57. Decimal points
    made it worse by inflating sentence counts. Uniform numeric density is
    therefore a hard invariant of this vocabulary, not a stylistic preference.
    """

    fact_id: str
    channels: frozenset[ReportType]
    renderings: tuple[str, ...]


FACTS: tuple[Fact, ...] = (
    Fact(
        "draft_mismatch",
        frozenset({ReportType.VESSEL_REPORT, ReportType.PORT_NOTICE, ReportType.SENSOR_TRACK}),
        (
            "At {time} the reported draft read {draft} metres, below the {laden} metre laden declaration.",
            "Draft was observed at {draft} metres at {time}, against {laden} metres declared as laden.",
            "By {time} the freeboard showed {draft} metres against a declared laden figure of {laden}.",
        ),
    ),
    Fact(
        "course_deviation",
        frozenset({ReportType.VESSEL_REPORT, ReportType.SENSOR_TRACK, ReportType.PARTNER_REPORT}),
        (
            "Course altered to {bearing} degrees at {time}, running {offset} degrees off the filed routing.",
            "Heading of {bearing} degrees at {time} sat some {offset} degrees off the declared route.",
            "At {time} the track turned to {bearing} degrees, a divergence of {offset} degrees from routing.",
        ),
    ),
    Fact(
        "unlit_contact",
        frozenset({ReportType.SENSOR_TRACK, ReportType.PARTNER_REPORT, ReportType.LIAISON_TIP}),
        (
            "An unlit contact closed to within {range} metres at {time} and held for {mins} minutes.",
            "An unlit contact came within {range} metres near {time} and was tracked {mins} minutes.",
            "A contact without lights reached {range} metres by {time}, observed over some {mins} minutes.",
        ),
    ),
    Fact(
        "pilot_absent",
        frozenset({ReportType.VESSEL_REPORT, ReportType.PORT_NOTICE}),
        (
            "Departure at {time} carried no pilot across {count} of {total} compulsory pilotage zones.",
            "No pilot embarked for the {time} sailing, covering {count} of {total} compulsory zones.",
            "The {time} departure went unpiloted through {count} of {total} pilotage zones.",
        ),
    ),
    Fact(
        "crew_substitution",
        frozenset({ReportType.CREW_MANIFEST, ReportType.DETENTION_RECORD}),
        (
            "An engineering substitution filed at {time} came {hours} hours before departure, {count} in total.",
            "The manifest logged {count} engineer changes at {time}, the last {hours} hours prior.",
            "A late engineering change lodged {time} was {count} of the period, {hours} hours ahead.",
        ),
    ),
    Fact(
        "certificate_unresolved",
        frozenset({ReportType.CREW_MANIFEST, ReportType.DETENTION_RECORD}),
        (
            "A registry query at {time} left {count} certificate numbers unresolved of {total} checked.",
            "At {time} the issuing registry returned no record for {count} of {total} certificates.",
            "Of {total} certificates checked at {time}, {count} failed to resolve against registry.",
        ),
    ),
    Fact(
        "berth_closure",
        frozenset({ReportType.PORT_NOTICE, ReportType.PRESS_ITEM}),
        (
            "From {time} berths {berth} and {berth2} are closed to all commercial traffic for scheduled works.",
            "Maintenance beginning {time} closes berths {berth} and {berth2} to all commercial movements.",
            "Commercial traffic is excluded from berths {berth} and {berth2} with effect from {time}.",
        ),
    ),
    Fact(
        "demurrage_dispute",
        frozenset({ReportType.PRESS_ITEM, ReportType.DETENTION_RECORD}),
        (
            "Trade press filed {time} reports demurrage disputes on {count} of {total} operator hulls.",
            "Reporting timed {time} places demurrage claims against {count} of {total} vessels.",
            "A {time} trade item lists {count} of {total} operator ships in demurrage dispute.",
        ),
    ),
    Fact(
        "reflagging",
        frozenset({ReportType.PRESS_ITEM, ReportType.VESSEL_REPORT}),
        (
            "Records up to {time} show {count} of {total} operator vessels were reflagged this period.",
            "As of {time}, {count} hulls of {total} under this operator had already changed flag.",
            "Flag changes recorded by {time} affected {count} of {total} vessels under this operator.",
        ),
    ),
    Fact(
        "ship_transfer_claim",
        frozenset({ReportType.LIAISON_TIP, ReportType.PARTNER_REPORT}),
        (
            "One account timed {time} places the hull at a transfer {range} metres out, {count} source only.",
            "An unverified report at {time} sites a transfer {range} metres offshore from {count} source.",
            "A single source timed {time} describes a transfer {range} metres out, {count} corroboration.",
        ),
    ),
    Fact(
        "shoal_activity",
        frozenset({ReportType.PARTNER_REPORT, ReportType.PORT_NOTICE, ReportType.LIAISON_TIP}),
        (
            "Small-craft movement in the closed shoals reached {count} of {total} sightings by {time}.",
            "By {time} the closed shoals had logged {count} small-craft sightings over {total} nights.",
            "Activity in the closed shoals hit {count} sightings across {total} nights as of {time}.",
        ),
    ),
    Fact(
        "inspection_history",
        frozenset({ReportType.DETENTION_RECORD, ReportType.CREW_MANIFEST}),
        (
            "Port state control logged {count} inspections across {total} operator hulls before {time}.",
            "Records up to {time} list {count} port state examinations across {total} operator vessels.",
            "As of {time}, {count} of {total} operator vessels had drawn a port state inspection.",
        ),
    ),
    Fact(
        "routine_transit",
        frozenset({ReportType.VESSEL_REPORT, ReportType.SENSOR_TRACK, ReportType.PORT_NOTICE}),
        (
            "Outbound transit ran {speed} knots at {time} on routing filed {hours} hours prior.",
            "The vessel made {speed} knots by {time}, on a route filed {hours} hours earlier.",
            "Transit held {speed} knots at {time} along routing lodged {hours} hours before.",
        ),
    ),
    Fact(
        "cargo_declared",
        frozenset({ReportType.PORT_NOTICE, ReportType.CREW_MANIFEST, ReportType.PRESS_ITEM}),
        (
            "Cargo declared at {time} matched {count} of {total} manifest lines lodged with port.",
            "At {time} the lodged manifest agreed on {count} of {total} declared cargo lines.",
            "Of {total} manifest lines checked at {time}, {count} matched the declared cargo.",
        ),
    ),
    Fact(
        "weather_note",
        frozenset({ReportType.PORT_NOTICE, ReportType.SENSOR_TRACK, ReportType.PRESS_ITEM}),
        (
            "Visibility held above {vis} nautical miles at {time} with a reported sea state of {count}.",
            "At {time} visibility stayed over {vis} nautical miles with sea state recorded at {count}.",
            "Conditions at {time} gave over {vis} nautical miles of visibility and sea state {count}.",
        ),
    ),
)

FACTS_BY_ID: dict[str, Fact] = {fact.fact_id: fact for fact in FACTS}

#: Every fact id, in declaration order. The padding pool for both classes.
ALL_FACT_IDS: tuple[str, ...] = tuple(fact.fact_id for fact in FACTS)

#: Facts outside the significant pattern, kept for callers that need the split.
FILLER_FACT_IDS: tuple[str, ...] = tuple(
    fact.fact_id for fact in FACTS if fact.fact_id not in SIGNIFICANT_PATTERN
)


@dataclass(frozen=True, slots=True)
class Event:
    """One underlying occurrence, significant or not, and the facts it presents.

    `significant` is the ground truth a triage specialist is scored against. It
    is true exactly when the fact set contains the whole significant pattern, so
    the label is a property of the world rather than an annotation laid over it.
    """

    event_id: str
    vessel: Vessel
    fact_ids: frozenset[str]

    @property
    def significant(self) -> bool:
        return self.fact_ids >= SIGNIFICANT_PATTERN
