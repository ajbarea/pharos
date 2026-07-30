"""Configuration-driven corpus generation.

Diversity is forced at generation time rather than filtered afterward: every
report crosses a report type, a source channel, and an officer voice, and one
underlying event is rendered through several of those crossings. Uncontrolled
generation is monotonous in precisely the way that manufactures the surface
regularities the shortcut gate exists to catch, so control is cheaper than
cleanup.

Plants traverse the identical path as background. The generator draws the same
number of facts per event and the same number of reports per event for both
classes, samples channels from the same distribution, and renders through the
same templates. The only difference is which facts co-occur. If that invariant
ever breaks, the gate fails and the corpus version is not usable.

Everything derives from one seeded `random.Random`, so a corpus is reproducible
from its `(seed, config)` pair alone.
"""

import random
from dataclasses import dataclass, field, replace

from pharos.labels import Capacity, Label
from pharos.world import (
    ALL_FACT_IDS,
    CENTERS,
    CHANNEL_LABELS,
    DECOY_PATTERNS,
    FACTS_BY_ID,
    FLAGS,
    HULL_TYPES,
    OPERATORS,
    SIGNIFICANT_PATTERN,
    VESSEL_NAMES,
    Center,
    Event,
    ReportType,
    Vessel,
    Voice,
)

#: Facts per event, identical for both classes so fact count carries no signal.
FACTS_PER_EVENT = 5

#: Reports per event, identical for both classes for the same reason.
REPORTS_PER_EVENT = 3

#: Fact sentences rendered into a single report. Holding this fixed is what keeps
#: report length independent of whether the underlying event was significant.
FACTS_PER_REPORT = 2


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Everything needed to reproduce a corpus."""

    seed: int
    n_events: int = 200
    plant_rate: float = 0.3
    centers: tuple[Center, ...] = CENTERS

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_events": self.n_events,
            "plant_rate": self.plant_rate,
            "centers": [c.center_id for c in self.centers],
        }


@dataclass(frozen=True, slots=True)
class Report:
    """One rendered report, carrying the label its channel confers."""

    report_id: str
    report_type: ReportType
    center: Center
    voice: Voice
    event_id: str
    vessel_name: str
    text: str
    label: Label
    is_plant: bool
    fact_ids: tuple[str, ...] = field(default=())

    def with_text(self, text: str) -> "Report":
        """A copy carrying different text, for tamper tests against the gate."""
        return replace(self, text=text)


def _vessels(rng: random.Random) -> tuple[Vessel, ...]:
    return tuple(
        Vessel(
            vessel_id=f"V-{index:03d}",
            name=name,
            hull_type=rng.choice(HULL_TYPES),
            flag=rng.choice(FLAGS),
            operator=rng.choice(OPERATORS),
        )
        for index, name in enumerate(VESSEL_NAMES)
    )


def _draw_fact_ids(rng: random.Random, *, plant: bool) -> frozenset[str]:
    """Exactly `FACTS_PER_EVENT` fact ids, containing the pattern only when `plant`.

    Both classes are built identically: one fixed triple plus two padding facts
    drawn from everything outside it. A plant's triple is the significant
    pattern; a background event's is one of the decoys, each of which carries at
    most two of the significant facts.

    That symmetry is the point. When only plants had a deterministic triple, any
    property of those three facts became a class signal and the shortcut probe
    found it however carefully the renderings were balanced.
    """
    triple = SIGNIFICANT_PATTERN if plant else rng.choice(DECOY_PATTERNS)
    chosen = set(triple)
    padding_pool = [fact_id for fact_id in ALL_FACT_IDS if fact_id not in chosen]
    chosen.update(rng.sample(padding_pool, FACTS_PER_EVENT - len(chosen)))
    if not plant and chosen >= SIGNIFICANT_PATTERN:
        # Padding completed the significant pattern by accident, which would make
        # a background event indistinguishable from a plant in ground truth.
        # Swap the offending padding fact for one that cannot complete it.
        surplus = sorted(chosen - triple)
        for fact_id in surplus:
            if fact_id in SIGNIFICANT_PATTERN:
                safe = [f for f in padding_pool if f not in SIGNIFICANT_PATTERN and f not in chosen]
                chosen.discard(fact_id)
                chosen.add(rng.choice(safe))
                break
    return frozenset(chosen)


def _events(rng: random.Random, config: GeneratorConfig) -> list[Event]:
    vessels = _vessels(rng)
    events: list[Event] = []
    for index in range(config.n_events):
        plant = rng.random() < config.plant_rate
        events.append(
            Event(
                event_id=f"E-{index:04d}",
                vessel=rng.choice(vessels),
                fact_ids=_draw_fact_ids(rng, plant=plant),
            )
        )
    return events


def _slot_values(rng: random.Random) -> dict[str, str]:
    """Values for every template slot, drawn once per report.

    All integers, no decimals. An earlier draft used decimal drafts and speeds,
    and the period inside "9.2" inflated the naive sentence count of any report
    carrying a number-dense fact, which handed the shortcut probe a tell. The
    gate's features are deliberately naive, so the corpus has to be clean under
    naive features rather than under clever ones.

    Every non-time slot is a two-digit integer, so each rendering carries exactly
    nine digits: five for the timestamp and two apiece for its other slots. This is
    the third leak the shortcut gate caught. Slot widths had varied (a range in
    hundreds of metres, a count in single digits), so which slots a fact used was a
    digit signature, and the significant facts averaged 9.00 digits against 8.25
    for fillers. Uniform width removes the signature at the source.
    """
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    draft = rng.randint(10, 13)
    count = rng.randint(10, 40)
    return {
        "draft": str(draft),
        "laden": str(rng.randint(draft + 1, 19)),
        "bearing": str(rng.randint(10, 99)),
        "offset": str(rng.randint(10, 90)),
        "time": f"{hour:02d}{minute:02d}Z",
        "range": str(rng.randint(20, 99)),
        "mins": str(rng.randint(10, 55)),
        "hours": str(rng.randint(10, 72)),
        "berth": str(rng.randint(10, 20)),
        "berth2": str(rng.randint(10, 20)),
        "count": str(count),
        "total": str(rng.randint(41, 99)),
        "speed": str(rng.randint(10, 15)),
        "vis": str(rng.randint(10, 12)),
    }


#: Per-voice header and closing phrasing. Written to comparable length across
#: voices so voice cannot become a length feature.
_HEADERS: dict[Voice, str] = {
    Voice.TERSE: "{kind} {rid}. {vessel}, {hull}, flag {flag}. Operator {operator}.",
    Voice.FORMAL: "{kind} reference {rid} concerns {vessel}, a {hull} under {flag} flag, operated by {operator}.",
    Voice.NARRATIVE: "{kind} {rid} follows the {hull} {vessel}, flagged {flag} and operated by {operator}.",
    Voice.CLIPPED: "{kind}/{rid}: {vessel} ({hull}) flag {flag} op {operator}.",
}

_CLOSINGS: dict[Voice, str] = {
    Voice.TERSE: "Watch center {center}. Entry ends.",
    Voice.FORMAL: "Recorded by watch center {center} for the current window.",
    Voice.NARRATIVE: "Logged at watch center {center} during this watch window.",
    Voice.CLIPPED: "Center {center}. End.",
}


def _render(
    rng: random.Random,
    *,
    report_type: ReportType,
    center: Center,
    event: Event,
    fact_ids: list[str],
    report_id: str,
) -> str:
    slots = _slot_values(rng)
    vessel = event.vessel
    header = _HEADERS[center.voice].format(
        kind=report_type.value.replace("_", " ").title(),
        rid=report_id,
        vessel=vessel.name,
        hull=vessel.hull_type,
        flag=vessel.flag,
        operator=vessel.operator.name,
    )
    body = [rng.choice(FACTS_BY_ID[fact_id].renderings).format(**slots) for fact_id in fact_ids]
    closing = _CLOSINGS[center.voice].format(center=center.center_id)
    return " ".join([header, *body, closing])


def _report_channels(rng: random.Random, event: Event) -> list[tuple[ReportType, list[str]]]:
    """Channel assignments for one event: which channel reports which facts.

    Channels are drawn from the full set for both classes. A fact is only placed
    on a channel that could plausibly observe it, and any shortfall is padded
    from the event's remaining facts, so every report carries exactly
    `FACTS_PER_REPORT` sentences regardless of class.
    """
    available = sorted(event.fact_ids)
    channels = rng.sample(sorted(ReportType), REPORTS_PER_EVENT)
    assignments: list[tuple[ReportType, list[str]]] = []
    for channel in channels:
        plausible = [f for f in available if channel in FACTS_BY_ID[f].channels]
        chosen = rng.sample(plausible, min(FACTS_PER_REPORT, len(plausible)))
        if len(chosen) < FACTS_PER_REPORT:
            # Pad from any fact renderable on this channel across the whole
            # vocabulary, keeping sentence count constant. Padding is drawn the
            # same way for both classes.
            pool = [
                f for f in FACTS_BY_ID if channel in FACTS_BY_ID[f].channels and f not in chosen
            ]
            if pool:
                chosen += rng.sample(pool, min(FACTS_PER_REPORT - len(chosen), len(pool)))
        assignments.append((channel, chosen))
    return assignments


def generate(config: GeneratorConfig) -> list[Report]:
    """The corpus for `config`, reproducible from its seed alone."""
    rng = random.Random(config.seed)
    reports: list[Report] = []
    counter = 0
    for event in _events(rng, config):
        for channel, fact_ids in _report_channels(rng, event):
            center = rng.choice(config.centers)
            sensitivity, compartments = CHANNEL_LABELS[channel]
            report_id = f"R-{counter:05d}"
            counter += 1
            reports.append(
                Report(
                    report_id=report_id,
                    report_type=channel,
                    center=center,
                    voice=center.voice,
                    event_id=event.event_id,
                    vessel_name=event.vessel.name,
                    text=_render(
                        rng,
                        report_type=channel,
                        center=center,
                        event=event,
                        fact_ids=fact_ids,
                        report_id=report_id,
                    ),
                    label=Label(sensitivity, compartments, Capacity.FREETEXT),
                    is_plant=event.significant,
                    fact_ids=tuple(fact_ids),
                )
            )
    return reports
