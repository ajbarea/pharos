"""Loading a world from a file, so a scenario is data rather than a code change.

Pharos began as one corpus with its world hardcoded. That made it a dataset. Moving
the world into a file makes it a generator, and lets someone specify a domain --
their channels, their compartment assignments, their fact vocabulary, their
significance rule -- without writing Python or forking anything.

**What is data and what is not.** The lattice algebra is not: `Sensitivity` is a
four-level ladder and `Compartment` a fixed set, both defined in `pharos.labels`,
because the joins and dominance rules over them are the thing under test rather than
a domain choice. Everything above that is data: which channels exist, what each
confers, the voices, the entities, the fact vocabulary, and which conjunction of
facts defines a significant event.

**Validation is the point, not a formality.** A scenario that violates the
generator's invariants produces a corpus that looks fine and measures nothing, and
the shortcut gate cannot catch most of it -- the gate tests whether *shape* predicts
the label, not whether the vocabulary is coherent. So `load` refuses:

- a significance rule referencing facts that do not exist
- a decoy containing the whole significant pattern, which would make background
  events indistinguishable from plants by definition
- a fact reachable through no channel, so it can never be rendered
- a channel no fact can use, so it never carries anything
- fewer than two distinct label cells, since a constant label cannot evaluate a
  disclosure boundary at all

Each of those was a way to build a broken corpus quietly.
"""

import string
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pharos.labels import Compartment, Sensitivity

#: Shipped scenarios live here; a path outside it is equally acceptable.
SCENARIO_DIR = Path(__file__).parent.parent.parent / "scenarios"

#: The world the published measurements were taken on.
DEFAULT_SCENARIO = "maritime-watch"

#: The only placeholders a rendering may use. This is a whitelist, and it exists for
#: a security reason rather than a tidiness one.
#:
#: Renderings are applied with `str.format(**slots)`, and `str.format` on an
#: attacker-controlled template is a known Python injection class: a field name may
#: traverse attributes and indices, so a rendering reading
#: `{vessel.__class__.__init__.__globals__[SECRET]}` walks the object graph out of
#: the template and into module state. Confirmed exploitable against this codebase
#: before the check existed.
#:
#: Scenarios are *designed* to be written by other people, which makes every
#: rendering untrusted input by construction. So each one is parsed at load time and
#: rejected unless every field is a bare name from this set: no dots, no brackets, no
#: conversions, no nested fields.
ALLOWED_SLOTS: frozenset[str] = frozenset(
    {
        "bearing",
        "berth",
        "berth2",
        "count",
        "draft",
        "hours",
        "laden",
        "mins",
        "offset",
        "range",
        "speed",
        "time",
        "total",
        "vis",
    }
)


class ScenarioError(ValueError):
    """A scenario that would generate a corpus nothing could be measured on."""


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    name: str
    sensitivity: Sensitivity
    compartments: frozenset[Compartment]


@dataclass(frozen=True, slots=True)
class CenterSpec:
    center_id: str
    name: str
    voice: str


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    operator_id: str
    name: str


@dataclass(frozen=True, slots=True)
class FactSpec:
    fact_id: str
    channels: frozenset[str]
    renderings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A complete world: everything the generator needs that is not algebra."""

    name: str
    description: str
    channels: tuple[ChannelSpec, ...]
    voices: tuple[str, ...]
    centers: tuple[CenterSpec, ...]
    operators: tuple[OperatorSpec, ...]
    vessel_names: tuple[str, ...]
    hull_types: tuple[str, ...]
    flags: tuple[str, ...]
    significant_pattern: frozenset[str]
    decoy_patterns: tuple[frozenset[str], ...]
    facts: tuple[FactSpec, ...]

    @property
    def channel_labels(self) -> dict[str, tuple[Sensitivity, frozenset[Compartment]]]:
        return {c.name: (c.sensitivity, c.compartments) for c in self.channels}

    @property
    def fact_ids(self) -> frozenset[str]:
        return frozenset(f.fact_id for f in self.facts)

    @property
    def label_cells(self) -> set[tuple[Sensitivity, frozenset[Compartment]]]:
        return {(c.sensitivity, c.compartments) for c in self.channels}

    def summary(self) -> dict[str, object]:
        """What to record alongside a corpus generated from this scenario."""
        return {
            "scenario": self.name,
            "channels": len(self.channels),
            "facts": len(self.facts),
            "centers": len(self.centers),
            "label_cells": len(self.label_cells),
            "significant_pattern": sorted(self.significant_pattern),
            "decoys": len(self.decoy_patterns),
        }


def _require(condition: object, message: str) -> None:
    """Raise unless `condition` is truthy.

    Typed as `object` rather than `bool` because callers pass collections directly --
    `_require(scenario.channels, ...)` reads better than wrapping every one in
    `bool(...)`, and truthiness is exactly the intended semantics. Declaring `bool`
    made the signature a lie the checker was right to flag.
    """
    if not condition:
        raise ScenarioError(message)


def _enum(kind, value: str, field: str):
    try:
        return kind[value] if kind is Sensitivity else kind(value)
    except (KeyError, ValueError) as exc:
        options = ", ".join(m.name for m in kind)
        raise ScenarioError(f"unknown {field} {value!r}; expected one of: {options}") from exc


def _check_rendering_is_safe(rendering: str, fact_id: str) -> None:
    """Refuse a template that could escape into the object graph.

    `str.Formatter().parse` gives the field names without evaluating anything, so
    this inspects a hostile template safely. A field is acceptable only when it is a
    bare name in `ALLOWED_SLOTS`; anything containing an attribute access, an index,
    or a conversion is rejected with the offending text quoted.
    """
    try:
        parsed = list(string.Formatter().parse(rendering))
    except ValueError as exc:
        raise ScenarioError(
            f"fact {fact_id!r} has a rendering that is not a valid template: {exc}"
        ) from exc

    for _literal, field, _spec, conversion in parsed:
        if field is None:
            continue
        if conversion is not None:
            raise ScenarioError(
                f"fact {fact_id!r} uses a conversion in {{{field}!{conversion}}}; "
                "renderings may only substitute plain values"
            )
        if not field.isidentifier():
            raise ScenarioError(
                f"fact {fact_id!r} uses the template field {{{field}}}, which is not a "
                "bare name. Attribute access and indexing in a format string can reach "
                "module globals, so only simple placeholders are allowed."
            )
        if field not in ALLOWED_SLOTS:
            raise ScenarioError(
                f"fact {fact_id!r} uses unknown placeholder {{{field}}}. "
                f"Available: {sorted(ALLOWED_SLOTS)}"
            )


def validate(scenario: Scenario) -> None:
    """Refuse a scenario that would generate an unmeasurable corpus.

    Every check corresponds to a way of building something that looks like a corpus
    and answers no question. The shortcut gate cannot substitute for these: it asks
    whether shape predicts the label, not whether the vocabulary is coherent.
    """
    known_facts = scenario.fact_ids
    channel_names = {c.name for c in scenario.channels}

    _require(scenario.channels, "a scenario needs at least one channel")
    _require(scenario.facts, "a scenario needs at least one fact")
    _require(scenario.centers, "a scenario needs at least one centre")

    missing = sorted(scenario.significant_pattern - known_facts)
    _require(
        not missing,
        f"the significance rule names facts that do not exist: {missing}. "
        "No event could ever satisfy it, so the positive class would be empty.",
    )
    _require(
        len(scenario.significant_pattern) >= 2,
        "a significance rule of fewer than two facts is a keyword match, not a "
        "conjunction, and a model can solve it without reasoning over sources",
    )

    for index, decoy in enumerate(scenario.decoy_patterns):
        unknown = sorted(decoy - known_facts)
        _require(not unknown, f"decoy {index} names facts that do not exist: {unknown}")
        _require(
            not scenario.significant_pattern <= decoy,
            f"decoy {index} contains the entire significance rule, so background "
            "events carrying it are significant by definition and the label is a lie",
        )

    for fact in scenario.facts:
        unknown = sorted(fact.channels - channel_names)
        _require(not unknown, f"fact {fact.fact_id!r} names unknown channels: {unknown}")
        _require(
            fact.channels,
            f"fact {fact.fact_id!r} is carried by no channel, so it can never be rendered",
        )
        _require(
            fact.renderings,
            f"fact {fact.fact_id!r} has no renderings, so it has no text to emit",
        )

    carried = {channel for fact in scenario.facts for channel in fact.channels}
    orphans = sorted(channel_names - carried)
    _require(
        not orphans,
        f"channels carry no fact and would emit nothing: {orphans}",
    )

    _require(
        len(scenario.label_cells) > 1,
        "every channel confers the same label, so the corpus has one label cell and "
        "cannot evaluate a disclosure boundary: every entry would be governed alike",
    )

    for fact in scenario.facts:
        for rendering in fact.renderings:
            _check_rendering_is_safe(rendering, fact.fact_id)

    voices = set(scenario.voices)
    unknown_voices = sorted({c.voice for c in scenario.centers} - voices)
    _require(
        not unknown_voices,
        f"centres use undeclared voices: {unknown_voices}. Declared: {sorted(voices)}. "
        "If that list is empty, check TOML table scoping: a bare key written after a "
        "[table] header belongs to that table, not the root, which silently hides it.",
    )


def load(source: str | Path = DEFAULT_SCENARIO) -> Scenario:
    """Read a scenario by name or path, and validate it before returning.

    A bare name resolves against the shipped `scenarios/` directory; anything with a
    separator or a `.toml` suffix is treated as a path, so a collaborator can keep
    their own world outside this repository.
    """
    path = Path(source)
    if path.suffix != ".toml" and not path.exists():
        path = SCENARIO_DIR / f"{source}.toml"
    if not path.exists():
        available = sorted(p.stem for p in SCENARIO_DIR.glob("*.toml"))
        raise ScenarioError(f"no scenario at {path}. Available: {available or 'none'}")

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    meta = raw.get("scenario", {})
    entities = raw.get("entities", {})
    patterns = raw.get("patterns", {})

    scenario = Scenario(
        name=meta.get("name", path.stem),
        description=meta.get("description", ""),
        channels=tuple(
            ChannelSpec(
                name=c["name"],
                sensitivity=_enum(Sensitivity, c["sensitivity"], "sensitivity"),
                compartments=frozenset(
                    _enum(Compartment, x, "compartment") for x in c.get("compartments", [])
                ),
            )
            for c in raw.get("channels", [])
        ),
        voices=tuple(meta.get("voices", raw.get("voices", []))),
        centers=tuple(
            CenterSpec(center_id=c["center_id"], name=c["name"], voice=c["voice"])
            for c in raw.get("centers", [])
        ),
        operators=tuple(
            OperatorSpec(operator_id=o["operator_id"], name=o["name"])
            for o in raw.get("operators", [])
        ),
        vessel_names=tuple(entities.get("vessel_names", [])),
        hull_types=tuple(entities.get("hull_types", [])),
        flags=tuple(entities.get("flags", [])),
        significant_pattern=frozenset(patterns.get("significant", [])),
        decoy_patterns=tuple(frozenset(d) for d in patterns.get("decoys", [])),
        facts=tuple(
            FactSpec(
                fact_id=f["fact_id"],
                channels=frozenset(f["channels"]),
                renderings=tuple(f["renderings"]),
            )
            for f in raw.get("facts", [])
        ),
    )
    validate(scenario)
    return scenario


def available() -> list[str]:
    """Names of the shipped scenarios."""
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.toml"))
