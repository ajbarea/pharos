"""The argument the findings make, assembled from the artifacts that make it.

`docs/findings.md` is twenty-four findings in reading order, which is the right shape
for someone auditing a claim and the wrong shape for someone deciding whether the
result matters. The argument is not twenty-four things. It is one tension and a list
of exits from it, each of which was built and measured rather than argued about.

This module is that list. It exists so the explorer can serve the argument rather than
only the corpus, and so a reader who will never run `make results` can still see which
escapes were tried and what each one cost.

**Every number here is read from a committed artifact.** The prose naming each exit is
editorial and lives below; the values beside it are not, and there is no code path that
lets one be typed in. That distinction is the whole design: this is a third rendering
of results that already appear in `docs/findings.md` and in two manuscripts, and a
third hand-maintained copy of a number is a number that will disagree with the other
two. `tests/test_escapes.py` asserts every field resolves and every artifact named
exists.

An exit whose artifact is missing is reported as unmeasured rather than omitted. A
list of escapes that silently shortens when a file is absent would misrepresent the
argument in the most flattering possible direction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

#: How an exit came out. Not a score: `bounded` is not a weaker `closes`, it is a
#: different statement -- the mechanism works and its scope condition is known, which
#: is the most useful thing a negative programme can produce.
VERDICTS = ("closes", "bounded", "open")


@dataclass(frozen=True)
class Reading:
    """One number, and where in its artifact it came from.

    `path` is a dotted route through the JSON rather than a value, so the value cannot
    be stale: it is resolved at request time from the file on disk.
    """

    label: str
    artifact: str
    path: str
    fmt: str = "{}"

    def resolve(self) -> dict[str, Any]:
        payload = _load(self.artifact)
        if payload is None:
            return {
                "label": self.label,
                "value": None,
                "artifact": self.artifact,
                "path": self.path,
            }
        value = _dig(payload, self.path)
        return {
            "label": self.label,
            "value": None if value is None else self.fmt.format(value),
            "raw": value,
            "artifact": self.artifact,
            "path": self.path,
        }


@dataclass(frozen=True)
class Escape:
    """An exit somebody would reach for, and what measuring it cost."""

    key: str
    objection: str
    findings: tuple[int, ...]
    verdict: str
    outcome: str
    readings: tuple[Reading, ...] = field(default_factory=tuple)

    def payload(self) -> dict[str, Any]:
        resolved = [reading.resolve() for reading in self.readings]
        return {
            "key": self.key,
            "objection": self.objection,
            "findings": list(self.findings),
            "verdict": self.verdict,
            "outcome": self.outcome,
            "readings": resolved,
            #: Whether every number behind this exit resolved. The page marks an exit
            #: unmeasured rather than dropping it, so an absent artifact reads as a gap
            #: in the evidence rather than as one fewer objection to answer.
            "measured": all(item["value"] is not None for item in resolved),
        }


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _dig(payload: Any, path: str) -> Any:
    """Follow a dotted path into an artifact.

    A segment is a mapping key, a list index, or a `field=value` selector that picks
    the first list element whose `field` equals `value`.

    The selector form is why this is not just `operator.attrgetter`. Addressing a row
    of `by_clearance_level` positionally works until somebody reorders the list, at
    which point the reading resolves to a *different* clearance level and the page
    shows a wrong number with no sign that anything moved. A selector cannot do that:
    it either finds the row it names or returns None.

    Returns None for any miss rather than raising, because a renamed artifact field
    should degrade one reading to "unmeasured" and not take the whole page down.
    """
    current = payload
    for segment in path.split("."):
        if isinstance(current, list):
            if "=" in segment:
                field_name, _, wanted = segment.partition("=")
                current = next(
                    (
                        row
                        for row in current
                        if isinstance(row, dict) and str(row.get(field_name)) == wanted
                    ),
                    None,
                )
                if current is None:
                    return None
                continue
            if not segment.lstrip("-").isdigit():
                return None
            index = int(segment)
            if not -len(current) <= index < len(current):
                return None
            current = current[index]
        elif isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
        else:
            return None
    return current


#: The tension every exit below is an exit from. Three steps, each with its own
#: measurement, and the third is why the first two cannot both be satisfied.
TENSION = (
    Escape(
        key="inheritance",
        objection="A fleet learns its analyst's tradecraft, including the mistakes.",
        findings=(10,),
        verdict="closes",
        outcome=(
            "A systematically mistaken teacher hands its error rate to the student with "
            "no dilution and no regression toward the model's prior. Personalization "
            "works exactly as specified, and that is the risk."
        ),
        readings=(
            Reading(
                "Largest inheritance gap",
                "teacher_fleet.json",
                "summary.inheritance_gap.largest_absolute",
                "{:.3f}",
            ),
        ),
    ),
    Escape(
        key="linkage",
        objection="The contribution stream identifies the analysts who produced it.",
        findings=(11,),
        verdict="closes",
        outcome=(
            "An attack that reads no content at all, against items each individually "
            "approved for release, recovers the most highly cleared analyst from which "
            "task identifiers appear under a pseudonym."
        ),
        readings=(
            Reading(
                "Recovery, most cleared",
                "fleet_linkage.json",
                "by_clearance_level.level=RESTRICTED.recovery.point",
                "{:.3f}",
            ),
            Reading(
                "Recovery, everyone else",
                "fleet_linkage.json",
                "by_clearance_level.level=OPEN.recovery.point",
                "{:.3f}",
            ),
        ),
    ),
    Escape(
        key="tension",
        objection="Fixing the first needs identity; fixing the second destroys it.",
        findings=(12,),
        verdict="closes",
        outcome=(
            "Estimating reliability from agreement instead needs no identity and fails "
            "exactly where it is needed: once a wrong standard is held widely enough, "
            "the estimate ratifies it with full confidence."
        ),
        readings=(),
    ),
)

#: The exits, ordered by how obvious the objection is rather than chronologically. A
#: reader's first four objections should be answered before their fifth occurs to them.
ESCAPES = (
    Escape(
        key="dp",
        objection="Add differential privacy.",
        findings=(15,),
        verdict="closes",
        outcome=(
            "The standard mechanism is inert here for a structural reason rather than a "
            "tuning one: noising the verdict spends the budget on a variable the attack "
            "never reads. Participation noise does bite, at a price that is a budget in "
            "name rather than a guarantee."
        ),
        readings=(),
    ),
    Escape(
        key="unlikely",
        objection="The cliff needs a majority, which is unlikely.",
        findings=(16,),
        verdict="closes",
        outcome=(
            "It is likelier than an i.i.d. draw suggests, and the understatement is "
            "largest at the lowest error rate, which is exactly the regime a deployment "
            "would point to as evidence that it is safe."
        ),
        readings=(),
    ),
    Escape(
        key="difficulty",
        objection="Use a better estimator: model item difficulty too.",
        findings=(17,),
        verdict="closes",
        outcome=(
            "The canonical joint model converts a wrong standard into a hard item. Worse, "
            "its per-reviewer ability score inverts, rating the mistaken reviewers as the "
            "more able, with no accompanying signal that it should be doubted."
        ),
        readings=(),
    ),
    Escape(
        key="classconditional",
        objection="Then condition on the class.",
        findings=(17,),
        verdict="open",
        outcome=(
            "The strongest form of the objection, and it is open on the record. The 2026 "
            "class-conditional model is bimodal across corpus draws here, so it is "
            "unidentified on this data and neither its successes nor its failures are "
            "quotable. Conceded rather than argued past."
        ),
        readings=(),
    ),
    Escape(
        key="secagg",
        objection="Compute the estimate under secure aggregation.",
        findings=(18,),
        verdict="closes",
        outcome=(
            "Our own held-in-reserve exit, closed in both directions at once. The port is "
            "exact, so privacy here is free, and the failure survives it untouched, "
            "which localises the failure to identifiability rather than disclosure."
        ),
        readings=(
            Reading(
                "Worst posterior gap",
                "secure_reliability.json",
                "worst_posterior_gap",
                "{:.1e}",
            ),
        ),
    ),
    Escape(
        key="authority",
        objection="Then use an authority of record.",
        findings=(19,),
        verdict="bounded",
        outcome=(
            "It works, and its price is a threshold rather than a rate. Reported as a "
            "median over 21 anchor draws with its range, because one draw is not a price "
            "and we have measured how badly a single draw misleads."
        ),
        readings=(),
    ),
    Escape(
        key="select",
        objection="Choose which items the authority rules on.",
        findings=(20,),
        verdict="bounded",
        outcome=(
            "Auditing where the fleet splits ties the oracle bound while reading only "
            "what the aggregator already holds. The claim that travels is conditional: "
            "when a wrong standard shows up as boundary disagreement, audit the "
            "disagreement."
        ),
        readings=(),
    ),
    Escape(
        key="blindspot",
        objection="Does that survive a shared blind spot?",
        findings=(21,),
        verdict="open",
        outcome=(
            "No, and this is what bounds the exit above. A channel blind spot corrupts a "
            "slice at the opposite difficulty extreme from the boundary, so the fleet is "
            "unanimous there and every disagreement policy drops to chance."
        ),
        readings=(),
    ),
    Escape(
        key="detect",
        objection="Then detect which regime the fleet is in.",
        findings=(22,),
        verdict="closes",
        outcome=(
            "A shared blind spot leaves no disagreement but does leave a conditional "
            "dependence between the verdict rate and the blinded channel, inside sums "
            "the secure aggregate already exposes. Detection does not weaken at the "
            "unanimity that defeats the policy."
        ),
        readings=(),
    ),
    Escape(
        key="channelaudit",
        objection="Then select on the channel the detector named.",
        findings=(23,),
        verdict="open",
        outcome=(
            "The selection half closes: provenance finds every corrupted item where "
            "uncertainty sampling is at chance, tying the oracle. The repair half does "
            "not. No unanchored label changes, and the oracle behaves identically, so "
            "the residual obstacle is not one of selection."
        ),
        readings=(),
    ),
    Escape(
        key="fleetsize",
        objection="Does any of this depend on the fleet being nine analysts?",
        findings=(24,),
        verdict="bounded",
        outcome=(
            "The wording, and more than we first reported. The crossing is not a fixed "
            "share but a distribution over corpus draws, and it grows less predictable "
            "as the fleet grows: at nine analysts every draw breaks at the same "
            "composition, at twenty-five a bare majority survives on half of them. Every "
            "other invariant survives at four fleet sizes."
        ),
        readings=(
            Reading(
                "Breaking share, median",
                "governance_sensitivity.json",
                "cliff_bracket.breaking_share_median",
                "{:.3f}",
            ),
            Reading(
                "Breaking share, lowest",
                "governance_sensitivity.json",
                "cliff_bracket.breaking_share_range.0",
                "{:.3f}",
            ),
            Reading(
                "Breaking share, highest",
                "governance_sensitivity.json",
                "cliff_bracket.breaking_share_range.1",
                "{:.3f}",
            ),
        ),
    ),
    Escape(
        key="initialisation",
        objection="Is the cliff real, or is it where you pointed the estimator?",
        findings=(25,),
        verdict="bounded",
        outcome=(
            "Bounded to one cell. Dawid-Skene here starts from the majority vote, which "
            "has no global-optimality guarantee, so the objection is fair and was never "
            "tested. Sweeping the start over an uninformative one, an adversarial one, "
            "thirty-two random restarts and the ground truth itself, an escape exists "
            "only at the crossing composition and only in a minority of draws. Past it "
            "the wrong answer is the better fit by tens of nats, so selecting by "
            "likelihood picks it even where the truth is reachable, and no initialiser "
            "helps a search whose objective prefers the wrong answer."
        ),
        readings=(
            Reading(
                "Compositions priced",
                "estimator_initialization.json",
                "priced_compositions",
            ),
            Reading(
                "Random restarts per cell",
                "estimator_initialization.json",
                "restarts",
            ),
            Reading(
                "Escape confined to the crossing",
                "estimator_initialization.json",
                "invariants.the_escape_is_confined_to_the_crossing",
            ),
        ),
    ),
)


def argument() -> dict[str, Any]:
    """The whole argument, resolved against whatever artifacts are on disk."""
    escapes = [escape.payload() for escape in ESCAPES]
    resolved_tension = [step.payload() for step in TENSION]
    return {
        "tension": resolved_tension,
        "escapes": escapes,
        "verdicts": list(VERDICTS),
        "counts": {
            verdict: sum(1 for e in escapes if e["verdict"] == verdict) for verdict in VERDICTS
        },
        #: Named so the page can say which exits have no evidence behind them right now,
        #: rather than presenting a partially-measured argument as a complete one.
        "unmeasured": [
            step["key"] for step in (*resolved_tension, *escapes) if not step["measured"]
        ],
    }
