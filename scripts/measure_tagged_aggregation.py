#!/usr/bin/env python3
"""Whether a reliability tag can replace contributor identity, and what it costs.

Three results in this repository box each other in. A learner acquires its teacher's
standard exactly, so reliability weighting needs contributor identity (finding 10).
Contributor identity is what makes a fleet's contributions attributable to a person,
and pooling is the only control that removes that at no cost in volume (finding 11).
Consensus cannot recover reliability from pooled outputs once the wrong standard holds
the majority (finding 12). Identity is also what an audit trail needs, which matters
because the interest this testbed serves asks for training signals that are both
personalized and *auditable*.

The remaining move is a tag coarser than a person. Label each contribution with the
contributor's reliability tier rather than their identity: the aggregator gets what it
needs for weighting, and nobody is named. This measures whether that works, and it
reports three things because the first two are not sufficient to answer the question.

  1. Does the tag buy anything? Weighting by tier has to recover the benefit that
     per-contributor weighting gives, or the tag is pure cost.
  2. Does it identify anyone? Finding 11's attack, run against tier-tagged streams.
  3. **Does it disclose clearance in aggregate?** This is the one that decides it, and
     the one finding 11's metric cannot see. That metric scores an analyst recovered
     only when the inference is unique *and* attributable to them, so any tag coarser
     than a person scores zero by construction. Zero there is the metric saturating,
     not a privacy property, and reading it as one would have shipped a false
     reassurance from this repository's own instrument.

The answer depends on something the mechanism does not control: whether tier and
clearance are independent. They need not be, and on a real watch floor seniority
plausibly drives both.

Needs no model and no network.

    uv run python scripts/measure_tagged_aggregation.py --out results/tagged.json
"""

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random

from pharos.disclosure import DROP_COMPARTMENTS
from pharos.fleet import Clearance, Contribution, assign_fleet, contribute, link
from pharos.generate import GeneratorConfig, generate
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200
FLEET_SEED = 11
FLEET_SIZE = 200

#: How many reliability tiers the aggregator distinguishes. Three is the coarsest
#: split that still supports "trust, discount, drop", which is what a weighting scheme
#: needs. Finer tiers approach per-person identity and coarser ones approach no tag.
TIERS = 3


@dataclass(frozen=True, slots=True)
class Scheme:
    """One way of labelling contributions before the aggregator sees them."""

    name: str
    describe: str
    assign: Callable[[Clearance, Random], str]


def _independent(_: Clearance, rng: Random) -> str:
    return f"tier{rng.randrange(TIERS)}"


def _correlated(clearance: Clearance, _: Random) -> str:
    """Tier tracks clearance level, the case where seniority drives both."""
    return f"tier{min(TIERS - 1, int(clearance.sensitivity))}"


SCHEMES: tuple[Scheme, ...] = (
    Scheme(
        "per_person",
        "contributor identity retained (finding 11 baseline)",
        lambda c, _: c.analyst_id,
    ),
    Scheme("pooled", "no tag at all (finding 11's free control)", lambda _c, _r: "POOLED"),
    Scheme("tier_independent", "reliability tier, independent of clearance", _independent),
    Scheme("tier_correlated", "reliability tier, correlated with clearance", _correlated),
)


def tag(
    stream: Sequence[Contribution], fleet: Sequence[Clearance], scheme: Scheme, *, seed: int
) -> tuple[Contribution, ...]:
    """Relabel every contribution's pseudonym under `scheme`, leaving truth intact."""
    rng = Random(seed)
    assigned = {c.analyst_id: scheme.assign(c, rng) for c in fleet}
    return tuple(replace(c, pseudonym=assigned[c.analyst_id]) for c in stream)


def clearance_inference(
    fleet: Sequence[Clearance], scheme: Scheme, *, seed: int
) -> tuple[float, int]:
    """How well a tag predicts a contributor's clearance level, and how many groups it makes.

    The adversary names the most common clearance within each tag group. This is the
    aggregate disclosure question, and it is deliberately *not* about identifying a
    person: learning that a contribution came from a group that is entirely
    top-clearance is a disclosure about everyone in it, whether or not any one of them
    is named.
    """
    rng = Random(seed)
    assigned = {c.analyst_id: scheme.assign(c, rng) for c in fleet}
    levels = {c.analyst_id: c.sensitivity for c in fleet}
    groups: dict[str, list[int]] = defaultdict(list)
    for analyst_id, group in assigned.items():
        groups[group].append(levels[analyst_id])
    correct = sum(Counter(v).most_common(1)[0][1] for v in groups.values())
    return correct / len(fleet), len(groups)


def prior_accuracy(fleet: Sequence[Clearance]) -> float:
    """Naming the fleet's most common clearance level, which any tag must beat to leak."""
    counts = Counter(c.sensitivity for c in fleet)
    return counts.most_common(1)[0][1] / len(fleet) if fleet else 0.0


@dataclass(frozen=True, slots=True)
class SchemeResult:
    """What one tagging scheme discloses, on both axes."""

    scheme: str
    describe: str
    identifies_individuals: float
    clearance_inference: float
    lift_over_prior: float
    groups: int

    @property
    def leaks_in_aggregate(self) -> bool:
        """Discloses clearance beyond the prior. 0.05 is a margin, not a threshold."""
        return self.lift_over_prior > 0.05

    @property
    def invisible_to_per_analyst_metric(self) -> bool:
        """Leaks in aggregate while finding 11's attack names nobody."""
        return self.leaks_in_aggregate and self.identifies_individuals == 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "describe": self.describe,
            "identifies_individuals": round(self.identifies_individuals, 4),
            "clearance_inference": round(self.clearance_inference, 4),
            "lift_over_prior": round(self.lift_over_prior, 4),
            "groups": self.groups,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET_SIZE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    fleet = assign_fleet(args.fleet, seed=FLEET_SEED)
    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    prior = prior_accuracy(fleet)

    print(f"{len(tasks)} tasks, fleet of {len(fleet)}, {TIERS} reliability tiers")
    print(f"naming the fleet's most common clearance level scores {prior:.3f}\n")
    print(f"  {'scheme':<20} {'identifies':>11} {'clearance':>11} {'groups':>7}")
    print("  " + "-" * 52)

    rows: list[SchemeResult] = []
    for scheme in SCHEMES:
        tagged = tag(stream, fleet, scheme, seed=SEED)
        linkages = link(tagged, tasks, fleet, policy=DROP_COMPARTMENTS)
        recovery = sum(link_.exact for link_ in linkages) / len(linkages)
        inferred, groups = clearance_inference(fleet, scheme, seed=SEED)
        rows.append(
            SchemeResult(
                scheme=scheme.name,
                describe=scheme.describe,
                identifies_individuals=recovery,
                clearance_inference=inferred,
                lift_over_prior=inferred - prior,
                groups=groups,
            )
        )
        print(f"  {scheme.name:<20} {recovery:>11.3f} {inferred:>11.3f} {groups:>7}")

    invisible = [r for r in rows if r.invisible_to_per_analyst_metric]
    for r in rows:
        record(
            "tagging.scheme",
            r.clearance_inference,
            scheme=r.scheme,
            identifies_individuals=r.identifies_individuals,
            lift_over_prior=round(r.lift_over_prior, 4),
            groups=r.groups,
        )
    for r in invisible:
        # The case the per-analyst metric scores as clean. Warned rather than merely
        # tabulated, because a reader trusting that metric would ship this.
        get_logger().warning(
            "tagging.aggregate_leak_invisible",
            extra={
                "event": "tagging.aggregate_leak_invisible",
                "scheme": r.scheme,
                "clearance_inference": r.clearance_inference,
                "identifies_individuals": r.identifies_individuals,
            },
        )

    print()
    print("Schemes that disclose clearance in aggregate while identifying nobody:")
    if invisible:
        for r in invisible:
            print(
                f"  {r.scheme}: clearance inference {r.clearance_inference:.3f} "
                f"against a prior of {prior:.3f} (+{r.lift_over_prior:.3f}), "
                f"individual recovery {r.identifies_individuals:.3f}"
            )
        print("  finding 11's per-analyst metric reports zero for each of these. It asks")
        print("  whether a person was named, not whether a group was characterised.")
    else:
        print("  none at this fleet composition")

    report = {
        "provenance": run_provenance(seed=SEED),
        "fleet": len(fleet),
        "events": args.events,
        "tiers": TIERS,
        "prior": round(prior, 4),
        "schemes": [r.as_dict() for r in rows],
        "aggregate_leak_invisible_to_finding_11": [r.scheme for r in invisible],
        "validity": check_sample_size(len(fleet), label="tagged aggregation").as_dict(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
