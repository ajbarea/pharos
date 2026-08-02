#!/usr/bin/env python3
"""What a privacy budget buys against the channel that actually leaks.

Finding 11's control ladder was assembled from our own reasoning: k-anonymity, rarity
suppression, subsampling, pooling. It omitted differential privacy, which is the
field's default control and the mechanism the motivating abstract promises. The
omission survived three findings because a self-generated list carries no signal that
it is incomplete.

Closing it turns out to be more interesting than adding a row, because the expected
form of the mechanism does nothing here at all.

**Value noise cannot help, at any epsilon.** The attack reads the map from pseudonym to
task-identifier set and never inspects a value. Randomized response on the verdict is
therefore the identity function as far as the attack is concerned. This is a
proposition rather than a measurement, and it is measured here anyway: a claim that a
standard defence is useless should be demonstrated rather than argued.

**Participation noise works, and the composed budget is what it costs.** Randomized
response over "did this contributor participate on this task" buys real deniability,
because it fabricates contributions on unreachable tasks as well as dropping reachable
ones. Its per-indicator epsilon looks reassuring. The attack observes every indicator
at once, so the figure that answers the adversary's question is the budget composed
over the tasks that distinguish two clearances, and that is hundreds of times larger.

Reporting only the per-indicator epsilon would describe a far stronger mechanism than
the one deployed. Both are reported, with the composed one marked as the operative
figure.

Needs no model and no network.

    uv run python scripts/measure_privacy_budget.py --out results/privacy_budget.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pharos.budget import (
    Budget,
    label_noise,
    randomized_participation,
    value_noise,
    widest_separation,
)
from pharos.disclosure import DROP_COMPARTMENTS
from pharos.fleet import FLEET_CEILING, assign_fleet, contribute, link
from pharos.generate import GeneratorConfig, generate
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200
FLEET_SEED = 11
FLEET_SIZE = 200

#: Flip probabilities for the value mechanism. Includes 0.5, which destroys the verdict
#: entirely: if even that leaves the attack untouched, no weaker setting can help.
VALUE_FLIPS = (0.0, 0.1, 0.25, 0.5)

#: (keep, fabricate) pairs for the participation mechanism, from no noise to heavy.
#: fabricate=0 is included deliberately: it is subsampling, and its epsilon is infinite.
PARTICIPATION = (
    (1.0, 0.0),
    (0.9, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """One mechanism setting, its cost, and what the attack still recovers.

    Keyword-only, for the reason given on `measure_teacher_transfer.Row`: `recovery`
    and `label_noise` are both rates and sit either side of an int.
    """

    mechanism: str
    setting: str
    recovery: float
    contributions: int
    label_noise: float
    budget: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "mechanism": self.mechanism,
            "setting": self.setting,
            "recovery": round(self.recovery, 4),
            "contributions": self.contributions,
            "label_noise": round(self.label_noise, 4),
            "budget": self.budget,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET_SIZE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    fleet = assign_fleet(args.fleet, seed=FLEET_SEED)
    baseline = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)

    def recovery_of(stream) -> float:
        linkages = link(stream, tasks, fleet, policy=DROP_COMPARTMENTS)
        return sum(x.exact for x in linkages) / len(linkages) if linkages else 0.0

    separation = widest_separation(fleet, tasks)
    base_recovery = recovery_of(baseline)
    print(f"{len(tasks)} tasks, fleet of {len(fleet)}, no mechanism: recovery {base_recovery:.3f}")
    print(f"widest clearance separation: {separation} tasks, which is what composition runs over\n")

    rows: list[Row] = []

    print("value noise on the verdict, the mechanism a reader expects:")
    for flip in VALUE_FLIPS:
        noised = value_noise(baseline, flip=flip, seed=SEED)
        rec = recovery_of(noised)
        rows.append(
            Row(
                mechanism="value",
                setting=f"flip={flip}",
                recovery=rec,
                contributions=len(noised),
                label_noise=0.0,
                budget=None,
            )
        )
        print(f"  flip {flip:<5} recovery {rec:.3f}")
    unchanged = {r.recovery for r in rows if r.mechanism == "value"}
    print(
        f"  -> {len(unchanged)} distinct value(s) across every flip rate: "
        + (
            "the attack is untouched, as the proposition requires"
            if len(unchanged) == 1
            else "UNEXPECTED"
        )
    )

    print("\nparticipation noise, which is the variable that leaks:")
    print(
        f"  {'keep':>5} {'fab':>5} {'recovery':>9} {'eps/ind':>9} {'eps composed':>13} {'noise':>7}"
    )
    for keep, fabricate in PARTICIPATION:
        budget = Budget(keep=keep, fabricate=fabricate)
        stream = randomized_participation(
            fleet,
            tasks,
            policy=DROP_COMPARTMENTS,
            ceiling=FLEET_CEILING,
            budget=budget,
            seed=SEED,
        )
        rec = recovery_of(stream)
        noise = label_noise(stream, fleet, tasks)
        spent = budget.as_dict(separation)
        rows.append(
            Row(
                mechanism="participation",
                setting=f"keep={keep},fab={fabricate}",
                recovery=rec,
                contributions=len(stream),
                label_noise=noise,
                budget=spent,
            )
        )
        per = spent["epsilon_per_indicator"]
        comp = spent["epsilon_effective"]
        print(
            f"  {keep:>5} {fabricate:>5} {rec:>9.3f} "
            f"{'inf' if per is None else f'{per:>9.2f}'} "
            f"{'inf' if comp is None else f'{comp:>13.1f}'} {noise:>7.3f}"
        )

    print(
        "\nthe per-indicator column is the one a mechanism advertises; the composed one"
        f"\nis what an adversary separating two clearances over {separation} tasks faces."
    )

    report = {
        "provenance": run_provenance(seed=SEED),
        "fleet": len(fleet),
        "events": args.events,
        "baseline_recovery": round(base_recovery, 4),
        "widest_separation": separation,
        "value_noise_is_inert": len(unchanged) == 1,
        "rows": [r.as_dict() for r in rows],
        "validity": check_sample_size(len(fleet), label="privacy budget").as_dict(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
