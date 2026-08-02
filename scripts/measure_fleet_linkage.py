#!/usr/bin/env python3
"""Whether a fleet's shared stream identifies the analysts contributing to it.

Every other gate in this repository decides one item at a time, and this asks what a
*stream* of individually-approved items discloses about who produced it. The attack
reads no content: it sees only which task identifiers appear under a pseudonym, and
recovers the contributor's compartment set by matching that against what each
candidate clearance would have been able to reach.

The question is the one the motivating abstract puts in its title -- can a fleet learn
across analysts without leaking what any analyst works on -- and it is not answered by
any per-item gate, because no item in the stream is the leak.

Three things are reported, and the second is the finding:

  1. Recovery against the prior. Guessing the most common compartment set is the
     floor a linkage attack has to beat to mean anything.
  2. Recovery **by the contributor's own clearance**, which is where the result
     lives: identifiability is not uniform across the fleet.
  3. A control ladder, each priced in retained training volume, because a control
     that protects by deleting the fleet's data is not a control.

Needs no model and no network. The corpus is regenerated from its seed and the
attack is deterministic given the fleet seed, so this reproduces exactly and runs in
CI.

    uv run python scripts/measure_fleet_linkage.py --out results/fleet_linkage.json
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pharos.disclosure import DROP_COMPARTMENTS, KEEP_COMPARTMENTS
from pharos.fleet import (
    Clearance,
    Contribution,
    Linkage,
    apply_k_anonymity,
    apply_pooling,
    apply_rarity_suppression,
    apply_subsample,
    assign_fleet,
    candidate_clearances,
    contribute,
    identifiability_ceiling,
    link,
)
from pharos.generate import GeneratorConfig, generate
from pharos.labels import DeclassificationPolicy
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import record
from pharos.uncertainty import Trial, summarize
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200
FLEET_SEED = 11
FLEET_SIZE = 200

#: Candidate compartment sets an adversary chooses among. The prior a linkage attack
#: must beat is guessing the most common one, not 1/16, because the fleet is not
#: uniform over the space once it is drawn.
N_COMPARTMENT_SETS = len({subset for _, subset in candidate_clearances()})


def recovery_rate(linkages: tuple[Linkage, ...]) -> float:
    return sum(link_.exact for link_ in linkages) / len(linkages) if linkages else 0.0


def mean_anonymity(linkages: tuple[Linkage, ...]) -> float:
    live = [link_ for link_ in linkages if not link_.silent]
    return sum(link_.anonymity_set for link_ in live) / len(live) if live else 0.0


def majority_prior(fleet: tuple[Clearance, ...]) -> float:
    """Rate an adversary reaches by always naming the fleet's most common beat."""
    counts = Counter(c.compartments for c in fleet)
    return counts.most_common(1)[0][1] / len(fleet) if fleet else 0.0


def trials(linkages: tuple[Linkage, ...]) -> list[Trial]:
    """One trial per analyst. The analyst is the cluster: tasks are shared across the
    fleet, so resampling tasks would resample the same person many times."""
    return [Trial(link_.analyst_id, link_.exact) for link_ in linkages]


#: Corpus sizes to test the structure at, and the seeds to test it across. The
#: structure saturates well inside this grid, so the grid shows both regimes.
SATURATION_EVENTS = (20, 40, 80, 150, 200, 400)
SATURATION_SEEDS = (1, 7, 11, 23, 101)


def saturation(
    *, policy: DeclassificationPolicy, seeds: Sequence[int] = SATURATION_SEEDS
) -> dict[str, object]:
    """Whether the identifiability structure depends on which corpus was drawn.

    It does not, above a size. The reachable-set *sizes* differ from corpus to
    corpus, but which compartment sets are mutually distinguishable does not, once
    the corpus is large enough that every compartment cell is populated. Below that
    the structure varies by seed and consistently understates the leak, which is the
    safe direction to be wrong in but worth knowing about before quoting a small run.

    Reported because it decides what the headline number is a property *of*. If the
    structure were corpus-dependent, the interval over analysts would be the wrong
    uncertainty and a corpus resample would be needed too.
    """
    # Kept as parallel typed lists rather than read back out of the JSON rows, so the
    # saturation point is computed from ints instead of from `object`.
    measured: list[tuple[int, int, list[float]]] = []
    for events in SATURATION_EVENTS:
        shapes: set[tuple[int, int, int]] = set()
        recoveries: set[float] = set()
        for seed in seeds:
            tasks = build_triage_tasks(generate(GeneratorConfig(seed=seed, n_events=events)))
            ceiling = identifiability_ceiling(tasks, policy=policy)
            shapes.add(
                (
                    ceiling["distinct_reachable_sets"],
                    ceiling["uniquely_identifying_sets"],
                    ceiling["largest_anonymity_class"],
                )
            )
            fleet = assign_fleet(FLEET_SIZE, seed=FLEET_SEED)
            linkages = link(contribute(fleet, tasks, policy=policy), tasks, fleet, policy=policy)
            recoveries.add(round(recovery_rate(linkages), 4))
        measured.append((events, len(shapes), sorted(recoveries)))

    # Smallest tested size at which every seed agrees and every larger one does too.
    # "And every larger one" matters: a single agreeing size below a disagreeing one
    # is a coincidence, not saturation.
    saturates_at = next(
        (
            events
            for events, distinct, _ in measured
            if distinct == 1 and all(d == 1 for e, d, _ in measured if e >= events)
        ),
        None,
    )
    return {
        "seeds": list(seeds),
        "grid": [
            {
                "events": events,
                "distinct_structures": distinct,
                "invariant": distinct == 1,
                "recoveries": recoveries,
            }
            for events, distinct, recoveries in measured
        ],
        "saturates_at": saturates_at,
    }


def evaluate(
    fleet: tuple[Clearance, ...],
    tasks: Sequence[TriageTask],
    stream: tuple[Contribution, ...],
    *,
    policy: DeclassificationPolicy,
    baseline_volume: int,
) -> dict[str, object]:
    linkages = link(stream, tasks, fleet, policy=policy)
    return {
        "recovery": round(recovery_rate(linkages), 4),
        "mean_anonymity_set": round(mean_anonymity(linkages), 2),
        "contributions": len(stream),
        "retained_volume": round(len(stream) / baseline_volume, 4) if baseline_volume else 0.0,
        "silenced_analysts": sum(link_.silent for link_ in linkages),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET_SIZE)
    parser.add_argument("--fleet-seed", type=int, default=FLEET_SEED)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    fleet = assign_fleet(args.fleet, seed=args.fleet_seed)
    prior = majority_prior(fleet)

    print(f"corpus {len(tasks)} tasks, fleet {len(fleet)} analysts")
    print(f"prior (always name the fleet's most common beat): {prior:.3f}")
    print(f"candidate compartment sets: {N_COMPARTMENT_SETS}\n")

    # The two rulings finding 2 showed federation is bimodal on, now priced for
    # disclosure as well as for eligibility.
    report: dict[str, object] = {}
    for name, policy in (("drop_compartments", DROP_COMPARTMENTS), ("keep", KEEP_COMPARTMENTS)):
        stream = contribute(fleet, tasks, policy=policy)
        linkages = link(stream, tasks, fleet, policy=policy)
        measured = summarize(trials(linkages), label=f"recovery-{name}")
        report[name] = {
            "recovery": measured.single_run.as_dict(),
            "mean_anonymity_set": round(mean_anonymity(linkages), 2),
            "contributions": len(stream),
        }
        print(
            f"{name:<18} contributions={len(stream):>6}  "
            f"recovery={measured.single_run.point:.3f} "
            f"[{measured.single_run.low:.3f}, {measured.single_run.high:.3f}]  "
            f"anonymity={mean_anonymity(linkages):.1f}"
        )

    # The finding: identifiability is not uniform across the fleet.
    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)
    linkages = link(stream, tasks, fleet, policy=DROP_COMPARTMENTS)
    by_id = {c.analyst_id: c for c in fleet}
    print("\nrecovery by the contributor's own clearance level:")
    by_level = []
    for level in sorted({c.sensitivity for c in fleet}, key=int):
        group = tuple(link_ for link_ in linkages if by_id[link_.analyst_id].sensitivity == level)
        if not group:
            continue
        measured = summarize(trials(group), label=f"recovery-{level.name}")
        by_level.append(
            {
                "level": level.name,
                "n": len(group),
                "recovery": measured.single_run.as_dict(),
                "mean_anonymity_set": round(mean_anonymity(group), 2),
            }
        )
        print(
            f"  {level.name:<11} n={len(group):>3}  recovery={measured.single_run.point:.3f} "
            f"[{measured.single_run.low:.3f}, {measured.single_run.high:.3f}]  "
            f"anonymity={mean_anonymity(group):>5.1f}"
        )
    report["by_clearance_level"] = by_level

    # The control ladder, each priced in retained volume.
    baseline = len(stream)
    controls: dict[str, object] = {}
    print("\ncontrols, each priced in retained training volume:")
    ladder: list[tuple[str, tuple[Contribution, ...]]] = [
        ("none", stream),
        *((f"k_anonymity_k{k}", apply_k_anonymity(stream, k)) for k in (10, 25, 50, 100)),
        *(
            (f"rarity_suppression_keep{keep}", apply_rarity_suppression(stream, keep))
            for keep in (0.75, 0.5, 0.25)
        ),
        *(
            (f"subsample_p{keep}", apply_subsample(stream, keep, seed=3))
            for keep in (0.5, 0.2, 0.05)
        ),
        ("pooled", apply_pooling(stream)),
    ]

    for name, controlled in ladder:
        linkages = link(controlled, tasks, fleet, policy=DROP_COMPARTMENTS)
        row = evaluate(fleet, tasks, controlled, policy=DROP_COMPARTMENTS, baseline_volume=baseline)
        controls[name] = row
        # Labelled, so a reader can tell which control produced which outcome. The
        # library emits the same numbers at DEBUG without knowing the control's name.
        record(
            "linkage.control",
            recovery_rate(linkages),
            control=name,
            retained_volume=len(controlled) / baseline if baseline else 0.0,
            mean_anonymity_set=round(mean_anonymity(linkages), 2),
            silenced=sum(1 for x in linkages if x.silent),
        )
        print(
            f"  {name:<30} recovery={row['recovery']:.3f}  "
            f"anonymity={row['mean_anonymity_set']:>6.1f}  "
            f"volume={row['retained_volume']:.3f}  silenced={row['silenced_analysts']}"
        )
    report["controls"] = controls

    validity = check_sample_size(len(fleet), label="fleet linkage")
    report["validity"] = validity.as_dict()
    report["structure"] = identifiability_ceiling(tasks, policy=DROP_COMPARTMENTS)
    saturation_report = saturation(policy=DROP_COMPARTMENTS)
    report["saturation"] = saturation_report
    print(
        "\nidentifiability ceiling, independent of which fleet was drawn:\n"
        f"  {report['structure']['candidate_clearances']} candidate clearances collapse to "
        f"{report['structure']['distinct_reachable_sets']} distinct reachable task sets\n"
        f"  {report['structure']['uniquely_identifying_sets']} of those identify a single "
        f"compartment set outright; the largest hides "
        f"{report['structure']['largest_anonymity_class']}"
    )
    print(
        f"  structure is invariant across {len(SATURATION_SEEDS)} corpus seeds at "
        f">= {saturation_report['saturates_at']} events; "
        "below that a corpus understates the leak"
    )
    report["prior"] = round(prior, 4)
    report["candidate_compartment_sets"] = N_COMPARTMENT_SETS
    report["provenance"] = run_provenance(seed=SEED)
    report["fleet_seed"] = args.fleet_seed
    report["events"] = args.events

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
