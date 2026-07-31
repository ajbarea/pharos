#!/usr/bin/env python3
"""Under content-provenance labelling, what fraction of turns can ever federate?

The companion question to label fidelity. Getting the label right is worthless if
every correct label turns out to be too restrictive to share, because then the
federated half of the design is dead and the system reduces to local learning.

Labels come from content provenance rather than ablation, for reasons measured in
`measure_label_fidelity.py`: leave-one-out attribution misses corroborated
sources and mislabels half of all turns in the under-restrictive direction.
Provenance labelling is conservative by construction, so this script asks what
that conservatism costs.

Run from the repo root with Ollama serving:

    uv run python scripts/measure_federation_eligibility.py --tasks 8
"""

import argparse
import json
import statistics
from pathlib import Path

from pharos.attribute import DEFAULT_ENDPOINT, DEFAULT_MODEL, generate_text, label_by_provenance
from pharos.detect import detect_facts, detector_accuracy
from pharos.generate import GeneratorConfig, generate
from pharos.labels import (
    Capacity,
    Compartment,
    DeclassificationPolicy,
    Label,
    Sensitivity,
    shared_eligible,
)
from pharos.models import resolve
from pharos.provenance import run_provenance
from pharos.tasks import build_tasks

#: Plausible fleet aggregators. An aggregator sits at or below the enclaves that
#: feed it, which is what makes sharing a downgrade in the first place, so a
#: ceiling cleared for everything is not a realistic case.
CEILINGS: tuple[tuple[str, Label], ...] = (
    ("OPEN[]", Label(Sensitivity.OPEN, frozenset(), Capacity.FREETEXT)),
    (
        "INTERNAL[SENSOR]",
        Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT),
    ),
    (
        "PROTECTED[SENSOR,LEGAL]",
        Label(
            Sensitivity.PROTECTED,
            frozenset({Compartment.SENSOR, Compartment.LEGAL}),
            Capacity.FREETEXT,
        ),
    ),
)

POLICIES: tuple[tuple[str, DeclassificationPolicy], ...] = (
    ("keep-compartments", DeclassificationPolicy()),
    ("drop-compartments", DeclassificationPolicy(drop_compartments=True)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--events", type=int, default=400)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    # Accept a registry key, a raw tag, or anything the backend knows.
    spec = resolve(args.model)
    args.model = spec.tag

    reports = generate(GeneratorConfig(seed=args.seed, n_events=args.events))
    accuracy = detector_accuracy(reports)
    if accuracy.f1 < 0.95:
        print(f"detector too weak to interpret results: {accuracy.as_dict()}")
        return 1
    print(f"detector: {accuracy.as_dict()}")

    tasks = build_tasks(reports, limit=args.tasks)
    print(f"{len(tasks)} tasks, {len(tasks[0].sources)} sources each, model {args.model}")
    print("one call per task: provenance labelling needs no ablation sweep\n")

    labels: list[tuple[str, Label]] = []
    compartment_counts: list[int] = []
    for task in tasks:
        summary = generate_text(task.prompt, endpoint=args.endpoint, model=args.model)
        asserted = detect_facts(summary)
        base = label_by_provenance(task, asserted, capacity=Capacity.FREETEXT)
        labels.append((task.task_id, base))
        compartment_counts.append(len(base.compartments))
        print(
            f"  {task.task_id}  facts={len(asserted):>2}  "
            f"label={base.sensitivity.name}[{','.join(sorted(base.compartments))}]"
        )

    print(
        f"\nmean compartments per turn: {statistics.mean(compartment_counts):.2f} "
        f"of {len(Compartment)}"
    )
    print(
        f"turns already at the top of the level ladder: "
        f"{sum(1 for _, lab in labels if lab.sensitivity is Sensitivity.RESTRICTED)}/{len(labels)}"
    )

    print("\n" + "=" * 78)
    print(f"{'ceiling':26} {'policy':20} {'capacity':10} {'eligible':>10}")
    grid: list[dict[str, object]] = []
    for ceiling_name, ceiling in CEILINGS:
        for policy_name, policy in POLICIES:
            for capacity in (Capacity.FREETEXT, Capacity.SPAN, Capacity.SCALAR, Capacity.ENUM):
                eligible = sum(
                    1
                    for _, base in labels
                    if shared_eligible(
                        Label(base.sensitivity, base.compartments, capacity), ceiling, policy
                    )
                )
                share = eligible / max(len(labels), 1)
                grid.append(
                    {
                        "ceiling": ceiling_name,
                        "policy": policy_name,
                        "capacity": capacity.name,
                        "eligible": eligible,
                        "share": round(share, 4),
                    }
                )
                print(
                    f"{ceiling_name:26} {policy_name:20} {capacity.name:10} "
                    f"{eligible:>4}/{len(labels)}  {share:>4.0%}"
                )
    print("=" * 78)

    keep = [row for row in grid if row["policy"] == "keep-compartments"]
    if all(row["eligible"] == 0 for row in keep):
        print(
            "\nEvery cell under keep-compartments is zero. With the fail-closed default,"
            "\nnothing ever federates and the federated half of the design is inert. The"
            "\nsystem's viability therefore rests on a single policy question: may a"
            "\nlow-capacity verdict shed the compartments of the sources behind it?"
        )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(
                        model=args.model, model_key=spec.key, endpoint=args.endpoint, seed=args.seed
                    ),
                    "model": args.model,
                    "seed": args.seed,
                    "n_tasks": len(labels),
                    "mean_compartments": round(statistics.mean(compartment_counts), 3),
                    "grid": grid,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
