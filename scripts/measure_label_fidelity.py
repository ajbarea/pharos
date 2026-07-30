#!/usr/bin/env python3
"""Does attribution recover a correct governed label?

This is the experiment the whole design rests on. A ledger entry's label is the
join of the labels of every source that fed the turn, so if attribution is wrong
the label is wrong, and the two error directions are not equally bad:

- Over-attribution inflates the join. The entry is needlessly restricted, which
  costs federation. Safe, and called `creep` here.
- Under-attribution deflates it. The entry looks more releasable than it is, which
  is how restricted material reaches a shared adapter. Called `leak`, and it must
  never happen.

Run from the repo root with Ollama serving:

    uv run python scripts/measure_label_fidelity.py --tasks 8
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from pharos.attribute import DEFAULT_ENDPOINT, DEFAULT_MODEL, attribute_leave_one_out
from pharos.detect import detector_accuracy
from pharos.generate import GeneratorConfig, generate
from pharos.tasks import build_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--events", type=int, default=400)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    reports = generate(GeneratorConfig(seed=args.seed, n_events=args.events))
    accuracy = detector_accuracy(reports)
    print(f"detector on this corpus: {accuracy.as_dict()}")
    if accuracy.f1 < 0.95:
        print("detector too weak to interpret attribution; aborting")
        return 1

    tasks = build_tasks(reports, limit=args.tasks)
    print(f"{len(tasks)} tasks, {len(tasks[0].sources)} sources each, model {args.model}\n")

    rows = []
    outcomes: Counter[str] = Counter()
    for task in tasks:
        attribution = attribute_leave_one_out(task, endpoint=args.endpoint, model=args.model)
        outcome = attribution.label_outcome(task)
        outcomes[outcome] += 1

        attributed = attribution.attributed_label(task)
        truth = attribution.true_label(task)
        # Redundancy: how many asserted facts are carried by more than one source.
        redundant = sum(
            1 for f in attribution.asserted_facts if len(task.sources_containing(f)) > 1
        )
        rows.append(
            {
                "task_id": task.task_id,
                "asserted_facts": len(attribution.asserted_facts),
                "redundant_facts": redundant,
                "truly_contributing": len(attribution.truly_contributing),
                "attributed": len(attribution.attributed_sources),
                "source_recall": round(attribution.source_recall, 4),
                "source_precision": round(attribution.source_precision, 4),
                "true_label": f"{truth.sensitivity.name}[{','.join(sorted(truth.compartments))}]",
                "attributed_label": (
                    f"{attributed.sensitivity.name}[{','.join(sorted(attributed.compartments))}]"
                ),
                "outcome": outcome,
                "calls": attribution.calls,
            }
        )
        print(
            f"{task.task_id}  facts={len(attribution.asserted_facts):>2} "
            f"redundant={redundant:>2}  true_sources={len(attribution.truly_contributing):>2} "
            f"attributed={len(attribution.attributed_sources):>2}  "
            f"recall={attribution.source_recall:.2f}  "
            f"{rows[-1]['true_label']:>28} -> {rows[-1]['attributed_label']:<28} {outcome.upper()}"
        )

    recalls = [r["source_recall"] for r in rows]
    precisions = [r["source_precision"] for r in rows]
    print("\n" + "=" * 78)
    print(f"tasks                    {len(rows)}")
    print(f"source recall            mean {statistics.mean(recalls):.4f}")
    print(f"source precision         mean {statistics.mean(precisions):.4f}")
    print(f"total model calls        {sum(r['calls'] for r in rows)}")
    print("label outcomes:")
    for name in ("exact", "creep", "leak", "incomparable"):
        count = outcomes[name]
        share = count / max(len(rows), 1)
        print(f"  {name:14} {count:>3}  ({share:.0%})")
    leaks = outcomes["leak"] + outcomes["incomparable"]
    print(
        f"\nLEAK RATE                {leaks / max(len(rows), 1):.0%}"
        "   <- any nonzero value means the label cannot be trusted as built"
    )
    print("=" * 78)

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "seed": args.seed,
                    "detector": accuracy.as_dict(),
                    "rows": rows,
                    "outcomes": dict(outcomes),
                    "source_recall_mean": round(statistics.mean(recalls), 4),
                    "source_precision_mean": round(statistics.mean(precisions), 4),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
