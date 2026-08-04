#!/usr/bin/env python3
"""Does an adapter inherit its teacher's standard, across a fleet rather than four cases?

Finding 10 established that a fleet learns its analyst's standard and not the world's,
on four teachers chosen to span target accuracy. Four adapters is four case studies:
federated-adaptation work runs twenty clients in its main experiments and extends to
fifty and a hundred, and the claim this project leads with rested on four.

This reads the grid the cluster array produces -- one adapter per point on the
standard-by-carefulness grid, the same grid `measure_review_sweep.py` sweeps -- and asks
the question at fleet scale:

- **Inheritance.** How closely does each adapter reproduce the teacher that taught it?
  Finding 10's claim is that the answer is "almost exactly", and at four teachers the
  correspondence read as an identity rather than a tendency. Twenty-four says whether
  that survives.
- **Direction.** Does the adapter track the teacher's agreement with the world, or is it
  pulled toward the world by the base model's own prior? These predict opposite things
  for the design and are distinguishable only by scoring the same decode twice.

Reads artifacts only; calls no model. The adapters themselves are cluster jobs.

    uv run python scripts/measure_teacher_fleet.py --out results/teacher_fleet.json
"""

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from pharos.provenance import run_provenance
from pharos.telemetry import get_logger, record

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOG = get_logger()

#: `review_adapter-t2s0.15.json` -- a grid teacher, as opposed to the four named ones,
#: which are kept out of this aggregate on purpose: they were trained at a different
#: time on a different corpus and mixing them in would be the cross-corpus comparison
#: this repository withdrew a claim for.
GRID = re.compile(r"^review_adapter-t(?P<threshold>[123])s(?P<slip>[0-9.]+)\.json$")


def rows() -> list[dict[str, Any]]:
    """One row per grid teacher, sorted by the standard then the carelessness."""
    out: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("review_adapter-t*.json")):
        spec = GRID.match(path.name)
        if spec is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        teacher = payload["teacher"]
        adapter = payload["adapter"]
        against_teacher = payload["adapter_vs_teacher"]
        out.append(
            {
                "reviewer": teacher["reviewer"],
                "threshold": int(spec["threshold"]),
                "slip_rate": float(spec["slip"]),
                #: What the teacher's own training targets agreed with the world on.
                #: The x-axis of the whole question.
                "teacher_agrees_with_world": teacher["train_target_agreement"],
                #: What the adapter trained on those targets agrees with the world on.
                "adapter_agrees_with_world": adapter["accuracy"],
                #: And with the teacher that taught it. Finding 10 says this is the
                #: number that stays high while the one above tracks the teacher down.
                "adapter_agrees_with_teacher": against_teacher["accuracy"],
                "adapter_recall": adapter["recall"],
                "adapter_unparsed": adapter["unparsed"],
                "quotable": (adapter.get("validity") or {}).get("quotable"),
            }
        )
    return sorted(out, key=lambda r: (r["threshold"], r["slip_rate"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    grid = rows()
    if not grid:
        raise SystemExit(
            "no grid adapters in results/. Run `sbatch --array=0-23%4 "
            "cluster/review-adapter.sbatch` and sync the artifacts back."
        )

    fidelity = [r["adapter_agrees_with_teacher"] for r in grid]
    #: How far each adapter's disagreement with the world differs from its teacher's.
    #: Near zero means the error rate was inherited rather than damped or amplified.
    inheritance_gap = [
        r["adapter_agrees_with_world"] - r["teacher_agrees_with_world"] for r in grid
    ]

    for r in grid:
        record(
            "teacher_fleet.inheritance",
            r["adapter_agrees_with_teacher"],
            reviewer=r["reviewer"],
        )

    payload = {
        "n_teachers": len(grid),
        "rows": grid,
        "summary": {
            "teacher_world_agreement_span": [
                min(r["teacher_agrees_with_world"] for r in grid),
                max(r["teacher_agrees_with_world"] for r in grid),
            ],
            "adapter_teacher_fidelity": {
                "min": min(fidelity),
                "median": statistics.median(fidelity),
                "max": max(fidelity),
            },
            "inheritance_gap": {
                "min": min(inheritance_gap),
                "median": statistics.median(inheritance_gap),
                "max": max(inheritance_gap),
                #: The claim in one number: the largest distance any adapter put between
                #: itself and its teacher's error rate.
                "largest_absolute": max(abs(g) for g in inheritance_gap),
            },
            "adapters_with_perfect_recall": sum(1 for r in grid if r["adapter_recall"] >= 1.0),
        },
        "provenance": run_provenance(n_teachers=len(grid)),
    }

    print(f"{len(grid)} grid teachers")
    print(
        f"  teacher agreement with world spans "
        f"{payload['summary']['teacher_world_agreement_span'][0]:.4f} to "
        f"{payload['summary']['teacher_world_agreement_span'][1]:.4f}"
    )
    f = payload["summary"]["adapter_teacher_fidelity"]
    print(f"  adapter reproduces its teacher: min {f['min']:.4f} median {f['median']:.4f}")
    g = payload["summary"]["inheritance_gap"]
    print(f"  largest gap between adapter and teacher error rate: {g['largest_absolute']:.4f}")

    if f["min"] < 0.9:
        LOG.warning(
            "teacher_fleet.inheritance_incomplete",
            extra={
                "event": "teacher_fleet.inheritance_incomplete",
                "min_fidelity": f["min"],
            },
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
