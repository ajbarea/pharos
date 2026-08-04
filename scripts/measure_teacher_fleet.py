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


def _ceiling_analysis(grid: list[dict[str, Any]]) -> dict[str, Any]:
    """How close each adapter comes to the best score obtainable against its teacher.

    Reading the fidelity column alone says inheritance decays: it runs from 1.000 at a
    teacher who never slips to about 0.5 at one who slips half the time. That reading is
    wrong, and the bound is why. A teacher slipping at rate `s` emits labels that
    disagree with its own rule that often, so an adapter that had learned the rule
    *perfectly* would still agree with those labels only `1 - s` of the time. `1 - s` is
    a ceiling on the column, not a target, and the distance below it is the only part
    that is about the adapter at all.

    The second comparison is the same point from the other side. Two independent passes
    of a teacher slipping at `s` agree with each other `(1-s)^2 + s^2` of the time. When
    the adapter beats that, it is reproducing the teacher more faithfully than the
    teacher reproduces itself -- which is what learning a rule from noisy draws of it
    looks like, and is not something a copier could do.
    """
    residual = [r["adapter_agrees_with_teacher"] - (1 - r["slip_rate"]) for r in grid]
    retest = [
        r["adapter_agrees_with_teacher"] - ((1 - r["slip_rate"]) ** 2 + r["slip_rate"] ** 2)
        for r in grid
    ]
    return {
        "residual_vs_denoised_ceiling": {
            "median": round(statistics.median(residual), 4),
            "min": round(min(residual), 4),
            "max": round(max(residual), 4),
            #: The claim in one number: no adapter sits further below the ceiling than
            #: this, so none of them is failing to inherit by more than that much.
            "largest_shortfall": round(-min(residual), 4),
        },
        "vs_teacher_self_agreement": {
            "median": round(statistics.median(retest), 4),
            #: Adapters that reproduce their teacher better than the teacher reproduces
            #: itself. A copier cannot exceed the thing it copies; a learner can.
            "n_exceeding": sum(1 for x in retest if x > 0),
            "n": len(retest),
        },
    }


def _by_threshold(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The inheritance gap conditioned on the standard rather than the carelessness.

    This is the split that changes sign, and averaging over the fleet hides it: a
    teacher applying the correct threshold is improved upon, one applying the strictest
    is made worse, and the fleet-wide median is positive either way. Training denoises a
    teacher who is right and careless; it cannot rescue one who is careful and wrong.
    """
    out = []
    for threshold in sorted({r["threshold"] for r in grid}):
        subset = [r for r in grid if r["threshold"] == threshold]
        gaps = [r["adapter_agrees_with_world"] - r["teacher_agrees_with_world"] for r in subset]
        out.append(
            {
                "threshold": threshold,
                "n": len(subset),
                "median_gap": round(statistics.median(gaps), 4),
                "adapters_better_than_their_teacher": sum(1 for g in gaps if g > 0.01),
                "quotable": sum(1 for r in subset if r["quotable"]),
            }
        )
    return out


def _fidelity_without_usefulness(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapters that reproduce their teacher well and are refused by the validity check.

    The governance result. An operator accepting a personalized model on agreement with
    the analyst who trained it would ship every row here, and every row here is refused
    for scoring at or beneath what answering "escalate" to everything scores. Fidelity
    and usefulness are separate measurements; only one of them can be an acceptance
    gate. Empty is a meaningful answer and would weaken the claim, so it is computed
    rather than asserted.
    """
    return [
        {
            "reviewer": r["reviewer"],
            "adapter_agrees_with_teacher": r["adapter_agrees_with_teacher"],
            "adapter_agrees_with_world": r["adapter_agrees_with_world"],
        }
        for r in sorted(grid, key=lambda r: -r["adapter_agrees_with_teacher"])
        if r["quotable"] is False and r["adapter_agrees_with_teacher"] >= 0.9
    ]


def _gap_summary(subset: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Inheritance gap over one slice of the grid.

    Positive means the adapter agrees with the world *better* than the teacher that
    taught it -- training repaired some of the teacher's error. Near zero means the
    error rate was handed over intact.
    """
    if not subset:
        return {"label": label, "n": 0}
    gaps = [r["adapter_agrees_with_world"] - r["teacher_agrees_with_world"] for r in subset]
    return {
        "label": label,
        "n": len(subset),
        "median_gap": round(statistics.median(gaps), 4),
        "max_gap": round(max(gaps), 4),
        "min_gap": round(min(gaps), 4),
        "adapters_better_than_their_teacher": sum(1 for g in gaps if g > 0.01),
    }


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
            #: Finding 10's actual claim is a dissociation, not a single number: a
            #: *systematic* error (the wrong threshold) is inherited almost exactly,
            #: while a *random* one (slipping) is partly repaired by training, because
            #: noise does not survive being averaged over 1,140 decisions and a wrong
            #: rule does. Four teachers could show that twice; the grid varies both axes
            #: independently, so the two groups below are the test.
            "dissociation": {
                "systematic_only": _gap_summary(
                    [r for r in grid if r["slip_rate"] == 0.0], "slip 0, threshold varies"
                ),
                "with_random_error": _gap_summary(
                    [r for r in grid if r["slip_rate"] > 0.0], "slip > 0"
                ),
            },
            #: Why the fidelity column falls without inheritance failing. See
            #: `_ceiling_analysis`.
            "ceiling": _ceiling_analysis(grid),
            #: The split that changes sign. See `_by_threshold`.
            "by_threshold": _by_threshold(grid),
            "quotable": sum(1 for r in grid if r["quotable"]),
            #: The governance result. See `_fidelity_without_usefulness`.
            "high_fidelity_but_refused": _fidelity_without_usefulness(grid),
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
    for slice_name in ("systematic_only", "with_random_error"):
        s = payload["summary"]["dissociation"][slice_name]
        if s["n"]:
            print(
                f"  {s['label']:24s} n={s['n']:2d}  median gap {s['median_gap']:+.4f}  "
                f"{s['adapters_better_than_their_teacher']} adapters beat their teacher"
            )

    c = payload["summary"]["ceiling"]
    print(
        f"  below the 1-s ceiling by a median of "
        f"{-c['residual_vs_denoised_ceiling']['median']:.4f}, never more than "
        f"{c['residual_vs_denoised_ceiling']['largest_shortfall']:.4f}"
    )
    v = c["vs_teacher_self_agreement"]
    print(f"  reproduce the teacher better than it reproduces itself: {v['n_exceeding']}/{v['n']}")
    print("  by threshold (the split that changes sign):")
    for row in payload["summary"]["by_threshold"]:
        print(
            f"    threshold {row['threshold']}  n={row['n']:2d}  median gap "
            f"{row['median_gap']:+.4f}  {row['adapters_better_than_their_teacher']} beat "
            f"their teacher  {row['quotable']} quotable"
        )
    refused = payload["summary"]["high_fidelity_but_refused"]
    print(f"  high fidelity but refused by the validity check: {len(refused)}")
    for row in refused:
        print(
            f"    {row['reviewer']:10s} reproduces its teacher at "
            f"{row['adapter_agrees_with_teacher']:.4f}, agrees with the world at "
            f"{row['adapter_agrees_with_world']:.4f}"
        )

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
