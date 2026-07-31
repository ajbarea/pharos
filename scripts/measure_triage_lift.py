#!/usr/bin/env python3
"""Can a specialist beat the surface baseline, and can its verdict federate?

The gate publishes a surface baseline: what a model scores on triage while reading
nothing. This asks the paired question, which is the one that matters for the
project. If a specialist cannot beat that baseline, the fleet has nothing worth
federating no matter how correct the labelling around it is.

It also records what each verdict's governed label would be, so lift and
releasability are measured on the same runs rather than argued about separately.

Run from the repo root with Ollama serving:

    uv run python scripts/measure_triage_lift.py --tasks 40
"""

import argparse
import json
from pathlib import Path

from pharos.attribute import DEFAULT_ENDPOINT, DEFAULT_MODEL, generate_text
from pharos.gate import run_gate
from pharos.generate import GeneratorConfig, generate
from pharos.labels import Capacity, Compartment, DeclassificationPolicy, Label, Sensitivity
from pharos.labels import shared_eligible as is_shared_eligible
from pharos.models import resolve
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks

#: A fleet aggregator cleared below the enclaves feeding it, which is what makes
#: sharing a downgrade at all.
CEILING = Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)


def parse_verdict(text: str) -> bool | None:
    """True for significant, False for routine, None when the model did not answer.

    Unparsable answers are counted rather than coerced: silently mapping them to
    one class would move the score, and how often a specialist fails to answer at
    all is a result in its own right.
    """
    upper = text.upper()
    said_significant = "SIGNIFICANT" in upper
    said_routine = "ROUTINE" in upper
    if said_significant == said_routine:
        return None
    return said_significant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=40)
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
    baseline = run_gate(reports, null_trials=10)
    print(f"surface baseline (AUC): {baseline.surface_baseline:.4f}")
    print(f"permutation null:       {baseline.null_mean:.4f} +/- {baseline.null_sd:.4f}\n")

    tasks = build_triage_tasks(reports, limit=args.tasks)
    print(f"{len(tasks)} triage tasks, model {args.model}")

    tp = fp = tn = fn = unparsed = 0
    rows = []
    for task in tasks:
        answer = generate_text(task.prompt, endpoint=args.endpoint, model=args.model, num_predict=8)
        verdict = parse_verdict(answer)
        if verdict is None:
            unparsed += 1
        elif verdict and task.significant:
            tp += 1
        elif verdict and not task.significant:
            fp += 1
        elif not verdict and task.significant:
            fn += 1
        else:
            tn += 1
        rows.append(
            {
                "task_id": task.task_id,
                "truth": task.significant,
                "verdict": verdict,
                "raw": answer.strip()[:40],
                "label": (
                    f"{task.label.sensitivity.name}[{','.join(sorted(task.label.compartments))}]"
                ),
            }
        )

    scored = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(scored, 1)
    prevalence = (tp + fn) / max(scored, 1)

    print("\n" + "=" * 74)
    print(f"scored / unparsable      {scored} / {unparsed}")
    print(f"confusion  tp={tp} fp={fp} tn={tn} fn={fn}")
    print(f"precision                {precision:.4f}")
    print(f"recall                   {recall:.4f}")
    print(f"f1                       {f1:.4f}")
    print(f"accuracy                 {accuracy:.4f}")
    print(f"majority-class accuracy  {max(prevalence, 1 - prevalence):.4f}  <- the floor to beat")
    print(f"surface-baseline AUC     {baseline.surface_baseline:.4f}")

    # Releasability of the verdicts themselves, on the same runs.
    strict = DeclassificationPolicy()
    permissive = DeclassificationPolicy(drop_compartments=True)
    eligible_strict = sum(1 for t in tasks if is_shared_eligible(t.label, CEILING, strict))
    eligible_drop = sum(1 for t in tasks if is_shared_eligible(t.label, CEILING, permissive))
    print(f"\nverdicts releasable, keep-compartments   {eligible_strict}/{len(tasks)}")
    print(f"verdicts releasable, drop-compartments   {eligible_drop}/{len(tasks)}")
    print("=" * 74)

    if accuracy <= max(prevalence, 1 - prevalence):
        print(
            "\nThe specialist does not beat majority-class guessing. Nothing here is worth"
            "\nfederating, and the labelling machinery around it is moot until this changes."
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
                    "surface_baseline": round(baseline.surface_baseline, 4),
                    "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                    "unparsed": unparsed,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "accuracy": round(accuracy, 4),
                    "majority_accuracy": round(max(prevalence, 1 - prevalence), 4),
                    "eligible_keep": eligible_strict,
                    "eligible_drop": eligible_drop,
                    "n_tasks": len(tasks),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
