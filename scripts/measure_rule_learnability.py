#!/usr/bin/env python3
"""Can the decision rule be learned from labelled examples alone?

This is the project's load-bearing premise, reduced to something measurable tonight.
The design claims a fleet learns analytic craft from an analyst's accept, revise, and
reject decisions. Withholding the rule from the prompt and supplying labelled
examples instead is the same question in a cheaper form: if examples cannot teach the
rule, adapters trained on the same signal are unlikely to do better, and the premise
needs rethinking before any federation work matters.

Three reference points, all with the rule withheld unless stated:

- **Floor**: zero examples, over-escalating at high recall.
- **Ceiling**: the rule stated with a checklist prompt, which reaches F1 1.000, so the
  task is known solvable by this model.
- **This script**: k labelled examples, rule never stated. Where it lands between
  those two is how much of the gap analyst feedback can close.

Examples are drawn from events disjoint from the evaluation set, and the example
block is class-balanced, because an unbalanced block teaches the prior rather than
the rule.

Run from the repo root with Ollama serving:

    uv run python scripts/measure_rule_learnability.py --shots 0 2 4 8 --tasks 30
"""

import argparse
import json
from pathlib import Path
from typing import Any

from pharos.attribute import DEFAULT_ENDPOINT, DEFAULT_MODEL, generate_text
from pharos.generate import GeneratorConfig, generate
from pharos.models import resolve
from pharos.prompting import (
    balanced_shots,
    build_prompt,
    parse_verdict,
)
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.uncertainty import Trial, resolves, summarize
from pharos.validity import check_classification

LOG = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--events", type=int, default=400)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "identical back-to-back passes per task. 1 is the publication setting: the "
            "interval resamples TASKS, and a real measurement calls each task once from "
            "cold. More passes measure the warm-up transition finding 9 retracted a "
            "claim over, and report it as uncertainty."
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    # Accept a registry key, a raw tag, or anything the backend knows.
    spec = resolve(args.model)
    args.model = spec.tag

    reports = generate(GeneratorConfig(seed=args.seed, n_events=args.events))
    all_tasks = build_triage_tasks(reports)

    # Disjoint splits: examples never come from the evaluation set.
    max_shots = max(args.shots)
    shot_pool = all_tasks[: max(max_shots * 4, 20)]
    evaluation = all_tasks[max(max_shots * 4, 20) :][: args.tasks]
    print(f"{len(evaluation)} evaluation tasks, {len(shot_pool)} available as examples")
    print(f"model {args.model}; rule NEVER stated in any condition\n")

    rows: list[dict[str, Any]] = []
    measurements = []
    for k in args.shots:
        shots = balanced_shots(shot_pool, k)
        tp = fp = tn = fn = unparsed = 0
        # One Trial per (task, pass). The confusion matrix keeps counting every
        # call, so the headline rates are the single-run estimand rather than a
        # vote -- which is what a fleet answering once per task actually gets.
        trials: list[Trial] = []
        for task in evaluation:
            for _ in range(args.repeats):
                answer = generate_text(
                    build_prompt(task, shots),
                    endpoint=args.endpoint,
                    model=args.model,
                    num_predict=260,
                )
                verdict = parse_verdict(answer)
                trials.append(
                    Trial(task.task_id, None if verdict is None else verdict == task.significant)
                )
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
        measurements.append(summarize(trials, label=f"{k}-shot"))

        scored = tp + fp + tn + fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(scored, 1)
        prevalence = (tp + fn) / max(scored, 1)
        majority = max(prevalence, 1 - prevalence)

        rows.append(
            {
                "shots": k,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "unparsed": unparsed,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                "majority": round(majority, 4),
                "accuracy_interval": measurements[-1].as_dict(),
                # Per shot count, because the conditions differ in exactly the ways
                # the check looks for: unparsed answers spike at four shots and a
                # condition below its own majority floor is not evidence of learning.
                # Computed here and carried into the artifact, so a published number
                # can be checked without rerunning the sweep.
                "validity": check_classification(
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    unparsed=unparsed,
                    label=f"learnability:{k}-shot",
                ).as_dict(),
            }
        )
        record("learnability.f1", f1, shots=k, model=args.model)
        record("learnability.accuracy", accuracy, shots=k, model=args.model)
        print(
            f"shots={k:>2}  acc={accuracy:.3f} (majority {majority:.3f})  "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}  "
            f"tp={tp} fp={fp} tn={tn} fn={fn} un={unparsed}"
        )

    print("\n" + "=" * 74)
    floor = next((r for r in rows if r["shots"] == 0), None)
    best = max(rows, key=lambda r: r["f1"])
    print(f"zero-shot floor    F1 {floor['f1']:.3f}" if floor else "zero-shot floor    not run")
    print(f"best few-shot      F1 {best['f1']:.3f} at {best['shots']} shots")
    print("rule-given ceiling F1 1.000  (measured separately, checklist prompt)")
    if floor:
        closed = (best["f1"] - floor["f1"]) / max(1.000 - floor["f1"], 1e-9)
        print(f"\nshare of the gap to the ceiling closed by examples alone: {closed:.0%}")
        if closed < 0.25:
            print(
                "Examples barely help. The premise that analyst decisions teach craft is in\n"
                "doubt for this task shape, and adapters trained on the same signal start\n"
                "from a weak position."
            )
        elif closed > 0.6:
            print(
                "Examples carry most of the gap, so the rule is learnable from labelled\n"
                "decisions alone. That is the premise the design needs, and it makes the\n"
                "adapter experiment worth running rather than speculative."
            )
    print("=" * 74)

    if args.repeats > 1:
        print("\n" + "=" * 74)
        print("Accuracy with a cluster-bootstrap 95% interval over tasks.")
        print("single-run is what one pass gets; consensus is a majority vote over passes.")
        print("=" * 74)
        for k, m in zip(args.shots, measurements, strict=True):
            i = m.single_run
            print(
                f"  {k:>2} shots  {i.point:.3f}  [{i.low:.3f}, {i.high:.3f}]"
                f"   consensus {m.consensus:.3f}   within-task variance"
                f" {m.variance.within_share:.0%}"
            )

        # Whether the ordering finding 5 reported survives its own noise. Reporting
        # the pairs that do NOT separate is the honest direction: a reader assumes
        # a table's ordering is real unless told otherwise.
        unresolved = [
            (a, b)
            for ia, a in enumerate(args.shots)
            for ib, b in enumerate(args.shots)
            if ia < ib and not resolves(measurements[ia].single_run, measurements[ib].single_run)
        ]
        if unresolved:
            pairs = ", ".join(f"{a} vs {b}" for a, b in unresolved)
            print(f"\n  NOT separated by the intervals: {pairs}")
            print("  Any ordering between those conditions is unsupported by this data.")
        else:
            print("\n  Every pair separates; the ordering is supported.")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(
                        model=args.model, model_key=spec.key, endpoint=args.endpoint, seed=args.seed
                    ),
                    "model": args.model,
                    "seed": args.seed,
                    # The corpus, not just how much of it was scored. `(seed,
                    # n_events)` regenerates the world; `n_eval` alone cannot.
                    "n_events": args.events,
                    "shots": args.shots,
                    "n_eval": len(evaluation),
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
