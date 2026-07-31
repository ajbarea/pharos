#!/usr/bin/env python3
"""Can the decision rule be learned from labelled examples alone?

This is the project's load-bearing premise, reduced to something measurable tonight.
The design claims a fleet learns analytic craft from an analyst's accept, revise, and
reject decisions. Withholding the rule from the prompt and supplying labelled
examples instead is the same question in a cheaper form: if examples cannot teach the
rule, adapters trained on the same signal are unlikely to do better, and the premise
needs rethinking before any federation work matters.

Three reference points, all with the rule withheld unless stated:

- **Floor**: zero examples. Measured previously at F1 0.537 plain, 0.688 with brief
  reasoning, over-escalating at recall 1.000.
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

from pharos.attribute import DEFAULT_ENDPOINT, DEFAULT_MODEL, generate_text
from pharos.generate import GeneratorConfig, generate
from pharos.models import resolve
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, record

LOG = get_logger()

INSTRUCTION_NO_RULE = (
    "You are a maritime watch officer. The numbered reports above all concern one "
    "vessel during one watch window. Decide whether they indicate a SIGNIFICANT event "
    "requiring escalation, or ROUTINE activity.\n\n"
    "Reason briefly about what the reports show, then finish with a line reading "
    "exactly 'VERDICT: SIGNIFICANT' or 'VERDICT: ROUTINE'."
)

EXAMPLE_PREAMBLE = (
    "Here are previous cases with the watch officer's own verdict. Infer the standard "
    "the officer applies, then apply that same standard to the new case.\n"
)


def _reports_block(task: TriageTask) -> str:
    return "\n\n".join(f"[{i + 1}] {r.text}" for i, r in enumerate(task.sources))


def build_prompt(target: TriageTask, shots: list[TriageTask]) -> str:
    """The target case, preceded by `shots` worked examples and no stated rule."""
    parts: list[str] = []
    if shots:
        parts.append(EXAMPLE_PREAMBLE)
        for index, shot in enumerate(shots, start=1):
            verdict = "SIGNIFICANT" if shot.significant else "ROUTINE"
            parts.append(
                f"--- CASE {index} ---\n{_reports_block(shot)}\nOFFICER'S VERDICT: {verdict}\n"
            )
        parts.append("--- NEW CASE ---")
    parts.append(_reports_block(target))
    parts.append(INSTRUCTION_NO_RULE)
    return "\n\n".join(parts)


def parse_verdict(text: str) -> bool | None:
    upper = text.upper()
    tail = upper.split("VERDICT")[-1] if "VERDICT" in upper else upper
    said_significant = "SIGNIFICANT" in tail
    said_routine = "ROUTINE" in tail
    if said_significant == said_routine:
        return None
    return said_significant


def balanced_shots(pool: list[TriageTask], k: int) -> list[TriageTask]:
    """`k` examples alternating between classes, so the block teaches the rule not the prior."""
    positives = [t for t in pool if t.significant]
    negatives = [t for t in pool if not t.significant]
    shots: list[TriageTask] = []
    while len(shots) < k and (positives or negatives):
        if len(shots) % 2 == 0 and positives:
            shots.append(positives.pop(0))
        elif negatives:
            shots.append(negatives.pop(0))
        elif positives:
            shots.append(positives.pop(0))
    return shots[:k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--tasks", type=int, default=30)
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
    all_tasks = build_triage_tasks(reports)

    # Disjoint splits: examples never come from the evaluation set.
    max_shots = max(args.shots)
    shot_pool = all_tasks[: max(max_shots * 4, 20)]
    evaluation = all_tasks[max(max_shots * 4, 20) :][: args.tasks]
    print(f"{len(evaluation)} evaluation tasks, {len(shot_pool)} available as examples")
    print(f"model {args.model}; rule NEVER stated in any condition\n")

    rows = []
    for k in args.shots:
        shots = balanced_shots(shot_pool, k)
        tp = fp = tn = fn = unparsed = 0
        for task in evaluation:
            answer = generate_text(
                build_prompt(task, shots),
                endpoint=args.endpoint,
                model=args.model,
                num_predict=260,
            )
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

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(
                        model=args.model, model_key=spec.key, endpoint=args.endpoint, seed=args.seed
                    ),
                    "model": args.model,
                    "seed": args.seed,
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
