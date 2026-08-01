#!/usr/bin/env python3
"""How much a score moves when nothing changes.

Every model-dependent number in this repo is a single pass: one call per task, one
verdict, one score. That is only honest if an identical call returns an identical
answer, and this measures whether it does by asking the same question several times
and counting how often the answer changes.

It does change, and how much depends on the decode rather than on the machine:

- **Rule stated, 8 tokens.** The triage conditions. Greedy decode of a single word
  from a prompt that already contains the rule.
- **Rule withheld, 320 tokens.** The in-context conditions. The model reasons first
  and emits a verdict line at the end, so any divergence early in the reasoning has
  hundreds of tokens to compound before it reaches the answer.

The distinction matters because the manuscript previously attributed a small number
of changed judgements to running on *different hardware*. That reading is available
only if same-machine repeats are stable, and one of these two regimes is not.

Temperature is 0.0 and the seed is fixed at 7 for every call, set in
`pharos.attribute.generate_text`. Nothing here relaxes that; the point is that
setting it is not sufficient.

    uv run python scripts/measure_decode_stability.py --out results/decode_stability.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from measure_rule_learnability import build_prompt, parse_verdict

from pharos.attribute import DEFAULT_ENDPOINT, DEFAULT_MODEL, generate_text
from pharos.generate import GeneratorConfig, generate
from pharos.models import resolve
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, progress, record

LOG = get_logger()

SEED = 7


def parse_short(text: str) -> bool | None:
    """The triage parser: a bare verdict word, no VERDICT: prefix expected."""
    upper = text.upper()
    said_significant = "SIGNIFICANT" in upper
    said_routine = "ROUTINE" in upper
    if said_significant == said_routine:
        return None
    return said_significant


@dataclass(frozen=True, slots=True)
class Regime:
    """One decode setting, and how it is scored."""

    name: str
    num_predict: int
    reasons: bool
    description: str


REGIMES: tuple[Regime, ...] = (
    Regime(
        "rule-stated-8",
        8,
        False,
        "the triage conditions: rule in the prompt, one word out",
    ),
    Regime(
        "rule-withheld-320",
        320,
        True,
        "the in-context conditions: reason first, verdict last",
    ),
)


@dataclass(frozen=True, slots=True)
class Stability:
    """How often an identical call returned a different answer."""

    regime: str
    num_predict: int
    n_tasks: int
    repeats: int
    unstable: int
    unparsed_any: int

    @property
    def unstable_share(self) -> float:
        return self.unstable / self.n_tasks if self.n_tasks else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime,
            "num_predict": self.num_predict,
            "n_tasks": self.n_tasks,
            "repeats": self.repeats,
            "unstable_tasks": self.unstable,
            "unstable_share": round(self.unstable_share, 4),
            "tasks_with_an_unparsed_call": self.unparsed_any,
        }


def measure(
    regime: Regime,
    tasks: list[TriageTask],
    *,
    repeats: int,
    model: str,
    endpoint: str,
) -> tuple[Stability, list[str]]:
    """Call each task `repeats` times and count the ones that disagree with themselves."""
    unstable: list[str] = []
    unparsed_any = 0
    for index, task in enumerate(tasks):
        prompt = build_prompt(task, []) if regime.reasons else task.prompt
        parse = parse_verdict if regime.reasons else parse_short
        answers = [
            parse(
                generate_text(
                    prompt, endpoint=endpoint, model=model, num_predict=regime.num_predict
                )
            )
            for _ in range(repeats)
        ]
        if any(a is None for a in answers):
            unparsed_any += 1
        if len(set(answers)) > 1:
            unstable.append(task.task_id)
        if (index + 1) % 10 == 0:
            progress(
                "decode_stability.progress",
                regime=regime.name,
                done=index + 1,
                total=len(tasks),
                unstable=len(unstable),
            )
    return (
        Stability(
            regime.name, regime.num_predict, len(tasks), repeats, len(unstable), unparsed_any
        ),
        unstable,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--events", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    spec = resolve(args.model)
    reports = generate(GeneratorConfig(seed=SEED, n_events=args.events))
    tasks = build_triage_tasks(reports, limit=args.tasks)

    print(f"{len(tasks)} tasks, {args.repeats} identical calls each, model {spec.tag}")
    print("temperature 0.0, seed 7, one machine, back to back\n")

    rows = []
    flipped: dict[str, list[str]] = {}
    for regime in REGIMES:
        stability, unstable = measure(
            regime, tasks, repeats=args.repeats, model=spec.tag, endpoint=args.endpoint
        )
        rows.append(stability)
        flipped[regime.name] = unstable
        print(f"{regime.name:20} {regime.description}")
        print(
            f"  {stability.unstable}/{stability.n_tasks} tasks disagreed with themselves "
            f"({stability.unstable_share:.1%})"
        )
        if unstable:
            print(f"  {', '.join(unstable)}")
        record(
            "decode.unstable_share",
            stability.unstable_share,
            regime=regime.name,
            num_predict=regime.num_predict,
        )
        print()

    short = next(r for r in rows if r.num_predict == min(x.num_predict for x in rows))
    long_ = next(r for r in rows if r.num_predict == max(x.num_predict for x in rows))
    print("=" * 74)
    if long_.unstable_share > short.unstable_share:
        print(
            f"The long decode is {long_.unstable_share:.1%} unstable against "
            f"{short.unstable_share:.1%} for the short one, on the same machine and the\n"
            "same prompt. Instability is a property of decode length, not of the platform."
        )
    else:
        print("The two regimes are comparably stable; the earlier reading does not reproduce.")
    print("=" * 74)

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(
                        model=spec.tag, model_key=spec.key, endpoint=args.endpoint, seed=SEED
                    ),
                    "seed": SEED,
                    "temperature": 0.0,
                    "regimes": [r.as_dict() for r in rows],
                    "unstable_task_ids": flipped,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
