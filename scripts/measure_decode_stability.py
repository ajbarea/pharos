#!/usr/bin/env python3
"""Whether a score moves when nothing changes, measured two ways that disagree.

**The measurement design decides the answer, and one of the two designs is wrong.**
This script runs both, because getting that wrong is how this repository briefly
published a false finding.

- **Repeat-one-prompt.** Call the same prompt several times in a row and count how
  often the answer changes. Intuitive, and misleading: the first call against a
  prompt differs from every call after it, so this design measures a warm-up
  transition and reports it as noise.
- **Repeat-the-whole-pass.** Run the entire measurement, each task called exactly
  once in order, then run it again and compare. This is how a measurement actually
  executes: every call is cold, so there is no warm call to disagree with.

The second is the reproducibility number that matters, and it comes out clean.
Finding 9 originally reported the first design's 10% as evidence that single-pass
scores are irreproducible. They are not. The retraction is recorded in
`docs/findings.md`; this script exists so the two designs can be compared rather
than argued about.

Temperature is 0.0 and the seed is fixed at 7 for every call, set in
`pharos.attribute.generate_text`.

    uv run python scripts/measure_decode_stability.py --out results/decode_stability.json
"""

import argparse
import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

from measure_rule_learnability import build_prompt, parse_verdict

from pharos.attribute import DEFAULT_ENDPOINT, DEFAULT_MODEL, generate_text
from pharos.generate import GeneratorConfig, generate
from pharos.models import resolve
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, progress, record
from pharos.validity import check_sample_size

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


@dataclass(frozen=True, slots=True, kw_only=True)
class Stability:
    """How often an identical call returned a different answer.

    Keyword-only: four consecutive count fields mean a positional construction that
    drifts out of order reports a different quantity without failing.
    """

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
            regime=regime.name,
            num_predict=regime.num_predict,
            n_tasks=len(tasks),
            repeats=repeats,
            unstable=len(unstable),
            unparsed_any=unparsed_any,
        ),
        unstable,
    )


def full_pass(
    regime: Regime, tasks: list[TriageTask], *, model: str, endpoint: str
) -> dict[str, bool | None]:
    """One complete measurement sweep: every task called exactly once, in order."""
    parse = parse_verdict if regime.reasons else parse_short
    answers: dict[str, bool | None] = {}
    for task in tasks:
        prompt = build_prompt(task, []) if regime.reasons else task.prompt
        answers[task.task_id] = parse(
            generate_text(prompt, endpoint=endpoint, model=model, num_predict=regime.num_predict)
        )
    return answers


def compare_passes(
    regime: Regime, tasks: list[TriageTask], *, passes: int, model: str, endpoint: str
) -> tuple[int, list[str]]:
    """How many tasks differ across `passes` complete sweeps.

    The honest reproducibility question. Each sweep visits every task once, so no
    call is ever answering a prompt the backend has just seen -- which is exactly
    the condition `measure` violates and the reason its number is larger.
    """
    runs = [full_pass(regime, tasks, model=model, endpoint=endpoint) for _ in range(passes)]
    differing = [task.task_id for task in tasks if len({run[task.task_id] for run in runs}) > 1]
    return len(differing), differing


def rate_upper_bound(differing: int, n: int, *, level: float = 0.95) -> float:
    """The largest disagreement rate this observation does not rule out.

    A count of zero is not a measurement of zero, and reporting it as though it were
    is how "the sweep reproduces" became a claim resting on 30 tasks. For the
    all-zero case this is the exact one-sided Clopper-Pearson bound, `1 - a**(1/n)`;
    otherwise it is solved for the same way, as the largest `p` under which seeing
    this few disagreements still has probability `1 - level`.
    """
    if n <= 0:
        return 1.0
    alpha = 1.0 - level
    if differing == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = differing / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, k) * mid**k * (1 - mid) ** (n - k) for k in range(differing + 1))
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # 300, not 30. Finding 9's replacement rests on "0 of N differ across full
    # sweeps", and 0 of 30 admits any rate up to 9.5% -- it cannot exclude the 10%
    # that finding retracted. 0 of 300 bounds it under 1%.
    parser.add_argument("--tasks", type=int, default=300)
    parser.add_argument("--events", type=int, default=400)
    parser.add_argument(
        "--repeats", type=int, default=3, help="calls per task, one prompt at a time"
    )
    parser.add_argument("--passes", type=int, default=2, help="complete sweeps to compare")
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

    # The design that matches how a measurement actually runs.
    print("=" * 74)
    print(f"Now the same question asked properly: {args.passes} complete sweeps, each task")
    print("called once per sweep, compared against each other.\n")
    pass_rows: list[dict[str, Any]] = []
    worst_differing = 0
    for regime in REGIMES:
        n_diff, which = compare_passes(
            regime, tasks, passes=args.passes, model=spec.tag, endpoint=args.endpoint
        )
        worst_differing = max(worst_differing, n_diff)
        bound = rate_upper_bound(n_diff, len(tasks))
        pass_rows.append(
            {
                "regime": regime.name,
                "differing": n_diff,
                "n_tasks": len(tasks),
                # A count of zero is not a measurement of zero. Carrying the bound in
                # the artifact is what stops the next reader quoting "0 differ" as
                # though the rate were known to be nil.
                "rate_upper_bound_95": round(bound, 4),
                "task_ids": which,
            }
        )
        print(
            f"  {regime.name:20} {n_diff}/{len(tasks)} tasks differ across full sweeps"
            f"   (rules out any rate above {bound:.2%})"
        )
        if which:
            print(f"    {', '.join(which)}")
        record(
            "decode.cross_pass_bound",
            bound,
            regime=regime.name,
            differing=n_diff,
            n_tasks=len(tasks),
        )

    print("\n" + "=" * 74)
    worst_repeat = max(r.unstable_share for r in rows)
    worst_pass = worst_differing / max(len(tasks), 1)
    worst_bound = max(rate_upper_bound(r["differing"], len(tasks)) for r in pass_rows)
    if worst_repeat > worst_pass:
        print(
            f"Repeating one prompt disagrees on up to {worst_repeat:.1%} of tasks; repeating the\n"
            f"whole sweep disagrees on {worst_pass:.1%}. The gap is a warm-up transition, not\n"
            "noise: the first call against a prompt differs from every call after it, so a\n"
            "design that repeats one prompt measures cold-versus-warm and reports it as\n"
            "irreproducibility."
        )
        print(
            f"\nWhat the sweep result licenses, precisely: a rate no higher than "
            f"{worst_bound:.2%}.\nNot zero. This measurement ran at 30 tasks once, where the "
            "same observation would\nhave licensed only 'no higher than 9.5%' -- a bound too "
            "loose to exclude the 10%\nthis finding originally reported and retracted."
        )
    else:
        print("Both designs agree; no warm-up transition is detectable at this sample size.")
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
                    "repeat_one_prompt": [r.as_dict() for r in rows],
                    "repeat_one_prompt_task_ids": flipped,
                    "repeat_whole_pass": pass_rows,
                    # The number of TASKS compared across sweeps. This read
                    # `len(pass_rows)` -- the two decode regimes -- so the artifact
                    # reported n=2 and marked itself unquotable for want of a sample
                    # it had. A validity check on the wrong quantity is worse than
                    # none: it discredits a sound measurement in the same voice.
                    "validity": check_sample_size(
                        len(tasks),
                        label="decode_stability",
                    ).as_dict(),
                    "passes": args.passes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
