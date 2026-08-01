#!/usr/bin/env python3
"""What each evaluation size can and cannot resolve.

Every model-dependent finding here runs at 24 to 60 tasks. Whether that is too
small has no single answer: it depends entirely on the size of the effect the
finding claims, and this repository's findings claim effects that differ by an order
of magnitude. So rather than argue the question, this computes it.

For a grid of evaluation sizes it reports the half-width of a 95% cluster-bootstrap
interval and the **minimum detectable effect**: the smallest difference between two
conditions that their intervals would separate. Then it checks every headline claim
against the size it was actually measured at, and says which are resolved, which are
not, and what size the unresolved ones would need.

The class balance is taken from the corpus rather than assumed, because the interval
width depends on it and this corpus is not balanced.

Needs no model and no network: outcomes are simulated at a stated rate and the
interval machinery is the same one the measurements use.

    uv run python scripts/measure_power.py --out results/power.json
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from pharos.generate import GeneratorConfig, generate
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.uncertainty import Trial, cluster_bootstrap

SEED = 7
EVENTS = 400

#: Evaluation sizes worth pricing. The low end is what findings here actually use;
#: 600 is the figure current evaluation guidance converges on for a stable estimate.
SIZES = (24, 30, 40, 60, 120, 240, 400, 600, 1000, 2000)

#: Bootstrap resamples per size. Lower than the measurement default because this is
#: a width estimate rather than a published interval.
RESAMPLES = 600


@dataclass(frozen=True, slots=True)
class Claim:
    """One headline claim, the size it was measured at, and the gap it rests on."""

    finding: str
    n: int
    effect: float
    description: str


#: Effects are the *gap the claim depends on*, not the score itself. A claim that a
#: model fails to clear a floor rests on the distance from the score to the floor; a
#: claim that two conditions differ rests on the distance between them.
CLAIMS: tuple[Claim, ...] = (
    Claim("3b", 40, 0.000, "qwen2.5-3b (0.625) clears the majority floor (0.625)"),
    Claim("3b", 40, 0.200, "mistral-7b (0.425) is below the majority floor (0.625)"),
    Claim("5", 30, 0.078, "8 shots (0.571) beats 0 shots (0.493)"),
    Claim("5", 30, 0.507, "8 shots (0.493) is below the stated-rule ceiling (1.000)"),
    Claim("6", 60, 0.531, "adapter (1.000) beats the base model (0.469)"),
    Claim("10", 60, 0.367, "any-one adapter matches teacher (1.000) not world (0.633)"),
    Claim("10", 60, 0.083, "inattentive adapter (0.883) beats its own teacher (0.800)"),
)


def class_balance(tasks) -> float:
    """Share of significant tasks. Interval width depends on it, so it is measured."""
    return sum(1 for t in tasks if t.significant) / len(tasks) if tasks else 0.5


def half_width(n: int, rate: float, *, resamples: int, seed: int) -> float:
    """Half-width of a 95% cluster-bootstrap interval for `n` tasks at `rate`.

    Simulated rather than taken from a formula so it exercises the same code the
    measurements use, including the clustering. With one run per task the clustering
    is a no-op, which is the honest case here: none of these findings repeat a task.
    """
    rng = random.Random(seed)
    trials = [Trial(f"t{i}", rng.random() < rate) for i in range(n)]
    interval = cluster_bootstrap(trials, resamples=resamples, seed=seed)
    return (interval.high - interval.low) / 2.0


def minimum_detectable(n: int, rate: float, *, resamples: int, seed: int) -> float:
    """Smallest gap two conditions of size `n` could separate.

    Two intervals separate when neither covers the other's point, so the gap must
    exceed roughly one half-width on each side. Conservative and matched to how
    `uncertainty.resolves` actually decides.
    """
    return 2.0 * half_width(n, rate, resamples=resamples, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=EVENTS)))
    rate = class_balance(tasks)
    print(f"corpus class balance {rate:.3f} significant, measured at {len(tasks)} tasks")
    print("worst-case width is near 0.5, so a balanced split is priced too\n")

    rows = []
    print(f"  {'n':>6} {'half-width':>12} {'min detectable':>16}")
    print("  " + "-" * 36)
    for n in SIZES:
        mde = minimum_detectable(n, rate, resamples=args.resamples, seed=SEED)
        hw = mde / 2.0
        rows.append({"n": n, "half_width": round(hw, 4), "min_detectable_effect": round(mde, 4)})
        print(f"  {n:>6} {hw:>12.3f} {mde:>16.3f}")

    print("\n" + "=" * 74)
    print("Every headline claim against the size it was measured at:")
    print("=" * 74)
    verdicts = []
    for claim in CLAIMS:
        mde = minimum_detectable(claim.n, rate, resamples=args.resamples, seed=SEED)
        resolved = claim.effect > mde
        # The smallest listed size that would resolve it, when the current one does not.
        needed = None
        if not resolved:
            for n in SIZES:
                if claim.effect > minimum_detectable(n, rate, resamples=args.resamples, seed=SEED):
                    needed = n
                    break
        verdicts.append(
            {
                "finding": claim.finding,
                "n": claim.n,
                "effect": claim.effect,
                "min_detectable": round(mde, 4),
                "resolved": resolved,
                "n_needed": needed,
                "description": claim.description,
            }
        )
        mark = "RESOLVED  " if resolved else "unresolved"
        tail = "" if resolved else (f"  needs n>={needed}" if needed else "  needs n>2000")
        print(f"  [{mark}] finding {claim.finding:3} n={claim.n:>3} gap={claim.effect:.3f}{tail}")
        print(f"              {claim.description}")

    unresolved = [v for v in verdicts if not v["resolved"]]
    print("\n" + "=" * 74)
    print(
        f"{len(verdicts) - len(unresolved)}/{len(verdicts)} headline claims are resolved at the size they were run."
    )
    if unresolved:
        print("Unresolved, and the size each would need:")
        for v in unresolved:
            need = v["n_needed"] or ">2000"
            print(f"  finding {v['finding']}: {v['description']}  ->  n={need}")
    print("=" * 74)

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(seed=SEED),
                    "class_balance": round(rate, 4),
                    "resamples": args.resamples,
                    "sizes": rows,
                    "claims": verdicts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
