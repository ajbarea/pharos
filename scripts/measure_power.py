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
    """One headline claim, the size it was measured at, and the gap it rests on.

    `against_constant` distinguishes the two questions this file prices, because they
    do not cost the same. Comparing two measured conditions needs the gap to clear a
    half-width on each side. Comparing one measured rate against a *fixed* reference
    -- a majority floor, a stated ceiling, a prior an adversary reaches by guessing --
    needs it to clear one half-width only, because the reference carries no sampling
    noise of its own. Treating the second as though it were the first would declare
    resolved claims unresolved, which is a failure of the instrument rather than
    conservatism.
    """

    finding: str
    n: int
    effect: float
    description: str
    against_constant: bool = False
    #: Outcome rate to price this claim at, when it is not the corpus class balance.
    #: Interval width depends on the rate, and a finding whose trials are analysts
    #: recovered rather than tasks answered has its own. Left None the corpus balance
    #: is used, which is the right reference for every task-scored finding here.
    rate: float | None = None

    def threshold(self, half: float) -> float:
        """The gap this claim must clear at a given half-width."""
        return half if self.against_constant else 2.0 * half


#: Effects are the *gap the claim depends on*, not the score itself. A claim that a
#: model fails to clear a floor rests on the distance from the score to the floor; a
#: claim that two conditions differ rests on the distance between them.
CLAIMS: tuple[Claim, ...] = (
    # The majority floor and the stated-rule ceiling are computed from a generated
    # corpus and are therefore exact, not estimated: claims against them pay one
    # half-width, not two.
    Claim("3b", 40, 0.000, "qwen2.5-3b (0.625) clears the majority floor (0.625)", True),
    Claim("3b", 40, 0.200, "mistral-7b (0.425) is below the majority floor (0.625)", True),
    # Bought and refuted 2026-08-01. Remeasured at 600 the gap is -0.009, so the
    # claim is not merely unresolved, it is gone. Kept in the table at its new size
    # because a claim that was retired by more data is exactly what this file is for.
    Claim("5", 600, 0.009, "8 shots (0.514) beats 0 shots (0.523) -- REFUTED at n=600"),
    Claim("5", 600, 0.055, "2 shots (0.468) is worse than 0 shots (0.523) -- direction only"),
    Claim("5", 600, 0.486, "8 shots (0.514) is below the stated-rule ceiling (1.000)", True),
    Claim("5", 600, 0.179, "8 shots (0.514) is below the majority floor (0.693)", True),
    Claim("6", 60, 0.531, "adapter (1.000) beats the base model (0.469)"),
    Claim("10", 600, 0.560, "any-one adapter matches teacher (1.000) not world (0.440)"),
    # Bought and CONFIRMED 2026-08-01, unlike finding 5's. Remeasured at 600 the gap
    # widened from 0.083 to 0.118 and now clears the bar. Both scorings are of one
    # decode over one evaluation set, so the pairing makes the real test tighter than
    # this table's independent one.
    Claim("10", 600, 0.118, "inattentive adapter (0.893) beats its own teacher (0.775)"),
    # Finding 11 clusters over analysts rather than tasks, so n is the fleet size.
    Claim(
        "11",
        200,
        0.100,
        "linkage recovery (0.205) beats the guessing prior (0.105)",
        True,
        rate=0.205,
    ),
    Claim(
        "11",
        50,
        0.820,
        "RESTRICTED analysts (0.820) are recovered where OPEN (0.000) are not",
        rate=0.820,
    ),
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

    Two half-widths, which is the strict end of the three rules in play. It exceeds
    the difference interval `uncertainty.resolves_difference` applies, and it is well
    above the point-coverage rule `uncertainty.resolves` applies. A claim this table
    calls resolved is therefore resolved under any of them, and a claim it calls
    unresolved may still separate under the difference test -- so the table under-
    reports rather than over-reports, which is the direction to err in.
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
        claim_rate = claim.rate if claim.rate is not None else rate
        half = half_width(claim.n, claim_rate, resamples=args.resamples, seed=SEED)
        required = claim.threshold(half)
        resolved = claim.effect > required
        # The smallest listed size that would resolve it, when the current one does not.
        needed = None
        if not resolved:
            for n in SIZES:
                candidate = half_width(n, claim_rate, resamples=args.resamples, seed=SEED)
                if claim.effect > claim.threshold(candidate):
                    needed = n
                    break
        verdicts.append(
            {
                "finding": claim.finding,
                "n": claim.n,
                "effect": claim.effect,
                "against_constant": claim.against_constant,
                "rate": round(claim_rate, 4),
                "required_gap": round(required, 4),
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
