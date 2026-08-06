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
from typing import Any

from pharos.generate import GeneratorConfig, generate
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.uncertainty import Trial, cluster_bootstrap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _fail(message: str) -> None:
    """Refuse rather than price a claim against a number nobody measured."""
    raise SystemExit(f"measure_power: {message}")


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


def _from(name: str) -> dict[str, Any]:
    """One measurement artifact, or a hard error naming what is missing.

    These claims used to be written out by hand, effect sizes and quoted scores
    together, in a table that feeds a *generated* block in `docs/findings.md`. That
    made them prose wearing a data structure's clothes: when the corpus was
    re-measured on 2026-08-04 every number here went stale silently, and the page
    published `qwen2.5-3b (0.625) clears the majority floor (0.625)` -- a claim of a
    dead heat -- while the artifact said 0.775 against a floor of 0.65. The generated
    block was faithfully rendering a hand-typed lie.

    So the numbers are read, and the *shape* of each claim is all that stays written.
    """
    path = RESULTS / name
    if not path.exists():
        _fail(f"power claims read {name}, which is missing; run its measurement first")
    return json.loads(path.read_text(encoding="utf-8"))


def _floor_claim(finding: str, n: int, artifact: str, label: str) -> Claim:
    """A score against the corpus's own majority floor, which is exact rather than
    estimated -- so the claim pays one half-width and not two."""
    payload = _from(artifact)
    score, floor = payload["accuracy"], payload["majority_accuracy"]
    verb = "clears" if score >= floor else "is below"
    return Claim(
        finding,
        n,
        abs(score - floor),
        f"{label} ({score:.3f}) {verb} the majority floor ({floor:.3f})",
        True,
    )


def _shot_gap(rows: dict[int, float], high: int, low: int) -> tuple[float, str]:
    """The gap between two few-shot conditions, and how it currently reads."""
    gap = rows[high] - rows[low]
    verdict = "beats" if gap > 0 else "is worse than"
    return abs(gap), f"{high} shots ({rows[high]:.3f}) {verdict} {low} shots ({rows[low]:.3f})"


def _claims() -> tuple[Claim, ...]:
    """Every claim this table prices, with its numbers read from the artifacts.

    Effects are the *gap the claim depends on*, not the score itself. A claim that a
    model fails to clear a floor rests on the distance from the score to the floor; a
    claim that two conditions differ rests on the distance between them.

    Kept a function rather than a module constant because it reads files, and a
    module-level read makes importing this script for a single helper do disk IO.
    """
    learn = _from("learnability.json")
    shots = {row["shots"]: row["accuracy"] for row in learn["rows"]}
    floor = max(row["majority"] for row in learn["rows"])
    adapter = _from("adapter_learnability.json")
    n_learn = learn["n_eval"] * len(learn["rows"])

    eight_v_zero, eight_v_zero_text = _shot_gap(shots, 8, 0)
    two_v_zero, two_v_zero_text = _shot_gap(shots, 2, 0)

    return (
        _floor_claim("3b", 40, "triage_lift-qwen2.5-3b.json", "qwen2.5-3b"),
        _floor_claim("3b", 40, "triage_lift-mistral-7b.json", "mistral-7b"),
        # Bought and refuted 2026-08-01, and it stays in the table at its new size:
        # a claim retired by more data is exactly what this file exists to record.
        Claim("5", n_learn, eight_v_zero, f"{eight_v_zero_text} -- REFUTED at n={n_learn}"),
        Claim("5", n_learn, two_v_zero, f"{two_v_zero_text} -- direction only"),
        Claim(
            "5",
            n_learn,
            abs(1.0 - shots[8]),
            f"8 shots ({shots[8]:.3f}) is below the stated-rule ceiling (1.000)",
            True,
        ),
        Claim(
            "5",
            n_learn,
            abs(floor - shots[8]),
            f"8 shots ({shots[8]:.3f}) is below the majority floor ({floor:.3f})",
            True,
        ),
        Claim(
            "6",
            adapter["adapter"]["n"],
            abs(adapter["adapter"]["f1"] - adapter["base"]["f1"]),
            f"adapter ({adapter['adapter']['f1']:.3f}) beats the base model "
            f"({adapter['base']['f1']:.3f})",
        ),
        *_teacher_claims(),
        *_linkage_claims(),
    )


def _teacher_claims() -> tuple[Claim, ...]:
    """Finding 10: an adapter inherits its teacher's standard rather than the world's."""
    payload = _from("review_adapter-t1s0.json")
    world = payload["adapter"]["accuracy"]
    teacher = payload["adapter_vs_teacher"]["accuracy"]
    inattentive = _from("review_adapter-t3s0.15.json")
    tuned = inattentive["adapter"]["accuracy"]
    its_teacher = inattentive["teacher"]["train_target_agreement"]
    n = payload["adapter"]["n"]
    return (
        Claim(
            "10",
            n,
            abs(teacher - world),
            f"any-one adapter matches teacher ({teacher:.3f}) not world ({world:.3f})",
        ),
        # Bought and CONFIRMED 2026-08-01, unlike finding 5's. Both scorings are of one
        # decode over one evaluation set, so the pairing makes the real test tighter
        # than this table's independent one.
        Claim(
            "10",
            n,
            abs(tuned - its_teacher),
            f"inattentive adapter ({tuned:.3f}) beats its own teacher ({its_teacher:.3f})",
        ),
    )


def _linkage_claims() -> tuple[Claim, ...]:
    """Finding 11 clusters over analysts rather than tasks, so n is the fleet size."""
    payload = _from("fleet_linkage.json")
    recovery = payload["drop_compartments"]["recovery"]["point"]
    prior = payload["prior"]
    by_level = {row["level"]: row for row in payload["by_clearance_level"]}
    restricted, openv = by_level.get("RESTRICTED"), by_level.get("OPEN")
    claims = [
        Claim(
            "11",
            payload["events"],
            abs(recovery - prior),
            f"linkage recovery ({recovery:.3f}) beats the guessing prior ({prior:.3f})",
            True,
            rate=recovery,
        )
    ]
    if restricted is not None and openv is not None:
        hit = restricted["recovery"]["point"]
        miss = openv["recovery"]["point"]
        claims.append(
            Claim(
                "11",
                restricted["n"],
                abs(hit - miss),
                f"RESTRICTED analysts ({hit:.3f}) are recovered where OPEN ({miss:.3f}) are not",
                rate=hit,
            )
        )
    return tuple(claims)


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
    for claim in _claims():
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
    # The headline: how many published claims their own samples support. A silent
    # change here moves what this project is allowed to say, and nothing else in a
    # run's output would show it.
    record(
        "power.claims_resolved",
        (len(verdicts) - len(unresolved)) / len(verdicts) if verdicts else 0.0,
        resolved=len(verdicts) - len(unresolved),
        total=len(verdicts),
        class_balance=round(rate, 4),
    )
    for v in unresolved:
        get_logger().warning(
            "power.claim_unresolved",
            extra={
                "event": "power.claim_unresolved",
                "finding": v["finding"],
                "n": v["n"],
                "effect": v["effect"],
                "n_needed": v["n_needed"],
            },
        )
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
