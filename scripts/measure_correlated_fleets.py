#!/usr/bin/env python3
"""What finding 12's cliff costs once analysts stop being independent draws.

Finding 12 established that consensus and Dawid-Skene both collapse once a wrong
standard holds the majority of a fleet. It swept the number of wrong analysts as a free
parameter, which answers "what happens at 5 of 9" and leaves open the question a
deployment actually faces: **how often does a fleet end up there?**

Every fleet in this repository so far has been drawn i.i.d. That assumption is doing
more work than it looks. Analysts are not independent draws from a policy grid: they
share a training pipeline, inherit a house style, and are corrected by the same
supervisors, so a wrong standard propagates through a cohort rather than appearing
independently in each person. The i.i.d. draw is the most favourable assumption
available and nothing in the design enforces it.

This measures the gap. For a population error rate, fleets are drawn two ways:

  independent   each analyst holds the wrong standard with probability p
  clustered     the fleet is partitioned into `schools`, and a whole school holds the
                wrong standard with probability p

Both have the same expected error rate. They differ only in how it is distributed, and
the difference decides how often the fleet crosses the majority.

Two quantities, computed two ways, because they have different natures and an earlier
version of this script got that wrong by sampling both.

**P(wrong majority) is exact.** It is a combinatorial property of the draw and needs no
simulation: a binomial over analysts when they are independent, and a binomial over
schools when they are not. A first attempt estimated it from 40 drawn fleets, whose
standard error near 0.1 is about 0.05, and duly produced clustered cells *below*
independent ones at two rates -- an ordering the mathematics forbids. Sampling a
quantity with a closed form is how a measurement invents its own noise.

**Agreement is measured**, on real fleets through the real pipeline, conditional on
whether the fleet crossed the majority. Expected agreement is then the exact
probability composed with the measured conditionals, which is both cheaper and free of
the sampling error that wrecked the first attempt.

Needs no model and no network.

    uv run python scripts/measure_correlated_fleets.py --out results/correlated.json
"""

import argparse
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random

from pharos.analyst import Action, AnalystPolicy, Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import WRONG_THRESHOLD as _WRONG_THRESHOLD
from pharos.governance import exact_wrong_majority
from pharos.inference import agreement_with, dawid_skene
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import record
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 120
FLEET = 9

#: The wrong standard, matching finding 12 so the two are directly comparable. Imported
#: rather than restated: "matching finding 12" is the whole point of the constant, and a
#: second `= 2` is the one way it can stop matching without anybody noticing.
WRONG_THRESHOLD = _WRONG_THRESHOLD

#: Population error rates worth pricing. 0.5 is included because it is where the two
#: structures must agree: at a coin flip, clustering cannot change the mean.
RATES = (0.1, 0.2, 0.3, 0.4, 0.5)

#: Fleet draws per cell. The quantity being estimated is a probability over draws, so
#: this is the sample size that matters, not the task count.
DRAWS = 60


@dataclass(frozen=True, slots=True)
class Cell:
    """One population rate under one correlation structure."""

    rate: float
    structure: str
    schools: int
    draws: int
    wrong_majority_rate: float
    mean_consensus: float
    #: Expected agreement for Dawid-Skene, composed the same way as consensus.
    mean_dawid_skene: float
    worst_consensus: float
    agreement_if_majority: float | None
    agreement_if_not: float | None
    expected_agreement: float

    def as_dict(self) -> dict[str, object]:
        return {
            "rate": self.rate,
            "structure": self.structure,
            "schools": self.schools,
            "draws": self.draws,
            "wrong_majority_rate": round(self.wrong_majority_rate, 4),
            "mean_consensus": round(self.mean_consensus, 4),
            "mean_dawid_skene": round(self.mean_dawid_skene, 4),
            "worst_consensus": round(self.worst_consensus, 4),
            "agreement_if_majority": (
                None if self.agreement_if_majority is None else round(self.agreement_if_majority, 4)
            ),
            "agreement_if_not": (
                None if self.agreement_if_not is None else round(self.agreement_if_not, 4)
            ),
            "expected_agreement": round(self.expected_agreement, 4),
        }


def draw_fleet(
    rate: float, *, schools: int, rng: Random, fleet: int = FLEET
) -> tuple[AnalystPolicy, ...]:
    """A fleet whose wrong standards are distributed over `schools` independent groups.

    `schools == fleet` is the i.i.d. case, one analyst per school. `schools == 1` is
    total correlation, the whole fleet right or wrong together. Sizes are equal, so the
    expected error rate is `rate` in every structure and only its distribution changes.
    """
    per_school = fleet // schools
    policies: list[AnalystPolicy] = []
    for school in range(schools):
        wrong = rng.random() < rate
        for member in range(per_school):
            name = f"s{school}m{member}"
            policies.append(
                AnalystPolicy(name, escalation_threshold=WRONG_THRESHOLD)
                if wrong
                else AnalystPolicy(name)
            )
    return tuple(policies)


def compose(exact: float, major: list[float], minor: list[float]) -> float:
    """Exact majority probability composed with whichever conditionals were observed.

    Every reported agreement column goes through this, so they are comparable. An
    earlier version composed one column and took a raw sample mean for the other, which
    put two differently-computed numbers side by side in a table that invites reading
    them against each other.

    Module-level rather than a closure over the loop: capturing `exact` late is a real
    bug class even where, as here, the call happens in the same iteration.
    """
    hi = statistics.mean(major) if major else None
    lo = statistics.mean(minor) if minor else None
    if hi is None:
        return lo if lo is not None else 0.0
    if lo is None:
        return hi
    return exact * hi + (1 - exact) * lo


def score_fleet(
    policies: tuple[AnalystPolicy, ...], tasks, proposals, truth, *, seed: int
) -> tuple[float, float, bool]:
    """Consensus and Dawid-Skene agreement for one fleet, plus whether it is wrong-majority."""
    grouped: dict[str, list[tuple[str, bool]]] = {}
    for policy in policies:
        for task in tasks:
            decision = policy.review(task, proposals[task.task_id], seed=seed)
            if decision.action is Action.ACCEPT:
                verdict: bool | None = proposals[task.task_id].verdict
            elif decision.action is Action.REVISE:
                verdict = decision.corrected_verdict
            else:
                verdict = None
            if verdict is not None:
                grouped.setdefault(task.task_id, []).append((policy.name, verdict))

    flat = [(tid, who, v) for tid, rows in grouped.items() for who, v in rows]
    if not flat:
        return 0.0, 0.0, False

    consensus = {
        tid: Counter(v for _, v in rows).most_common(1)[0][0] for tid, rows in grouped.items()
    }
    estimate = dawid_skene(flat)
    n_wrong = sum(1 for p in policies if p.escalation_threshold == WRONG_THRESHOLD)
    return (
        agreement_with(consensus, truth),
        agreement_with(estimate.labels(), truth),
        n_wrong * 2 > len(policies),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    structures = (("independent", args.fleet), ("three schools", 3), ("one culture", 1))
    print(f"{len(tasks)} tasks, fleets of {args.fleet}, {args.draws} draws per cell")
    print(f"  {'rate':>5} {'structure':>14} {'P(wrong maj)':>13} {'E[agree]':>10} {'D-S':>8}")
    print("  " + "-" * 56)

    cells: list[Cell] = []
    for rate in RATES:
        for name, schools in structures:
            rng = Random(SEED * 1000 + int(rate * 100) * 10 + schools)
            exact = exact_wrong_majority(rate, schools=schools, fleet=args.fleet)

            # Conditional agreements, measured. Drawn fleets are grouped by whether
            # they crossed the majority, so each conditional is estimated on the
            # fleets it describes rather than smeared across both regimes.
            with_majority: list[float] = []
            without: list[float] = []
            ds_major: list[float] = []
            ds_minor: list[float] = []
            for draw in range(args.draws):
                drawn = draw_fleet(rate, schools=schools, rng=rng, fleet=args.fleet)
                consensus, ds, crossed = score_fleet(
                    drawn, tasks, proposals, truth, seed=SEED + draw
                )
                (with_majority if crossed else without).append(consensus)
                (ds_major if crossed else ds_minor).append(ds)

            if_major = statistics.mean(with_majority) if with_majority else None
            if_not = statistics.mean(without) if without else None
            # Expected agreement composes the EXACT probability with the measured
            # conditionals. Where a regime was never drawn its conditional is unknown,
            # and the other one carries the estimate rather than a zero standing in.
            expected = compose(exact, with_majority, without)
            expected_ds = compose(exact, ds_major, ds_minor)

            observed = with_majority + without
            cell = Cell(
                rate=rate,
                structure=name,
                schools=schools,
                draws=args.draws,
                wrong_majority_rate=exact,
                mean_consensus=statistics.mean(observed) if observed else 0.0,
                mean_dawid_skene=expected_ds,
                worst_consensus=min(observed) if observed else 0.0,
                agreement_if_majority=if_major,
                agreement_if_not=if_not,
                expected_agreement=expected,
            )
            cells.append(cell)
            print(
                f"  {rate:>5} {name:>14} {exact:>13.3f} "
                f"{cell.expected_agreement:>10.3f} {cell.mean_dawid_skene:>8.3f}"
            )

    for cell in cells:
        record(
            "correlated.composition",
            cell.wrong_majority_rate,
            rate=cell.rate,
            structure=cell.structure,
            expected_agreement=round(cell.expected_agreement, 4),
        )

    print("\nhow much the i.i.d. assumption understates the majority risk (exact):")
    for rate in RATES:
        by = {c.structure: c for c in cells if c.rate == rate}
        indep = by["independent"].wrong_majority_rate
        for name in ("three schools", "one culture"):
            got = by[name].wrong_majority_rate
            ratio = (got / indep) if indep > 0 else float("inf")
            shown = "inf" if ratio == float("inf") else f"{ratio:.1f}x"
            print(f"  rate {rate}: {name:<14} {got:.3f} against {indep:.3f} -> {shown}")
            record(
                "correlated.understatement",
                ratio if ratio != float("inf") else -1.0,
                rate=rate,
                structure=name,
                clustered=round(got, 4),
                independent=round(indep, 4),
            )

    report = {
        "provenance": run_provenance(seed=SEED),
        "fleet": args.fleet,
        "events": args.events,
        "draws_per_cell": args.draws,
        "wrong_threshold": WRONG_THRESHOLD,
        "cells": [c.as_dict() for c in cells],
        "validity": check_sample_size(args.draws, label="correlated fleets").as_dict(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
