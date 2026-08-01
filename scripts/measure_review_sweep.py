#!/usr/bin/env python3
"""How wrong a reviewer can be before their decisions stop teaching the rule.

Finding 7 reported target accuracy for eight named reviewers. Each was one point
in a space, and a point cannot say where the edge is. This sweeps the two axes
that move the number and reports the region in which a review stream still
supports the world's rule.

The two axes are different in kind, and separating them is the point:

- **Standard** (`escalation_threshold`) is how many of the three defining facts the
  reviewer requires. Getting it wrong is a *systematic* error: the reviewer is
  consistently, reproducibly applying a different rule.
- **Carefulness** (`slip_rate`) is how often the reviewer fails to apply their own
  standard. Getting it wrong is *random* error, symmetric and unpatterned.

The learning-from-noisy-labels literature holds that systematic annotator bias
damages a model more than random noise of the same magnitude, because the errors
are class-dependent and the model learns the pattern rather than averaging it out.
That is a qualitative claim. This measures the exchange rate.

Needs no model and no network: reviewers are decision procedures and the corpus is
regenerated from its seed.

    uv run python scripts/measure_review_sweep.py --out results/review_sweep.json
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from pharos.analyst import AnalystPolicy, Proposal, supervision_yield
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.validity import check_sample_size
from pharos.world import SIGNIFICANT_PATTERN

#: The corpus finding 7 was measured on, so the two are directly comparable.
SEED = 7
EVENTS = 400
TASKS = 40

#: Review seeds. The slip draw is keyed by `(seed, name, task_id)`, so varying the
#: review seed is the only way to resample which tasks a reviewer slips on. Five is
#: enough to show a spread without implying more precision than 40 tasks support.
REVIEW_SEEDS = (1, 7, 23, 101, 202)

#: Every standard a reviewer could hold over a three-fact conjunction.
THRESHOLDS = tuple(range(1, len(SIGNIFICANT_PATTERN) + 1))

#: Carefulness, from perfect to badly inattentive.
SLIP_RATES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


@dataclass(frozen=True, slots=True)
class Cell:
    """One (standard, carefulness) pair, over the review seeds."""

    threshold: int
    slip_rate: float
    accuracies: tuple[float, ...]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.accuracies)

    @property
    def sd(self) -> float:
        return statistics.stdev(self.accuracies) if len(self.accuracies) > 1 else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "slip_rate": self.slip_rate,
            "target_accuracy_mean": round(self.mean, 4),
            "target_accuracy_sd": round(self.sd, 4),
            "per_seed": [round(a, 4) for a in self.accuracies],
        }


def majority_floor(tasks: list[TriageTask]) -> float:
    """The accuracy of always answering the larger class.

    The bar a target stream has to clear to be worth training on at all: a learner
    handed targets below this would do better ignoring them and guessing.
    """
    if not tasks:
        # Not 1.0. `max(0, 1-0)` is the arithmetic answer and a nonsense floor: it
        # says guessing scores perfectly, which would mark every real measurement
        # as failing to clear the bar. An empty set sets no bar.
        return 0.0
    share = sum(1 for t in tasks if t.significant) / len(tasks)
    return max(share, 1.0 - share)


def sweep(tasks: list[TriageTask], proposals: dict[str, Proposal]) -> list[Cell]:
    """Target accuracy at every (threshold, slip) cell, over the review seeds."""
    truth = {task.task_id: task.significant for task in tasks}
    cells: list[Cell] = []
    for threshold in THRESHOLDS:
        for slip in SLIP_RATES:
            accuracies = []
            for review_seed in REVIEW_SEEDS:
                policy = AnalystPolicy(
                    f"t{threshold}-s{slip}", escalation_threshold=threshold, slip_rate=slip
                )
                decisions = [
                    policy.review(t, proposals[t.task_id], seed=review_seed)
                    for t in tasks
                    if t.task_id in proposals
                ]
                accuracies.append(supervision_yield(decisions, truth).target_accuracy)
            cells.append(Cell(threshold, slip, tuple(accuracies)))
    return cells


def equivalent_slip(cells: list[Cell], threshold: int, floor: float) -> float | None:
    """The slip rate at which a correct standard falls to `threshold`'s accuracy.

    The exchange rate between the two kinds of error. Reported as the largest
    measured slip whose accuracy still exceeds the wrong standard's, so it is a
    lower bound on the grid rather than an interpolation between cells.
    """
    target = next((c.mean for c in cells if c.threshold == threshold and c.slip_rate == 0.0), None)
    if target is None:
        return None
    correct = sorted(
        (c for c in cells if c.threshold == max(THRESHOLDS)), key=lambda c: c.slip_rate
    )
    worse = [c.slip_rate for c in correct if c.mean <= target]
    return min(worse) if worse else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    reports = generate(GeneratorConfig(seed=SEED, n_events=EVENTS))
    tasks = build_triage_tasks(reports, limit=TASKS)
    floor = majority_floor(tasks)

    # The proposal a reviewer is handed. Target accuracy does not depend on it --
    # a corrected verdict is the reviewer's own call and an accepted one is a call
    # they agreed with -- and finding 7 verified that empirically across six models.
    # One fixed proposal set therefore suffices, and using the fail-closed default
    # keeps the sweep on the same footing as that measurement.
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }

    cells = sweep(tasks, proposals)
    validity = check_sample_size(len(tasks), label="review_sweep")

    print(f"{len(tasks)} tasks at seed {SEED}, {len(REVIEW_SEEDS)} review seeds per cell")
    print(f"majority floor {floor:.3f} -- a target stream below this is worse than guessing\n")
    header = "  slip ".ljust(9) + "".join(f"{s:>10.2f}" for s in SLIP_RATES)
    print(header)
    for threshold in THRESHOLDS:
        row = [c for c in cells if c.threshold == threshold]
        label = f"  need {threshold}/3"
        cells_text = "".join(
            f"{c.mean:>9.3f}{'*' if c.mean >= floor else ' '}"
            for c in sorted(row, key=lambda c: c.slip_rate)
        )
        print(f"{label:9}{cells_text}")
    print("\n  * clears the majority floor")

    exchange = {}
    for threshold in THRESHOLDS:
        if threshold == max(THRESHOLDS):
            continue
        slip = equivalent_slip(cells, threshold, floor)
        exchange[str(threshold)] = slip
        wrong = next(c.mean for c in cells if c.threshold == threshold and c.slip_rate == 0.0)
        if slip is None:
            print(
                f"\n  A reviewer needing {threshold}/3 scores {wrong:.3f}. No measured slip rate "
                f"up to {max(SLIP_RATES):.0%} drives a correct standard that low."
            )
        else:
            print(
                f"\n  A reviewer needing {threshold}/3 scores {wrong:.3f}, which a correct "
                f"standard reaches only at a slip rate of {slip:.0%}."
            )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(seed=SEED),
                    "seed": SEED,
                    "n_tasks": len(tasks),
                    "review_seeds": list(REVIEW_SEEDS),
                    "majority_floor": round(floor, 4),
                    "validity": validity.as_dict(),
                    "cells": [c.as_dict() for c in cells],
                    "equivalent_slip": exchange,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
