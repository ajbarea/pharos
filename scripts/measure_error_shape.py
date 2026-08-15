#!/usr/bin/env python3
"""Whether a fleet can tell how much of its error is shared, from the aggregate alone.

Finding 28 leaves a fork a deployment cannot resolve from inside: withholding by the named
channel is right where the error is shared, withholding by confidence is right where it is
independent, and each is wrong in the other regime. That measurement decided which regime
it was in by checking a *healthy control fleet*, which an experimenter has and a deployment
does not.

The statistic is an index of dispersion over per-task vote sums, conditioned on the
evidence stratum. Within a stratum a fleet applying one rule votes identically, so
independent slips are exactly binomial; a shared standard is not, because it splits the
stratum into two deterministic groups and a mixture of two rates carries more variance
than a binomial at their mean. It reads vote sums, contributor counts and public evidence
counts -- strictly less than finding 22's detector, which also needs a channel to name.

The null is parametric rather than a permutation: dispersion has no second label to
shuffle, so the null is "these counts are binomial at this stratum's own rate" and it is
simulated. That assumes a shape a permutation would not, which is a real weakening.

**Predictions, before the run.** (1) On a fleet with no shared blind spot the index sits at
1 wherever there is any dispersion at all. (2) It rises with the share carrying the blind
spot and falls as independent noise rises. (3) Thresholded against its own null it picks
the rule finding 28 says wins, and its failures land in cells where neither rule dominates.
(4) A fleet that never disagrees is reported *undiagnosable* rather than clean, because
there is no variance to compare against.

If (1) and (2) hold and (3) fails, the shape is visible and not actionable, and finding
28's advice stays conditional on something a deployment cannot check. Say that rather than
softening it.

Needs no model and no network.

    uv run python scripts/measure_error_shape.py --out results/error_shape.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from pharos.analyst import Proposal, evidence_shown
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import (
    ALPHA,
    BLIND,
    MIN_STRATUM,
    NULL_DRAWS,
    REFUSED_EXIT,
    REPORT_BUDGET,
    UNIFORM_SEEDS,
    ChannelUnusableError,
    ServerObservation,
    assert_channel_usable,
    beats_every_draw,
    dispersion,
    fleet_view,
    score,
    select,
)
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, progress, record
from pharos.validity import check_sample_size

LOG = get_logger()

SEED = 7
EVENTS = 200
FLEET = 9

#: The grid. Shares of the fleet carrying the shared blind spot, crossed with the rate at
#: which every analyst slips independently. The point of the cross is that neither
#: dimension alone poses the question: finding 28 measured one column of this and had to
#: name the regime from outside.
SHARES = (0, 1, 3, 5, 7, 9)
SLIP_RATES = (0.0, 0.05, 0.15, 0.25, 0.40)

#: What this sweep varies and what it pins.
MULTIVERSE = {
    "swept": {"blind_share": list(SHARES), "slip_rate": list(SLIP_RATES)},
    "pinned": {
        "corpus_seed": {
            "value": SEED,
            "why": "the corpus dimension belongs to measure_corpus_sensitivity.py",
        },
        "fleet": {
            "value": FLEET,
            "why": "the size every published governance number was measured at",
        },
        "blind_compartment": {
            "value": BLIND.value,
            "why": "the one channel on this corpus whose evidence is not entangled with "
            "item difficulty",
        },
    },
}


@dataclass(frozen=True, slots=True)
class Outcome:
    """Which rule wins here, and what each one leaves on the table.

    The risks travel with the verdict because the verdict alone cannot price a wrong
    call. A predictor that names the losing rule in a cell where both are within a
    thousandth of each other has made a different mistake from one that names it where
    the gap is a tenth, and only the second is worth a deployment's attention.
    """

    winner: str | None
    risk: dict[str, float | None]


def winning_rule(
    view: ServerObservation,
    labels: dict[str, bool],
    truth: dict[str, bool],
) -> Outcome:
    """Which of finding 28's rules actually lowers risk here, scored with ground truth.

    Not deployable and not proposed: this is the answer the dispersion index is trying to
    predict without it. `None` where the estimator makes no errors -- a fleet with nothing
    wrong prices no rule, and scoring one would mark both down for failing to remove
    errors that do not exist.
    """
    pool = tuple(sorted(labels))
    wrong = frozenset(task for task, value in labels.items() if value != truth[task])
    if not wrong:
        return Outcome(winner=None, risk={})

    def risk_of(policy: str) -> float | None:
        held = select(policy, view, truth, REPORT_BUDGET, seed=UNIFORM_SEEDS[0])
        return score(
            n_blind=0, slip_rate=0.0, policy=policy, withheld=held, pool=pool, wrong=wrong
        ).risk

    uniform_draws = []
    for draw_seed in UNIFORM_SEEDS:
        held = select("uniform", view, truth, REPORT_BUDGET, seed=draw_seed)
        uniform_draws.append(
            score(
                n_blind=0,
                slip_rate=0.0,
                policy="uniform",
                withheld=held,
                pool=pool,
                wrong=wrong,
            ).risk
        )

    channel_risk = risk_of("channel")
    confidence_risk = min(
        (r for r in (risk_of("margin"), risk_of("posterior")) if r is not None),
        default=None,
    )
    channel_wins = beats_every_draw(channel_risk, uniform_draws)
    confidence_wins = beats_every_draw(confidence_risk, uniform_draws)
    risk = {
        "channel": channel_risk,
        "confidence": confidence_risk,
        "uniform_best": min(uniform_draws) if uniform_draws else None,
    }
    if channel_wins and not confidence_wins:
        return Outcome(winner="channel", risk=risk)
    if confidence_wins and not channel_wins:
        return Outcome(winner="confidence", risk=risk)
    if channel_wins and confidence_wins:
        return Outcome(winner="either", risk=risk)
    return Outcome(winner="neither", risk=risk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--draws", type=int, default=NULL_DRAWS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=args.seed, n_events=args.events)))
    # The same precondition finding 21 refuses on, applied here for the same reason: a
    # corpus whose blinded channel is entangled with item difficulty cannot host any of
    # this, and would still produce a well-formed artifact.
    try:
        check = assert_channel_usable(tasks)
    except ChannelUnusableError as refusal:
        print(refusal, file=sys.stderr)
        raise SystemExit(REFUSED_EXIT) from refusal
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}
    evidence = {t.task_id: len(evidence_shown(t)) for t in tasks}

    floor = 1.0 / (args.draws + 1)
    if floor >= ALPHA:
        # A test whose smallest attainable p-value is at or above its own threshold is a
        # test that reports "not significant" for every input, including the ones it was
        # built to catch. Refusing is the only honest response; the alternative is a grid
        # of confident negatives.
        LOG.error(
            "error_shape.null_too_coarse",
            extra={
                "event": "error_shape.null_too_coarse",
                "draws": args.draws,
                "floor": floor,
                "alpha": ALPHA,
            },
        )
        print(
            f"{args.draws} null draws floor the p-value at {floor:.5f}, at or above alpha "
            f"{ALPHA}; every cell would read as not overdispersed",
            file=sys.stderr,
        )
        raise SystemExit(REFUSED_EXIT)

    shares = tuple(n for n in SHARES if n <= args.fleet)
    cells: list[dict[str, Any]] = []
    for slip_rate in SLIP_RATES:
        for n_blind in shares:
            progress("error_shape.cell", n_blind=n_blind, slip_rate=slip_rate)
            view, labels, converged = fleet_view(
                tasks,
                proposals,
                n_blind=n_blind,
                fleet=args.fleet,
                slip_rate=slip_rate,
                seed=args.seed,
            )
            spread = dispersion(view, evidence, draws=args.draws, seed=args.seed)
            errors = sum(1 for task, value in labels.items() if value != truth[task])
            outcome = winning_rule(view, labels, truth)
            actual = outcome.winner
            # The deployable call: overdispersed against its own null means part of the
            # error is shared, which is the regime the provenance rule is for.
            predicted = (
                None
                if spread.p_value is None
                else ("channel" if spread.p_value <= ALPHA else "confidence")
            )
            cells.append(
                {
                    "n_blind": n_blind,
                    "slip_rate": slip_rate,
                    "dispersion": spread.as_dict(),
                    "base_errors": errors,
                    "converged": converged,
                    "actual_winner": actual,
                    "predicted_rule": predicted,
                    "risk": outcome.risk,
                    #: What following the prediction costs here, in published error rate,
                    #: against following the rule that actually wins. Zero where the
                    #: prediction is right, and the number a deployment would feel where
                    #: it is not. A rate of correct calls says nothing about this.
                    "cost_of_following_the_prediction": _cost(outcome, actual, predicted),
                    #: Scored only where a rule was actually decided. "either" and
                    #: "neither" are not wrong answers to predict against -- they are
                    #: cells where the question has no single right answer, and counting
                    #: them either way would manufacture an accuracy.
                    "decidable": actual in ("channel", "confidence"),
                    "correct": (
                        actual in ("channel", "confidence")
                        and predicted is not None
                        and predicted == actual
                    ),
                }
            )
            record(
                "error_shape.dispersion",
                spread.index if spread.index is not None else -1.0,
                n_blind=n_blind,
                slip_rate=slip_rate,
            )

    print(f"{len(tasks)} tasks, fleet of {args.fleet}, blind spot on {BLIND.value}")
    print(f"{check.affected} verdicts change under a fleet-wide blind spot\n")
    print("  dispersion index by share and slip ('--' = no variance to compare against)")
    header = "".join(f"{rate:>9}" for rate in SLIP_RATES)
    print(f"    {'blind':>6}{header}")
    print("    " + "-" * (6 + 9 * len(SLIP_RATES)))
    for n_blind in shares:
        row = ""
        for rate in SLIP_RATES:
            cell = next(c for c in cells if c["n_blind"] == n_blind and c["slip_rate"] == rate)
            value = cell["dispersion"]["index"]
            row += f"{'--':>9}" if value is None else f"{value:>9.2f}"
        print(f"    {n_blind:>6}{row}")

    print("\n  which rule wins / which the index predicts")
    print(f"    {'blind':>6}{header}")
    print("    " + "-" * (6 + 9 * len(SLIP_RATES)))
    for n_blind in shares:
        row = ""
        for rate in SLIP_RATES:
            cell = next(c for c in cells if c["n_blind"] == n_blind and c["slip_rate"] == rate)
            actual = {"channel": "C", "confidence": "F", "either": "b", "neither": "-"}.get(
                cell["actual_winner"] or "", "?"
            )
            guess = {"channel": "C", "confidence": "F"}.get(cell["predicted_rule"] or "", "?")
            row += f"{actual + '/' + guess:>9}"
        print(f"    {n_blind:>6}{row}")
    print("    C = provenance, F = confidence, b = both work, - = neither, ? = undiagnosable")

    decidable = [c for c in cells if c["decidable"]]
    correct = [c for c in decidable if c["correct"]]
    healthy = [c for c in cells if c["n_blind"] == 0 and c["dispersion"]["index"] is not None]
    findings = {
        #: Prediction 1. The index sits at 1 where nothing is shared.
        "index_is_calibrated_on_a_healthy_fleet": bool(healthy)
        and all(c["dispersion"]["p_value"] > ALPHA for c in healthy),
        #: Prediction 2, on the noiseless column where the shared component is unmixed.
        "index_rises_with_the_shared_share": _monotone(
            [
                c["dispersion"]["index"]
                for c in sorted(
                    (c for c in cells if c["slip_rate"] == min(SLIP_RATES)),
                    key=lambda c: c["n_blind"],
                )
                if c["dispersion"]["index"] is not None
            ]
        ),
        #: Prediction 3, and the one the deployment depends on.
        "the_index_picks_the_rule_that_wins": bool(decidable) and len(correct) == len(decidable),
        #: Prediction 4. A fleet with no disagreement anywhere cannot be diagnosed, and
        #: says so instead of reporting a low index.
        "a_silent_fleet_reports_undiagnosable": any(
            c["dispersion"]["index"] is None for c in cells
        ),
    }

    print("\n  predictions, as measured")
    for name, value in findings.items():
        print(f"    {name:<48} {value}")
    print(
        f"\n  decidable cells: {len(decidable)} of {len(cells)}; predicted correctly: {len(correct)}"
    )
    missed = [c for c in decidable if not c["correct"]]
    if missed:
        print("\n  where it is wrong, and what following it costs")
        print(f"    {'blind':>6}{'slip':>7}{'wins':>12}{'says':>12}{'cost':>8}")
        for cell in missed:
            cost = cell["cost_of_following_the_prediction"]
            print(
                f"    {cell['n_blind']:>6}{cell['slip_rate']:>7}"
                f"{cell['actual_winner']:>12}{cell['predicted_rule']:>12}"
                + (f"{cost:>8.3f}" if cost is not None else f"{'--':>8}")
            )

    if decidable and len(correct) != len(decidable):
        LOG.warning(
            "error_shape.rule_choice_missed",
            extra={
                "event": "error_shape.rule_choice_missed",
                "missed": [
                    f"{c['n_blind']}@{c['slip_rate']}" for c in decidable if not c["correct"]
                ],
            },
        )

    smallest = min((c["dispersion"]["tasks"] for c in cells if c["dispersion"]["tasks"]), default=0)
    report = {
        "provenance": run_provenance(seed=args.seed),
        "fleet": args.fleet,
        "events": args.events,
        "blind_compartment": BLIND.value,
        "shares": list(shares),
        "slip_rates": list(SLIP_RATES),
        "null_draws": args.draws,
        "alpha": ALPHA,
        "min_stratum": MIN_STRATUM,
        "multiverse": MULTIVERSE,
        "cells": cells,
        "findings": findings,
        "decidable_cells": len(decidable),
        "correct_predictions": len(correct),
        #: The worst a wrong call costs anywhere on the grid, in published error rate.
        #: Published as a number rather than left to a reader to compute: a rate of
        #: correct calls says nothing about what the wrong ones are worth, and this
        #: predictor's errors are confident rather than abstaining.
        "worst_cost_of_a_wrong_call": max(
            (
                c["cost_of_following_the_prediction"]
                for c in decidable
                if not c["correct"] and c["cost_of_following_the_prediction"] is not None
            ),
            default=None,
        ),
        "validity": check_sample_size(smallest, label="error shape").as_dict(),
        "tasks_min": smallest,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _cost(outcome: Outcome, actual: str | None, predicted: str | None) -> float | None:
    """Published error rate under the predicted rule, less the rate under the winner.

    None wherever the comparison is not defined -- an undecidable cell, an undiagnosable
    fleet, or a rule whose risk was never computed -- rather than a zero, which would read
    as "the prediction cost nothing" in exactly the cells where it was never tested.
    """
    if actual not in ("channel", "confidence") or predicted is None:
        return None
    theirs, best = outcome.risk.get(predicted), outcome.risk.get(actual)
    if theirs is None or best is None:
        return None
    return round(theirs - best, 4)


def _monotone(values: list[float]) -> bool:
    """Non-decreasing, and not flat. A constant sequence is not a rise."""
    if len(values) < 2:
        return False
    return all(b >= a for a, b in pairwise(values)) and values[-1] > values[0]


if __name__ == "__main__":
    sys.exit(main())
