#!/usr/bin/env python3
"""Whether a fleet can tell how much of its error is shared, from the aggregate alone.

Finding 28 ends on a fork it cannot resolve. Withholding by the channel the detector
named is the right move where the shared blind spot is the whole of the estimator's
error; withholding by confidence is the right move where the error is independent noise;
each is wrong in the other regime. The measurement establishes which regime it is in by
*checking a healthy fleet's error count* --- which is an experimenter's move, not a
deployment's. A deployment has one fleet and no control.

So the question this script asks is the one left over: **is the shape of the error
estimable from what the aggregator already sees?**

**The statistic, and why it is the right shape.** Condition on the evidence stratum, as
finding 22's detector does. Within a stratum every task shows the same number of defining
facts, so a fleet applying one rule votes the same way on all of them, and the per-task
vote sum varies only as its analysts slip. Independent slips are exactly binomial: the
variance of the vote sum is $np(1-p)$. A *shared* error is not --- it splits the stratum
into two deterministic groups, the tasks the shared standard corrupts and the tasks it
does not, and a mixture of two rates has more variance than a binomial at their mean.

So the index of dispersion --- observed variance over binomial variance, pooled across
strata --- is near 1 when the error is independent and above 1 when part of it is shared.
The instrument is standard (overdispersion diagnostics are routine wherever counts are
modelled); the application is the part we did not find prior work for, and that is stated
as a search rather than as a claim about the literature.

**What it needs, and what it does not.** Per-task vote sums, per-task contributor counts,
and the public evidence count of each task. That is strictly *less* than finding 22's
detector, which also needs a channel to name. No per-analyst stream, no ground truth, no
healthy reference fleet. If this works, a deployment can pick between finding 28's rules
with what it already has.

**The null is parametric, not a permutation, and that is a deliberate difference.**
Finding 22 shuffles channel labels within a stratum because it tests an association
between two things it can both see. Dispersion has no second label to shuffle: the null
is "these counts are binomial at this stratum's own rate", so it is simulated. Reported
as such, because a parametric null assumes a shape where a permutation null does not, and
that is a real weakening.

**The predictions, stated before the run.**

1. On a fleet with no shared blind spot, the dispersion index sits at 1 at every slip
   rate that produces any dispersion at all.
2. It rises with the share of the fleet carrying the blind spot, at a fixed slip rate,
   and falls as the slip rate rises at a fixed share --- noise fills in the gap between
   the two groups the shared error creates.
3. Thresholding it against its own null picks the rule finding 28 says wins, in most
   cells of the grid, and the cells it gets wrong are the mixed ones where neither rule
   dominates. That last clause is a prediction about *where* it fails, and it is the
   half most likely to be refuted.
4. On a noiseless healthy fleet the statistic is undefined rather than low: with every
   analyst deterministic and identical there is no variance anywhere to compare against.
   That is reported as *not diagnosable*, which is a different answer from "no shared
   component", and conflating the two would be the same defect as reading a silent guard
   as a passing one.

If 1 and 2 hold and 3 fails, the honest conclusion is that the shape is *visible* and not
*actionable*, and finding 28's advice stays conditional on something a deployment cannot
check. Say so rather than softening it.

Needs no model and no network.

    uv run python scripts/measure_error_shape.py --out results/error_shape.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from random import Random
from typing import Any

from measure_audit_policy import ServerObservation
from measure_blind_spot import BLIND, REFUSED_EXIT, assert_channel_usable
from measure_selective_risk import (
    REPORT_BUDGET,
    UNIFORM_SEEDS,
    beats_every_draw,
    score,
)
from measure_selective_risk import (
    _fleet_view as fleet_view,
)

from pharos.analyst import Proposal, evidence_shown
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
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

#: Draws of the parametric null per cell. The p-value floors at 1/(m+1), so this bounds
#: the smallest claim the test can make, and it has to sit *below* the alpha it is
#: compared against or the test cannot fire at all.
#:
#: The first version of this line said 400 draws put the floor "comfortably below" an
#: alpha of 0.001. It does not: 1/401 is 0.0025, which is larger, so every cell would have
#: read as not-overdispersed no matter how extreme -- including the fleet whose index is
#: 9.00. The arithmetic is now asserted at run time rather than described here, in the
#: same form measure_corpus_sensitivity uses on finding 22's permutation count.
NULL_DRAWS = 2000

#: Significance for "this fleet is overdispersed". Matched to finding 22's alpha so the
#: two detectors are read on the same scale.
ALPHA = 0.001

#: A stratum contributes only if it has at least this many tasks. A dispersion index over
#: two tasks is a number, and it is not an estimate.
MIN_STRATUM = 10

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
class Dispersion:
    """The index, its null, and the evidence base it was computed over."""

    index: float | None
    p_value: float | None
    strata: int
    tasks: int
    #: Strata dropped because a rate of exactly 0 or 1 leaves no binomial variance to
    #: compare against. Reported rather than filtered silently: on a noiseless healthy
    #: fleet this is *every* stratum, and that is the finding rather than a missing row.
    degenerate_strata: int

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "p_value": self.p_value,
            "strata": self.strata,
            "tasks": self.tasks,
            "degenerate_strata": self.degenerate_strata,
        }


def dispersion(
    view: ServerObservation,
    evidence: dict[str, int],
    *,
    draws: int = NULL_DRAWS,
    seed: int = 0,
) -> Dispersion:
    """Observed variance of the vote sums over the binomial variance at the same rate.

    Pooled across strata rather than averaged: a stratum of eighty tasks and a stratum of
    twelve are not two equal observations of the same quantity, and averaging their
    indices would let the small one move the answer as much as the large one.

    The null simulates each stratum's counts as binomial at that stratum's own observed
    rate, which is the hypothesis being tested: one rate per stratum, analysts
    independent. A p-value is the share of simulated fleets at least as dispersed as this
    one, computed as (b+1)/(m+1) so it is never zero.
    """
    strata: dict[int, list[tuple[float, float]]] = {}
    for task, votes in view.votes.items():
        seen = view.seen.get(task, 0.0)
        if seen <= 0:
            continue
        strata.setdefault(evidence.get(task, -1), []).append((votes, seen))

    observed = 0.0
    expected = 0.0
    used: list[tuple[list[tuple[float, float]], float]] = []
    degenerate = 0
    tasks = 0
    for rows in strata.values():
        if len(rows) < MIN_STRATUM:
            continue
        total_votes = sum(v for v, _ in rows)
        total_seen = sum(n for _, n in rows)
        rate = total_votes / total_seen
        variance = sum(n * rate * (1.0 - rate) for _, n in rows)
        if variance <= 0.0:
            # A stratum every analyst answered identically. There is no binomial variance
            # to compare against, so it carries no information about dispersion -- which
            # is different from carrying information that there is none.
            degenerate += 1
            continue
        observed += sum((v - n * rate) ** 2 for v, n in rows)
        expected += variance
        used.append((rows, rate))
        tasks += len(rows)

    if not used:
        return Dispersion(index=None, p_value=None, strata=0, tasks=0, degenerate_strata=degenerate)

    index = observed / expected
    rng = Random(seed)  # noqa: S311  -- a null distribution, not a security boundary
    at_least_as_extreme = 0
    for _ in range(draws):
        null_observed = 0.0
        null_expected = 0.0
        for rows, rate in used:
            for _, n in rows:
                drawn = sum(1 for _ in range(int(n)) if rng.random() < rate)
                null_observed += (drawn - n * rate) ** 2
                null_expected += n * rate * (1.0 - rate)
        if null_observed / null_expected >= index:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (draws + 1)
    return Dispersion(
        index=round(index, 4),
        p_value=round(p_value, 6),
        strata=len(used),
        tasks=tasks,
        degenerate_strata=degenerate,
    )


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
        from measure_audit_policy import select

        held = select(policy, view, truth, REPORT_BUDGET, seed=UNIFORM_SEEDS[0])
        return score(
            n_blind=0, slip_rate=0.0, policy=policy, withheld=held, pool=pool, wrong=wrong
        ).risk

    uniform_draws = []
    for draw_seed in UNIFORM_SEEDS:
        from measure_audit_policy import select

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
    check = assert_channel_usable(tasks)
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
