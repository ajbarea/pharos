#!/usr/bin/env python3
"""What a deployment can do once the detector fires, when the audit budget buys nothing.

Findings 19 to 21 priced an *authority of record* and found the same wall from three
directions: at unanimity no budget on the ladder repairs a label, in any of eight corpus
draws, and every selection policy that reads disagreement falls to chance because a
unanimously blind fleet has no disagreement to read. Finding 22 then showed the regime is
nonetheless *detectable* from the aggregate alone --- a conditional dependence between the
verdict rate and the blinded channel, inside the per-task sums finding 18's secure
aggregation already reveals. That left the question this script exists for, and it was
carried as the open problem: **a fleet that knows which regime it is in still has nothing
to select on, so what should it do?**

Auditing is not the only action available. A deployment that cannot correct a label can
still decline to publish it. That is selective prediction, and it is the standard move:
predict where the model is confident, abstain where it is not. The 2026 form of the rule
states it as agreement --- predict only when the label is forced, that is when every
consistent hypothesis agrees, and abstain otherwise [Khosravani, arXiv:2605.02611]. Read
across to a fleet of analysts, that rule says publish where the fleet is unanimous.

This corpus is built to say what happens to that rule when unanimity is the failure.

**What is measured.** No authority, no anchors, no re-estimation: the estimator runs once
per fleet and each policy chooses a set of tasks to *withhold*. Two numbers per cell, and
neither is readable without the other:

  selective risk   errors among the labels still published, over the labels published
  coverage         how many labels are published at all

**Withholding is deletion, and here that is the point rather than the defect.** Findings
19 and 20 scored agreement over unanchored tasks, so anchoring a task the estimator got
wrong lifted the score without correcting anything, and that was an artifact because those
findings claimed *repair*. Nothing here claims repair. A withheld label is withheld, the
coverage column prices exactly what was given up to remove it, and a policy that withholds
correct labels raises its own risk rather than lowering it. That asymmetry is what makes
the metric honest without an oracle.

**The prediction, stated before the run.**

1. On a fleet whose errors are *random* --- healthy analysts slipping independently ---
   confidence-based abstention works: `margin` and `posterior` lower selective risk well
   below `uniform`, because a slip shows up as disagreement.
2. On a fleet whose errors are *shared* and unanimous, the same rule does nothing. The
   corrupted items are the ones the fleet agrees on hardest, so a confidence ranking sorts
   them to the safe end and abstention removes correct labels instead.
3. `consensus` --- withhold where the fleet agrees most --- is the textbook inversion and
   will beat confidence at unanimity, but only weakly, because at unanimity almost
   everything is unanimous and the ranking is nearly arbitrary within that set.
4. `channel` --- withhold what the detector named --- is the only deployable rule that
   lowers risk in the shared regime, and its coverage cost is set by how prevalent the
   channel is rather than by how many labels are wrong. That is a worse trade than any
   audit would be and it is available, which is the whole claim.
5. On a healthy fleet the same rule is pure cost: a false detection withholds a slice of
   the corpus and removes nothing, and that price is reported rather than assumed small.

If 2 and 4 hold, the answer to the open problem is that detection converts into
*coverage* rather than into correction, and the exchange rate is the channel's prevalence.
If 2 fails --- if confidence-based abstention works at unanimity after all --- then finding
21's collapse is a property of audit budgets rather than of the signal, which would be a
larger result than this one and must be reported as such.

Needs no model and no network. One EM fit per fleet, so the sweep is cheap: every policy
and every budget is set arithmetic over the same posterior.

    uv run python scripts/measure_selective_risk.py --out results/selective_risk.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

from measure_audit_policy import POLICY_SEED, ServerObservation, observe, select
from measure_authority_anchors import ladder
from measure_blind_spot import (
    BLIND,
    BLIND_RUNGS,
    REFUSED_EXIT,
    assert_channel_usable,
    blind_fleet,
)
from measure_secure_reliability import MASK_SEED, contributions_for

from pharos.analyst import Proposal, evidence_shown
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.inference import federated_dawid_skene, partition_by_contributor
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, progress, record
from pharos.validity import check_sample_size

LOG = get_logger()

SEED = 7
EVENTS = 200
FLEET = 9

#: How much of the fleet carries the blind spot. Finding 21's ladder, unchanged, so the
#: two measurements are read against each other rather than against different fleets.
SHARES_RUNGS = BLIND_RUNGS

#: How often an analyst slips at random, independently of anyone else. The contrast this
#: script rests on: 0.0 is a fleet whose only error is the shared one, and 0.15 is finding
#: 22's realistic rate, where random error and shared error are present at once. A
#: confidence rule that cannot tell them apart is the failure being measured.
#:
#: 0.40 is here because the control needs to exist. Aggregation absorbs independent noise,
#: which is what it is for, and at 0.15 a fleet with no shared blind spot has *zero*
#: estimator errors -- so the regime where confidence-based abstention is supposed to work
#: had nothing in it to work on, and prediction 1 would have read as refuted by a cell
#: that was never populated. The rate that breaks a healthy fleet is the rate the control
#: has to be measured at.
SLIP_RATES = (0.0, 0.15, 0.40)

#: Labels withheld. Truncated at run time to the pool the estimator actually covers, for
#: the reason finding 26 gives: that pool is 83 to 99 tasks depending on the draw, and a
#: constant ladder sized for one draw exits non-zero on the others.
WITHHELD = (0, 2, 5, 8, 12, 20, 30, 45, 60)

#: Policies, all of them reading only what finding 18's aggregator can see. `oracle` is a
#: bound and is reported apart from the rest: it withholds exactly the wrong labels, so it
#: says how much of the risk any abstention rule could remove at a given coverage, not how
#: to remove it.
DEPLOYABLE = ("uniform", "margin", "posterior", "consensus", "channel")
BOUND = "oracle"

#: Draws of the uniform baseline. Uniform is a sample where every other policy here is an
#: exact function of the aggregate, and comparing one draw of a variable quantity against
#: an exact one is how a policy gets credited with the margin the draw supplied. Finding
#: 19 measured that spread at 2 to 30 items on this corpus, so it is reported as a median
#: with its range here too.
UNIFORM_SEEDS = tuple(POLICY_SEED + i for i in range(21))

#: The budget every summary column is quoted at. A constant rather than a literal in
#: three places: the artifact publishes it, and the docs block and both manuscripts read
#: it from there, so a change moves the tables instead of making them disagree.
REPORT_BUDGET = 20

#: What counts as risk removed. Half the errors the fleet started with --- a chosen
#: constant, published so a reader can move it, in the same spirit as the 0.95 agreement
#: bar findings 19 and 20 report their thresholds against.
HALVED = 0.5

#: What this sweep varies and what it holds fixed. Declared in the artifact rather than
#: left to be read off the code, because a multiverse chosen after seeing which dimensions
#: are kind to the result is not a robustness check.
MULTIVERSE = {
    "swept": {
        "blind_share": "none to unanimous, the ladder finding 21 uses",
        "slip_rate": list(SLIP_RATES),
        "withheld": list(WITHHELD),
        "uniform_draws": len(UNIFORM_SEEDS),
    },
    "pinned": {
        "corpus_seed": {
            "value": SEED,
            "why": "the corpus dimension belongs to measure_corpus_sensitivity.py, which "
            "sweeps this script over eight draws; crossing them here would confound "
            "which one moved a result",
        },
        "fleet": {
            "value": FLEET,
            "why": "the size every published governance number was measured at; the fleet "
            "dimension belongs to measure_governance_sensitivity.py",
        },
        "blind_compartment": {
            "value": BLIND.value,
            "why": "the one channel on this corpus whose evidence is not entangled with "
            "item difficulty, which measure_blind_spot asserts before it runs",
        },
    },
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Cell:
    """One policy at one budget, on one fleet."""

    n_blind: int
    slip_rate: float
    policy: str
    withheld: int
    coverage: float
    published: int
    risk: float
    errors_published: int
    caught: int
    precision: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_blind": self.n_blind,
            "slip_rate": self.slip_rate,
            "policy": self.policy,
            "withheld": self.withheld,
            "coverage": self.coverage,
            "published": self.published,
            "risk": self.risk,
            "errors_published": self.errors_published,
            "caught": self.caught,
            "precision": self.precision,
        }


def score(
    *,
    n_blind: int,
    slip_rate: float,
    policy: str,
    withheld: tuple[str, ...],
    pool: tuple[str, ...],
    wrong: frozenset[str],
) -> Cell:
    """What one withholding decision published, and what it got wrong anyway.

    Risk is over *published* labels, so withholding a correct label raises it. That is
    what stops this metric from rewarding abstention for its own sake, and it is why no
    oracle is needed to read the table: a policy that withholds indiscriminately is
    visibly punished in the same column it would otherwise be flattered in.
    """
    held = set(withheld)
    published = [task for task in pool if task not in held]
    errors = sum(1 for task in published if task in wrong)
    return Cell(
        n_blind=n_blind,
        slip_rate=slip_rate,
        policy=policy,
        withheld=len(held),
        coverage=round(len(published) / len(pool), 4) if pool else 0.0,
        published=len(published),
        risk=round(errors / len(published), 4) if published else 0.0,
        errors_published=errors,
        caught=len(held & wrong),
        precision=round(len(held & wrong) / len(held), 4) if held else 0.0,
    )


def first_budget_halving(cells: list[Cell], base_errors: int) -> int | None:
    """Smallest budget whose published labels carry at most half the errors it started with.

    Counted in *errors*, not in the risk rate. The rate falls when correct labels are
    withheld as well, so a threshold read off the rate would credit a policy for shrinking
    the corpus. This is the deletion artifact findings 19 and 20 had to retract, in the one
    place on this page where it could still get in.
    """
    target = base_errors * HALVED
    return next(
        (
            c.withheld
            for c in sorted(cells, key=lambda c: c.withheld)
            if c.errors_published <= target
        ),
        None,
    )


def beats_every_draw(theirs: float | None, draws: list[float]) -> bool:
    """Whether a targeted risk is below the *best* of the untargeted draws it is compared to.

    A module-level function rather than a closure so it can be tested against the case it
    exists for: a policy that merely beats the median of a variable baseline. Withholding
    20 random labels of two hundred removes an error now and then, so the median uniform
    risk is not a floor. At unanimity `margin` came in at 0.094 against a median 0.100 and
    a best draw of 0.089, and reading the first comparison as a win credited the policy
    with one task's worth of luck.
    """
    return theirs is not None and bool(draws) and theirs < min(draws)


def _fleet_view(
    tasks: list[Any],
    proposals: dict[str, Proposal],
    *,
    n_blind: int,
    fleet: int,
    slip_rate: float,
    seed: int,
) -> tuple[ServerObservation, dict[str, bool], bool]:
    """The aggregator's view of one fleet, the labels it would publish, and whether it converged.

    Convergence travels with the labels because it has to: at the slip rates where a
    healthy fleet breaks at all, the EM fit stops converging, and a risk column computed
    off a fit that ran out of iterations is a different quantity from one computed off a
    fit that finished. This project already publishes flags rather than filtering on them,
    and this is one more.
    """
    flat = contributions_for(
        blind_fleet(n_blind, fleet, slip_rate=slip_rate), tasks, proposals, seed=seed
    )
    partitioned = partition_by_contributor(flat)
    view = observe(partitioned)
    by_id = {t.task_id: t for t in tasks}
    # What the detector hands over, and only where it has fired. At n_blind = 0 there is
    # no channel to name, and a policy selecting by provenance on a fleet nobody has shown
    # to be channel-blind would be an oracle wearing a method's name. The one exception is
    # the false-detection control below, which supplies it deliberately and says so.
    carries = {
        task: any(BLIND in r.label.compartments for r in by_id[task].sources)
        for task in view.posterior
    }
    evidence = {task: len(evidence_shown(by_id[task])) for task in view.posterior}
    estimate = federated_dawid_skene(partitioned, seed=MASK_SEED)
    return replace(view, carries=carries, evidence=evidence), estimate.labels(), estimate.converged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--slip-rates", type=float, nargs="+", default=list(SLIP_RATES))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    shares = ladder(args.fleet, SHARES_RUNGS)

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=args.seed, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    # The same precondition finding 21 refuses on, applied here rather than copied here.
    # A draw whose blinded channel is entangled with item difficulty cannot host this
    # experiment either, and a script that skipped the check would still write a
    # well-formed artifact -- which is the failure mode worth guarding, because nothing
    # downstream could tell it apart from a valid one.
    check = assert_channel_usable(tasks)
    print(
        f"{len(tasks)} tasks, fleet of {args.fleet}, blind spot on {BLIND.value}: "
        f"{check.affected} verdicts change, on a slice sitting at "
        f"{check.affected_mean:.2f} defining facts against a corpus mean of "
        f"{check.corpus_mean:.2f}"
    )

    cells: list[Cell] = []
    fleets: list[dict[str, Any]] = []
    uniform_spread: list[dict[str, Any]] = []

    for slip_rate in args.slip_rates:
        for n_blind in shares:
            progress("selective_risk.fleet", n_blind=n_blind, slip_rate=slip_rate)
            view, labels, converged = _fleet_view(
                tasks,
                proposals,
                n_blind=n_blind,
                fleet=args.fleet,
                slip_rate=slip_rate,
                seed=args.seed,
            )
            pool = tuple(sorted(labels))
            wrong = frozenset(task for task, value in labels.items() if value != truth[task])
            budgets = tuple(b for b in WITHHELD if b <= len(pool))
            base = score(
                n_blind=n_blind,
                slip_rate=slip_rate,
                policy="none",
                withheld=(),
                pool=pool,
                wrong=wrong,
            )

            # A fleet the estimator already gets right prices no abstention rule, and
            # scoring one would mark every policy down for failing to remove errors that
            # do not exist. Recorded as a property of the fleet rather than silently
            # dropped, the way measure_corpus_sensitivity records a refused draw.
            nothing_to_withhold = not wrong

            per_policy: dict[str, dict[str, Any]] = {}
            for policy in (*DEPLOYABLE, BOUND):
                if policy == "uniform":
                    continue
                # `channel` reads the detector's output, which exists only where the
                # detector fired. At n_blind = 0 it is handed the channel anyway: that is
                # the false-detection control, and its cost is the number reported.
                policy_view = view
                block: list[Cell] = []
                for budget in budgets:
                    held = select(policy, policy_view, truth, budget, seed=POLICY_SEED)
                    cell = score(
                        n_blind=n_blind,
                        slip_rate=slip_rate,
                        policy=policy,
                        withheld=held,
                        pool=pool,
                        wrong=wrong,
                    )
                    cells.append(cell)
                    block.append(cell)
                    record(
                        "selective_risk.cell",
                        cell.risk,
                        policy=policy,
                        n_blind=n_blind,
                        withheld=budget,
                    )
                per_policy[policy] = {
                    "halved_at": None
                    if nothing_to_withhold
                    else first_budget_halving(block, len(wrong)),
                    "precision_at_20": next(
                        (c.precision for c in block if c.withheld == REPORT_BUDGET), None
                    ),
                    "risk_at_20": next(
                        (c.risk for c in block if c.withheld == REPORT_BUDGET), None
                    ),
                }

            # Uniform is a draw, so it is re-run per seed and reported as a median with
            # its range. The median draw is what the comparison row uses.
            draws: list[int | None] = []
            uniform_cells: dict[int, list[Cell]] = {b: [] for b in budgets}
            for seed in UNIFORM_SEEDS:
                block = []
                for budget in budgets:
                    held = select("uniform", view, truth, budget, seed=seed)
                    cell = score(
                        n_blind=n_blind,
                        slip_rate=slip_rate,
                        policy="uniform",
                        withheld=held,
                        pool=pool,
                        wrong=wrong,
                    )
                    block.append(cell)
                    uniform_cells[budget].append(cell)
                draws.append(
                    None if nothing_to_withhold else first_budget_halving(block, len(wrong))
                )
            reached = sorted(d for d in draws if d is not None)
            per_policy["uniform"] = {
                "halved_at": (
                    sorted(
                        [*reached, *([None] * (len(draws) - len(reached)))],
                        key=lambda v: (v is None, v),
                    )[(len(draws) - 1) // 2]
                ),
                "precision_at_20": round(
                    median(c.precision for c in uniform_cells.get(REPORT_BUDGET, [])), 4
                )
                if uniform_cells.get(REPORT_BUDGET)
                else None,
                "risk_at_20": round(median(c.risk for c in uniform_cells.get(REPORT_BUDGET, [])), 4)
                if uniform_cells.get(REPORT_BUDGET)
                else None,
                #: Every draw, not the median. A targeted policy is compared against the
                #: *best* of these, and the comparison is unreadable without them: at
                #: unanimity the first version of this script credited `margin` with
                #: beating uniform on a gap of 0.006, which is one task and well inside
                #: the spread of the draw it was beating.
                "risk_at_20_draws": sorted(
                    round(c.risk, 4) for c in uniform_cells.get(REPORT_BUDGET, [])
                ),
            }
            uniform_spread.append(
                {
                    "n_blind": n_blind,
                    "slip_rate": slip_rate,
                    "draws": len(draws),
                    "reached": len(reached),
                    "lowest": reached[0] if reached else None,
                    "highest": reached[-1] if reached else None,
                    "risk_at_20_range": (
                        [
                            min(c.risk for c in uniform_cells[REPORT_BUDGET]),
                            max(c.risk for c in uniform_cells[REPORT_BUDGET]),
                        ]
                        if uniform_cells.get(REPORT_BUDGET)
                        else None
                    ),
                }
            )
            # The median uniform cell per budget, so the grid carries one uniform row
            # rather than 21 and the artifact still says which draws it came from.
            for budget in budgets:
                block = uniform_cells[budget]
                cells.append(
                    Cell(
                        n_blind=n_blind,
                        slip_rate=slip_rate,
                        policy="uniform",
                        withheld=budget,
                        coverage=round(median(c.coverage for c in block), 4),
                        published=round(median(c.published for c in block)),
                        risk=round(median(c.risk for c in block), 4),
                        errors_published=round(median(c.errors_published for c in block)),
                        caught=round(median(c.caught for c in block)),
                        precision=round(median(c.precision for c in block), 4),
                    )
                )

            fleets.append(
                {
                    "n_blind": n_blind,
                    "slip_rate": slip_rate,
                    "pool": len(pool),
                    "base_errors": len(wrong),
                    "base_risk": base.risk,
                    "converged": converged,
                    "budgets": list(budgets),
                    "nothing_to_withhold": nothing_to_withhold,
                    "policies": per_policy,
                }
            )

    if not any(f["base_errors"] for f in fleets):
        # Nothing anywhere to abstain from is not a result about abstention, and an
        # artifact reporting flat zeros would read as one.
        LOG.error(
            "selective_risk.no_errors_anywhere",
            extra={"event": "selective_risk.no_errors_anywhere", "seed": args.seed},
        )
        raise SystemExit(REFUSED_EXIT)

    unanimous = max(shares)
    for slip_rate in args.slip_rates:
        print(f"\n  slip rate {slip_rate}: risk among published labels, 20 of the pool withheld")
        print(f"    {'blind':>6}{'base':>8}" + "".join(f"{p:>11}" for p in (*DEPLOYABLE, BOUND)))
        print("    " + "-" * (14 + 11 * (len(DEPLOYABLE) + 1)))
        for n_blind in shares:
            entry = next(
                f for f in fleets if f["n_blind"] == n_blind and f["slip_rate"] == slip_rate
            )
            row = "".join(
                f"{entry['policies'][p]['risk_at_20']:>11.3f}"
                if entry["policies"][p]["risk_at_20"] is not None
                else f"{'-':>11}"
                for p in (*DEPLOYABLE, BOUND)
            )
            print(f"    {n_blind:>6}{entry['base_risk']:>8.3f}{row}")

    def fleet_at(n_blind: int, slip_rate: float) -> dict[str, Any]:
        return next(f for f in fleets if f["n_blind"] == n_blind and f["slip_rate"] == slip_rate)

    def at(n_blind: int, slip_rate: float, policy: str, field: str) -> Any:
        return fleet_at(n_blind, slip_rate)["policies"][policy][field]

    def lowers_risk(n_blind: int, slip_rate: float, policy: str) -> bool:
        """Whether a policy beats *every* uniform draw at 20 withheld, on a fleet with errors.

        Against uniform rather than against the base rate, because withholding anything at
        all moves the rate a little and the question is whether the *ranking* carried
        information.

        Against the best of 21 draws rather than their median, which is the correction
        finding 20 already had to make and which this script needed too. Withholding 20
        random labels of a hundred removes an error now and then, so the median uniform
        risk is not a floor: at unanimity `margin` came in at 0.094 against a median
        0.100, and reading that as a win credited the policy with 0.006 --- one task,
        inside the spread of the thing it was beating. A policy that cannot beat the
        luckiest untargeted draw has told the deployment nothing.

        A fleet the estimator already gets right is not a win for anybody: every policy
        scores 0.000 there, and `<` on two zeros is False, but saying so out loud is
        cheaper than rediscovering why a control row read as a refutation.
        """
        if not fleet_at(n_blind, slip_rate)["base_errors"]:
            return False
        return beats_every_draw(
            at(n_blind, slip_rate, policy, "risk_at_20"),
            at(n_blind, slip_rate, "uniform", "risk_at_20_draws"),
        )

    healthy = min(shares)

    # The random-error control has to be *constructible* before it can be passed or
    # failed. Aggregation is built to absorb independent noise and it does: on a fleet
    # with no shared blind spot, every slip rate up to 0.15 leaves the estimator with zero
    # errors, so there is nothing for any abstention rule to remove and a False here would
    # report "the rule failed" where the honest reading is "the experiment was not run".
    # The control is therefore the healthy fleet at whatever swept rate actually breaks
    # the estimator, and None where none of them does.
    healthy_with_errors = [
        rate for rate in args.slip_rates if fleet_at(healthy, rate)["base_errors"] > 0
    ]
    control_rate = max(healthy_with_errors, default=None)
    if control_rate is None:
        LOG.warning(
            "selective_risk.control_not_constructible",
            extra={
                "event": "selective_risk.control_not_constructible",
                "slip_rates": list(args.slip_rates),
            },
        )

    # What a false detection costs, as a number rather than as a claim. On a healthy fleet
    # the channel policy withholds a slice of the corpus and can remove nothing the shared
    # blind spot put there, because there is no shared blind spot -- so a boolean saying
    # "removes nothing" would be true by construction, which is the kind of check this
    # project refuses to publish. The quantity with content is the coverage given up.
    false_detection = [
        {
            "slip_rate": rate,
            "base_errors": fleet_at(healthy, rate)["base_errors"],
            "coverage_at_20": next(
                (
                    c.coverage
                    for c in cells
                    if c.n_blind == healthy
                    and c.slip_rate == rate
                    and c.policy == "channel"
                    and c.withheld == REPORT_BUDGET
                ),
                None,
            ),
            "caught_at_20": next(
                (
                    c.caught
                    for c in cells
                    if c.n_blind == healthy
                    and c.slip_rate == rate
                    and c.policy == "channel"
                    and c.withheld == REPORT_BUDGET
                ),
                None,
            ),
        }
        for rate in args.slip_rates
    ]
    # Where the shared blind spot is the *whole* of what the estimator gets wrong: the slip
    # rates at which a fleet with no blind spot has zero errors, so every error at any
    # other share is attributable to the shared standard. The criterion is read off the
    # healthy row rather than chosen, which matters, because the alternative is picking the
    # regimes a claim survives in after seeing it survive there.
    #
    # This distinction was forced by the measurement and is not cosmetic. The first version
    # quantified every claim over all three slip rates and reported prediction 4 as
    # refuted, on the strength of a rate that had been added as prediction 1's control: at
    # 0.40 a healthy fleet already carries 74 estimator errors, so the shared blind spot is
    # a small minority of the damage and no rule aimed at it can move the column. That is a
    # real bound and it is published below as its own line, rather than folded into the
    # claim it does not bear on.
    shared_only = [rate for rate in args.slip_rates if fleet_at(healthy, rate)["base_errors"] == 0]
    if not shared_only:
        # A refusal, not a crash, and it carries the declared exit code so a sweep counts
        # it as a draw that could not host the experiment rather than as a bug. Both are
        # non-zero and only one of them belongs in the denominator.
        LOG.error(
            "selective_risk.no_shared_only_regime",
            extra={
                "event": "selective_risk.no_shared_only_regime",
                "slip_rates": list(args.slip_rates),
            },
        )
        print(
            "no swept slip rate leaves a healthy fleet error-free, so no cell isolates "
            "the shared blind spot and every claim would be about mixed error",
            file=sys.stderr,
        )
        raise SystemExit(REFUSED_EXIT)
    unconverged = [
        (f["n_blind"], f["slip_rate"]) for f in fleets if not f["converged"] and f["base_errors"]
    ]
    if unconverged:
        LOG.warning(
            "selective_risk.estimator_did_not_converge",
            extra={
                "event": "selective_risk.estimator_did_not_converge",
                "cells": [f"{n}@{rate}" for n, rate in unconverged],
            },
        )

    findings = {
        #: Prediction 1. The rule works where the error is random, which is the control
        #: that makes its failure below a statement about the error's shape rather than
        #: about this corpus. None where no swept slip rate breaks a healthy fleet.
        "confidence_abstention_works_on_random_error": None
        if control_rate is None
        else (
            lowers_risk(healthy, control_rate, "margin")
            or lowers_risk(healthy, control_rate, "posterior")
        ),
        #: Prediction 2, the one the open problem turns on. `any` rather than `all`: one
        #: regime where the textbook rule works at unanimity is enough to refute it.
        "confidence_abstention_works_at_unanimity": any(
            lowers_risk(unanimous, rate, policy)
            for rate in shared_only
            for policy in ("margin", "posterior")
        ),
        #: Prediction 3.
        "consensus_abstention_works_at_unanimity": any(
            lowers_risk(unanimous, rate, "consensus") for rate in shared_only
        ),
        #: Prediction 4: the answer to the open problem, if it holds. Quantified over the
        #: regimes the shared blind spot is the whole of the error in, which is where
        #: finding 22's detector is the instrument that fired.
        "provenance_abstention_works_at_unanimity": all(
            lowers_risk(unanimous, rate, "channel") for rate in shared_only
        ),
        #: The bound on prediction 4, and the reason the claim is conditional. Where random
        #: error dominates, a rule aimed at the shared component addresses a minority of
        #: what is wrong. Expected false, published as a limit rather than as a failure.
        "provenance_abstention_survives_dominant_random_error": None
        if control_rate is None
        else lowers_risk(unanimous, control_rate, "channel"),
    }

    print("\n  predictions, as measured")
    for name, value in findings.items():
        print(f"    {name:<52} {value}")
    print("\n  prediction 5: what a false detection costs on a fleet with no blind spot")
    print(f"    {'slip':>6}{'errors':>9}{'coverage':>11}{'caught':>9}")
    for entry in false_detection:
        coverage = "-" if entry["coverage_at_20"] is None else f"{entry['coverage_at_20']:.3f}"
        print(
            f"    {entry['slip_rate']:>6}{entry['base_errors']:>9}"
            f"{coverage:>11}{entry['caught_at_20']!s:>9}"
        )

    if not findings["provenance_abstention_works_at_unanimity"]:
        LOG.warning(
            "selective_risk.no_deployable_action",
            extra={
                "event": "selective_risk.no_deployable_action",
                "n_blind": unanimous,
            },
        )
    if findings["confidence_abstention_works_at_unanimity"]:
        # Louder than the line above, because it would be the larger result: finding 21's
        # collapse would be a property of audit budgets rather than of the signal.
        LOG.warning(
            "selective_risk.confidence_survives_unanimity",
            extra={
                "event": "selective_risk.confidence_survives_unanimity",
                "n_blind": unanimous,
            },
        )

    published = [c.published for c in cells if c.published]
    report = {
        "provenance": run_provenance(seed=args.seed),
        "fleet": args.fleet,
        "events": args.events,
        "blind_compartment": BLIND.value,
        "shares": list(shares),
        "slip_rates": list(args.slip_rates),
        "withheld_requested": list(WITHHELD),
        "deployable": list(DEPLOYABLE),
        "bound": BOUND,
        "halved_threshold": HALVED,
        "report_budget": REPORT_BUDGET,
        "uniform_seeds": list(UNIFORM_SEEDS),
        "multiverse": MULTIVERSE,
        "fleets": fleets,
        "uniform_spread": uniform_spread,
        "findings": findings,
        "random_error_control": control_rate,
        "shared_only_slip_rates": shared_only,
        "unconverged_cells": [{"n_blind": n, "slip_rate": rate} for n, rate in unconverged],
        "false_detection": false_detection,
        "grid": [c.as_dict() for c in cells],
        # Over the smallest published set anywhere in the grid, not the corpus: the
        # question is whether the thinnest cell quoted is one a rate may be read off.
        "validity": check_sample_size(min(published, default=0), label="selective risk").as_dict(),
        "published_min": min(published, default=0),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
