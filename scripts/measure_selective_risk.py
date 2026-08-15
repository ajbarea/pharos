#!/usr/bin/env python3
"""What a deployment can do once the detector fires, when the audit budget buys nothing.

Findings 19 to 21 close off correction at unanimity from three directions: no budget on
the anchor ladder repairs a label, every disagreement-reading policy falls to chance, and
both hold across corpus draws. Finding 22 detects the regime anyway. That leaves the
question this script answers: a fleet that knows which regime it is in still has nothing
to select on, so what should it do?

Auditing is not the only action. A deployment that cannot correct a label can decline to
publish it. That is selective prediction, whose 2026 form states the licence as agreement
-- predict only where the label is *forced*, abstain otherwise [Khosravani,
arXiv:2605.02611]. Read across to a fleet of analysts, that rule says publish where the
fleet is unanimous, and this corpus makes unanimity the failure.

No authority, no anchors, no re-estimation: the estimator runs once per fleet and each
policy chooses labels to withhold. Two numbers per cell and neither is readable alone --
the errors among labels still published, and how many are published at all. Withholding
here is deletion by design rather than by accident: nothing claims a repair, and a policy
that withholds correct labels raises its own risk, which is what makes the metric honest
without an oracle.

**Predictions, before the run.** (1) Where the error is random, confidence-based
abstention works. (2) Where it is shared and unanimous, the same rule does nothing,
because the corrupted items are the ones the fleet agrees on hardest. (3) Withholding
where the fleet agrees *most* beats confidence at unanimity, weakly. (4) Withholding by
the channel the detector named is the only deployable rule that lowers risk there, and its
cost is set by the channel's prevalence rather than by how many labels are wrong. (5) On a
healthy fleet that same rule is pure cost, and the price is reported rather than assumed
small.

If (2) and (4) hold, detection converts into coverage rather than into correction. If (2)
fails, finding 21's collapse is a property of audit budgets rather than of the signal,
which is a larger result and must be reported as one.

Needs no model and no network. One EM fit per fleet, so every policy and budget is set
arithmetic over the same posterior.

    uv run python scripts/measure_selective_risk.py --out results/selective_risk.json
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

from pharos.analyst import Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import (
    BLIND,
    BLIND_RUNGS,
    HALVED,
    POLICY_SEED,
    REFUSED_EXIT,
    REPORT_BUDGET,
    UNIFORM_SEEDS,
    AbstentionCell,
    ChannelUnusableError,
    assert_channel_usable,
    beats_every_draw,
    first_budget_halving,
    fleet_view,
    ladder,
    score,
    select,
)
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, progress, record
from pharos.validity import check_sample_size

LOG = get_logger()

#: Finding 22's artifact lives here, and is read to say where the `channel` policy is
#: licensed at all.
RESULTS = Path(__file__).resolve().parents[1] / "results"

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


def detector_fired(slip_rate: float, n_blind: int, *, seed: int) -> bool | None:
    """Whether finding 22's detector actually names this channel on this fleet.

    The `channel` policy is deployable *because* finding 22 supplies the channel name
    from data the aggregator already holds. That licence is not uniform across this
    grid: detection reaches one blind analyst in nine on a noiseless fleet and four of
    nine at a realistic slip rate, so there are cells here where the policy is scored on
    a channel no deployment would have been told about.

    Read from `channel_bias.json` rather than recomputed, because a second implementation
    of a detection threshold is a second thing to drift. Returns None where that artifact
    cannot answer -- it is missing, or carries no cell at this slip rate and share -- and
    None is reported rather than assumed false, since "not measured" and "measured and
    silent" are different claims and only one of them is about the detector.
    """
    path = RESULTS / "channel_bias.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    # The committed artifact is one corpus draw, and this script is swept over eight.
    # Without this line every draw would be handed seed 7's detections, and the sweep
    # would report a licence that was measured on a different corpus -- which is the
    # cross-corpus confound this project has already withdrawn a claim over. A mismatch
    # is "cannot say", not "did not fire".
    if payload.get("provenance", {}).get("seed") != seed:
        return None
    for cell in payload.get("sweep", []):
        if cell.get("slip_rate") != slip_rate or cell.get("n_blind") != n_blind:
            continue
        for detection in cell.get("detections", []):
            if detection.get("channel") == BLIND.value:
                return bool(detection.get("detected"))
    return None


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
    try:
        check = assert_channel_usable(tasks)
    except ChannelUnusableError as refusal:
        # The exit code is this script's contract with the corpus sweep; the library
        # raises and the protocol stays here.
        print(refusal, file=sys.stderr)
        raise SystemExit(REFUSED_EXIT) from refusal
    print(
        f"{len(tasks)} tasks, fleet of {args.fleet}, blind spot on {BLIND.value}: "
        f"{check.affected} verdicts change, on a slice sitting at "
        f"{check.affected_mean:.2f} defining facts against a corpus mean of "
        f"{check.corpus_mean:.2f}"
    )

    cells: list[AbstentionCell] = []
    fleets: list[dict[str, Any]] = []
    uniform_spread: list[dict[str, Any]] = []

    for slip_rate in args.slip_rates:
        for n_blind in shares:
            progress("selective_risk.fleet", n_blind=n_blind, slip_rate=slip_rate)
            view, labels, converged = fleet_view(
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
                block: list[AbstentionCell] = []
                for budget in budgets:
                    held = select(policy, view, truth, budget, seed=POLICY_SEED)
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
            uniform_cells: dict[int, list[AbstentionCell]] = {b: [] for b in budgets}
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
                    AbstentionCell(
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
                    #: Whether finding 22's detector names this channel on this fleet.
                    #: The `channel` column is a proposal only where this is true.
                    "detector_fired": detector_fired(slip_rate, n_blind, seed=args.seed),
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
    # Where the shared blind spot is the *whole* of the estimator's error: the slip rates
    # at which a healthy fleet has none. Read off the healthy row rather than chosen, so
    # the regimes a claim is quantified over cannot be picked after seeing where it
    # survives. Quantifying over the high-noise control instead reported the provenance
    # result as refuted by a regime it was never aimed at; that bound is published below
    # as its own line.
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

    # The licence check, on the cells the claims are quantified over. A claim resting on
    # a cell where finding 22's detector never fired would be proposing a policy no
    # deployment could have known to run.
    #
    # Two states, not one. `False` is finding 22 saying it does not detect this fleet,
    # which would make the policy undeployable on the cell a claim rests on. `None` is
    # its artifact being unable to answer -- a different corpus draw, or a share it did
    # not sweep -- which is missing evidence rather than negative evidence. Collapsing
    # them would either cry wolf on every swept draw or hide a real gap on the committed
    # one, and only one of those is loud enough to notice.
    unlicensed = [
        (n, rate)
        for rate in shared_only
        for n in (unanimous,)
        if fleet_at(n, rate)["detector_fired"] is False
    ]
    unverified = [
        (n, rate)
        for rate in shared_only
        for n in (unanimous,)
        if fleet_at(n, rate)["detector_fired"] is None
    ]
    if unlicensed:
        LOG.warning(
            "selective_risk.policy_unlicensed_here",
            extra={
                "event": "selective_risk.policy_unlicensed_here",
                "cells": [f"{n}@{rate}" for n, rate in unlicensed],
            },
        )

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
        #: Cells a claim is quantified over where the detector did not fire, or where
        #: finding 22's artifact cannot say. Empty is the expected state and is published
        #: rather than assumed, because the licence is what makes the policy a proposal.
        "unlicensed_claim_cells": [{"n_blind": n, "slip_rate": rate} for n, rate in unlicensed],
        #: Claim cells where finding 22's artifact cannot answer at this corpus draw.
        #: Expected to be every cell on a swept draw and none on the committed one.
        "unverified_claim_cells": [{"n_blind": n, "slip_rate": rate} for n, rate in unverified],
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
