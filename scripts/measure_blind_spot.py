#!/usr/bin/env python3
"""The corpus finding 20's audit policy cannot handle, built on purpose.

Finding 20 reported that auditing where the fleet splits repairs the estimate at a
fraction of a uniform budget, and tied the oracle bound exactly --- then carried a
scope condition saying why. Every wrong standard measured up to that point was a
shifted `escalation_threshold`, which keys on *how much evidence a task shows*. Two
reviewers differing on that axis disagree precisely on the boundary items, so "the
fleet is split" and "the fleet is wrong" were the same set by construction, and a
policy selecting the first got the second for free.

That is a property of the generator, not of the world, and the honest way to find out
what it is worth is to remove it. This script builds a wrong standard of a different
*shape* and asks whether the policy survives.

**The blind spot.** A reviewer who discounts a channel rather than misjudging a
quantity: they read every report and decline to credit the ones arriving through one
compartment. Nothing about them is noisy or inconsistent. On this corpus PARTNER is
the honest choice of channel --- mean evidence shown is 1.88 on tasks carrying it
against 1.72 on tasks that do not, where SENSOR would be 2.00 against 0.48 and would
smuggle the difficulty confound back in. A fleet-wide PARTNER blind spot changes 20
verdicts of 200, and **every one of them on a task showing all three defining facts**:
the corrupted slice is the *unambiguous* end of the corpus, which is exactly where a
threshold-shifted reviewer is always right and where a fleet is otherwise unanimous.

**What is swept.** The share of the fleet carrying the blind spot, from none to all.
While a sighted minority remains, the corrupted tasks still draw dissent and the
disagreement signal still points at them. When the blind spot becomes unanimous the
signal does not weaken --- it ceases to exist, because there is nothing left to
disagree. Uniform selection never depended on it.

**The prediction, stated before the run.** `margin` should lose its advantage and then
invert it, and the crossing should be at unanimity rather than gradual. `uniform`
should be unaffected in kind. The oracle should be unaffected entirely, because it
reads the answer rather than a proxy for it. If that holds, finding 20's conditional
is exactly the right shape and the condition is the load-bearing half.

Needs no model and no network.

    uv run python scripts/measure_blind_spot.py --out results/blind_spot.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from pharos.analyst import Proposal, evidence_shown
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import (
    BLIND,
    BLIND_RUNGS,
    DEPLOYABLE,
    POLICY_SEED,
    REFUSED_EXIT,
    REPAIRED,
    ChannelUnusableError,
    assert_channel_usable,
    baseline_errors,
    blind_fleet,
    contributions_for,
    evaluate_audit,
    ladder,
    observe,
    select,
)
from pharos.inference import partition_by_contributor
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200
FLEET = 9

SHARES = ladder(FLEET, BLIND_RUNGS)

#: Budgets swept. Capped under the auditable pool, as in finding 20.
BUDGETS = (0, 2, 5, 8, 12, 20, 30, 45, 60, 80, 95)


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    n_blind: int
    policy: str
    budget: int
    agreement: float
    repaired: bool
    scored_tasks: int
    remaining_errors: int
    hits: int
    mechanical: float
    corrected: int

    def as_dict(self) -> dict[str, object]:
        return {
            "n_blind": self.n_blind,
            "policy": self.policy,
            "budget": self.budget,
            "agreement": self.agreement,
            "repaired": self.repaired,
            "scored_tasks": self.scored_tasks,
            "remaining_errors": self.remaining_errors,
            "hits": self.hits,
            "mechanical": self.mechanical,
            "corrected": self.corrected,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    shares = ladder(args.fleet, BLIND_RUNGS)

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=args.seed, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    # The orthogonality claim, asserted rather than trusted. If the affected slice is
    # picked out by difficulty after all, this experiment measures the same confound it
    # was built to escape and its result would be meaningless in a way no later check
    # would reveal.
    try:
        check = assert_channel_usable(tasks)
    except ChannelUnusableError as refusal:
        # The exit code is this script's contract with `measure_corpus_sensitivity`,
        # which matches on it to tell a draw that cannot host the experiment from a draw
        # that crashed. The library raises; the protocol lives here.
        print(refusal, file=sys.stderr)
        raise SystemExit(REFUSED_EXIT) from refusal

    print(
        f"{len(tasks)} tasks, fleet of {args.fleet}, blind spot on {BLIND.value}: "
        f"{check.affected} verdicts change. Channel evidence {check.mean_with:.2f} carrying "
        f"vs {check.mean_without:.2f} not. Affected slice sits at {check.affected_mean:.2f} "
        f"against a corpus mean of {check.corpus_mean:.2f} -- the opposite extreme from the "
        "boundary items a threshold error hits, which is the anti-correlation this rests on."
    )

    # `channel` is finding 22's answer wired into finding 21's failure. Every policy in
    # DEPLOYABLE reads disagreement, and a unanimously blind fleet has none -- which is
    # the whole of finding 21. This one reads the provenance the detector names instead,
    # so it is evaluated here and nowhere else: in finding 20's regime there is no
    # channel to detect and it would degrade to an arbitrary draw.
    policies_here = (*DEPLOYABLE, "channel", "oracle")

    rows: list[Row] = []
    thresholds: dict[str, dict[int, int | None]] = {p: {} for p in policies_here}
    hit_rate: dict[int, dict[str, float]] = {}

    for n_blind in shares:
        flat = contributions_for(blind_fleet(n_blind, args.fleet), tasks, proposals, seed=args.seed)
        partitioned = partition_by_contributor(flat)
        view = observe(partitioned)
        # What the detector hands over. Only populated where finding 22 actually fires;
        # at n_blind = 0 there is no detection and the policy must not select this way.
        by_id = {t.task_id: t for t in tasks}
        carries = {
            task: any(BLIND in r.label.compartments for r in by_id[task].sources)
            for task in view.posterior
        }
        evidence = {task: len(evidence_shown(by_id[task])) for task in view.posterior}
        view = replace(
            view, carries=carries if n_blind else {}, evidence=evidence if n_blind else {}
        )
        baseline = baseline_errors(partitioned, truth)
        wrong = {t for t, p in view.posterior.items() if (p >= 0.5) != truth[t]}

        # The mechanism, measured directly rather than inferred from the outcome: what
        # share of a policy's picks are actually corrupted items. This is the number
        # that explains the thresholds below, and in finding 20 it was 100%.
        hit_rate[n_blind] = {}
        for name in policies_here:
            picked = set(select(name, view, truth, 20, seed=POLICY_SEED))
            hit_rate[n_blind][name] = round(len(picked & wrong) / 20, 3) if wrong else 0.0

        for name in policies_here:
            found: int | None = None
            for budget in BUDGETS:
                out = evaluate_audit(
                    partitioned,
                    view,
                    truth,
                    policy=name,
                    budget=budget,
                    error=0.0,
                    baseline=baseline,
                )
                # `repaired` now requires a label to have actually changed. Scoring
                # over unanchored tasks alone lets a policy climb by deleting the
                # errors it anchors, and at every share and budget here that is
                # precisely what happened: not one unanchored corrupted label is
                # corrected by any policy, so the old threshold table reported a
                # denominator artifact as a repair.
                repaired = out.agreement >= REPAIRED and out.corrected > 0
                rows.append(
                    Row(
                        n_blind=n_blind,
                        policy=name,
                        budget=budget,
                        agreement=out.agreement,
                        repaired=repaired,
                        scored_tasks=out.scored,
                        remaining_errors=out.remaining_errors,
                        hits=out.hits,
                        mechanical=out.mechanical,
                        corrected=out.corrected,
                    )
                )
                if repaired and found is None:
                    found = budget
            thresholds[name][n_blind] = found
            record("blindspot.threshold", float(found or -1), policy=name, n_blind=n_blind)

    order = policies_here
    print(f"\n  budget to repair (lower is better), '-' = not reached within {max(BUDGETS)}")
    print(f"    {'blind':>6}" + "".join(f"{p:>12}" for p in order))
    print("    " + "-" * (6 + 12 * len(order)))
    for n_blind in shares:
        cells = []
        for name in order:
            value = thresholds[name].get(n_blind)
            cells.append(f"{'-' if value is None else value:>12}")
        print(f"    {n_blind:>6}" + "".join(cells))

    print("\n  share of a 20-item audit that lands on a corrupted task")
    print(f"    {'blind':>6}" + "".join(f"{p:>12}" for p in order))
    print("    " + "-" * (6 + 12 * len(order)))
    for n_blind in shares:
        print(f"    {n_blind:>6}" + "".join(f"{hit_rate[n_blind][p]:>12.2f}" for p in order))

    unanimous = max(shares)
    margin_at_unanimity = thresholds["margin"].get(unanimous)
    uniform_at_unanimity = thresholds["uniform"].get(unanimous)

    # Three outcomes, not two. A boolean here would call "both policies failed" an
    # inversion, which is a different claim from "the targeted policy did worse than
    # the untargeted one" and would overstate the result. This repository has already
    # had one finding decided by a two-valued return; naming the third case is cheap.
    if margin_at_unanimity is None and uniform_at_unanimity is None:
        verdict = "lost"
    elif margin_at_unanimity is None:
        verdict = "inverted"
    elif uniform_at_unanimity is None:
        verdict = "retained"
    elif margin_at_unanimity > uniform_at_unanimity:
        verdict = "inverted"
    elif margin_at_unanimity < uniform_at_unanimity:
        verdict = "retained"
    else:
        verdict = "lost"

    if verdict != "retained":
        # The finding, as an event. Finding 20's advice is safe only inside its stated
        # condition, and this is the run that shows what happens outside it.
        get_logger().warning(
            "blindspot.policy_advantage_inverted",
            extra={
                "event": "blindspot.policy_advantage_inverted",
                "n_blind": unanimous,
                "verdict": verdict,
                "margin": margin_at_unanimity,
                "uniform": uniform_at_unanimity,
            },
        )
    print(
        f"\n  at {unanimous} of {args.fleet} blind: margin needs "
        f"{'more than the sweep' if margin_at_unanimity is None else margin_at_unanimity}, "
        f"uniform needs "
        f"{'more than the sweep' if uniform_at_unanimity is None else uniform_at_unanimity}"
    )

    report = {
        "provenance": run_provenance(seed=args.seed),
        "fleet": args.fleet,
        "events": args.events,
        "blind_compartment": BLIND.value,
        "affected_verdicts": check.affected,
        "affected_mean_evidence": round(check.affected_mean, 3),
        "corpus_mean_evidence": round(check.corpus_mean, 3),
        "channel_mean_with": round(check.mean_with, 3),
        "channel_mean_without": round(check.mean_without, 3),
        "shares": list(shares),
        "budgets": list(BUDGETS),
        "repaired_threshold": REPAIRED,
        "thresholds": {k: {str(n): v for n, v in d.items()} for k, d in thresholds.items()},
        "audit_hit_rate": {str(k): v for k, v in hit_rate.items()},
        "advantage_at_unanimity": verdict,
        "grid": [r.as_dict() for r in rows],
        "validity": check_sample_size(
            min((r.scored_tasks for r in rows), default=0), label="blind spot"
        ).as_dict(),
        "scored_tasks_min": min((r.scored_tasks for r in rows), default=0),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
