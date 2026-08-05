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
from dataclasses import dataclass
from pathlib import Path

from measure_audit_policy import DEPLOYABLE, POLICY_SEED, evaluate, observe, select
from measure_authority_anchors import REPAIRED
from measure_secure_reliability import contributions_for

from pharos.analyst import AnalystPolicy
from pharos.disclosure import DROP_COMPARTMENTS, KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.inference import partition_by_contributor
from pharos.labels import Compartment, declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200
FLEET = 9

#: The channel the blind fleet discounts. Chosen for orthogonality to difficulty and
#: asserted below rather than trusted, because the whole experiment is void if the
#: replacement confound is the one it was built to remove.
BLIND = Compartment.PARTNER

#: How much of the fleet carries the blind spot. The interesting end is the top: a
#: sighted minority is what keeps the disagreement signal alive.
SHARES = (0, 3, 5, 7, 8, 9)

#: Budgets swept. Capped under the auditable pool, as in finding 20.
BUDGETS = (0, 2, 5, 8, 12, 20, 30, 45, 60, 80, 95)

#: How far the mean evidence on blind-affected tasks may sit from the corpus mean
#: before the channel is judged to have reintroduced the difficulty confound. The
#: affected tasks here sit *above* the mean, which is the opposite of the boundary
#: items a threshold error hits, so the guard is two-sided and generous.
ORTHOGONALITY_SLACK = 1.5


def blind_fleet(n_blind: int, size: int) -> tuple[AnalystPolicy, ...]:
    """A fleet holding the correct threshold, `n_blind` of them discounting a channel.

    Every reviewer here applies the world's own rule. That is the point: this is not a
    fleet of over-escalators, it is a fleet of careful analysts who share one habit.

    **Compartment shedding is required, not incidental.** Under the fail-closed default
    every one of the affected tasks is escalated on disclosure grounds and contributes
    no verdict, so the blind spot reaches the aggregator on exactly zero tasks and the
    experiment silently measures nothing --- which is what the first run of this script
    did. The cause is a confound worth naming: blinding a compartment selects tasks
    *carrying* that compartment, and carrying a compartment is what makes a task
    unreleasable, so the blind spot was perfectly correlated with the release gate.
    Under the shedding ruling of finding 2 all 200 tasks are observed and all 20
    affected verdicts enter the stream. Finding 18's readership measurement runs under
    the same ruling for the same structural reason.
    """
    sighted = [
        AnalystPolicy(f"sighted-{i}", release_policy=DROP_COMPARTMENTS)
        for i in range(size - n_blind)
    ]
    blind = [
        AnalystPolicy(f"blind-{i}", blind_compartment=BLIND, release_policy=DROP_COMPARTMENTS)
        for i in range(n_blind)
    ]
    return tuple(sighted + blind)


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    n_blind: int
    policy: str
    budget: int
    agreement: float
    repaired: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "n_blind": self.n_blind,
            "policy": self.policy,
            "budget": self.budget,
            "agreement": self.agreement,
            "repaired": self.repaired,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    from pharos.analyst import Proposal, evidence_shown

    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    # The orthogonality claim, asserted rather than trusted. If the affected slice is
    # picked out by difficulty after all, this experiment measures the same confound it
    # was built to escape and its result would be meaningless in a way no later check
    # would reveal.
    reference = AnalystPolicy("reference")
    blind_reviewer = AnalystPolicy("blind", blind_compartment=BLIND)
    affected = [
        t
        for t in tasks
        if blind_reviewer.evidence_visible_to(t) != reference.evidence_visible_to(t)
        and (len(blind_reviewer.evidence_visible_to(t)) >= reference.escalation_threshold)
        != (len(reference.evidence_visible_to(t)) >= reference.escalation_threshold)
    ]
    corpus_mean = sum(len(evidence_shown(t)) for t in tasks) / len(tasks)
    affected_mean = (
        sum(len(evidence_shown(t)) for t in affected) / len(affected) if affected else 0.0
    )
    if not affected:
        get_logger().error(
            "blindspot.no_effect",
            extra={"event": "blindspot.no_effect", "compartment": BLIND.value},
        )
        raise SystemExit(
            f"a fleet-wide {BLIND.value} blind spot changes no verdict on this corpus; "
            "there is nothing to measure"
        )
    if abs(affected_mean - corpus_mean) > ORTHOGONALITY_SLACK:
        get_logger().warning(
            "blindspot.difficulty_confounded",
            extra={
                "event": "blindspot.difficulty_confounded",
                "affected_mean": round(affected_mean, 3),
                "corpus_mean": round(corpus_mean, 3),
            },
        )

    print(
        f"{len(tasks)} tasks, fleet of {args.fleet}, blind spot on {BLIND.value}: "
        f"{len(affected)} verdicts change, mean evidence {affected_mean:.2f} "
        f"against a corpus mean of {corpus_mean:.2f}"
    )

    rows: list[Row] = []
    thresholds: dict[str, dict[int, int | None]] = {p: {} for p in (*DEPLOYABLE, "oracle")}
    hit_rate: dict[int, dict[str, float]] = {}

    for n_blind in SHARES:
        if n_blind > args.fleet:
            continue
        flat = contributions_for(blind_fleet(n_blind, args.fleet), tasks, proposals, seed=SEED)
        partitioned = partition_by_contributor(flat)
        view = observe(partitioned)
        wrong = {t for t, p in view.posterior.items() if (p >= 0.5) != truth[t]}

        # The mechanism, measured directly rather than inferred from the outcome: what
        # share of a policy's picks are actually corrupted items. This is the number
        # that explains the thresholds below, and in finding 20 it was 100%.
        hit_rate[n_blind] = {}
        for name in (*DEPLOYABLE, "oracle"):
            picked = set(select(name, view, truth, 20, seed=POLICY_SEED))
            hit_rate[n_blind][name] = round(len(picked & wrong) / 20, 3) if wrong else 0.0

        for name in (*DEPLOYABLE, "oracle"):
            found: int | None = None
            for budget in BUDGETS:
                agreement, _ = evaluate(
                    partitioned, view, truth, policy=name, budget=budget, error=0.0
                )
                repaired = agreement >= REPAIRED
                rows.append(
                    Row(
                        n_blind=n_blind,
                        policy=name,
                        budget=budget,
                        agreement=agreement,
                        repaired=repaired,
                    )
                )
                if repaired and found is None:
                    found = budget
            thresholds[name][n_blind] = found
            record("blindspot.threshold", float(found or -1), policy=name, n_blind=n_blind)

    order = (*DEPLOYABLE, "oracle")
    print(f"\n  budget to repair (lower is better), '-' = not reached within {max(BUDGETS)}")
    print(f"    {'blind':>6}" + "".join(f"{p:>12}" for p in order))
    print("    " + "-" * (6 + 12 * len(order)))
    for n_blind in SHARES:
        if n_blind > args.fleet:
            continue
        cells = []
        for name in order:
            value = thresholds[name].get(n_blind)
            cells.append(f"{'-' if value is None else value:>12}")
        print(f"    {n_blind:>6}" + "".join(cells))

    print("\n  share of a 20-item audit that lands on a corrupted task")
    print(f"    {'blind':>6}" + "".join(f"{p:>12}" for p in order))
    print("    " + "-" * (6 + 12 * len(order)))
    for n_blind in SHARES:
        if n_blind > args.fleet:
            continue
        print(f"    {n_blind:>6}" + "".join(f"{hit_rate[n_blind][p]:>12.2f}" for p in order))

    unanimous = max(SHARES)
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
        "provenance": run_provenance(seed=SEED),
        "fleet": args.fleet,
        "events": args.events,
        "blind_compartment": BLIND.value,
        "affected_verdicts": len(affected),
        "affected_mean_evidence": round(affected_mean, 3),
        "corpus_mean_evidence": round(corpus_mean, 3),
        "shares": list(SHARES),
        "budgets": list(BUDGETS),
        "repaired_threshold": REPAIRED,
        "thresholds": {k: {str(n): v for n, v in d.items()} for k, d in thresholds.items()},
        "audit_hit_rate": {str(k): v for k, v in hit_rate.items()},
        "advantage_at_unanimity": verdict,
        "grid": [r.as_dict() for r in rows],
        "validity": check_sample_size(len(tasks), label="blind spot").as_dict(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
