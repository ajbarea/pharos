#!/usr/bin/env python3
"""Which items an authority should rule on, and what happens when it is wrong too.

Finding 19 priced an authority of record and drew its anchors uniformly, on the
explicit grounds that an authority auditing the *hardest* items would be assuming the
question: knowing which items are hard is knowing where the fleet is wrong, which is
what the estimate was supposed to establish. Uniform is an honest floor and a bad
proposal, and this repository has carried "can a selection policy beat it" as the
successor question ever since.

Two things are measured here.

**Which items to rule on.** Five policies, four of them deployable and one a bound:

  uniform    finding 19's floor, drawn without regard to anything
  margin     the tasks whose vote is most evenly split. This is textbook uncertainty
             sampling, the default an experienced practitioner reaches for
  posterior  the tasks whose current EM posterior is nearest 0.5, which is the same
             instinct expressed against the estimator rather than the raw votes
  consensus  the tasks the fleet agrees on *most*. Deliberately the opposite of the
             textbook answer, and the prediction this script exists to test
  oracle     the tasks the fleet actually gets wrong. Not deployable and not proposed:
             it is handed the answer any real policy would have to guess, so it bounds
             what selection can buy rather than offering a way to buy it

**Deployability is a hard constraint, not a preference.** A policy may read only what
the aggregator can see under the protocol of finding 18: per-task vote sums and
per-task contributor counts. It may not read a per-analyst stream, because that stream
does not exist, and it may not read ground truth, because then it is the oracle. Every
policy below is annotated with which of those it uses, and the oracle is reported apart
from the rest for the same reason finding 12 reports its oracle apart from its methods.

**The prediction, stated before the run, and refuted by it.** The prediction was that
uncertainty sampling would *lose* to the policy that inverts it: the failure this line
of work is about is a wrong standard held confidently by a majority, so the items it
corrupts ought to be the items the fleet agrees on, and a budget spent where the fleet
disagrees ought to be spent where the fleet is already right.

That is wrong, and it is wrong because it conflated two different things. A wrong
majority means the *votes* on a corrupted item break the wrong way; it does not mean
the item is unanimous. Here the mistaken reviewers differ from the correct ones by one
escalation threshold, so they diverge only on boundary items --- and a boundary item is
exactly an item the fleet splits on. Disagreement is therefore not orthogonal to the
failure, it is the failure's signature. `margin` finds those items; `consensus` finds
the items both standards already agree about, which teach nothing and, worse, actively
mislead: anchoring them tells the estimator its contributors are reliable, which makes
it trust the wrong majority harder on the boundary items it never audited.

**So the transferable claim is conditional and must be stated that way.** In this
corpus `margin` selects a *subset of the items the fleet gets wrong* at every budget
that fits inside that set --- 20 of 20, 30 of 30, up to the 33 items the estimator gets
wrong, above which neither it nor the oracle can, since there is nothing left to pick
--- which is why it ties the oracle exactly. That is a
property of a corpus whose difficulty structure is discrete and known, where the hard
items and the reviewer's blind spot coincide by construction. Finding 17 already names
that coincidence. The claim that travels is *when a wrong standard manifests as
boundary disagreement, audit where the fleet splits*; the claim that does not travel is
that disagreement sampling is optimal in general.

**And an authority is not an oracle either.** Finding 19 gave the authority perfect
ground truth. Chew and Williams (arXiv:2607.15455, July 2026) make the opposite
assumption central: audit labels are themselves noisy, and disagreement among auditors
reflects genuine ambiguity rather than only random error. Their remedy is expert
adjudication of a subset; they sample the audit set by probability rather than
selecting it. So the second half of this script asks what a *fallible* authority buys,
by sweeping its error rate --- which is the first question a sponsor asks and the one
finding 19 assumed away.

Needs no model and no network.

    uv run python scripts/measure_audit_policy.py --out results/audit_policy.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pharos.analyst import Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import (
    DEPLOYABLE,
    POLICY_SEED,
    REPAIRED,
    UNIFORM_SEEDS,
    ThresholdSpread,
    baseline_errors,
    contributions_for,
    evaluate_audit,
    fleet_of,
    ladder,
    observe,
    summarize_thresholds,
    threshold,
)
from pharos.inference import (
    partition_by_contributor,
)
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200
FLEET = 9

#: Anchor budgets swept, dense at the low end because a threshold that moves at all moves
#: first where the budget is small.
#:
#: Capped below the *auditable* pool rather than the corpus: only tasks some contributor
#: reported on constrain a confusion matrix, so an anchor anywhere else buys nothing. That
#: pool is a property of the draw (83 to 99 over eight), not a constant, so the ladder is
#: truncated at run time and the artifact records both forms.
BUDGETS = (0, 2, 5, 8, 12, 20, 30, 45, 60, 80, 95)

#: Compositions worth pricing. 5 is the bare majority finding 19 repairs cheaply, 6 and
#: 7 are where its price exploded, so they are where a policy has room to help.
AUDIT_RUNGS = ("majority", "two-thirds", "seven-ninths")
COMPOSITIONS = ladder(FLEET, AUDIT_RUNGS)

#: Authority error rates swept in the second half. 0.0 reproduces finding 19.
AUTHORITY_ERROR = (0.0, 0.05, 0.1, 0.2)


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    policy: str
    n_wrong: int
    budget: int
    error: float
    scored_tasks: int
    agreement: float
    repaired: bool
    remaining_errors: int
    hits: int
    mechanical: float
    corrected: int

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "n_wrong": self.n_wrong,
            "budget": self.budget,
            "error": self.error,
            "scored_tasks": self.scored_tasks,
            "agreement": self.agreement,
            "repaired": self.repaired,
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
    compositions = ladder(args.fleet, AUDIT_RUNGS)

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=args.seed, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    # The auditable pool: tasks any contributor actually reported on. An anchor
    # anywhere else constrains no confusion matrix, so this and not the corpus size is
    # the denominator an audit budget is a fraction of.
    probe = partition_by_contributor(
        contributions_for(fleet_of(compositions[0], args.fleet), tasks, proposals, seed=args.seed)
    )
    auditable = len({t for rows in probe.values() for t, _ in rows})
    # `select` refuses a budget past the pool rather than clipping it, deliberately: a
    # threshold reported at an untested budget is worse than a missing row. So the
    # ladder is truncated here instead, and both forms are published.
    budgets = tuple(b for b in BUDGETS if b <= auditable)
    print(
        f"{len(tasks)} tasks, {auditable} auditable, fleet of {args.fleet}, "
        "scored over UNANCHORED tasks only"
    )
    if len(budgets) < len(BUDGETS):
        get_logger().warning(
            "audit.budget_ladder_truncated",
            extra={
                "event": "audit.budget_ladder_truncated",
                "auditable": auditable,
                # From the full ladder: `budgets` is already filtered by this predicate,
                # so reading it here reported an empty list on every truncation.
                "dropped": [b for b in BUDGETS if b > auditable],
            },
        )
    rows: list[Row] = []
    thresholds: dict[str, dict[int, int | None]] = {}

    for name in (*DEPLOYABLE, "oracle"):
        thresholds[name] = {}
        print(f"\n  {name}")
        print(f"    {'wrong':>6}" + "".join(f"{b:>7}" for b in budgets))
        print("    " + "-" * (6 + 7 * len(budgets)))
        for n_wrong in compositions:
            if n_wrong > args.fleet:
                continue
            flat = contributions_for(
                fleet_of(n_wrong, args.fleet), tasks, proposals, seed=args.seed
            )
            partitioned = partition_by_contributor(flat)
            view = observe(partitioned)
            baseline = baseline_errors(partitioned, truth)

            cells, block = [], []
            for budget in budgets:
                out = evaluate_audit(
                    partitioned,
                    view,
                    truth,
                    policy=name,
                    budget=budget,
                    error=0.0,
                    baseline=baseline,
                )
                agreement = out.agreement
                row = Row(
                    policy=name,
                    n_wrong=n_wrong,
                    budget=budget,
                    error=0.0,
                    scored_tasks=out.scored,
                    agreement=agreement,
                    repaired=agreement >= REPAIRED,
                    remaining_errors=out.remaining_errors,
                    hits=out.hits,
                    mechanical=out.mechanical,
                    corrected=out.corrected,
                )
                rows.append(row)
                block.append(row)
                cells.append(f"{agreement:>7.3f}")
                record("audit.cell", agreement, policy=name, n_wrong=n_wrong, budget=budget)
            thresholds[name][n_wrong] = threshold(block)
            print(f"    {n_wrong:>6}" + "".join(cells))

    # The baseline's threshold is a draw, and the targeted policies' are not. Comparing
    # one sample against an exact number credits a policy with whatever margin the draw
    # happened to supply, which is not a small worry here: on the same corpus finding 19
    # measured this quantity ranging 2 to 30 across draws. So the baseline is re-run per
    # seed and reported with its spread; the row below is unchanged, and is now the
    # median draw rather than the only one.
    uniform_spread: dict[int, ThresholdSpread] = {}
    print(f"\n  uniform baseline over {len(UNIFORM_SEEDS)} draws")
    print(f"    {'wrong':>6}  median   range        reached")
    print("    " + "-" * 44)
    for n_wrong in compositions:
        flat = contributions_for(fleet_of(n_wrong, args.fleet), tasks, proposals, seed=args.seed)
        partitioned = partition_by_contributor(flat)
        view = observe(partitioned)
        baseline = baseline_errors(partitioned, truth)
        draws: list[int | None] = []
        for seed in UNIFORM_SEEDS:
            block: list[Row] = []
            for budget in budgets:
                out = evaluate_audit(
                    partitioned,
                    view,
                    truth,
                    policy="uniform",
                    budget=budget,
                    error=0.0,
                    baseline=baseline,
                    seed=seed,
                )
                block.append(
                    Row(
                        policy="uniform",
                        n_wrong=n_wrong,
                        budget=budget,
                        error=0.0,
                        scored_tasks=out.scored,
                        agreement=out.agreement,
                        repaired=out.agreement >= REPAIRED,
                        remaining_errors=out.remaining_errors,
                        hits=out.hits,
                        mechanical=out.mechanical,
                        corrected=out.corrected,
                    )
                )
            draws.append(threshold(block))
        spread = summarize_thresholds(n_wrong, draws)
        uniform_spread[n_wrong] = spread
        # The comparison row, and everything chosen from it, uses the median draw. The
        # printed uniform grid above is still one draw, the same way finding 19 prints
        # one seed's agreement grid beside a median over 21.
        thresholds["uniform"][n_wrong] = spread.median
        span = f"{spread.lowest}-{spread.highest}" if spread.reached else "—"
        median = "none" if spread.median is None else str(spread.median)
        print(f"    {n_wrong:>6}  {median:>6}   {span:>10}   {spread.reached} of {spread.seeds}")
        record(
            "audit.uniform_spread",
            float(spread.median if spread.median is not None else -1),
            n_wrong=n_wrong,
            reached=spread.reached,
        )

    print("\n  budget to repair, by policy (lower is better)")
    print(f"    {'wrong':>6}" + "".join(f"{p:>12}" for p in (*DEPLOYABLE, "oracle")))
    print("    " + "-" * (6 + 12 * (len(DEPLOYABLE) + 1)))
    for n_wrong in compositions:
        cells = []
        for name in (*DEPLOYABLE, "oracle"):
            value = thresholds[name].get(n_wrong)
            cells.append(f"{'none' if value is None else value:>12}")
        print(f"    {n_wrong:>6}" + "".join(cells))

    # Which deployable policy to carry into the fallibility sweep. Chosen from the
    # measurement rather than named in advance, so a corpus change that reordered the
    # policies would move this too instead of leaving a stale winner hard-coded.
    def total_cost(name: str) -> tuple[int, int]:
        values = [thresholds[name].get(n) for n in compositions]
        unrepaired = sum(1 for v in values if v is None)
        return unrepaired, sum(v for v in values if v is not None)

    best = min(DEPLOYABLE, key=total_cost)
    uniform_cost, best_cost = total_cost("uniform"), total_cost(best)
    print(f"\n  best deployable policy: {best}")
    if best != "uniform" and best_cost < uniform_cost:
        get_logger().warning(
            "audit.policy_beats_uniform",
            extra={"event": "audit.policy_beats_uniform", "policy": best, "against": "uniform"},
        )

    print(f"\n  a fallible authority, using {best}")
    print(f"    {'wrong':>6}" + "".join(f"{e:>9}" for e in AUTHORITY_ERROR))
    print("    " + "-" * (6 + 9 * len(AUTHORITY_ERROR)))
    fallible: dict[int, dict[str, int | None]] = {}
    for n_wrong in compositions:
        flat = contributions_for(fleet_of(n_wrong, args.fleet), tasks, proposals, seed=args.seed)
        partitioned = partition_by_contributor(flat)
        view = observe(partitioned)
        baseline = baseline_errors(partitioned, truth)
        cells, per_error = [], {}
        for error in AUTHORITY_ERROR:
            block = []
            for budget in budgets:
                out = evaluate_audit(
                    partitioned,
                    view,
                    truth,
                    policy=best,
                    budget=budget,
                    error=error,
                    baseline=baseline,
                )
                row = Row(
                    policy=best,
                    n_wrong=n_wrong,
                    budget=budget,
                    error=error,
                    scored_tasks=out.scored,
                    agreement=out.agreement,
                    repaired=out.agreement >= REPAIRED,
                    remaining_errors=out.remaining_errors,
                    hits=out.hits,
                    mechanical=out.mechanical,
                    corrected=out.corrected,
                )
                rows.append(row)
                block.append(row)
            value = threshold(block)
            per_error[str(error)] = value
            cells.append(f"{'none' if value is None else value:>9}")
            record("audit.fallible", float(value or -1), n_wrong=n_wrong, error=error)
        fallible[n_wrong] = per_error
        print(f"    {n_wrong:>6}" + "".join(cells))

    report = {
        "provenance": run_provenance(seed=args.seed),
        "fleet": args.fleet,
        "events": args.events,
        "budgets": list(budgets),
        "budgets_requested": list(BUDGETS),
        "auditable_pool": auditable,
        "compositions": list(compositions),
        "authority_error": list(AUTHORITY_ERROR),
        "repaired_threshold": REPAIRED,
        "policy_seed": POLICY_SEED,
        "uniform_seeds": list(UNIFORM_SEEDS),
        "deployable": list(DEPLOYABLE),
        # `thresholds["uniform"]` is the median over `uniform_seeds`; every other policy
        # is deterministic given the aggregate and has one exact value.
        "thresholds": {k: {str(n): v for n, v in d.items()} for k, d in thresholds.items()},
        "uniform_spread": {str(n): s.as_dict() for n, s in uniform_spread.items()},
        "best_deployable": best,
        "fallible_authority": {str(k): v for k, v in fallible.items()},
        "grid": [r.as_dict() for r in rows],
        # The corpus is 200; the estimator covers ~97 and anchoring shrinks that
        # further, to 2 in the worst published cell. Scoring validity on len(tasks)
        # reported quotable: true for thresholds resting on a handful of tasks, which
        # is exactly what this gate exists to prevent.
        "validity": check_sample_size(
            min((r.scored_tasks for r in rows), default=0), label="audit policy"
        ).as_dict(),
        "scored_tasks_min": min((r.scored_tasks for r in rows), default=0),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
