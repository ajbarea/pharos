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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from measure_authority_anchors import (
    REPAIRED,
    ThresholdSpread,
    choose_anchors,
    ladder,
    summarize_thresholds,
)
from measure_secure_reliability import MASK_SEED, contributions_for, fleet_of

from pharos.analyst import Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.inference import (
    agreement_with,
    federated_dawid_skene,
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

#: Anchor budgets swept. Denser than finding 19's at the low end, because the question
#: here is whether a policy moves the *threshold*, and a threshold that moves at all
#: moves first where the budget is small.
#:
#: Capped below the *auditable* pool rather than the corpus. Only about half of the 200
#: tasks are ever observed by the aggregator at these compositions -- the rest were
#: rejected by every reviewer, so no contributor's confusion matrix touches them and an
#: anchor placed there constrains nothing. Finding 19 drew uniformly from all 200 and
#: therefore spent about half its budget on tasks that could not repay it, which is
#: worth knowing before reading its 50% threshold as the price of an audit.
#:
#: The ladder is truncated at run time to the pool this corpus actually has, and the
#: artifact records both. The pool is a property of the draw, not a constant: it is 97
#: at the committed seed and ranges 83 to 99 over eight draws. Hard-coding the top rung
#: at 95 made this script exit non-zero on four of those eight, which is the same defect
#: as the absolute composition constants -- a sweepable dimension with a constant sized
#: for one draw of it.
BUDGETS = (0, 2, 5, 8, 12, 20, 30, 45, 60, 80, 95)

#: Compositions worth pricing. 5 is the bare majority finding 19 repairs cheaply, 6 and
#: 7 are where its price exploded, so they are where a policy has room to help.
AUDIT_RUNGS = ("majority", "two-thirds", "seven-ninths")
COMPOSITIONS = ladder(FLEET, AUDIT_RUNGS)

#: Authority error rates swept in the second half. 0.0 reproduces finding 19.
AUTHORITY_ERROR = (0.0, 0.05, 0.1, 0.2)

#: Seed for the authority's own slips, and for uniform selection.
POLICY_SEED = 4242

#: Draws of the uniform baseline. The targeted policies are deterministic given the
#: aggregate, so their thresholds are exact; `uniform` is a sample, and comparing an
#: exact number against one sample of a variable one is how a policy gets credited with
#: a margin the draw supplied. Finding 19 measured that spread on the same corpus and
#: it is wide -- the bare-majority price ranged 2 to 30 items over 21 draws -- so the
#: baseline column here is reported the same way, as a median with its range.
UNIFORM_SEEDS = tuple(POLICY_SEED + i for i in range(21))


@dataclass(frozen=True, slots=True)
class ServerObservation:
    """Exactly what a selection policy is allowed to read.

    A dataclass rather than a loose dict because the deployability constraint is the
    substance of this measurement: a policy that reaches past these fields is reading
    something the aggregator does not have, and the result would be an oracle wearing a
    method's name. Finding 12 made that mistake in the other direction and had to be
    corrected; making the permitted view a type is how this one avoids it.

    `votes` and `seen` are the two per-task sums the protocol already reveals.
    `posterior` is the estimator's own output, which the server holds by construction.
    `evidence` and `carries` are public corpus structure -- how much evidence a task
    shows, and whether it carries a named channel -- which finding 22's detector
    already reads and is deployable for reading. They are empty unless a caller
    supplies them, and `observe` never does.

    What must never appear is a per-analyst field. That is the line, and it is asserted
    on `__slots__` in the tests rather than left to this docstring.
    """

    votes: dict[str, float]
    seen: dict[str, float]
    posterior: dict[str, float]
    #: How much evidence each task shows, which is public corpus structure and the
    #: same quantity finding 22's detector conditions on. Empty where unused.
    evidence: dict[str, int] = field(default_factory=dict)
    #: Which tasks carry the channel finding 22's detector named, when it has fired.
    #: Empty until then, and empty is the honest default: a policy may not select by
    #: provenance on a fleet nobody has shown to be channel-blind.
    #:
    #: Reading this does NOT widen what a policy may see. Finding 22 established that
    #: the public structure of the corpus is available to a deployment -- its detector
    #: reads exactly this, conditions the verdict rate on it, and is deployable for
    #: that reason. What stays forbidden is ground truth and the per-analyst stream.
    carries: dict[str, bool] = field(default_factory=dict)

    def margin(self, task: str) -> float:
        """How evenly the fleet split on this task. 0.0 is a dead heat."""
        n = self.seen.get(task, 0.0)
        if not n:
            return 1.0
        return abs(2.0 * self.votes.get(task, 0.0) - n) / n


def observe(partitioned: dict[str, list[tuple[str, bool]]]) -> ServerObservation:
    """The aggregator's view, assembled the way finding 18's protocol produces it."""
    votes: dict[str, float] = {}
    seen: dict[str, float] = {}
    for rows in partitioned.values():
        for task, verdict in rows:
            votes[task] = votes.get(task, 0.0) + (1.0 if verdict else 0.0)
            seen[task] = seen.get(task, 0.0) + 1.0
    estimate = federated_dawid_skene(partitioned, seed=MASK_SEED)
    return ServerObservation(votes, seen, dict(estimate.posterior))


#: A policy scores every task; the `count` lowest scores are audited. Ordering ties by
#: task id keeps the draw reproducible, which a sweep over budgets needs -- a policy
#: whose 8-item selection is not a superset of its 5-item selection would confound
#: "more anchors" with "different anchors".
Policy = Callable[[ServerObservation, dict[str, bool]], dict[str, float]]


def policy_channel(view: ServerObservation, _truth: dict[str, bool]) -> dict[str, float]:
    """Audit the channel the detector named, which is finding 22's answer to finding 21.

    Every other deployable policy here reads *disagreement*, and a unanimously blind
    fleet has none -- which is why they all fall to chance in finding 21. This one reads
    provenance instead. It selects tasks carrying the discounted channel, breaking ties
    toward the ones the fleet split on, and it is only usable at all because finding 22
    supplies the channel name from data the aggregator already holds.

    That is the pairing: the detector says *which regime* and *which channel*, and this
    turns the second half into a selection rule. It is not a better estimator and not a
    better uncertainty signal; it is the observation that once provenance is known, the
    corrupted slice can be addressed by provenance rather than found by disagreement.
    """
    if not view.carries:
        # No detection, no licence to select this way. Scoring every task equally would
        # silently degrade to an arbitrary draw wearing a policy's name.
        return dict.fromkeys(view.posterior, 1.0)

    # Carrying the channel is necessary and not sufficient, which the first version of
    # this policy demonstrated: selecting on provenance alone landed 0.50 of its audit
    # on corrupted tasks against uniform's 0.10 -- a real gain, and half the budget
    # still spent on tasks a blind analyst never had cause to change.
    #
    # A blind analyst only flips a verdict where the discounted channel was doing the
    # work, and in this corpus that is the high-evidence end: the affected slice sits at
    # 3.00 defining facts against a corpus mean of 1.75. So order by evidence within the
    # carrying set. Evidence count is public corpus structure, the same quantity finding
    # 22's detector already conditions on, so this reads nothing new.
    if not view.evidence:
        return {
            task: (0.0 if view.carries.get(task) else 1.0) + view.margin(task) * 1e-3
            for task in view.posterior
        }
    deepest = max(view.evidence.values(), default=0) or 1
    return {
        task: (0.0 if view.carries.get(task) else 1.0)
        + (1.0 - view.evidence.get(task, 0) / deepest)
        for task in view.posterior
    }


def policy_margin(view: ServerObservation, _truth: dict[str, bool]) -> dict[str, float]:
    """Textbook uncertainty sampling: audit where the fleet is most split."""
    return {task: view.margin(task) for task in view.posterior}


def policy_posterior(view: ServerObservation, _truth: dict[str, bool]) -> dict[str, float]:
    """The same instinct against the estimator: audit where the posterior is nearest 0.5."""
    return {task: abs(p - 0.5) for task, p in view.posterior.items()}


def policy_consensus(view: ServerObservation, _truth: dict[str, bool]) -> dict[str, float]:
    """The inversion: audit where the fleet agrees most. Negated margin, so it sorts."""
    return {task: -view.margin(task) for task in view.posterior}


def policy_oracle(view: ServerObservation, truth: dict[str, bool]) -> dict[str, float]:
    """A bound, not a method: audit exactly the tasks the fleet currently gets wrong.

    Reads ground truth, so it is not deployable and is never proposed. It is here to
    say how much of the gap between uniform and perfect any selection could close, so
    that a policy scoring near it is known to be near the ceiling rather than merely
    better than the floor.
    """
    return {task: 0.0 if (p >= 0.5) != truth[task] else 1.0 for task, p in view.posterior.items()}


POLICIES: dict[str, Policy] = {
    "channel": policy_channel,
    "margin": policy_margin,
    "posterior": policy_posterior,
    "consensus": policy_consensus,
    "oracle": policy_oracle,
}

#: Which policies may be proposed *in this regime*. Two names in `POLICIES` are absent
#: for different reasons, and both matter. `oracle` is excluded by construction:
#: reporting it inside the same set would let a summary line quote a number no
#: deployment can have. `channel` is excluded because it needs finding 22's detector to
#: have named a channel first, which has not happened in a fleet holding a *threshold*
#: error -- there is no channel to name, so it would degrade to an arbitrary draw
#: wearing a policy's name. `measure_blind_spot` adds it back where the detector fires.
DEPLOYABLE = ("uniform", "margin", "posterior", "consensus")


def select(
    name: str,
    view: ServerObservation,
    truth: dict[str, bool],
    count: int,
    *,
    seed: int,
) -> tuple[str, ...]:
    """The `count` tasks `name` would have the authority rule on."""
    pool = sorted(view.posterior)
    if count <= 0:
        return ()
    if count > len(pool):
        # Loud rather than clipped. A budget past the auditable pool is a question the
        # sweep cannot answer, and silently returning the whole pool would report a
        # threshold as if it had been reached at a budget that was never tested.
        raise ValueError(
            f"budget {count} exceeds the {len(pool)} auditable tasks; "
            "lower BUDGETS or widen the corpus"
        )
    if name == "uniform":
        return choose_anchors(pool, count, seed=seed)
    scored = POLICIES[name](view, truth)
    return tuple(sorted(sorted(scored, key=lambda t: (scored[t], t))[:count]))


def authority_ruling(
    anchored: Sequence[str], truth: dict[str, bool], *, error: float, seed: int
) -> dict[str, bool]:
    """What the authority actually asserts, which is the truth only when it is right.

    Finding 19 assumed error 0. Chew and Williams (arXiv:2607.15455) build their method
    on the opposite assumption, and a sponsor asks this question before any other, so it
    is swept rather than assumed.
    """
    if error <= 0.0:
        return {task: truth[task] for task in anchored}
    rng = Random((seed, round(error * 1000)).__hash__())  # noqa: S311
    return {task: (not truth[task]) if rng.random() < error else truth[task] for task in anchored}


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


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one audit budget actually bought, separated from what it merely hid.

    Agreement over unanchored tasks was finding 19's scoring rule and it has a failure
    the rule was not designed for. Excluding an anchored task removes it from the
    denominator, so anchoring a *wrong* item raises agreement **without correcting
    anything**. A policy that targets errors perfectly therefore climbs toward 1.000 by
    deletion alone, and both findings 19 and 20 read that climb as repair.

    `mechanical` is what the score would be if not one unanchored label changed:
    the baseline errors minus the ones anchored away, over the surviving pool. So
    `corrected` -- baseline errors, less those anchored away, less those still wrong --
    is the only number here that counts a label the authority actually fixed.
    """

    agreement: float
    scored: int
    remaining_errors: int
    hits: int
    mechanical: float
    corrected: int

    @property
    def genuine(self) -> bool:
        """Whether any unanchored label was actually corrected."""
        return self.corrected > 0


def baseline_errors(
    partitioned: dict[str, list[tuple[str, bool]]], truth: dict[str, bool]
) -> tuple[int, int]:
    """The pool the estimator covers, and how many of it it gets wrong, at zero anchors.

    The pool is *not* the corpus. The estimator only produces labels for tasks some
    contributor reported on, and every rate here has to be read against that.
    """
    estimate = federated_dawid_skene(partitioned, seed=MASK_SEED)
    labels = estimate.labels()
    return len(labels), sum(1 for task, value in labels.items() if value != truth[task])


def evaluate(
    partitioned: dict[str, list[tuple[str, bool]]],
    view: ServerObservation,
    truth: dict[str, bool],
    *,
    policy: str,
    budget: int,
    error: float,
    baseline: tuple[int, int] | None = None,
    seed: int = POLICY_SEED,
) -> Outcome:
    """What one policy at one budget bought, and what it only appeared to buy.

    `baseline` is `(pool, errors)` at zero anchors, from `baseline_errors`. It is the
    reference the mechanical score is computed against; passed in rather than recomputed
    because it is the same for every budget of a given fleet and costs an EM run.

    `seed` moves the uniform draw and the authority's slips together, which is what a
    replication of this cell means: the targeted policies ignore it entirely, because
    their selection is a function of the aggregate rather than of chance.
    """
    anchored = select(policy, view, truth, budget, seed=seed)
    anchors = authority_ruling(anchored, truth, error=error, seed=seed)
    estimate = federated_dawid_skene(partitioned, seed=MASK_SEED, anchors=anchors)
    scored = {t: v for t, v in truth.items() if t not in anchors}
    labels = {t: v for t, v in estimate.labels().items() if t not in anchors}
    agreement = round(agreement_with(labels, scored), 4)
    remaining = sum(1 for task, value in labels.items() if value != scored[task])

    if baseline is None:
        baseline = baseline_errors(partitioned, truth)
    pool, errors = baseline
    # How many of the anchors landed on a task the estimator was getting wrong. Those
    # are the ones whose removal flatters the score without changing an estimate.
    zero = federated_dawid_skene(partitioned, seed=MASK_SEED).labels()
    hits = sum(1 for task in anchors if task in zero and zero[task] != truth[task])
    surviving = pool - sum(1 for task in anchors if task in zero)
    mechanical = round((surviving - (errors - hits)) / surviving, 4) if surviving else 0.0
    return Outcome(
        agreement=agreement,
        scored=len(labels),
        remaining_errors=remaining,
        hits=hits,
        mechanical=mechanical,
        corrected=(errors - hits) - remaining,
    )


def threshold(rows: Sequence[Row]) -> int | None:
    """Smallest budget that clears the bar AND actually corrected something.

    `repaired` alone is a score crossing, and a score can cross by deletion: anchoring
    a wrong task removes it from the denominator. Requiring `corrected > 0` is what
    makes this a threshold for repair rather than for successful targeting.
    """
    return next(
        (r.budget for r in sorted(rows, key=lambda r: r.budget) if r.repaired and r.corrected > 0),
        None,
    )


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
                out = evaluate(
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
                out = evaluate(
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
                out = evaluate(
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
