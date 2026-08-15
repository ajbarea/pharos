"""Selecting which items to act on, from the aggregator's view and nothing else.

Five policies and a bound. Each is a function of a `ServerObservation`, which is the whole
point: a policy that needs more than the view is not deployable, and making the permitted
input a type is how that stays true when somebody adds a sixth.

These were defined in `scripts/measure_audit_policy.py` and imported from there by the
blind-spot, abstention and shape measurements. The selection rule is a claim this work
makes about what a deployment can do, so it belongs in the package that work ships.
"""

from collections.abc import Callable, Sequence
from random import Random

from pharos.governance.view import ServerObservation

__all__ = [
    "DEPLOYABLE",
    "POLICIES",
    "POLICY_SEED",
    "UNIFORM_SEEDS",
    "Policy",
    "choose_anchors",
    "policy_channel",
    "policy_consensus",
    "policy_deviation",
    "policy_margin",
    "policy_oracle",
    "policy_posterior",
    "policy_shortfall",
    "select",
]

#: Seed for the authority's own slips, and for uniform selection.
POLICY_SEED = 4242

#: Draws of the uniform baseline. The targeted policies are deterministic given the
#: aggregate, so their thresholds are exact; `uniform` is a sample, and comparing an exact
#: number against one sample of a variable one is how a policy gets credited with a margin
#: the draw supplied. Finding 19 measured that spread on this corpus and it is wide -- the
#: bare-majority price ranged 2 to 30 items over 21 draws -- so every comparison against it
#: uses the same 21.
UNIFORM_SEEDS = tuple(POLICY_SEED + i for i in range(21))


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


def policy_deviation(view: ServerObservation, _truth: dict[str, bool]) -> dict[str, float]:
    """Localization without a partition: audit where the vote sum deviates from its stratum.

    The one policy here that needs neither a channel to name nor a disagreement to read.
    Finding 29's index of dispersion is a *sum* of squared within-stratum residuals, and it
    detects a shared error from vote sums alone. This asks the obvious next question: the
    summands are per task, so ranking by them should point at the tasks the shared standard
    corrupts, and finding 28's remedy would no longer need a name to withhold by.

    Within an evidence stratum every analyst applying one rule votes the same way, so the
    stratum's rate is what the fleet does on tasks of that difficulty and a task far from
    it is a task something else acted on. Reads per-task vote sums, contributor counts and
    the public evidence count -- the same three inputs as the index, and strictly less than
    `policy_channel` needs.

    Ties break toward the lower task id, as every policy here does, so a budget sweep stays
    nested.
    """
    if not view.evidence:
        # No strata, no residual. Scoring every task equally rather than pooling them into
        # one stratum: an unstratified residual is a different statistic, and reporting it
        # under this name is how a policy gets credited for a comparison it did not make.
        return dict.fromkeys(view.posterior, 1.0)

    totals: dict[int, tuple[float, float]] = {}
    for task in view.posterior:
        level = view.evidence.get(task, -1)
        votes, seen = totals.get(level, (0.0, 0.0))
        totals[level] = (votes + view.votes.get(task, 0.0), seen + view.seen.get(task, 0.0))

    scores: dict[str, float] = {}
    for task in view.posterior:
        level = view.evidence.get(task, -1)
        votes, seen = totals[level]
        rate = votes / seen if seen else 0.0
        expected = view.seen.get(task, 0.0) * rate
        # Negated: `select` audits the lowest scores, and the candidates here are the
        # tasks furthest from what their stratum did.
        scores[task] = -abs(view.votes.get(task, 0.0) - expected)
    return scores


def policy_shortfall(view: ServerObservation, _truth: dict[str, bool]) -> dict[str, float]:
    """`policy_deviation` read one-sided: audit where the stratum votes *low*, not far.

    The two-sided form is what the index of dispersion literally sums, and it carries the
    index's blindness to direction with it: once the corrupted tasks are the majority of
    their stratum, they set the rate and the *clean* tasks become the outliers, so the rule
    inverts. Signing it removes that, because a blind spot is a failure to credit evidence
    and a reviewer who credits less escalates less. The corrupted tasks vote low whether
    they are two of a stratum or fifty.

    That is a directional assumption and it is the scope condition on everything this
    policy buys. It is not a new one: finding 22's detector is already one-sided, for
    exactly this reason and stated in exactly these terms. A shared error that *raised* the
    rate -- a fleet crediting something it should not -- would need the mirror of this rule,
    and reading which of the two a fleet has is not something this signal supplies.
    """
    if not view.evidence:
        return dict.fromkeys(view.posterior, 1.0)

    totals: dict[int, tuple[float, float]] = {}
    for task in view.posterior:
        level = view.evidence.get(task, -1)
        votes, seen = totals.get(level, (0.0, 0.0))
        totals[level] = (votes + view.votes.get(task, 0.0), seen + view.seen.get(task, 0.0))

    scores: dict[str, float] = {}
    for task in view.posterior:
        votes, seen = totals[view.evidence.get(task, -1)]
        rate = votes / seen if seen else 0.0
        # Not negated, unlike `policy_deviation`: `select` audits the lowest scores, and
        # the candidates here are the tasks that fell furthest *below* their stratum.
        scores[task] = view.votes.get(task, 0.0) - view.seen.get(task, 0.0) * rate
    return scores


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
    "deviation": policy_deviation,
    "margin": policy_margin,
    "shortfall": policy_shortfall,
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
#:
#: `deviation` is absent for a third reason, and it is the one worth stating: it *is*
#: deployable, needing no prior detection at all. What this tuple actually names is the
#: policy set findings 19 through 28 proposed, which four committed artifacts are measured
#: against. Adding a fifth member here would silently move all four. Finding 30 names its
#: own list instead, and this constant stays the historical one.
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


def choose_anchors(task_ids: Sequence[str], count: int, *, seed: int) -> tuple[str, ...]:
    """Which tasks the authority rules on, drawn without regard to difficulty.

    A uniform draw on purpose. An authority that audited the *hardest* items would
    look better and would be assuming the thing in question: knowing which items are
    hard is knowing where the fleet is wrong, which is what the estimate was supposed
    to establish. Uniform is the honest floor, and a targeted policy can only beat it.

    Nested across budgets, which `random.sample` is not. One shuffled order is drawn
    and sliced, so the draw at budget `b` is a subset of the draw at any larger budget
    -- the same property `select` gives the targeted policies by construction, and the
    reason `tests/test_audit_policy.py` could only assert nesting for those. A fresh
    `sample` per budget returns a uniform subset of the right size, so each cell was
    individually honest, but the sweep then moved two things at once: a threshold read
    off it could sit where a *different set of items* was audited rather than where
    more of them were. Slicing one order isolates the budget, and each prefix is still
    a uniform random subset of its size.
    """
    if not count:
        return ()
    order = list(task_ids)
    # `sample` raised on an oversized count; a slice would quietly return fewer and
    # report a budget that was never spent, so the loudness is kept explicitly.
    if count > len(order):
        raise ValueError(f"anchor budget {count} exceeds the {len(order)} available tasks")
    Random(seed).shuffle(order)  # noqa: S311
    return tuple(sorted(order[:count]))
