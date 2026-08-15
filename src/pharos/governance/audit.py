"""Scoring an audit: what a budget bought, and what it only appeared to buy.

Agreement over unanchored tasks was the original scoring rule and it has a failure the
rule was not designed for. Excluding an anchored task removes it from the denominator, so
anchoring a *wrong* item raises agreement without correcting anything, and two findings
read that climb as repair before it was caught. `AuditOutcome` separates the two: `mechanical`
is what the score would be if not one unanchored label changed, and `corrected` is the
only number here that counts a label an authority actually fixed.

Library rather than script for the same reason as the rest of this package: four
measurements score audits, and the definition of "repaired" is a claim this work makes
rather than a detail of one experiment.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol

from pharos.governance.fleet import MASK_SEED
from pharos.governance.policy import POLICY_SEED, select
from pharos.governance.view import ServerObservation
from pharos.inference import agreement_with, federated_dawid_skene

__all__ = [
    "REPAIRED",
    "AuditOutcome",
    "Scored",
    "ThresholdSpread",
    "authority_ruling",
    "baseline_errors",
    "evaluate_audit",
    "summarize_thresholds",
    "threshold",
]

#: Agreement counted as repaired. The pre-cliff level is 1.000 and the post-cliff level
#: is 0.660; anything at or above this is nearer the first than the second by a wide
#: margin, and the exact threshold is published so a reader can move it.
REPAIRED = 0.95


class Scored(Protocol):
    """What `threshold` needs from a row, which is less than any caller's row type.

    Typing this against one script's dataclass is how a library ends up importing from an
    experiment. Three fields is the whole contract: what the budget was, whether the score
    cleared the bar, and whether anything was actually corrected.
    """

    @property
    def budget(self) -> int: ...

    @property
    def repaired(self) -> bool: ...

    @property
    def corrected(self) -> int: ...


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


@dataclass(frozen=True, slots=True)
class AuditOutcome:
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


def evaluate_audit(
    partitioned: dict[str, list[tuple[str, bool]]],
    view: ServerObservation,
    truth: dict[str, bool],
    *,
    policy: str,
    budget: int,
    error: float,
    baseline: tuple[int, int] | None = None,
    seed: int = POLICY_SEED,
) -> AuditOutcome:
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
    return AuditOutcome(
        agreement=agreement,
        scored=len(labels),
        remaining_errors=remaining,
        hits=hits,
        mechanical=mechanical,
        corrected=(errors - hits) - remaining,
    )


def threshold(rows: Sequence[Scored]) -> int | None:
    """Smallest budget that clears the bar AND actually corrected something.

    `repaired` alone is a score crossing, and a score can cross by deletion: anchoring
    a wrong task removes it from the denominator. Requiring `corrected > 0` is what
    makes this a threshold for repair rather than for successful targeting.
    """
    return next(
        (r.budget for r in sorted(rows, key=lambda r: r.budget) if r.repaired and r.corrected > 0),
        None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ThresholdSpread:
    """What the audited-items price looks like across draws, rather than in one.

    `reached` is the censoring: a draw that never repairs within the sweep has no
    finite threshold, and averaging it in at the sweep's maximum would report the
    experiment's edge as a measurement. So the median is taken over the ordered draws
    with unreached ones sorted last, which yields `None` when more than half of them
    never repair -- the honest answer being "usually not reached", not a number.
    """

    n_wrong: int
    seeds: int
    reached: int
    median: int | None
    lowest: int | None
    highest: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "n_wrong": self.n_wrong,
            "seeds": self.seeds,
            "reached": self.reached,
            "median": self.median,
            "lowest": self.lowest,
            "highest": self.highest,
        }


def summarize_thresholds(n_wrong: int, thresholds: Sequence[int | None]) -> ThresholdSpread:
    """Median and range of a censored threshold, over one draw per seed."""
    if not thresholds:
        raise ValueError("no draws to summarize")
    reached = sorted(t for t in thresholds if t is not None)
    # Unreached sort after every finite value, so the median lands on `None` exactly
    # when at least half the draws failed to repair.
    ordered: list[int | None] = [*reached, *([None] * (len(thresholds) - len(reached)))]
    return ThresholdSpread(
        n_wrong=n_wrong,
        seeds=len(thresholds),
        reached=len(reached),
        median=ordered[(len(ordered) - 1) // 2],
        lowest=reached[0] if reached else None,
        highest=reached[-1] if reached else None,
    )
