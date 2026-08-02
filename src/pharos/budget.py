"""A privacy budget over the variable that actually leaks, and what it really costs.

Finding 11's attack recovers an analyst's compartment set from which task identifiers
appear under a pseudonym. It reads no values: not the verdict, not the released label,
not a word of report text.

**Proposition (value noise is useless here).** Let `A` be the attack of
`fleet.link`. `A` is a function of the observed map from pseudonym to task-identifier
set. Any mechanism `M` acting only on contribution *values* leaves that map unchanged,
so `A(M(S)) = A(S)` pointwise, for every mechanism and every privacy parameter. This is
not an empirical finding and no epsilon appears in it. `tests/test_fleet.py` enforces
the premise by flipping every verdict in a stream and requiring identical output, and
`tests/test_budget.py` enforces the conclusion for the mechanism itself.

The consequence is worth stating plainly, because the expected move in this literature
is to calibrate noise into the values a client uploads, usually model updates. Against
this channel that spends the budget on the wrong variable and buys nothing at any
epsilon. What needs protecting is **participation**: the fact that a contribution about
task T exists under pseudonym P at all.

So the mechanism is randomized response over the participation indicator. Each eligible
task is contributed with probability `keep`; each task the analyst could *not* read is
contributed anyway with probability `fabricate`. The second half buys the deniability
and is what plain subsampling lacks, since dropping contributions makes a stream
sparser without making any present contribution deniable.

**The composition trap, which is the substantive part.** Randomized response bounds the
likelihood ratio for *one* indicator. The attack observes all of them. Two clearances
are distinguished by the tasks where their reachability differs, and there can be
hundreds of those, so the guarantee against an adversary trying to tell one clearance
from another is the per-indicator budget composed over that whole set. Reporting the
per-indicator figure alone would describe a mechanism far stronger than the one
deployed. Both are computed here and the composed one is the one that answers the
question the attack asks.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from random import Random

from pharos.fleet import Clearance, Contribution
from pharos.labels import Capacity, DeclassificationPolicy, Label, join, shared_eligible
from pharos.tasks import TriageTask
from pharos.telemetry import get_logger, record

#: Failure probability for the advanced-composition bound. The usual convention is
#: delta well below 1/n for a population of n; 1e-5 is comfortably under that for the
#: fleet sizes measured here.
DEFAULT_DELTA = 1e-5


@dataclass(frozen=True, slots=True)
class Budget:
    """What a participation mechanism spends, per indicator and composed.

    Kept as one object because quoting either number without the other misleads in
    opposite directions: the per-indicator figure flatters the mechanism, and the
    composed figure is meaningless without knowing how many indicators it covers.
    """

    keep: float
    fabricate: float

    @property
    def epsilon(self) -> float:
        """Per-indicator randomized-response bound.

        Infinite when `fabricate` is zero, and that is substantive rather than an edge
        case: with no fabricated contributions every observed contribution is proof of
        reachability, so no amount of *dropping* buys a finite guarantee. It is why
        subsampling sits in finding 11's ladder as a cost with no protection beside it.
        """
        if self.fabricate <= 0.0 or self.keep >= 1.0:
            return math.inf
        present = self.keep / self.fabricate
        absent = (1.0 - self.fabricate) / (1.0 - self.keep)
        return math.log(max(present, absent))

    def basic_composition(self, indicators: int) -> float:
        """Sequential composition over `indicators` independent indicators."""
        return indicators * self.epsilon

    def advanced_composition(self, indicators: int, *, delta: float = DEFAULT_DELTA) -> float:
        """Advanced composition, which is tighter than basic for many indicators.

        The Dwork-Rothblum-Vadhan bound. Holds with probability `1 - delta`, which is
        the price of the improvement and is why the basic bound is still reported: the
        two answer slightly different questions and the smaller number is not
        automatically the honest one.
        """
        eps = self.epsilon
        if math.isinf(eps) or indicators <= 0:
            return eps if indicators > 0 else 0.0
        return math.sqrt(2 * indicators * math.log(1 / delta)) * eps + indicators * eps * (
            math.exp(eps) - 1
        )

    def effective_epsilon(self, indicators: int, *, delta: float = DEFAULT_DELTA) -> float:
        """The bound a deployment would actually claim: the better of the two."""
        return min(
            self.basic_composition(indicators),
            self.advanced_composition(indicators, delta=delta),
        )

    def as_dict(self, indicators: int, *, delta: float = DEFAULT_DELTA) -> dict[str, object]:
        def clean(value: float) -> float | None:
            return None if math.isinf(value) else round(value, 4)

        return {
            "keep": self.keep,
            "fabricate": self.fabricate,
            "epsilon_per_indicator": clean(self.epsilon),
            "indicators": indicators,
            "epsilon_basic": clean(self.basic_composition(indicators)),
            "epsilon_advanced": clean(self.advanced_composition(indicators, delta=delta)),
            "epsilon_effective": clean(self.effective_epsilon(indicators, delta=delta)),
            "delta": delta,
        }


def _readable(clearance: Clearance, task: TriageTask) -> bool:
    return clearance.label.dominates(
        join([r.label for r in task.sources], capacity=Capacity.FREETEXT)
    )


def distinguishing_tasks(first: Clearance, second: Clearance, tasks: Sequence[TriageTask]) -> int:
    """How many tasks separate two clearances, which is what composition runs over.

    The adversary's question is which of two clearances produced a stream, and only the
    tasks where their reachability differs carry any signal. This counts them, so the
    composed budget is reported over the set that actually distinguishes the pair
    rather than over the corpus, which would overstate the spend.
    """
    return sum(1 for t in tasks if _readable(first, t) != _readable(second, t))


def widest_separation(fleet: Sequence[Clearance], tasks: Sequence[TriageTask]) -> int:
    """The largest distinguishing set in the fleet: the worst case to quote."""
    return max(
        (distinguishing_tasks(a, b, tasks) for i, a in enumerate(fleet) for b in fleet[i + 1 :]),
        default=0,
    )


def randomized_participation(
    fleet: Sequence[Clearance],
    tasks: Sequence[TriageTask],
    *,
    policy: DeclassificationPolicy,
    ceiling: Label,
    budget: Budget,
    seed: int,
) -> tuple[Contribution, ...]:
    """Contribute under randomized response on the participation indicator.

    Built from the fleet rather than by filtering an existing stream, because the
    fabricated half does not exist in one: a contribution on a task the analyst cannot
    read is exactly what a stream of genuine contributions never contains.

    Eligibility at the ceiling binds on both halves. A fabricated contribution is a lie
    about *who saw what*, which is the deniability being bought. It is not licence to
    federate something the disclosure gate refused, which would be a leak rather than a
    control.
    """
    if math.isinf(budget.epsilon):
        # Worth a warning rather than a silent pass. A mechanism running with no
        # fabrication still drops contributions and still looks like a privacy
        # control from the outside, while offering no finite guarantee at all.
        get_logger().warning(
            "budget.no_finite_guarantee",
            extra={
                "event": "budget.no_finite_guarantee",
                "keep": budget.keep,
                "fabricate": budget.fabricate,
                "why": "fabricate=0 makes every observed contribution proof of reachability",
            },
        )
    rng = Random(seed)
    out: list[Contribution] = []
    for clearance in fleet:
        for task in tasks:
            if not shared_eligible(task.label, ceiling, policy):
                continue
            reachable = _readable(clearance, task)
            if rng.random() >= (budget.keep if reachable else budget.fabricate):
                continue
            out.append(
                Contribution(
                    analyst_id=clearance.analyst_id,
                    pseudonym=clearance.analyst_id,
                    task_id=task.task_id,
                    # A fabricated contribution carries a verdict the analyst never
                    # formed. Filling it with the corpus truth would make the noise
                    # free and the utility cost fictional, so it is drawn at random.
                    verdict=task.significant if reachable else rng.random() < 0.5,
                )
            )
    record(
        "budget.participation",
        len(out),
        keep=budget.keep,
        fabricate=budget.fabricate,
        epsilon=None if math.isinf(budget.epsilon) else round(budget.epsilon, 4),
    )
    return tuple(out)


def label_noise(
    stream: Iterable[Contribution], fleet: Sequence[Clearance], tasks: Sequence[TriageTask]
) -> float:
    """Share of contributions made on tasks the contributor could not read.

    The utility side of the budget, measured rather than derived from `fabricate`,
    because what reaches the aggregator depends on how many unreachable tasks each
    clearance had to fabricate from, which varies across the fleet.
    """
    by_id = {c.analyst_id: c for c in fleet}
    by_task = {t.task_id: t for t in tasks}
    contributions = list(stream)
    if not contributions:
        return 0.0
    fabricated = sum(
        1
        for c in contributions
        if c.analyst_id in by_id
        and c.task_id in by_task
        and not _readable(by_id[c.analyst_id], by_task[c.task_id])
    )
    return fabricated / len(contributions)


def value_noise(
    stream: Iterable[Contribution], *, flip: float, seed: int
) -> tuple[Contribution, ...]:
    """Randomized response on the *verdict*: the mechanism a reader expects.

    Present so its uselessness against this channel is demonstrated rather than
    asserted. By the proposition in this module's docstring it cannot change the
    attack's output at any flip probability, while degrading the training signal at the
    usual rate.
    """
    rng = Random(seed)
    return tuple(replace(c, verdict=not c.verdict) if rng.random() < flip else c for c in stream)
