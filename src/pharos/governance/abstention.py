"""Scoring a decision to withhold, which is deletion by design rather than by accident.

Two numbers per cell and neither is readable without the other: the errors among labels
still published, and how many labels are published at all. Withholding removes errors by
removing labels, and an earlier finding in this work had to retract a repair claim that
was exactly that effect -- so the coverage column is not decoration, it is what stops the
risk column from reading as a repair.

A policy that withholds *correct* labels raises its own risk. That asymmetry is what makes
the metric honest without an oracle, and it is the reason these live in the package rather
than beside one experiment.
"""

from dataclasses import dataclass

__all__ = [
    "HALVED",
    "REPORT_BUDGET",
    "AbstentionCell",
    "beats_every_draw",
    "first_budget_halving",
    "score",
]

#: The budget every summary column is quoted at. A constant rather than a literal in three
#: places: the artifact publishes it, and the docs block and both manuscripts read it from
#: there, so a change moves the tables instead of making them disagree.
REPORT_BUDGET = 20

#: What counts as risk removed. Half the errors the fleet started with -- a chosen
#: constant, published so a reader can move it.
HALVED = 0.5


@dataclass(frozen=True, slots=True, kw_only=True)
class AbstentionCell:
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
) -> AbstentionCell:
    """What one withholding decision published, and what it got wrong anyway.

    Risk is over *published* labels, so withholding a correct label raises it. That is
    what stops this metric from rewarding abstention for its own sake, and it is why no
    oracle is needed to read the table: a policy that withholds indiscriminately is
    visibly punished in the same column it would otherwise be flattered in.
    """
    held = set(withheld)
    published = [task for task in pool if task not in held]
    errors = sum(1 for task in published if task in wrong)
    return AbstentionCell(
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


def first_budget_halving(cells: list[AbstentionCell], base_errors: int) -> int | None:
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
