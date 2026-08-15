"""Whether the shape of a fleet's error is legible in the aggregate.

Within an evidence stratum a fleet applying one rule votes identically, so independent
slips are exactly binomial and the variance of a per-task vote sum is $np(1-p)$. A shared
standard is not binomial: it splits the stratum into the tasks it corrupts and the tasks
it does not, and a mixture of two rates carries more variance than a binomial at their
mean. The index of dispersion therefore reads 1 under independent error and above it when
part of the error is shared.

The inputs are per-task vote sums, per-task contributor counts, and public evidence
counts -- strictly less than the channel detector needs, since nothing here has to name a
channel. The instrument is standard wherever counts are modelled; the use of it to choose
between remedies is what this testbed adds.
"""

from dataclasses import dataclass
from random import Random

from pharos.governance.channel import ALPHA
from pharos.governance.view import ServerObservation

__all__ = ["ALPHA", "MIN_STRATUM", "NULL_DRAWS", "Dispersion", "dispersion"]

#: Draws of the parametric null. A simulated p-value floors at 1/(m+1), and that floor has
#: to sit below the alpha it is compared against or the test cannot fire at all. Callers
#: that lower this must check the arithmetic; the measurement script asserts it.
NULL_DRAWS = 2000

# `ALPHA` is re-exported from the channel detector rather than restated. The two are read
# on the same scale on purpose, and two literals that must agree are two literals that
# will eventually disagree.

#: A stratum contributes only if it has at least this many tasks. A dispersion index over
#: two tasks is a number, and it is not an estimate.
MIN_STRATUM = 10


@dataclass(frozen=True, slots=True)
class Dispersion:
    """The index, its null, and the evidence base it was computed over."""

    index: float | None
    p_value: float | None
    strata: int
    tasks: int
    #: Strata dropped because a rate of exactly 0 or 1 leaves no binomial variance to
    #: compare against. Reported rather than filtered silently: on a noiseless healthy
    #: fleet this is *every* stratum, and that is the finding rather than a missing row.
    degenerate_strata: int

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "p_value": self.p_value,
            "strata": self.strata,
            "tasks": self.tasks,
            "degenerate_strata": self.degenerate_strata,
        }


def dispersion(
    view: ServerObservation,
    evidence: dict[str, int],
    *,
    draws: int = NULL_DRAWS,
    seed: int = 0,
) -> Dispersion:
    """Observed variance of the vote sums over the binomial variance at the same rate.

    Pooled across strata rather than averaged: a stratum of eighty tasks and a stratum of
    twelve are not two equal observations of the same quantity, and averaging their
    indices would let the small one move the answer as much as the large one.

    The null simulates each stratum's counts as binomial at that stratum's own observed
    rate, which is the hypothesis being tested: one rate per stratum, analysts
    independent. A p-value is the share of simulated fleets at least as dispersed as this
    one, computed as (b+1)/(m+1) so it is never zero.
    """
    strata: dict[int, list[tuple[float, float]]] = {}
    for task, votes in view.votes.items():
        seen = view.seen.get(task, 0.0)
        if seen <= 0:
            continue
        strata.setdefault(evidence.get(task, -1), []).append((votes, seen))

    observed = 0.0
    expected = 0.0
    used: list[tuple[list[tuple[float, float]], float]] = []
    degenerate = 0
    tasks = 0
    for rows in strata.values():
        if len(rows) < MIN_STRATUM:
            continue
        total_votes = sum(v for v, _ in rows)
        total_seen = sum(n for _, n in rows)
        rate = total_votes / total_seen
        variance = sum(n * rate * (1.0 - rate) for _, n in rows)
        if variance <= 0.0:
            # A stratum every analyst answered identically. There is no binomial variance
            # to compare against, so it carries no information about dispersion -- which
            # is different from carrying information that there is none.
            degenerate += 1
            continue
        observed += sum((v - n * rate) ** 2 for v, n in rows)
        expected += variance
        used.append((rows, rate))
        tasks += len(rows)

    if not used:
        return Dispersion(index=None, p_value=None, strata=0, tasks=0, degenerate_strata=degenerate)

    index = observed / expected
    rng = Random(seed)  # noqa: S311  -- a null distribution, not a security boundary
    at_least_as_extreme = 0
    for _ in range(draws):
        null_observed = 0.0
        null_expected = 0.0
        for rows, rate in used:
            for _, n in rows:
                drawn = sum(1 for _ in range(int(n)) if rng.random() < rate)
                null_observed += (drawn - n * rate) ** 2
                null_expected += n * rate * (1.0 - rate)
        if null_observed / null_expected >= index:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (draws + 1)
    return Dispersion(
        index=round(index, 4),
        p_value=round(p_value, 6),
        strata=len(used),
        tasks=tasks,
        degenerate_strata=degenerate,
    )
