"""Whether a fleet's verdict rate depends on a channel once difficulty is held fixed.

Finding 22's detector. For an unbiased fleet, whether a task happens to carry a
particular channel should tell you nothing about the verdict once you already know how
much evidence the task shows. Writing $V$ for the fleet's significant-rate, $C$ for
whether a task carries the channel, and $E$ for the count of defining facts visible:

    V  independent of  C  |  E

A shared channel blind spot breaks exactly that, and it survives unanimity because it is
a property of the *level* of the fleet's verdict rather than of the spread.

The inputs are the per-task vote sums the aggregator already holds and public corpus
structure. Not a per-analyst stream, which secure aggregation does not produce, and not
ground truth, which would make the question circular.

This lived in `scripts/measure_channel_bias.py`. It moved here when a second measurement
needed it: finding 30 asks what the same detector does when the shared error follows no
enumerable channel, and the answer is only meaningful if it is the *same* detector rather
than a copy that drifted. A test asserts no script imports another, so the alternative was
a second copy.
"""

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from random import Random

from pharos.governance.view import observe
from pharos.labels import Compartment
from pharos.tasks import TriageTask

__all__ = [
    "ALPHA",
    "PERMUTATIONS",
    "PERMUTATION_SEED",
    "ChannelCarriage",
    "Detection",
    "compartment_carriage",
    "detect",
    "scan_channels",
    "stratified_delta",
    "verdict_rates",
]

#: Channel name to which tasks carry it. The scan's whole input, named because the size of
#: this family is the assumption finding 30 tests rather than an implementation detail.
type ChannelCarriage = dict[str, dict[str, bool]]

#: Significance level for a detection. The gate's own convention elsewhere in this repo is
#: three sigma, whose one-sided normal tail is 0.00135, so 0.001 is the nearest round
#: threshold at least as strict. It is deliberately not tuned: a threshold picked after
#: seeing the effect is not a threshold. `PERMUTATIONS` has to be large enough to resolve
#: it, since a permutation p-value cannot go below 1 / (m + 1).
ALPHA = 0.001

#: Permutations in the null. One pooled null rather than several small ones: this drew 21
#: nulls of 200 and reported the median z, which spends the same budget to estimate the
#: same quantity less precisely. The floor on an achievable p-value is 1 / (m + 1), so this
#: resolves down to 2.4e-4 and can therefore actually decide ALPHA.
PERMUTATIONS = 4200

PERMUTATION_SEED = 90210


@dataclass(frozen=True, slots=True)
class Detection:
    """The conditional-independence statistic for one channel, against its null."""

    channel: str
    delta: float
    null_mean: float
    p_value: float
    extreme: int
    permutations: int
    detected: bool
    strata: int

    def as_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            # The effect, and the thing to read for *extent*: it is linear in the blind
            # share. A p-value cannot report extent, because it saturates at its own
            # floor once the effect is comfortably significant.
            "delta": round(self.delta, 4),
            "null_mean": round(self.null_mean, 4),
            # Not rounded to a fixed number of places: these span several orders of
            # magnitude and 2.4e-4 is the floor, so a fixed rounding would flatten the
            # strong results into one another.
            "p_value": float(f"{self.p_value:.3g}"),
            "extreme": self.extreme,
            "permutations": self.permutations,
            "detected": self.detected,
            "strata": self.strata,
        }


def verdict_rates(partitioned: dict[str, list[tuple[str, bool]]]) -> dict[str, float]:
    """Each task's share of significant verdicts, as the aggregator already sees it.

    Read from the per-task vote sums of finding 18's protocol. No contributor is
    distinguishable here, which is the point: the statistic must survive the protocol that
    made finding 11's attack impossible.

    The arithmetic is `ServerObservation.rates`; this is the entry point for a caller
    holding the partitioned stream rather than a view. A caller who already has the view
    should call the method, and not hand the stream back to be re-observed.
    """
    return observe(partitioned).rates()


def stratified_delta(
    rates: dict[str, float],
    carries: dict[str, bool],
    evidence: dict[str, int],
) -> tuple[float, int]:
    """Mean gap in verdict rate between carrying and non-carrying tasks, within strata.

    Signed so that a *negative* delta means tasks carrying the channel are called
    significant less often than equally-evidenced tasks that do not carry it, which is the
    direction a blind spot produces. Strata with either side empty contribute nothing
    rather than zero: an absent comparison is not a null result.
    """
    gaps: list[float] = []
    used = 0
    for level in sorted(set(evidence.values())):
        with_channel = [rates[t] for t in rates if evidence[t] == level and carries[t]]
        without = [rates[t] for t in rates if evidence[t] == level and not carries[t]]
        if not with_channel or not without:
            continue
        gaps.append(statistics.fmean(with_channel) - statistics.fmean(without))
        used += 1
    return (statistics.fmean(gaps) if gaps else 0.0), used


def detect(
    rates: dict[str, float],
    carries: dict[str, bool],
    evidence: dict[str, int],
    *,
    permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> Detection | None:
    """The observed stratified gap against a within-stratum permutation null.

    Shuffling channel membership *within* an evidence level preserves how difficulty is
    distributed and destroys only the association being tested, so a channel that merely
    correlates with difficulty cannot score here. The statistic itself -- the mean gap
    between carrying and non-carrying tasks within a stratum, averaged across strata -- is
    the standard one for this design.

    **Significance is a permutation p-value, not a z-score.** This reported
    `z = (null_mean - observed) / null_sd` until 2026-08-06, which was wrong in three ways
    at once and wrong in the same place each time. A permutation test exists precisely so
    the null's shape need not be assumed; standardizing against its mean and standard
    deviation puts the normality assumption back. It is undefined when the null has no
    spread, which is the case both of finding 22's negative controls sit in, so the
    controls "passed" from a division the code special-cased rather than from evidence.
    And it invited a fix in kind: the previous attempt drew the null 21 times and reported
    the median z, which spends 4200 permutations to estimate a quantity one pooled null of
    4200 estimates better.

        p = (b + 1) / (m + 1)

    with `b` the number of permutations at least as extreme as the observed gap. The `+1`
    on both sides is Phipson and Smyth (2010): the permuted draws generate an exact
    discrete null distribution rather than an estimate of a tail probability, so the
    observed value is one of its own draws. The naive `b / m` understates by about `1/m`
    and can report zero, which is a claim no finite number of permutations supports. The
    floor here is `1 / (m + 1)`.

    The degenerate case then needs no handling at all. If every permutation returns the
    observed gap -- a noiseless fleet, where each task's rate is fixed by its evidence
    stratum and there is nothing left to shuffle -- then every draw is at least as
    extreme, `b = m`, and `p = 1.0`. No detection, correctly, and by construction rather
    than by a special case.

    One-sided: a blind spot *depresses* the rate on carrying tasks, so a gap at least as
    extreme is one at least as negative. Reporting a two-sided result would let an elevated
    rate read as the same finding.
    """
    observed, strata = stratified_delta(rates, carries, evidence)
    if strata == 0:
        return None

    by_level: dict[int, list[str]] = {}
    for task in rates:
        by_level.setdefault(evidence[task], []).append(task)

    rng = Random(seed)  # noqa: S311  -- a null distribution, not a security boundary
    null: list[float] = []
    for _ in range(permutations):
        shuffled: dict[str, bool] = {}
        for tasks_at_level in by_level.values():
            flags = [carries[t] for t in tasks_at_level]
            rng.shuffle(flags)
            shuffled.update(dict(zip(tasks_at_level, flags, strict=True)))
        null.append(stratified_delta(rates, shuffled, evidence)[0])

    at_least_as_extreme = sum(1 for gap in null if gap <= observed)
    p_value = (at_least_as_extreme + 1) / (permutations + 1)
    return Detection(
        channel="",
        delta=observed,
        null_mean=statistics.fmean(null),
        p_value=p_value,
        extreme=at_least_as_extreme,
        permutations=permutations,
        detected=p_value <= ALPHA,
        strata=strata,
    )


def scan_channels(
    rates: dict[str, float],
    carriage: ChannelCarriage,
    evidence: dict[str, int],
    *,
    permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> list[Detection]:
    """Every candidate channel tested against the same fleet, so false positives show.

    `carriage` maps a channel name to which tasks carry it. Passed in rather than derived
    from the corpus, because the set of channels a detector may scan is the substance of
    finding 30: the family has to be small and enumerable for the scan to be a method at
    all, and handing it in is what makes that assumption visible at the call site.
    """
    found: list[Detection] = []
    for channel, carries in carriage.items():
        result = detect(rates, carries, evidence, permutations=permutations, seed=seed)
        if result is not None:
            # `replace` rather than re-listing the fields: this rebuild used to name every
            # field by hand, so adding one to Detection meant silently dropping it here
            # unless the author remembered both sites.
            found.append(replace(result, channel=channel))
    return found


def compartment_carriage(tasks: Sequence[TriageTask], over: Iterable[str]) -> ChannelCarriage:
    """Which of `over` carries each compartment, for the enumerable scan finding 22 runs.

    The compartments of this corpus are the whole family a deployment can scan, and that
    ceiling is the premise finding 30 tests rather than an implementation detail. Built
    here rather than in each measurement because two of them now scan the same family, and
    a second copy is a copy that drifts.
    """
    by_id = {t.task_id: t for t in tasks}
    pool = list(over)
    return {
        channel.value: {
            task: any(channel in r.label.compartments for r in by_id[task].sources) for task in pool
        }
        for channel in Compartment
    }
