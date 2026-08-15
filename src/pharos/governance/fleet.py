"""How a fleet is built, what it contributes, and when a corpus can host the experiment.

Every governance measurement in this testbed starts by constructing a fleet of simulated
analysts and collecting what they contribute. That construction was defined inside
`scripts/measure_secure_reliability.py` and `scripts/measure_blind_spot.py` and imported
from there by most of the other measurement scripts, which made a library out of two
experiment files: seven scripts imported the first, four the second, and one of them
reached past a leading underscore to do it. Anyone using this testbed to measure their own
policy needs these, and needed a copy of a script to get them.

The constants describing *positions* in a fleet live here too. Findings 19 through 23 were
measured at nine analysts and wrote their compositions as absolute tuples, which read as
arbitrary until you know that five is the majority crossing at nine. Written as rungs, the
same experiment runs at any fleet size.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import comb
from random import Random

from pharos.analyst import Action, AnalystPolicy, Proposal, evidence_shown
from pharos.disclosure import DROP_COMPARTMENTS
from pharos.labels import Compartment
from pharos.tasks import TriageTask
from pharos.telemetry import get_logger

__all__ = [
    "BLIND",
    "BLIND_RUNGS",
    "CHANNEL_ENTANGLEMENT_SLACK",
    "LATENT_ATTEMPTS",
    "LATENT_CARRIAGE_QUANTILE",
    "LATENT_NULL_DRAWS",
    "LATENT_SEED_STRIDE",
    "LATENT_SLICE",
    "MASK_SEED",
    "REFUSED_EXIT",
    "RUNGS",
    "WRONG_THRESHOLD",
    "ChannelCheck",
    "ChannelUnusableError",
    "LatentSlice",
    "assert_channel_usable",
    "blind_fleet",
    "contributions_for",
    "draw_balanced_slice",
    "draw_latent_slice",
    "exact_wrong_majority",
    "fleet_of",
    "ladder",
    "latent_blind_fleet",
    "majority",
]

#: The escalation threshold a mistaken analyst holds, one below the world's rule.
WRONG_THRESHOLD = 2

#: Mask seed for the aggregation rounds. Distinct from the corpus seed so a corpus
#: change cannot silently alter the protocol's randomness as well.
MASK_SEED = 4242

#: The channel a blind fleet discounts.
BLIND = Compartment.PARTNER

#: How much of the fleet carries the blind spot. The interesting end is the top: a
#: sighted minority is what keeps the disagreement signal alive.
BLIND_RUNGS = ("none", "one-third", "majority", "seven-ninths", "all-but-one", "unanimous")

#: How far the mean evidence on tasks CARRYING the channel may sit from the mean on tasks
#: that do not, before the channel is judged too entangled with item difficulty to serve
#: as an independent axis.
#:
#: This deliberately measures the channel, not the affected slice. An earlier version
#: guarded the affected slice's mean against the corpus mean with a slack of 1.5, and that
#: guard could not fail: blinding only ever removes evidence and the threshold is 3 of 3,
#: so a verdict can flip only on a task whose visible evidence was exactly 3. The affected
#: mean is therefore 3.00 for every compartment, and SENSOR -- the channel the docs single
#: out as unusable -- passed identically. The statistic below does discriminate: PARTNER
#: 1.88 against 1.72, SENSOR 2.00 against 0.48.
CHANNEL_ENTANGLEMENT_SLACK = 0.5

#: Exit code for "this corpus cannot host the experiment", as against a crash.
#:
#: Both are non-zero, and a sweep that cannot tell them apart reports a bug as a corpus
#: property: the draw drops out of the denominator and the rate above it looks unchanged.
#: A caller matches this code and re-raises anything else.
REFUSED_EXIT = 3


class ChannelUnusableError(Exception):
    """This corpus cannot host a blind-spot experiment on this channel.

    An exception rather than a `SystemExit`, because the exit code is a command-line
    contract and this is a library. Every measurement script catches it, prints the
    reason to stderr and exits `REFUSED_EXIT`, which is what `measure_corpus_sensitivity`
    matches on to tell a draw that cannot host the experiment from a draw that crashed.
    Those are different results and only one of them belongs in a denominator.
    """


def majority(fleet: int) -> int:
    """Smallest number of analysts that carries a vote. Fleets here are odd, so no ties."""
    return fleet // 2 + 1


#: Fleet positions as functions of the size, rather than as counts.
#:
#: Written as counts, `--fleet` was misleading rather than merely unused: a composition
#: larger than the fleet was skipped and printed as "none", which elsewhere means swept
#: and never repaired, so an unmeasured cell read as a measured failure. As positions the
#: ladder is the same experiment at any size, and reproduces every committed constant at
#: nine (asserted in the tests).
RUNGS: dict[str, Callable[[int], int]] = {
    "none": lambda fleet: 0,
    "one": lambda fleet: 1,
    "two": lambda fleet: 2,
    "one-third": lambda fleet: round(fleet / 3),
    "below": lambda fleet: majority(fleet) - 1,
    "majority": majority,
    "two-thirds": lambda fleet: round(2 * fleet / 3),
    "seven-ninths": lambda fleet: round(7 * fleet / 9),
    "all-but-one": lambda fleet: fleet - 1,
    "unanimous": lambda fleet: fleet,
}


def ladder(fleet: int, rungs: Sequence[str]) -> tuple[int, ...]:
    """The named positions as counts at this fleet, deduplicated and ordered.

    Deduplication is not incidental: at small fleets distinct rungs collide (at five,
    the majority and two-thirds are both 3), and a sweep that measured the same
    composition twice would report it as two agreeing observations.
    """
    return tuple(sorted({min(fleet, max(0, RUNGS[rung](fleet))) for rung in rungs}))


def fleet_of(n_wrong: int, size: int) -> tuple[AnalystPolicy, ...]:
    """`n_wrong` analysts holding the wrong standard, the rest holding the right one.

    One definition rather than two. This was reproduced in a second script with a comment
    explaining that it had to be, because the original was written against its own
    module-level constant; the constant moved here with it and the copy is gone.
    """
    right = [AnalystPolicy(f"right-{i}") for i in range(size - n_wrong)]
    wrong = [
        AnalystPolicy(f"wrong-{i}", escalation_threshold=WRONG_THRESHOLD) for i in range(n_wrong)
    ]
    return tuple(right + wrong)


def contributions_for(
    policies: Sequence[AnalystPolicy],
    tasks: Sequence[TriageTask],
    proposals: dict[str, Proposal],
    *,
    seed: int,
) -> list[tuple[str, str, bool]]:
    """Flat `(task, contributor, verdict)` rows, as finding 12 builds them.

    A rejection contributes nothing rather than counting as a vote, which is finding
    7's result and is preserved here so the two measurements see the same stream.
    """
    rows: list[tuple[str, str, bool]] = []
    for policy in policies:
        for task in tasks:
            decision = policy.review(task, proposals[task.task_id], seed=seed)
            if decision.action is Action.ACCEPT:
                verdict: bool | None = proposals[task.task_id].verdict
            elif decision.action is Action.REVISE:
                verdict = decision.corrected_verdict
            else:
                verdict = None
            if verdict is not None:
                rows.append((task.task_id, policy.name, verdict))
    return rows


def blind_fleet(n_blind: int, size: int, *, slip_rate: float = 0.0) -> tuple[AnalystPolicy, ...]:
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

    `slip_rate` defaults to zero, which is what finding 21 measured and must keep
    measuring. It is a parameter because a noiseless fleet turns out to be a special
    case in a way that matters elsewhere: with every analyst deterministic and
    identical, each task's verdict rate is a function of its evidence stratum alone, so
    a within-stratum statistic has exactly zero variance to work against. Finding 22
    needs a fleet that can disagree with itself in order to have a null at all.
    """
    sighted = [
        AnalystPolicy(f"sighted-{i}", release_policy=DROP_COMPARTMENTS, slip_rate=slip_rate)
        for i in range(size - n_blind)
    ]
    blind = [
        AnalystPolicy(
            f"blind-{i}",
            blind_compartment=BLIND,
            release_policy=DROP_COMPARTMENTS,
            slip_rate=slip_rate,
        )
        for i in range(n_blind)
    ]
    return tuple(sighted + blind)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelCheck:
    """Whether this corpus can host a blind-spot experiment at all, and on what evidence."""

    affected: int
    affected_mean: float
    corpus_mean: float
    mean_with: float
    mean_without: float


def assert_channel_usable(
    tasks: list[TriageTask], *, compartment: Compartment = BLIND
) -> ChannelCheck:
    """Refuse a corpus where the blinded channel is not an independent axis.

    Two preconditions, both of which some corpus draws fail, and both of which are
    refusals rather than failures: a draw that cannot host the experiment says nothing
    about the finding, which is different from a draw that contradicts it.

    Shared rather than copied. Every measurement that blinds a channel needs exactly this
    guard, and a second copy is a copy that drifts -- the failure mode being silent by
    construction, since a script running without the guard produces a well-formed artifact
    from an invalid construction. `measure_selective_risk` is the second caller.
    """
    reference = AnalystPolicy("reference")
    blind_reviewer = AnalystPolicy("blind", blind_compartment=compartment)
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

    # The statistic that actually distinguishes one channel from another: evidence on
    # tasks carrying it against tasks that do not. `affected_mean` cannot do this job
    # -- it is 3.00 for every compartment by construction (see the slack constant).
    carrying = [t for t in tasks if any(compartment in r.label.compartments for r in t.sources)]
    not_carrying = [t for t in tasks if t not in set(carrying)]
    mean_with = sum(len(evidence_shown(t)) for t in carrying) / len(carrying) if carrying else 0.0
    mean_without = (
        sum(len(evidence_shown(t)) for t in not_carrying) / len(not_carrying)
        if not_carrying
        else 0.0
    )
    if not affected:
        get_logger().error(
            "blindspot.no_effect",
            extra={"event": "blindspot.no_effect", "compartment": compartment.value},
        )
        raise ChannelUnusableError(
            f"a fleet-wide {compartment.value} blind spot changes no verdict on this "
            "corpus; there is nothing to measure"
        )
    if abs(mean_with - mean_without) > CHANNEL_ENTANGLEMENT_SLACK:
        # A hard stop, not a warning. An earlier version logged and continued, while the
        # docs described it as "asserted rather than trusted" -- prose describing code that
        # did not exist. A channel this entangled with difficulty cannot support the
        # argument, so producing an artifact from it is worse than failing.
        get_logger().error(
            "blindspot.channel_entangled",
            extra={
                "event": "blindspot.channel_entangled",
                "compartment": compartment.value,
                "mean_with": round(mean_with, 3),
                "mean_without": round(mean_without, 3),
            },
        )
        raise ChannelUnusableError(
            f"{compartment.value} carries mean evidence {mean_with:.2f} against "
            f"{mean_without:.2f} without it; that channel is too entangled with "
            "difficulty to serve as an independent axis"
        )
    return ChannelCheck(
        affected=len(affected),
        affected_mean=affected_mean,
        corpus_mean=corpus_mean,
        mean_with=mean_with,
        mean_without=mean_without,
    )


def exact_wrong_majority(rate: float, *, schools: int, fleet: int) -> float:
    """Exact probability that a wrong standard holds the majority. No sampling.

    `fleet` is a required argument here rather than defaulting to nine. The default was a
    module-level constant in the script this came from, and a sweep over fleet sizes that
    forgot to pass it would have silently priced every size at nine.

    Each school is wrong independently with probability `rate`, and every school is the
    same size, so the fleet crosses the majority when more than half its *members* are
    wrong. With equal schools that is a binomial over schools rather than over people,
    which is precisely why clustering matters: the same expected error rate is carried
    by fewer, larger, all-or-nothing draws.
    """
    per_school = fleet // schools
    needed = fleet // 2 + 1
    return float(
        sum(
            comb(schools, j) * rate**j * (1 - rate) ** (schools - j)
            for j in range(schools + 1)
            if j * per_school >= needed
        )
    )


#: How many verdicts a latent blind spot corrupts, matched to what `BLIND` corrupts on
#: the committed corpus. Finding 21 measured PARTNER changing 20 verdicts of 200, all of
#: them on tasks showing all three defining facts. The latent construction below corrupts
#: the same count in the same stratum, so the two differ in exactly one respect: whether
#: the corrupted slice follows a channel a detector can name.
LATENT_SLICE = 20

#: How lopsided a drawn slice may be, as a quantile of what uniform draws from the same
#: pool produce. Not a gap in carriage share: a fixed one was tried first and refused the
#: committed corpus at 0.155 against a slack of 0.15, which was the guard measuring
#: sampling noise rather than lopsidedness. Twenty tasks drawn from a pool of a few dozen
#: put a compartment's carriage share several points either side of the pool's by
#: arithmetic, and eight compartments give the maximum of eight such gaps.
#:
#: So the threshold is read off the draw's own null. The slice is a uniform draw from the
#: eligible stratum, so balance holds in expectation by construction and this guard exists
#: only against a *seed* that drew a genuinely lopsided one -- a slice loading onto LIAISON
#: would be findable by a channel scan, and finding 30 would report the scan succeeding as
#: the scan failing. A draw more extreme than this share of its own null is refused, and
#: every draw's percentile is published whether it passes or not.
LATENT_CARRIAGE_QUANTILE = 0.99

#: Uniform draws used to calibrate that quantile.
LATENT_NULL_DRAWS = 2000

#: Draws `draw_balanced_slice` may refuse before giving up. Generous, because refusing is
#: cheap and the alternative -- a structural refusal reported as bad luck -- is not.
LATENT_ATTEMPTS = 50

#: Gap between the seeds a rejection rule walks. Consecutive integers would make two
#: measurements at neighbouring seeds share their retry sequences, so a corpus sweep would
#: quietly correlate its draws.
LATENT_SEED_STRIDE = 1009


@dataclass(frozen=True, slots=True, kw_only=True)
class LatentSlice:
    """A shared blind spot with no channel behind it, and the evidence that it has none."""

    #: The reports every blind analyst declines to credit.
    distrusted: frozenset[str]
    #: The tasks whose verdict that changes, which is the oracle for this construction.
    corrupted: frozenset[str]
    #: Tasks the draw could have picked from: those showing all three defining facts, the
    #: only ones where discounting a report can change a verdict at all.
    eligible: int
    #: Largest gap, over every compartment, between the slice's carriage of it and that of
    #: the eligible tasks it was drawn from.
    worst_carriage_gap: float
    #: Where that gap sits in the distribution of the same statistic over uniform draws
    #: from the same pool. This is the number to read: the gap alone has no scale, and a
    #: reader cannot tell 0.16 on this corpus from 0.16 on another.
    carriage_percentile: float
    #: Draws refused by the balance precondition before this one was accepted. Published
    #: because a rejection rule is a selection rule: a reader is entitled to know how many
    #: slices this corpus had to discard, and a count that climbs is a corpus where a
    #: balanced slice of this size barely exists.
    rejected: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "distrusted": len(self.distrusted),
            "corrupted": len(self.corrupted),
            "eligible": self.eligible,
            "worst_carriage_gap": round(self.worst_carriage_gap, 4),
            "carriage_percentile": round(self.carriage_percentile, 4),
            "rejected": self.rejected,
        }


def draw_latent_slice(
    tasks: Sequence[TriageTask],
    *,
    size: int = LATENT_SLICE,
    seed: int,
) -> LatentSlice:
    """Reports to distrust so that `size` verdicts flip, on a slice nothing names.

    Discounting reporting can only change a verdict on a task that shows all three
    defining facts, because the world's rule is a conjunction over three and a reviewer
    who loses one falls below it. That is the same stratum `BLIND` acts on, and drawing
    from it uniformly is what makes the resulting slice unpredictable from public
    structure: within the stratum, membership is a coin flip that no column of the corpus
    records.

    For each drawn task one defining fact is chosen and every report carrying it is
    distrusted. Removing the fact rather than one report is required, not tidier:
    corroborated facts reach a reviewer through more than one report --- which is finding
    1's result --- so distrusting a single report often changes nothing, and a draw that
    silently changed fewer verdicts than it reports would price the experiment wrong.

    Raises `ChannelUnusableError` where the corpus cannot supply the slice, on the same
    terms as `assert_channel_usable`: a corpus that cannot host the construction says
    nothing about the finding.
    """
    reference = AnalystPolicy("reference")
    eligible = [t for t in tasks if len(evidence_shown(t)) >= reference.escalation_threshold]
    if len(eligible) < size:
        get_logger().error(
            "latent.pool_too_small",
            extra={"event": "latent.pool_too_small", "eligible": len(eligible), "size": size},
        )
        raise ChannelUnusableError(
            f"a latent slice of {size} needs that many tasks showing all "
            f"{reference.escalation_threshold} defining facts; this corpus has {len(eligible)}"
        )

    rng = Random(seed)  # noqa: S311  -- an experimental draw, not a security boundary
    drawn = rng.sample(sorted(eligible, key=lambda t: t.task_id), size)
    distrusted: set[str] = set()
    corrupted: set[str] = set()
    for task in drawn:
        fact = rng.choice(sorted(evidence_shown(task)))
        distrusted |= {r.report_id for r in task.sources if fact in r.fact_ids}
        corrupted.add(task.task_id)

    blinded = AnalystPolicy("blind", distrusted_reports=frozenset(distrusted))
    actually = {
        t.task_id
        for t in tasks
        if (len(blinded.evidence_visible_to(t)) >= reference.escalation_threshold)
        != (len(reference.evidence_visible_to(t)) >= reference.escalation_threshold)
    }
    if actually != corrupted:
        # Loud rather than reconciled. The two disagree only if a report id is shared
        # across tasks or a fact survives its own removal, and either means the
        # construction is not what this docstring says it is.
        raise ChannelUnusableError(
            f"the draw intended to flip {len(corrupted)} verdicts and flips "
            f"{len(actually)}; the slice is not the set it was drawn as"
        )

    pool = sorted({t.task_id for t in eligible})
    by_id = {t.task_id: t for t in tasks}
    carries = {
        compartment: {
            task: any(compartment in r.label.compartments for r in by_id[task].sources)
            for task in pool
        }
        for compartment in Compartment
    }

    def worst_gap(slice_ids: set[str]) -> float:
        """Largest carriage gap between a slice of the pool and the rest of it."""
        rest = [t for t in pool if t not in slice_ids]
        inside = sorted(slice_ids)
        gaps = []
        for flags in carries.values():
            here = sum(flags[t] for t in inside) / len(inside) if inside else 0.0
            there = sum(flags[t] for t in rest) / len(rest) if rest else 0.0
            gaps.append(abs(here - there))
        return max(gaps, default=0.0)

    worst = worst_gap(corrupted)
    null_rng = Random(seed + 1)  # noqa: S311  -- a null distribution, not a security boundary
    null = [worst_gap(set(null_rng.sample(pool, size))) for _ in range(LATENT_NULL_DRAWS)]
    percentile = sum(1 for g in null if g <= worst) / len(null)
    if percentile > LATENT_CARRIAGE_QUANTILE:
        get_logger().error(
            "latent.slice_nameable",
            extra={
                "event": "latent.slice_nameable",
                "worst_carriage_gap": round(worst, 4),
                "carriage_percentile": round(percentile, 4),
            },
        )
        raise ChannelUnusableError(
            f"the drawn slice's worst carriage gap of {worst:.2f} sits at the "
            f"{percentile:.1%} point of uniform draws from the same pool; a channel scan "
            "could plausibly name it, which is the construction this measurement avoids"
        )
    return LatentSlice(
        distrusted=frozenset(distrusted),
        corrupted=frozenset(corrupted),
        eligible=len(eligible),
        worst_carriage_gap=worst,
        carriage_percentile=percentile,
    )


def draw_balanced_slice(
    tasks: Sequence[TriageTask],
    *,
    size: int = LATENT_SLICE,
    seed: int,
    attempts: int = LATENT_ATTEMPTS,
) -> LatentSlice:
    """The first draw at or below the balance quantile, and how many were refused first.

    A rejection rule rather than a hand-picked seed, and the difference matters. Balance is
    a *precondition* of this construction -- a slice a channel scan could name measures the
    scan working and reports it as the scan failing -- so it is fixed before the run and is
    independent of everything the experiment goes on to measure. Choosing a seed after
    seeing which one produced the nicer result would be the opposite, and this project has
    retracted enough single draws to want the distinction written down.

    Small slices need it most: at 20 of 69 the first draw passes at the 79th percentile,
    and at 5 of 69 the same seed lands at the 99.7th, because the worst of eight carriage
    gaps over a five-task draw is wide by arithmetic. The count of refusals is carried on
    the result rather than logged and forgotten.
    """
    rejected = 0
    for offset in range(attempts):
        try:
            drawn = draw_latent_slice(tasks, size=size, seed=seed + offset * LATENT_SEED_STRIDE)
        except ChannelUnusableError as refusal:
            if "carriage gap" not in str(refusal):
                # A pool too small, or a draw that did not flip what it drew. Neither is
                # fixed by another seed, and retrying would turn a structural refusal into
                # `attempts` identical failures and then a misleading message.
                raise
            rejected += 1
            continue
        return replace(drawn, rejected=rejected)
    raise ChannelUnusableError(
        f"no balanced slice of {size} in {attempts} draws; on this corpus a slice that "
        "size cannot be drawn without a compartment scan being able to name it"
    )


def latent_blind_fleet(
    n_blind: int,
    size: int,
    slice_: LatentSlice,
    *,
    slip_rate: float = 0.0,
) -> tuple[AnalystPolicy, ...]:
    """`blind_fleet`, with the shared blind spot keyed on reports instead of a channel.

    Everything else is held: the same fleet size, the same compartment-shedding ruling
    (required for the same structural reason -- under the fail-closed default the affected
    tasks are escalated on disclosure grounds and contribute no verdict at all), and the
    same independent slip rate. The single difference is what the blind analysts discount,
    which is what makes this the negative control for a detector that scans channels.
    """
    sighted = [
        AnalystPolicy(f"sighted-{i}", release_policy=DROP_COMPARTMENTS, slip_rate=slip_rate)
        for i in range(size - n_blind)
    ]
    blind = [
        AnalystPolicy(
            f"blind-{i}",
            distrusted_reports=slice_.distrusted,
            release_policy=DROP_COMPARTMENTS,
            slip_rate=slip_rate,
        )
        for i in range(n_blind)
    ]
    return tuple(sighted + blind)
