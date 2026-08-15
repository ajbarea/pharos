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
from dataclasses import dataclass
from math import comb

from pharos.analyst import Action, AnalystPolicy, Proposal, evidence_shown
from pharos.disclosure import DROP_COMPARTMENTS
from pharos.labels import Compartment
from pharos.tasks import TriageTask
from pharos.telemetry import get_logger

__all__ = [
    "BLIND",
    "BLIND_RUNGS",
    "CHANNEL_ENTANGLEMENT_SLACK",
    "MASK_SEED",
    "REFUSED_EXIT",
    "RUNGS",
    "WRONG_THRESHOLD",
    "ChannelCheck",
    "ChannelUnusableError",
    "assert_channel_usable",
    "blind_fleet",
    "contributions_for",
    "exact_wrong_majority",
    "fleet_of",
    "ladder",
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


@dataclass(frozen=True, slots=True)
class AbstentionCell:
    """One population rate under one correlation structure."""

    rate: float
    structure: str
    schools: int
    draws: int
    wrong_majority_rate: float
    mean_consensus: float
    #: Expected agreement for Dawid-Skene, composed the same way as consensus.
    mean_dawid_skene: float
    worst_consensus: float
    agreement_if_majority: float | None
    agreement_if_not: float | None
    expected_agreement: float

    def as_dict(self) -> dict[str, object]:
        return {
            "rate": self.rate,
            "structure": self.structure,
            "schools": self.schools,
            "draws": self.draws,
            "wrong_majority_rate": round(self.wrong_majority_rate, 4),
            "mean_consensus": round(self.mean_consensus, 4),
            "mean_dawid_skene": round(self.mean_dawid_skene, 4),
            "worst_consensus": round(self.worst_consensus, 4),
            "agreement_if_majority": (
                None if self.agreement_if_majority is None else round(self.agreement_if_majority, 4)
            ),
            "agreement_if_not": (
                None if self.agreement_if_not is None else round(self.agreement_if_not, 4)
            ),
            "expected_agreement": round(self.expected_agreement, 4),
        }
