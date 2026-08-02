"""What a fleet's contribution stream discloses about the analysts contributing to it.

Every gate in this package decides one item at a time. `labels.shared_eligible` asks
whether *this* entry may train an adapter released at a ceiling; `disclosure.decide`
asks whether *this* derived label may leave, and at whose authority. Both are
per-item by construction, and both are correct at what they do.

A fleet does not disclose one item. It discloses a *stream*, and a stream carries
something no item in it carries: the set of tasks a given contributor was able to
weigh in on at all. That set is a function of the contributor's clearance, because an
analyst can only decide what an analyst can read. So a composition of individually
releasable items can identify the reader even when every item in it was ruled
releasable on its own.

This module measures that. The attack in `link` uses **no content whatsoever** -- not
the report text, not the verdict, not the label on the released output. It uses only
which task identifiers appear under a pseudonym. That is the sharpest available form
of the claim, because it cannot be answered by redacting text or by lowering output
capacity: there is no text in it to redact.

The adversary is assumed to know the corpus's label structure. For Pharos that is not
a modelling convenience but a fact about the artifact, since the corpus is published
with its labels. The federated-inference literature grants an adversary "auxiliary
public information" as a matter of course; here the auxiliary information is the
benchmark itself.
"""

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from random import Random

from pharos.labels import (
    Capacity,
    Compartment,
    DeclassificationPolicy,
    Label,
    Sensitivity,
    join,
    shared_eligible,
)
from pharos.tasks import TriageTask
from pharos.telemetry import record_routine

#: The ceiling a shared adapter is released under. OPEN with no compartments is the
#: only ceiling under which federation means anything: a shared model released into a
#: compartment is not shared, it is a second silo. Capacity is carried for the type
#: and ignored by `dominates`.
FLEET_CEILING = Label(Sensitivity.OPEN, frozenset(), Capacity.ENUM)


@dataclass(frozen=True, slots=True)
class Clearance:
    """One analyst's standing authority, and the beat it implies.

    The compartment set is the interesting part and the part the attack recovers. An
    analyst holding LIAISON is, in the fiction, the officer who works liaison
    reporting, so recovering the set is recovering what they work on.
    """

    analyst_id: str
    sensitivity: Sensitivity
    compartments: frozenset[Compartment]

    @property
    def label(self) -> Label:
        """The clearance as a lattice element, for `Label.dominates`."""
        return Label(self.sensitivity, self.compartments, Capacity.FREETEXT)

    def can_read(self, task: TriageTask) -> bool:
        """Whether this analyst may read *every* source of `task`.

        Every, not any. The triage rule is a conjunction over the reports, so an
        analyst who sees only part of an event is not deciding the same task, they
        are deciding a different and easier one. Admitting partial views would make
        the contribution set depend on a modelling choice rather than on the
        clearance, and the clearance is the variable under study.
        """
        return self.label.dominates(
            join([r.label for r in task.sources], capacity=Capacity.FREETEXT)
        )


@dataclass(frozen=True, slots=True)
class Contribution:
    """One analyst's decision on one task, as it reaches the aggregator.

    `analyst_id` is ground truth and is never read by the attack. `pseudonym` is what
    the aggregator actually sees. Keeping them as separate fields is what lets a
    control that breaks the link between them be measured rather than described:
    under pooling the pseudonym stops distinguishing anyone while the truth needed to
    score the attack survives outside the adversary's view.
    """

    analyst_id: str
    pseudonym: str
    task_id: str
    verdict: bool


class Mitigation(StrEnum):
    """A control applied to the stream before the aggregator sees it."""

    NONE = "none"
    K_ANONYMITY = "k_anonymity"
    SUBSAMPLE = "subsample"
    POOLED = "pooled"


def candidate_clearances() -> tuple[tuple[Sensitivity, frozenset[Compartment]], ...]:
    """Every clearance an adversary would enumerate.

    The full product of levels and compartment subsets, which is the honest candidate
    space: nothing about the design tells an adversary which cells are unused, so an
    attack searching a smaller space would be crediting itself with knowledge it was
    never given.
    """
    members = tuple(Compartment)
    subsets = [
        frozenset(combo)
        for size in range(len(members) + 1)
        for combo in itertools.combinations(members, size)
    ]
    return tuple((level, subset) for level in Sensitivity for subset in subsets)


def assign_fleet(n_analysts: int, *, seed: int) -> tuple[Clearance, ...]:
    """A fleet of `n_analysts` drawn from the candidate space.

    Drawn with replacement on purpose: two analysts sharing a beat is the ordinary
    case on a real watch floor, and it is what gives the guessing prior something to
    guess. Note that sharing a beat is *not* protective here. The attack recovers a
    compartment set rather than an identity, so a duplicate clearance under its own
    pseudonym is recovered exactly as easily as a unique one; only sharing a
    pseudonym helps, which is what `apply_pooling` does.
    """
    rng = Random(seed)
    space = candidate_clearances()
    return tuple(
        Clearance(f"A-{i:03d}", *space[rng.randrange(len(space))]) for i in range(n_analysts)
    )


def contribute(
    fleet: Sequence[Clearance],
    tasks: Sequence[TriageTask],
    *,
    policy: DeclassificationPolicy,
    ceiling: Label = FLEET_CEILING,
) -> tuple[Contribution, ...]:
    """The stream reaching the aggregator, after every per-item gate has passed.

    Two conditions, both already enforced elsewhere in this package: the analyst must
    be able to read the task, and the task's derived verdict must be eligible to leave
    at `ceiling`. Nothing here weakens either. The measurement's point is that
    satisfying both, item by item, is not sufficient.
    """
    return tuple(
        Contribution(c.analyst_id, c.analyst_id, t.task_id, t.significant)
        for c in fleet
        for t in tasks
        if c.can_read(t) and shared_eligible(t.label, ceiling, policy)
    )


def _observed(stream: Iterable[Contribution]) -> dict[str, frozenset[str]]:
    """Task identifiers per pseudonym. The entire adversary observation."""
    grouped: dict[str, set[str]] = {}
    for c in stream:
        grouped.setdefault(c.pseudonym, set()).add(c.task_id)
    return {k: frozenset(v) for k, v in grouped.items()}


def _reachable(
    level: Sensitivity,
    compartments: frozenset[Compartment],
    tasks: Sequence[TriageTask],
    *,
    policy: DeclassificationPolicy,
    ceiling: Label,
) -> frozenset[str]:
    """The task set a given clearance would produce, computed from public structure."""
    probe = Clearance("probe", level, compartments)
    return frozenset(
        t.task_id for t in tasks if probe.can_read(t) and shared_eligible(t.label, ceiling, policy)
    )


def _jaccard(predicted: frozenset[str], seen: frozenset[str]) -> float:
    """Agreement between a candidate's reachable set and what was observed.

    Two empty sets agree perfectly under the usual definition, and scoring that 1.0
    would report a contributor who was seen doing nothing as fully identified. It is
    0.0 here: no evidence is not agreement.
    """
    union = len(predicted | seen)
    return (len(predicted & seen) / union) if union else 0.0


@dataclass(frozen=True, slots=True)
class Linkage:
    """What the adversary concluded about one analyst.

    One per analyst rather than per pseudonym, because a control that merges
    pseudonyms must be scored as hiding people rather than as producing fewer of them.
    """

    analyst_id: str
    truth: frozenset[Compartment]
    inferred: frozenset[Compartment]
    anonymity_set: int
    exact: bool
    silent: bool


def link(
    stream: Iterable[Contribution],
    tasks: Sequence[TriageTask],
    fleet: Sequence[Clearance],
    *,
    policy: DeclassificationPolicy,
    ceiling: Label = FLEET_CEILING,
) -> tuple[Linkage, ...]:
    """Recover each analyst's compartment set from which tasks their pseudonym touched.

    For every candidate clearance the adversary computes the task set it would produce
    and scores it against what was observed under that pseudonym. Jaccard rather than
    exact match, so the attack degrades gracefully under controls that remove
    contributions without changing the clearance behind them.

    `anonymity_set` is the more informative of the two outputs. Exact recovery says
    the attack won; the anonymity set says by how much, and it is what a control
    actually moves. A control leaving the true clearance ranked first but tied with
    fifteen others has done real work that an accuracy column alone scores as a loss.
    It counts candidates tied at the top score, multiplied by the number of analysts
    sharing the pseudonym, since an inference that cannot be attributed to a person
    has not identified one.

    Compartment sets are compared, not levels. A level is a rank and is rarely secret;
    what an analyst is read into is the thing this testbed's motivating abstract
    promises not to leak.
    """
    observed = _observed(stream)
    sharing: dict[str, set[str]] = {}
    for c in stream:
        sharing.setdefault(c.pseudonym, set()).add(c.analyst_id)
    pseudonym_of = {c.analyst_id: c.pseudonym for c in stream}

    reachable = {
        (level, subset): _reachable(level, subset, tasks, policy=policy, ceiling=ceiling)
        for level, subset in candidate_clearances()
    }

    results: list[Linkage] = []
    for analyst in fleet:
        pseudonym = pseudonym_of.get(analyst.analyst_id)
        if pseudonym is None:
            # Contributed nothing the aggregator ever saw. Unlinkable for the one
            # reason no control can take credit for, so it is reported separately
            # rather than folded into the protected count.
            results.append(
                Linkage(analyst.analyst_id, analyst.compartments, frozenset(), 0, False, True)
            )
            continue

        seen = observed[pseudonym]
        scored = [
            (_jaccard(predicted, seen), subset) for (_, subset), predicted in reachable.items()
        ]
        best = max(score for score, _ in scored)
        tied = {subset for score, subset in scored if score == best}
        # A tie resolves toward the smallest compartment set: an adversary with no
        # separating evidence guesses the least-read-in analyst, which is the
        # conservative reading and declines to credit the attack for a coin flip.
        inferred = min(tied, key=lambda s: (len(s), sorted(s)))
        crowd = len(sharing[pseudonym])
        results.append(
            Linkage(
                analyst_id=analyst.analyst_id,
                truth=analyst.compartments,
                inferred=inferred,
                anonymity_set=len(tied) * crowd,
                exact=(len(tied) == 1 and crowd == 1 and inferred == analyst.compartments),
                silent=False,
            )
        )
    exact = sum(1 for r in results if r.exact)
    live = [r for r in results if not r.silent]
    # The attack's own outcome. Without it a change to the tie-breaking rule, or to
    # what counts as an identification, moves every published recovery figure with
    # nothing in a run's output to show that it did.
    # Routine: a control ladder calls this once per control, and an unlabelled row per
    # call is unreadable. The caller knows which control it is running and logs the
    # labelled outcome at INFO; this stays available at DEBUG for a single direct call.
    record_routine(
        "fleet.linkage",
        exact / len(results) if results else 0.0,
        analysts=len(results),
        exact=exact,
        silent=sum(1 for r in results if r.silent),
        pseudonyms=len(observed),
        mean_anonymity_set=(
            round(sum(r.anonymity_set for r in live) / len(live), 3) if live else 0.0
        ),
    )
    return tuple(results)


def identifiability_ceiling(
    tasks: Sequence[TriageTask],
    *,
    policy: DeclassificationPolicy,
    ceiling: Label = FLEET_CEILING,
) -> dict[str, int]:
    """How identifiable the corpus makes a clearance, before any fleet is drawn.

    A recovery rate depends on which analysts a draw happened to produce. This does
    not. It collapses the candidate clearances by the task set each would reach and
    counts how many compartment sets share each collapse, which bounds what any
    attack against a contributor-segmented stream could achieve. A reachable set
    belonging to exactly one compartment set identifies its holder outright, and no
    control that leaves the stream segmented by contributor prevents that.
    """
    collapsed: dict[frozenset[str], set[frozenset[Compartment]]] = {}
    for level, subset in candidate_clearances():
        reachable = _reachable(level, subset, tasks, policy=policy, ceiling=ceiling)
        collapsed.setdefault(reachable, set()).add(subset)
    sizes = sorted(len(v) for v in collapsed.values())
    return {
        "candidate_clearances": len(candidate_clearances()),
        "distinct_reachable_sets": len(collapsed),
        "uniquely_identifying_sets": sum(1 for s in sizes if s == 1),
        "largest_anonymity_class": max(sizes) if sizes else 0,
    }


def apply_k_anonymity(stream: Iterable[Contribution], k: int) -> tuple[Contribution, ...]:
    """Drop contributions on any task fewer than `k` distinct analysts reached.

    The control matched to the leak: the identifying signal is a rare task, since a
    task only one clearance can read names that clearance outright. Tasks everyone can
    read carry no information about who read them.
    """
    contributions = list(stream)
    reach: dict[str, set[str]] = {}
    for c in contributions:
        reach.setdefault(c.task_id, set()).add(c.analyst_id)
    return tuple(c for c in contributions if len(reach[c.task_id]) >= k)


def apply_rarity_suppression(
    stream: Iterable[Contribution], keep_fraction: float
) -> tuple[Contribution, ...]:
    """Keep only the `keep_fraction` of tasks the most analysts reached.

    The same idea as `apply_k_anonymity` with the threshold chosen by rank instead
    of as an absolute count, which is what makes it portable: the count that
    protects a fleet of 200 does nothing to a fleet of 20. It is also strictly
    better calibrated here, reaching the same protection for a quarter of the volume
    cost, because it removes the rarest tasks first rather than everything below a
    line drawn without reference to the distribution.

    Ties in reachability are broken by task identifier so the control is
    deterministic; an arbitrary but stable choice is what makes the artifact
    reproducible.
    """
    contributions = list(stream)
    reach: dict[str, set[str]] = {}
    for c in contributions:
        reach.setdefault(c.task_id, set()).add(c.analyst_id)
    ranked = sorted(reach, key=lambda t: (-len(reach[t]), t))
    kept = set(ranked[: max(1, int(len(ranked) * keep_fraction))])
    return tuple(c for c in contributions if c.task_id in kept)


def apply_subsample(
    stream: Iterable[Contribution], keep: float, *, seed: int
) -> tuple[Contribution, ...]:
    """Keep each contribution independently with probability `keep`.

    The cheap control, and the one to be suspicious of: it removes volume uniformly
    while the signal it needs to remove is concentrated in the rare tasks.
    """
    rng = Random(seed)
    return tuple(c for c in stream if rng.random() < keep)


def apply_pooling(stream: Iterable[Contribution]) -> tuple[Contribution, ...]:
    """Replace every pseudonym with one, the secure-aggregation analogue.

    Costs no volume at all. Whether it costs anything that matters depends on whether
    the aggregator needed per-contributor identity, and in this design it does not:
    personalization is the part that stays local, so the shared side never had a
    reason to know who sent what.
    """
    return tuple(replace(c, pseudonym="POOLED") for c in stream)
