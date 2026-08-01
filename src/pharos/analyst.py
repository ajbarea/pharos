"""The analyst whose accept, revise, and reject decisions a fleet learns from.

Step 3 of the build order asks whether the decision rule is learnable from an
analyst's *review of a proposal* rather than from a clean label. Finding 6 settled
the easier question: on clean labels the rule is learnable by gradient descent.
Review is a weaker signal in three separable ways, and this module exists to make
each of them a parameter rather than an assumption.

**A policy, not a person.** Persona-prompted language models are the usual way to
manufacture annotator variation, and the perspectivist literature is consistent
that they explain only a small share of real annotator variance and compress
disagreement rather than reproducing it. A simulated analyst here is therefore a
decision procedure with named parameters, in the same spirit as the rest of the
corpus: ground truth by construction. It supports claims of the form *at this noise
rate and this feedback bandwidth the rule stops being recoverable*, and it supports
no claim whatever about what a human analyst would do. The corpus metadata already
caps behavioural claims for that reason (`rai:dataLimitations`, which names
synthetic analysts explicitly); nothing here relaxes that cap.

The three ways a review is weaker than a label:

- **Indirect.** A reviewer is shown one proposed verdict and says whether it will
  do. A rejection names a wrong answer, not the right one.
- **Ambiguous.** A proposal carries a verdict *and* the label it would be released
  under, and a reviewer can object to either. When the objection does not say
  which, the learner has a credit-assignment problem that a clean label never
  poses. `names_grounds` is that switch, and it is the axis this module was built
  to expose: it is a property of the governed boundary, not of triage, so it has no
  analogue in the usual preference-learning setup.
- **Noisy and scarce.** Reviewers slip, and only sometimes take the trouble to
  supply a correction. `slip_rate` and `revision_rate` are those two.

Feedback shapes, and what each supports downstream: an accept or a bare reject is
an unpaired binary judgement on one response, which is the signal KTO consumes
(Ethayarajh et al., arXiv:2402.01306); a revision additionally carries a corrected
verdict, which is the supervised case finding 6 already measured. `supervision_yield`
is how much of the second survives a given parameter setting.

Deterministic: every draw is keyed by `(seed, analyst name, task id)`, so a
decision does not depend on iteration order, ensemble membership, or how many
tasks were reviewed before it.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from pharos.labels import (
    Capacity,
    Compartment,
    DeclassificationPolicy,
    Label,
    Sensitivity,
    declassify,
)
from pharos.tasks import TriageTask
from pharos.world import SIGNIFICANT_PATTERN

#: A fleet aggregator cleared below the enclaves feeding it, which is what makes
#: sharing a downgrade at all. Same ceiling as `scripts/measure_triage_lift.py`,
#: so a review runs against the boundary the triage numbers were measured at.
DEFAULT_CEILING = Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)

#: The fail-closed default from finding 2: an eligible output drops to OPEN but
#: keeps every compartment it inherited.
KEEP_COMPARTMENTS = DeclassificationPolicy()

#: The one policy ruling finding 2 showed the federation's viability turns on.
DROP_COMPARTMENTS = DeclassificationPolicy(drop_compartments=True)


class Action(StrEnum):
    """What a reviewer did with a proposal."""

    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


class Ground(StrEnum):
    """What a reviewer objected to.

    Two grounds rather than one because a proposal is two assertions. A verdict can
    be right while the label it would be released under is wrong, and the reverse,
    and a learner that cannot tell those apart is being asked to fix an error it
    cannot locate.
    """

    VERDICT = "verdict"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class Proposal:
    """One verdict offered for review, and the label it would be released under."""

    task_id: str
    verdict: bool
    release: Label


@dataclass(frozen=True, slots=True)
class Decision:
    """A review, as the learner receives it.

    Deliberately carries nothing the reviewer did not disclose. `grounds` is empty
    on a rejection by a reviewer who does not name grounds, which is
    indistinguishable in this record from an accept's empty grounds *except* by
    `action` -- exactly the ambiguity being measured. Ground truth for scoring is
    recomputed by the measurement, never read off the decision.
    """

    task_id: str
    analyst: str
    action: Action
    grounds: frozenset[Ground] = frozenset()
    corrected_verdict: bool | None = None
    corrected_release: Label | None = None

    @property
    def is_supervised(self) -> bool:
        """Whether this decision hands the learner a usable target verdict.

        An accept does: the proposal it accepted is the target. A revision does.
        A bare rejection does not, and on a binary output that is a narrower gap
        than it looks -- the complement of a rejected verdict is the other class.
        The gap is real for the release label, where the complement of one wrong
        label is the rest of the lattice.
        """
        return self.corrected_verdict is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "analyst": self.analyst,
            "action": str(self.action),
            "grounds": sorted(str(g) for g in self.grounds),
            "corrected_verdict": self.corrected_verdict,
            "corrected_release": _label_text(self.corrected_release),
        }


def _label_text(label: Label | None) -> str | None:
    if label is None:
        return None
    inner = ",".join(sorted(str(c) for c in label.compartments))
    return f"{label.sensitivity.name}[{inner}]@{label.capacity.name}"


def evidence_shown(task: TriageTask) -> frozenset[str]:
    """The defining facts a reviewer can actually see in the task's reports.

    Read from the reports rather than from the underlying event, because a reviewer
    decides from what is in front of them. That distinction is not academic here:
    before the coverage fix behind finding 3, only 34% of significant events
    rendered all three of their defining facts, and an analyst scored against the
    event rather than against the page would have been marked wrong for reading
    correctly.
    """
    shown: set[str] = set()
    for report in task.sources:
        shown |= set(report.fact_ids)
    return frozenset(shown & SIGNIFICANT_PATTERN)


@dataclass(frozen=True, slots=True)
class AnalystPolicy:
    """One reviewer's standard, made explicit.

    `escalation_threshold` is how many of the three defining facts the reviewer
    needs before calling an event significant. Three is the world's own rule. Two
    and one are the over-escalating reviewer, and they matter because that is the
    same axis every model in finding 3b failed on: recall 1.000 at precision 0.395
    to 0.500 is what a threshold-of-one reviewer looks like. A reviewer and a model
    that fail the same way agree for the wrong reason, and an ensemble that cannot
    represent that cannot detect it.

    `release_policy` is the declassification ruling the reviewer would apply, and
    `release_ceiling` the aggregator they would release to. Finding 2 showed
    eligibility is bimodal on precisely this ruling, so two reviewers who agree on
    every verdict can still disagree on every release.
    """

    name: str
    escalation_threshold: int = len(SIGNIFICANT_PATTERN)
    release_policy: DeclassificationPolicy = KEEP_COMPARTMENTS
    release_ceiling: Label = DEFAULT_CEILING
    slip_rate: float = 0.0
    revision_rate: float = 1.0
    names_grounds: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.escalation_threshold <= len(SIGNIFICANT_PATTERN):
            raise ValueError(
                f"escalation_threshold must be 1..{len(SIGNIFICANT_PATTERN)}, "
                f"got {self.escalation_threshold}"
            )
        for field_name in ("slip_rate", "revision_rate"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1], got {value}")

    def verdict_for(self, task: TriageTask, rng: random.Random) -> bool:
        """This reviewer's own call on `task`, after their slip rate.

        The slip flips the call rather than abstaining. An analyst who notices they
        are unsure asks someone; the failure worth modelling is the one that reaches
        the record looking exactly like a considered judgement.
        """
        held = len(evidence_shown(task)) >= self.escalation_threshold
        return not held if rng.random() < self.slip_rate else held

    def release_for(self, task: TriageTask) -> Label:
        """The label this reviewer would release the verdict under."""
        return declassify(task.label, self.release_policy)

    def permits(self, release: Label) -> bool:
        """Whether this reviewer would let `release` leave at their ceiling."""
        return self.release_ceiling.dominates(release)

    def review(self, task: TriageTask, proposal: Proposal, *, seed: int) -> Decision:
        """Review one proposal.

        Objections are collected on both grounds before any decision is taken, so a
        reviewer who dislikes the verdict *and* the release does not silently drop
        the second objection. Whether the learner is told which is the
        `names_grounds` switch, applied last.
        """
        rng = random.Random(f"{seed}:{self.name}:{task.task_id}")

        own_verdict = self.verdict_for(task, rng)
        own_release = self.release_for(task)

        grounds: set[Ground] = set()
        if own_verdict != proposal.verdict:
            grounds.add(Ground.VERDICT)
        if not self.permits(proposal.release):
            grounds.add(Ground.RELEASE)

        if not grounds:
            return Decision(
                task_id=task.task_id,
                analyst=self.name,
                action=Action.ACCEPT,
                corrected_verdict=proposal.verdict,
                corrected_release=proposal.release,
            )

        # The revision draw is taken from the same stream whether or not it is
        # used, so turning `names_grounds` off does not shift every later draw and
        # silently change the verdicts too. The two axes have to move independently
        # or the experiment cannot attribute anything to either.
        revises = rng.random() < self.revision_rate
        disclosed = frozenset(grounds) if self.names_grounds else frozenset()
        if not revises:
            return Decision(
                task_id=task.task_id,
                analyst=self.name,
                action=Action.REJECT,
                grounds=disclosed,
            )
        return Decision(
            task_id=task.task_id,
            analyst=self.name,
            action=Action.REVISE,
            grounds=disclosed,
            corrected_verdict=own_verdict,
            corrected_release=own_release,
        )


#: A grid over the axes rather than a cast of characters.
#:
#: Every member differs from `by-the-book` in exactly one respect, so a difference
#: in the divergence report attributes to one axis. Naming them after people would
#: invite reading the spread as a finding about analysts; it is a sensitivity
#: analysis over parameters this module chose.
DEFAULT_ENSEMBLE: tuple[AnalystPolicy, ...] = (
    AnalystPolicy("by-the-book"),
    AnalystPolicy("two-of-three", escalation_threshold=2),
    AnalystPolicy("any-one", escalation_threshold=1),
    AnalystPolicy("releaser", release_policy=DROP_COMPARTMENTS),
    AnalystPolicy("inattentive", slip_rate=0.15),
    AnalystPolicy("terse", revision_rate=0.25),
    AnalystPolicy("unexplained", names_grounds=False),
)


def review_all(
    policies: Iterable[AnalystPolicy],
    tasks: Sequence[TriageTask],
    proposals: dict[str, Proposal],
    *,
    seed: int,
) -> list[Decision]:
    """Every policy's review of every task that has a proposal.

    Tasks without a proposal are skipped rather than given a default one: a review
    of a verdict nobody offered is not an observation.
    """
    return [
        policy.review(task, proposals[task.task_id], seed=seed)
        for policy in policies
        for task in tasks
        if task.task_id in proposals
    ]


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """How much an ensemble's reviews coincide, and how much of that is chance."""

    n_items: int
    n_raters: int
    observed: float
    expected: float
    kappa: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_items": self.n_items,
            "n_raters": self.n_raters,
            "observed_agreement": round(self.observed, 4),
            "expected_agreement": round(self.expected, 4),
            "fleiss_kappa": round(self.kappa, 4),
        }


def action_agreement(decisions: Iterable[Decision]) -> AgreementReport:
    """Fleiss' kappa over the accept / revise / reject trichotomy.

    Items reviewed by fewer than two raters are dropped, and items are required to
    carry the same number of raters, because Fleiss' correction is defined per item
    on a fixed rater count. A ragged panel would make the chance term depend on
    which items happened to be reviewed twice.

    Returns kappa 0.0 when every rater chose the same category everywhere: the
    chance term is then 1 and the statistic is undefined rather than perfect.
    """
    by_task: dict[str, list[Action]] = defaultdict(list)
    for decision in decisions:
        by_task[decision.task_id].append(decision.action)

    counts = Counter(len(actions) for actions in by_task.values())
    if not counts:
        return AgreementReport(0, 0, 0.0, 0.0, 0.0)
    n_raters = max(counts, key=lambda size: (counts[size], size))
    items = [actions for actions in by_task.values() if len(actions) == n_raters]
    if n_raters < 2 or not items:
        return AgreementReport(len(items), n_raters, 0.0, 0.0, 0.0)

    categories = tuple(Action)
    tallies = [Counter(actions) for actions in items]
    n_items = len(items)

    observed = (
        sum(
            (sum(tally[c] ** 2 for c in categories) - n_raters) / (n_raters * (n_raters - 1))
            for tally in tallies
        )
        / n_items
    )
    expected = sum(
        (sum(tally[c] for tally in tallies) / (n_items * n_raters)) ** 2 for c in categories
    )
    kappa = 0.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return AgreementReport(n_items, n_raters, observed, expected, kappa)


@dataclass(frozen=True, slots=True)
class SupervisionYield:
    """How much of a review stream a learner can actually train on."""

    n_decisions: int
    accepted: int
    revised: int
    rejected: int
    supervised: int
    located: int
    correct_targets: int

    @property
    def supervised_share(self) -> float:
        return self.supervised / self.n_decisions if self.n_decisions else 0.0

    @property
    def located_share(self) -> float:
        """Share of objections whose grounds the learner was told."""
        objections = self.revised + self.rejected
        return self.located / objections if objections else 0.0

    @property
    def target_accuracy(self) -> float:
        """Of the targets a learner would train on, the share that are right.

        The number that decides whether review is usable at all. A supervised share
        of 1.0 is worthless if the targets it hands over are wrong, and an
        over-escalating reviewer hands over wrong targets confidently.
        """
        return self.correct_targets / self.supervised if self.supervised else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "n_decisions": self.n_decisions,
            "accepted": self.accepted,
            "revised": self.revised,
            "rejected": self.rejected,
            "supervised": self.supervised,
            "supervised_share": round(self.supervised_share, 4),
            "located": self.located,
            "located_share": round(self.located_share, 4),
            "correct_targets": self.correct_targets,
            "target_accuracy": round(self.target_accuracy, 4),
        }


def supervision_yield(decisions: Iterable[Decision], truth: dict[str, bool]) -> SupervisionYield:
    """Score a review stream for what a learner could do with it.

    `truth` is the world's own significance per task, supplied by the caller rather
    than read from a decision, so a reviewer's confidence never enters its own score.
    """
    n = accepted = revised = rejected = supervised = located = correct = 0
    for decision in decisions:
        n += 1
        if decision.action is Action.ACCEPT:
            accepted += 1
        elif decision.action is Action.REVISE:
            revised += 1
        else:
            rejected += 1
        if decision.action is not Action.ACCEPT and decision.grounds:
            located += 1
        if decision.corrected_verdict is not None:
            supervised += 1
            if decision.corrected_verdict == truth.get(decision.task_id):
                correct += 1
    return SupervisionYield(n, accepted, revised, rejected, supervised, located, correct)


@dataclass(frozen=True, slots=True)
class ReleaseRecovery:
    """Whether objecting to a release ever produces one that can leave.

    Finding 2 showed federation eligibility is bimodal on a single ruling. Under
    review that bimodality reappears as two reviewers raising the *same* objection
    and supplying incompatible corrections: one that keeps the compartment and so
    stays blocked, one that sheds it and so clears the ceiling. An objection rate
    alone cannot distinguish them, which is why this is counted separately.
    """

    objections: int
    corrections: int
    releasable_after: int

    @property
    def recovery_rate(self) -> float:
        return self.releasable_after / self.objections if self.objections else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "objections": self.objections,
            "corrections": self.corrections,
            "releasable_after": self.releasable_after,
            "recovery_rate": round(self.recovery_rate, 4),
        }


def release_recovery(
    decisions: Iterable[Decision],
    proposals: dict[str, Proposal],
    *,
    ceiling: Label = DEFAULT_CEILING,
) -> ReleaseRecovery:
    """Count blocked releases, and how many corrections clear `ceiling`.

    A decision counts here when the proposal it reviewed was itself unreleasable,
    which is a fact about the proposal rather than about what the reviewer chose to
    disclose. Reading it off `grounds` instead would silently exclude every reviewer
    who does not name grounds, and those are the reviewers the comparison is about.
    """
    objections = corrections = releasable = 0
    for decision in decisions:
        proposal = proposals.get(decision.task_id)
        if proposal is None or ceiling.dominates(proposal.release):
            continue
        objections += 1
        if decision.corrected_release is None:
            continue
        corrections += 1
        if ceiling.dominates(decision.corrected_release):
            releasable += 1
    return ReleaseRecovery(objections, corrections, releasable)


def with_name(policy: AnalystPolicy, name: str) -> AnalystPolicy:
    """`policy` under a different name, for sweeps that vary one field at a time."""
    return replace(policy, name=name)
