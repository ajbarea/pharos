"""What the aggregator may see, and nothing else.

This module exists because the deployability constraint is the substance of several
findings rather than an implementation detail of one script. A selection policy, a
detector, or an abstention rule is only a proposal if it reads what a real aggregator
holds: per-task vote sums, per-task contributor counts, the estimator's own posterior,
and public corpus structure. A method that reaches past those is an oracle wearing a
method's name.

It lived in `scripts/measure_audit_policy.py` and was imported from there by seven other
scripts, which made the line it draws a property of an experiment file rather than of the
package. Anyone using this testbed to measure their own policy needs the type, not a copy
of it.
"""

from dataclasses import dataclass, field, replace

from pharos.analyst import Proposal, evidence_shown
from pharos.governance.fleet import BLIND, MASK_SEED, blind_fleet, contributions_for
from pharos.inference import FederatedDawidSkene, federated_dawid_skene, partition_by_contributor
from pharos.tasks import TriageTask

__all__ = ["ServerObservation", "fleet_view", "observe", "observe_with_estimate"]


@dataclass(frozen=True, slots=True)
class ServerObservation:
    """Exactly what a selection policy is allowed to read.

    A dataclass rather than a loose dict because the deployability constraint is the
    substance of this measurement: a policy that reaches past these fields is reading
    something the aggregator does not have, and the result would be an oracle wearing a
    method's name. Finding 12 made that mistake in the other direction and had to be
    corrected; making the permitted view a type is how this one avoids it.

    `votes` and `seen` are the two per-task sums the protocol already reveals.
    `posterior` is the estimator's own output, which the server holds by construction.
    `evidence` and `carries` are public corpus structure -- how much evidence a task
    shows, and whether it carries a named channel -- which finding 22's detector
    already reads and is deployable for reading. They are empty unless a caller
    supplies them, and `observe` never does.

    What must never appear is a per-analyst field. That is the line, and it is asserted
    on `__slots__` in the tests rather than left to this docstring.
    """

    votes: dict[str, float]
    seen: dict[str, float]
    posterior: dict[str, float]
    #: How much evidence each task shows, which is public corpus structure and the
    #: same quantity finding 22's detector conditions on. Empty where unused.
    evidence: dict[str, int] = field(default_factory=dict)
    #: Which tasks carry the channel finding 22's detector named, when it has fired.
    #: Empty until then, and empty is the honest default: a policy may not select by
    #: provenance on a fleet nobody has shown to be channel-blind.
    #:
    #: Reading this does NOT widen what a policy may see. Finding 22 established that
    #: the public structure of the corpus is available to a deployment -- its detector
    #: reads exactly this, conditions the verdict rate on it, and is deployable for
    #: that reason. What stays forbidden is ground truth and the per-analyst stream.
    carries: dict[str, bool] = field(default_factory=dict)

    def margin(self, task: str) -> float:
        """How evenly the fleet split on this task. 0.0 is a dead heat."""
        n = self.seen.get(task, 0.0)
        if not n:
            return 1.0
        return abs(2.0 * self.votes.get(task, 0.0) - n) / n

    def rates(self) -> dict[str, float]:
        """Each task's share of significant verdicts, for tasks anybody saw.

        Arithmetic on the two sums this class already holds, so it belongs beside `margin`
        rather than in the channel detector that happens to be its heaviest reader. It sat
        there as a free function taking a view, which meant reading a rate off a view
        required importing the module for scanning channels.
        """
        return {task: self.votes[task] / self.seen[task] for task in self.seen if self.seen[task]}


def observe_with_estimate(
    partitioned: dict[str, list[tuple[str, bool]]],
) -> tuple[ServerObservation, FederatedDawidSkene]:
    """The aggregator's view *and* the fit behind it, for callers that need both.

    `observe` discards the estimate, so a caller wanting `converged` or the estimate's
    own labels had no way to ask for them and re-ran the fit instead. Finding 30's
    measurement did exactly that three times per cell -- once inside `observe`, once for
    the labels, and once more inside `verdict_rates`. Returning both is how one fit serves
    every reader of it.

    `observe` stays the narrow door: a selection policy must not see the estimate object,
    only the posterior the server holds. This is for measurement scripts assembling a
    cell, not for anything a policy is handed.
    """
    votes: dict[str, float] = {}
    seen: dict[str, float] = {}
    for rows in partitioned.values():
        for task, verdict in rows:
            votes[task] = votes.get(task, 0.0) + (1.0 if verdict else 0.0)
            seen[task] = seen.get(task, 0.0) + 1.0
    estimate = federated_dawid_skene(partitioned, seed=MASK_SEED)
    return ServerObservation(votes, seen, dict(estimate.posterior)), estimate


def observe(partitioned: dict[str, list[tuple[str, bool]]]) -> ServerObservation:
    """The aggregator's view, assembled the way finding 18's protocol produces it."""
    return observe_with_estimate(partitioned)[0]


def fleet_view(
    tasks: list[TriageTask],
    proposals: dict[str, Proposal],
    *,
    n_blind: int,
    fleet: int,
    slip_rate: float,
    seed: int,
) -> tuple[ServerObservation, dict[str, bool], bool]:
    """One fleet's aggregate view, the labels it would publish, and whether EM converged.

    Convergence travels with the labels because it has to: at the noise levels where a
    healthy fleet breaks at all, the fit stops converging, and a risk column computed off
    a fit that ran out of iterations is a different quantity from one computed off a fit
    that finished. This package publishes flags rather than filtering on them.

    The channel map is supplied at every share, including shares where finding 22's
    detector would not have fired, because two of the cells the abstention measurement
    needs are exactly those: a healthy fleet handed a channel is the false-detection
    control, and its cost is a number that measurement reports.
    """
    flat = contributions_for(
        blind_fleet(n_blind, fleet, slip_rate=slip_rate), tasks, proposals, seed=seed
    )
    partitioned = partition_by_contributor(flat)
    view = observe(partitioned)
    by_id = {t.task_id: t for t in tasks}
    carries = {
        task: any(BLIND in r.label.compartments for r in by_id[task].sources)
        for task in view.posterior
    }
    evidence = {task: len(evidence_shown(by_id[task])) for task in view.posterior}
    estimate = federated_dawid_skene(partitioned, seed=MASK_SEED)
    return (
        replace(view, carries=carries, evidence=evidence),
        estimate.labels(),
        estimate.converged,
    )
