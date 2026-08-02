"""Truth inference from disagreeing contributors, including the canonical estimator.

Finding 12 first compared three ways of turning a fleet's contributions into training
targets: pool everything, drop low-agreement contributors using ground truth, or take
each task's majority vote. Only the middle one estimates reliability, and it does so by
being handed the answer, which makes it a bound rather than a method.

That comparison had a hole, and the hole was the field's standard answer. Dawid and
Skene (1979) estimate per-contributor error rates *and* the true labels jointly by
expectation-maximization, using no ground truth at all. Surveys of truth inference
report it outperforming majority voting, and any claim that reliability cannot be
estimated has to survive it before it means anything.

It is implemented here rather than cited past for two reasons. It is the strongest
identity-based method, so it is the right upper bound for what identity buys, replacing
an oracle that cheats. And it is the method whose failure mode is most interesting: EM
started from the majority vote is drawn toward whatever the majority believes, so it
should inherit the same cliff at the majority crossing rather than escape it. Asserting
that without running it would be exactly the shortcut this module exists to close.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Laplace smoothing on the confusion counts. Without it a contributor who is right on
#: every task they touched gets a zero-probability cell, and one disagreement later
#: drives the posterior to a hard 0 or 1 that no further evidence can move.
SMOOTHING = 1.0

#: EM iteration cap. Convergence on this problem is fast; the cap exists so a
#: pathological input cannot spin, not because the default is expected to bind.
MAX_ITERS = 100

#: Stop when no posterior moves by more than this.
TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class DawidSkene:
    """Estimated truth and per-contributor error rates, with no ground truth used.

    `posterior[task]` is P(task is significant) given every contribution about it and
    the current error-rate estimates. `error_rates[who]` is a two-by-two confusion
    matrix indexed `[true][reported]`, which is the per-contributor reliability the
    weighting scheme in finding 10 would want.
    """

    posterior: dict[str, float]
    error_rates: dict[str, tuple[tuple[float, float], tuple[float, float]]]
    prevalence: float
    iterations: int
    converged: bool

    def labels(self) -> dict[str, bool]:
        """Hard labels, which is what a learner would actually train on."""
        return {task: p >= 0.5 for task, p in self.posterior.items()}

    def reliability(self, who: str) -> float:
        """A contributor's chance of reporting correctly, averaged over both classes.

        The scalar a reliability-weighted aggregator needs. Averaged over classes
        rather than weighted by prevalence so a contributor who is accurate only on the
        common class does not read as broadly reliable.
        """
        matrix = self.error_rates.get(who)
        if matrix is None:
            return 0.0
        return (matrix[0][0] + matrix[1][1]) / 2.0


def dawid_skene(
    contributions: Sequence[tuple[str, str, bool]],
    *,
    max_iters: int = MAX_ITERS,
    tolerance: float = TOLERANCE,
) -> DawidSkene:
    """Joint EM over true labels and per-contributor error rates.

    `contributions` is `(task, contributor, verdict)`. Nothing else is supplied: no
    ground truth, no prior on who is reliable, no identity beyond a stable handle to
    group a contributor's own reports.

    Initialized from the majority vote, which is the conventional start and is also the
    honest one for the question being asked here. If EM merely reproduces its
    initialization when the majority is wrong, that is a property of the method worth
    measuring rather than a choice to be tuned away.
    """
    if not contributions:
        return DawidSkene({}, {}, 0.0, 0, True)

    tasks = sorted({t for t, _, _ in contributions})
    workers = sorted({w for _, w, _ in contributions})
    by_task: dict[str, list[tuple[str, bool]]] = {t: [] for t in tasks}
    for task, who, verdict in contributions:
        by_task[task].append((who, verdict))

    # Majority vote initialization.
    posterior = {
        t: (sum(1 for _, v in rows if v) / len(rows) if rows else 0.5)
        for t, rows in by_task.items()
    }

    error_rates: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    prevalence = 0.5
    converged = False

    iterations_run = 0
    for _step in range(1, max_iters + 1):
        iterations_run += 1
        # M step: error rates and class prevalence from the current soft labels.
        counts = {w: [[SMOOTHING, SMOOTHING], [SMOOTHING, SMOOTHING]] for w in workers}
        for task, rows in by_task.items():
            p = posterior[task]
            for who, verdict in rows:
                reported = 1 if verdict else 0
                counts[who][1][reported] += p
                counts[who][0][reported] += 1.0 - p
        error_rates = {
            w: (
                (counts[w][0][0] / sum(counts[w][0]), counts[w][0][1] / sum(counts[w][0])),
                (counts[w][1][0] / sum(counts[w][1]), counts[w][1][1] / sum(counts[w][1])),
            )
            for w in workers
        }
        prevalence = sum(posterior.values()) / len(posterior)
        prevalence = min(max(prevalence, 1e-6), 1 - 1e-6)

        # E step: re-estimate each task's posterior from every report about it.
        moved = 0.0
        for task, rows in by_task.items():
            pos, neg = prevalence, 1.0 - prevalence
            for who, verdict in rows:
                reported = 1 if verdict else 0
                pos *= error_rates[who][1][reported]
                neg *= error_rates[who][0][reported]
            total = pos + neg
            updated = 0.5 if total <= 0 else pos / total
            moved = max(moved, abs(updated - posterior[task]))
            posterior[task] = updated

        if moved < tolerance:
            converged = True
            break

    return DawidSkene(posterior, error_rates, prevalence, iterations_run, converged)


def weighted_targets(
    contributions: Sequence[tuple[str, str, bool]],
    estimate: DawidSkene,
    *,
    floor: float,
) -> list[tuple[str, str, bool]]:
    """Contributions from contributors the estimate rates at or above `floor`.

    The deployable counterpart to finding 12's oracle: same shape, but the reliability
    is estimated from disagreement rather than read off the answer key.
    """
    return [c for c in contributions if estimate.reliability(c[1]) >= floor]


def agreement_with(labels: Mapping[str, bool], truth: Mapping[str, bool]) -> float:
    """Share of inferred labels matching the world. Scoring only, never fed to EM."""
    shared = [t for t in labels if t in truth]
    if not shared:
        return 0.0
    return sum(1 for t in shared if labels[t] == truth[t]) / len(shared)
