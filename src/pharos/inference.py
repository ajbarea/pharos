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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pharos.telemetry import get_logger, record_routine

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

    if not converged:
        # A silent quality failure otherwise. EM that hits the cap has not found a
        # fixed point, so the error rates and labels below are wherever the last
        # iteration happened to leave them, and nothing in the returned object's
        # numbers looks different from a converged run.
        get_logger().warning(
            "inference.em_did_not_converge",
            extra={
                "event": "inference.em_did_not_converge",
                "iterations": iterations_run,
                "max_iters": max_iters,
                "tasks": len(tasks),
                "workers": len(workers),
            },
        )
    # Routine: a fleet sweep runs EM once per drawn fleet, which was 900 identical
    # lines in one script and drowned that script's own result. Non-convergence above
    # stays at WARNING, because that one is never routine.
    record_routine(
        "inference.dawid_skene",
        prevalence,
        converged=converged,
        iterations=iterations_run,
        tasks=len(tasks),
        workers=len(workers),
    )
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


# --------------------------------------------------------------------- GLAD -----
#
# Dawid-Skene attributes every disagreement to the annotator. The other standard
# explanation is the item: some cases are genuinely near the boundary and everyone
# struggles with them. Whitehill et al. (NIPS 2009) estimate labeler ability, item
# difficulty and the true label jointly, which is the model to reach for when both
# explanations are live.
#
# Both are live here, and worse than live: they coincide. This corpus's significant
# class is a conjunction of three facts, and several background patterns carry two of
# the three, so the objectively hardest routine items are exactly the ones a reviewer
# holding a two-of-three standard gets wrong. An estimator asked to apportion blame
# between "this analyst is wrong" and "this item is hard" is being asked to separate
# two hypotheses that predict identical data.
#
# Implemented so that can be measured rather than asserted.

#: Ability starts slightly positive: a labeler assumed adversarial at initialisation
#: converges to an inverted solution that fits equally well and is not the one meant.
INITIAL_ABILITY = 1.0

#: Log-difficulty starts at zero, i.e. every item equally hard, so any structure in the
#: estimate is learned rather than assumed.
INITIAL_LOG_DIFFICULTY = 0.0

#: M-step gradient ascent. GLAD has no closed form for its parameters.
GLAD_STEPS = 40
GLAD_RATE = 0.05


@dataclass(frozen=True, slots=True)
class Glad:
    """Estimated truth, per-labeler ability, and per-item difficulty."""

    posterior: dict[str, float]
    ability: dict[str, float]
    log_difficulty: dict[str, float]
    iterations: int
    converged: bool

    def labels(self) -> dict[str, bool]:
        return {task: p >= 0.5 for task, p in self.posterior.items()}

    def difficulty(self, task: str) -> float:
        """Item difficulty on the natural scale. Higher is harder."""
        return math.exp(self.log_difficulty.get(task, 0.0))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    e = math.exp(max(x, -60.0))
    return e / (1.0 + e)


def glad(
    contributions: Sequence[tuple[str, str, bool]],
    *,
    max_iters: int = MAX_ITERS,
    tolerance: float = TOLERANCE,
) -> Glad:
    """Joint EM over true labels, labeler ability, and item difficulty.

    Probability that labeler `j` reports item `i` correctly is
    `sigmoid(ability_j * exp(-log_difficulty_i))`, the parameterisation of Whitehill
    et al. Ability may go negative, which represents a labeler who is reliably wrong;
    difficulty is positive by construction.

    Takes no ground truth, in the same way `dawid_skene` takes none. The point of
    running both is that they differ in what they are allowed to blame.
    """
    if not contributions:
        return Glad({}, {}, {}, 0, True)

    tasks = sorted({t for t, _, _ in contributions})
    workers = sorted({w for _, w, _ in contributions})
    by_task: dict[str, list[tuple[str, bool]]] = {t: [] for t in tasks}
    for task, who, verdict in contributions:
        by_task[task].append((who, verdict))

    posterior = {
        t: (sum(1 for _, v in rows if v) / len(rows) if rows else 0.5)
        for t, rows in by_task.items()
    }
    ability = dict.fromkeys(workers, INITIAL_ABILITY)
    log_difficulty = dict.fromkeys(tasks, INITIAL_LOG_DIFFICULTY)

    converged = False
    iterations_run = 0
    for _step in range(1, max_iters + 1):
        iterations_run += 1

        # M step: gradient ascent on the expected complete-data log-likelihood.
        for _ in range(GLAD_STEPS):
            grad_a = dict.fromkeys(workers, 0.0)
            grad_b = dict.fromkeys(tasks, 0.0)
            for task, rows in by_task.items():
                inv = math.exp(-log_difficulty[task])
                for who, verdict in rows:
                    p_correct = _sigmoid(ability[who] * inv)
                    # Expected indicator that this report was correct.
                    target = posterior[task] if verdict else 1.0 - posterior[task]
                    residual = target - p_correct
                    grad_a[who] += residual * inv
                    grad_b[task] += -residual * ability[who] * inv
            for w in workers:
                ability[w] += GLAD_RATE * grad_a[w]
            for t in tasks:
                log_difficulty[t] += GLAD_RATE * grad_b[t]

        # E step.
        moved = 0.0
        for task, rows in by_task.items():
            inv = math.exp(-log_difficulty[task])
            log_pos = log_neg = 0.0
            for who, verdict in rows:
                p_correct = min(max(_sigmoid(ability[who] * inv), 1e-9), 1 - 1e-9)
                log_pos += math.log(p_correct if verdict else 1 - p_correct)
                log_neg += math.log((1 - p_correct) if verdict else p_correct)
            top = max(log_pos, log_neg)
            pos, neg = math.exp(log_pos - top), math.exp(log_neg - top)
            updated = pos / (pos + neg)
            moved = max(moved, abs(updated - posterior[task]))
            posterior[task] = updated

        if moved < tolerance:
            converged = True
            break

    if not converged:
        get_logger().warning(
            "inference.glad_did_not_converge",
            extra={
                "event": "inference.glad_did_not_converge",
                "iterations": iterations_run,
                "max_iters": max_iters,
            },
        )
    record_routine(
        "inference.glad",
        float(iterations_run),
        converged=converged,
        tasks=len(tasks),
        workers=len(workers),
    )
    return Glad(posterior, ability, log_difficulty, iterations_run, converged)
