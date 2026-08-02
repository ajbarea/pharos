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

#: Ability starts at the prior mean below: a labeler assumed adversarial at
#: initialisation converges to an inverted solution that fits equally well and is not
#: the one meant.
INITIAL_ABILITY = 1.0

#: Log-difficulty starts at the prior mean, so any structure in the estimate is learned
#: rather than assumed.
INITIAL_LOG_DIFFICULTY = 1.0

#: The priors Whitehill et al. specify in section 3.1, and which this omitted at first.
#: Their words: "In our implementation we used Gaussian priors (mu = 1, sigma = 1) for
#: alpha. For beta, we need a prior that does not generate negative values. To do so we
#: re-parameterized beta = e^beta' and imposed a Gaussian prior (mu = 1, sigma = 1) on
#: beta'."
#:
#: Leaving them out does not give a slightly different GLAD, it gives an unregularised
#: MLE with unbounded parameters, and this repository published a convergence failure
#: as a property of the method on that basis. The log-difficulty here IS their beta',
#: since difficulty is reported as exp(log_difficulty), so the same prior applies
#: directly.
PRIOR_MEAN = 1.0
PRIOR_SIGMA = 1.0

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
            # The log-prior gradients from section 3.1. d/dx of log N(x; mu, sigma) is
            # -(x - mu) / sigma^2, which is what stops the ascent walking off: without
            # these terms the residual never reaches zero as the sigmoid saturates, so
            # the parameters climb without bound and EM never reaches a fixed point.
            precision = 1.0 / (PRIOR_SIGMA**2)
            for w in workers:
                grad_a[w] -= precision * (ability[w] - PRIOR_MEAN)
                ability[w] += GLAD_RATE * grad_a[w]
            for t in tasks:
                grad_b[t] -= precision * (log_difficulty[t] - PRIOR_MEAN)
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


# ----------------------------------------------------------------- CC-Rasch -----
#
# GLAD gives each annotator one ability number. Singer et al. (arXiv:2607.24622,
# 2026) point out what that costs: "a single ability parameter per annotator ...
# prevents them from distinguishing majority-class competence from minority-class
# competence." Their CC-Rasch model makes both ability and difficulty class-specific,
#
#     P(labels correctly | true class k) = sigmoid(ability[r][k] - difficulty[i][k])
#
# which is the natural fit for the failure this repository keeps measuring. A reviewer
# holding a two-of-three standard is not globally unreliable. They are exactly right on
# the significant class and exactly wrong on the routine items near the boundary, which
# is a class-conditional error and the one shape GLAD's single parameter cannot
# represent. Finding 17 noted that a different estimator might resist the confound; this
# is the strongest available candidate, so the caveat can be measured rather than
# stated.
#
# Implemented from the model definition, not adapted from their code.

#: Class-level means, from which per-annotator and per-item deviations are measured:
#: `ability[r][k] = mean_ability[k] + g[r][k]` and `difficulty[i][k] = mean_difficulty[k]
#: + h[i][k]`, exactly the decomposition in their equations for alpha and beta.
INITIAL_CLASS_ABILITY = 1.0
INITIAL_CLASS_DIFFICULTY = 0.0

#: The identifiability machinery, which is not optional and which a first version of
#: this omitted. Singer et al.: *"To interpret mu_alpha and mu_beta as the average
#: ability (respectively difficulty) we impose for any k: sum_i h_{i,k} = sum_r g_{r,k}
#: = 0"*, and *"Following Whitehill et al. (2009); Liu et al. (2026), we fix this
#: translation invariance with Gaussian priors: g_{r,k} ~ N(0, sigma^2_{a,k}), h_{i,k} ~
#: N(0, sigma^2_{b,k})"*.
#:
#: Without both, the model is translation-invariant per class and EM wanders between
#: mirror-image solutions: an early run here scored 1.000 on one composition and 0.717
#: on another with near-identical parameters, which is label switching rather than a
#: result.
CC_RASCH_DEVIATION_SIGMA = 1.0

#: A light ridge on the class means themselves, their `lambda_mu` term. The means are
#: identified once the deviations are centred, so this only keeps them finite.
CC_RASCH_MEAN_PENALTY = 0.01

CC_RASCH_STEPS = 40
CC_RASCH_RATE = 0.05


@dataclass(frozen=True, slots=True)
class CCRasch:
    """Estimated truth, per-labeler ability per class, per-item difficulty per class."""

    posterior: dict[str, float]
    ability: dict[str, dict[bool, float]]
    difficulty: dict[str, dict[bool, float]]
    iterations: int
    converged: bool

    def labels(self) -> dict[str, bool]:
        return {task: p >= 0.5 for task, p in self.posterior.items()}

    def mean_ability(self, who: str) -> float:
        """Ability averaged over classes, for comparison against GLAD's single number."""
        by_class = self.ability.get(who)
        return sum(by_class.values()) / len(by_class) if by_class else 0.0


def cc_rasch(
    contributions: Sequence[tuple[str, str, bool]],
    *,
    max_iters: int = MAX_ITERS,
    tolerance: float = TOLERANCE,
) -> CCRasch:
    """Joint EM over true labels, class-conditional ability, and class-conditional difficulty.

    The class-dependent counterpart to `glad`. Takes no ground truth, like the other
    two, so the three differ only in what they are permitted to blame: the annotator
    (`dawid_skene`), the annotator and the item (`glad`), or the annotator and the item
    separately per class (this).
    """
    if not contributions:
        return CCRasch({}, {}, {}, 0, True)

    classes = (False, True)
    tasks = sorted({t for t, _, _ in contributions})
    workers = sorted({w for _, w, _ in contributions})
    by_task: dict[str, list[tuple[str, bool]]] = {t: [] for t in tasks}
    for task, who, verdict in contributions:
        by_task[task].append((who, verdict))

    posterior = {
        t: (sum(1 for _, v in rows if v) / len(rows) if rows else 0.5)
        for t, rows in by_task.items()
    }
    ability = {w: dict.fromkeys(classes, INITIAL_CLASS_ABILITY) for w in workers}
    difficulty = {t: dict.fromkeys(classes, INITIAL_CLASS_DIFFICULTY) for t in tasks}

    converged = False
    iterations_run = 0
    for _step in range(1, max_iters + 1):
        iterations_run += 1

        # M step. The gradient is taken per class and weighted by the posterior mass
        # currently assigned to that class, which is what makes the two ability numbers
        # separable at all: a task the fit believes is routine contributes only to the
        # routine-class parameters.
        for _ in range(CC_RASCH_STEPS):
            grad_a = {w: dict.fromkeys(classes, 0.0) for w in workers}
            grad_b = {t: dict.fromkeys(classes, 0.0) for t in tasks}
            for task, rows in by_task.items():
                for k in classes:
                    weight = posterior[task] if k else 1.0 - posterior[task]
                    if weight <= 0.0:
                        continue
                    for who, verdict in rows:
                        p_correct = _sigmoid(ability[who][k] - difficulty[task][k])
                        residual = weight * ((1.0 if verdict == k else 0.0) - p_correct)
                        grad_a[who][k] += residual
                        grad_b[task][k] -= residual
            # The penalty, applied to the DEVIATION from each class mean rather than to
            # the parameter, which is what their `R(theta)` does. Penalising the raw
            # parameter would instead shrink every annotator toward zero ability.
            precision = 1.0 / (CC_RASCH_DEVIATION_SIGMA**2)
            for k in classes:
                mean_a = sum(ability[w][k] for w in workers) / len(workers)
                mean_b = sum(difficulty[t][k] for t in tasks) / len(tasks)
                for w in workers:
                    grad_a[w][k] -= precision * (ability[w][k] - mean_a)
                for t in tasks:
                    grad_b[t][k] -= precision * (difficulty[t][k] - mean_b)
                # lambda_mu, keeping the class means themselves finite.
                for w in workers:
                    grad_a[w][k] -= CC_RASCH_MEAN_PENALTY * mean_a / len(workers)
                for t in tasks:
                    grad_b[t][k] -= CC_RASCH_MEAN_PENALTY * mean_b / len(tasks)

            for w in workers:
                for k in classes:
                    ability[w][k] += CC_RASCH_RATE * grad_a[w][k]
            for t in tasks:
                for k in classes:
                    difficulty[t][k] += CC_RASCH_RATE * grad_b[t][k]

        # The centring constraint, imposed after the ascent rather than during it.
        #
        # The model depends on ability and difficulty only through their DIFFERENCE:
        # `sigmoid(ability[r][k] - difficulty[i][k])` is unchanged by adding the same
        # constant to both, which is the translation invariance the constraint exists
        # to remove. So the shift has to move both in the SAME direction. A first
        # version moved them in opposite directions by half the drift each, which
        # changes every sigmoid by the full drift and is not a reparameterisation at
        # all -- the observed-data log-likelihood fell between iterations, which EM
        # cannot do, and that is how it was found.
        for k in classes:
            gauge = sum(ability[w][k] for w in workers) / len(workers) - INITIAL_CLASS_ABILITY
            for w in workers:
                ability[w][k] -= gauge
            for t in tasks:
                difficulty[t][k] -= gauge

        # E step.
        moved = 0.0
        for task, rows in by_task.items():
            log_pos = log_neg = 0.0
            for who, verdict in rows:
                for k, accumulate in ((True, "pos"), (False, "neg")):
                    p = min(max(_sigmoid(ability[who][k] - difficulty[task][k]), 1e-9), 1 - 1e-9)
                    contribution = math.log(p if verdict == k else 1 - p)
                    if accumulate == "pos":
                        log_pos += contribution
                    else:
                        log_neg += contribution
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
            "inference.cc_rasch_did_not_converge",
            extra={
                "event": "inference.cc_rasch_did_not_converge",
                "iterations": iterations_run,
                "max_iters": max_iters,
            },
        )
    record_routine(
        "inference.cc_rasch",
        float(iterations_run),
        converged=converged,
        tasks=len(tasks),
        workers=len(workers),
    )
    return CCRasch(posterior, ability, difficulty, iterations_run, converged)
