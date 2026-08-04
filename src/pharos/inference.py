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

from pharos.secagg import MIN_PARTICIPANTS, secure_sum
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


# ------------------------------------------------- Dawid-Skene under a sum -----
#
# The estimator above needs a stable handle per contributor to group their reports,
# and finding 11 is that the handle is the leak. The open problem this repository
# has carried since is whether the estimate can be computed *under* aggregation
# instead of recovered after it.
#
# It can, and the reason is a property of the algorithm rather than a new
# approximation of it. Dawid-Skene's M step is per contributor and touches only that
# contributor's own reports, so it can run on the contributor's own machine and never
# be transmitted. Its E step needs, for each task, a *product* over contributors --
# which is a sum in logs, and a sum over clients is exactly what secure aggregation
# reveals and all it reveals. Split along that seam, the server holds two vectors of
# per-task log-likelihood and nobody's confusion matrix.
#
# The nearest published relative is FedDS (Dong, Zhu, Shang and Xue, *Information
# Sciences* 745:123425, 2026), which brings Dawid-Skene to federated learning to
# weight clients by estimated reliability without a labelled public dataset. It runs
# EM *at the server* over each client's prediction vector on an unlabelled public set,
# so the server sees precisely the per-client stream that finding 11 attacks, and the
# paper does not discuss secure aggregation. Its identifiability result (§4.2, Eq. 16)
# assumes each confusion matrix is diagonally dominant. That assumption is the subject
# of finding 12's cliff, and finding 16 prices how often a fleet violates it, which is
# why moving the computation under a sum was never going to fix the cliff and why
# measuring both at once is the honest way to report it.


def partition_by_contributor(
    contributions: Sequence[tuple[str, str, bool]],
) -> dict[str, list[tuple[str, bool]]]:
    """Regroup flat `(task, contributor, verdict)` rows into per-contributor streams.

    A conversion for callers holding finding 12's shape. The federated estimator takes
    the partitioned form because that is the form the protocol actually has -- each
    analyst holds only their own rows -- and building the flat list first is the
    centralization the protocol exists to avoid.
    """
    partitioned: dict[str, list[tuple[str, bool]]] = {}
    for task, who, verdict in contributions:
        partitioned.setdefault(who, []).append((task, verdict))
    return partitioned


@dataclass(frozen=True, slots=True)
class FederatedDawidSkene:
    """What the server holds after running Dawid-Skene without seeing a contributor.

    **There is no `error_rates` field and no `reliability(who)` method.** Their absence
    is the result, not an omission: each contributor's confusion matrix is computed and
    kept on that contributor's own machine, so a server-side object that could return
    one would be reporting something the protocol does not produce. A deployment that
    wants reliability weighting applies it locally, where the matrix lives.

    `readership` is the opposite case and is present for the opposite reason. The
    server genuinely does learn how many contributors reported on each task, because
    majority-vote initialization is a per-task mean and a mean needs its denominator.
    Carrying it here is what let finding 18 measure the channel instead of claiming
    the protocol has none.
    """

    posterior: dict[str, float]
    prevalence: float
    iterations: int
    converged: bool
    participants: int
    readership: dict[str, int]
    anchored: frozenset[str]

    def labels(self) -> dict[str, bool]:
        """Hard labels, which is what a learner would actually train on."""
        return {task: p >= 0.5 for task, p in self.posterior.items()}


def _stable_posterior(log_pos: float, log_neg: float) -> float:
    """P(significant) from two log-likelihoods, without leaving log space.

    The centralized implementation multiplies probabilities directly and falls back to
    0.5 when the product underflows to zero on both sides. This cannot underflow, so a
    fleet wide enough to underflow a product of per-contributor likelihoods would make
    the two disagree, and there this one is right.

    **That regime is not measured.** Finding 18 compares the two at a fleet of nine,
    where they agree to 3.8e-14, and sweeps the wrong-standard share rather than the
    fleet size. So the equivalence claim is scoped to the size it was run at, and the
    crossover point is unknown rather than absent.
    """
    difference = log_neg - log_pos
    if difference > 0:
        return 1.0 / (1.0 + math.exp(difference))
    scaled = math.exp(-difference)
    return scaled / (1.0 + scaled)


def federated_dawid_skene(
    partitioned: Mapping[str, Sequence[tuple[str, bool]]],
    *,
    seed: int = 0,
    max_iters: int = MAX_ITERS,
    tolerance: float = TOLERANCE,
    anchors: Mapping[str, bool] | None = None,
    min_participants: int = MIN_PARTICIPANTS,
) -> FederatedDawidSkene:
    """Dawid-Skene where the server learns only per-task sums.

    `partitioned` maps a contributor to their own `(task, verdict)` rows. Every
    quantity crossing to the server does so through `pharos.secagg.secure_sum`, whose
    return value has no per-client field to read, so the separation is enforced by the
    types rather than by care.

    `anchors` are tasks whose true label is supplied by an authority of record. Their
    posteriors are clamped and never revised, which is what breaks the relabelling
    degeneracy the wrong majority otherwise exploits. Finding 19 prices how many are
    needed.

    The mask seed advances every round. Reusing masks across rounds would let a server
    difference two rounds and recover a client's change, so the freshness is a property
    of the protocol and not a detail of the loop.
    """
    contributors = sorted(partitioned)
    tasks = sorted({task for rows in partitioned.values() for task, _ in rows})
    if not contributors or not tasks:
        return FederatedDawidSkene({}, 0.0, 0, True, len(contributors), {}, frozenset())

    index = {task: i for i, task in enumerate(tasks)}
    width = len(tasks)
    anchors = dict(anchors or {})
    anchored = frozenset(anchors) & set(tasks)

    def _clamp(posterior: dict[str, float]) -> None:
        """The authority's rulings, reasserted after every step that could move them."""
        for task in anchored:
            posterior[task] = 1.0 if anchors[task] else 0.0

    # Round zero: the majority vote, as two aggregates. `votes` sums the verdicts and
    # `seen` sums participation, and the server divides them. `seen` is the readership
    # channel in the open: a per-task mean cannot be formed without its denominator.
    vote_vectors: list[list[float]] = []
    seen_vectors: list[list[float]] = []
    for who in contributors:
        votes = [0.0] * width
        seen = [0.0] * width
        for task, verdict in partitioned[who]:
            votes[index[task]] += 1.0 if verdict else 0.0
            seen[index[task]] += 1.0
        vote_vectors.append(votes)
        seen_vectors.append(seen)

    vote_view = secure_sum(vote_vectors, seed=seed, min_participants=min_participants)
    seen_view = secure_sum(seen_vectors, seed=seed + 1, min_participants=min_participants)
    readership = {task: round(seen_view.total[index[task]]) for task in tasks}

    posterior = {
        task: (
            vote_view.total[index[task]] / seen_view.total[index[task]]
            if seen_view.total[index[task]] > 0
            else 0.5
        )
        for task in tasks
    }
    _clamp(posterior)

    prevalence = 0.5
    converged = False
    iterations_run = 0

    for step in range(1, max_iters + 1):
        iterations_run += 1

        # M step, entirely local. Each contributor updates their own confusion matrix
        # from their own rows and the broadcast posterior. Nothing here is sent.
        local_rates: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        for who in contributors:
            counts = [[SMOOTHING, SMOOTHING], [SMOOTHING, SMOOTHING]]
            for task, verdict in partitioned[who]:
                p = posterior[task]
                reported = 1 if verdict else 0
                counts[1][reported] += p
                counts[0][reported] += 1.0 - p
            local_rates[who] = (
                (counts[0][0] / sum(counts[0]), counts[0][1] / sum(counts[0])),
                (counts[1][0] / sum(counts[1]), counts[1][1] / sum(counts[1])),
            )

        prevalence = sum(posterior.values()) / len(posterior)
        prevalence = min(max(prevalence, 1e-6), 1 - 1e-6)

        # E step. Each contributor emits the log-likelihood their own reports assign to
        # each hypothesis, zero on every task they did not touch, and only the sum
        # crosses. Two aggregates rather than one interleaved vector so that a
        # coordinate means the same thing in both.
        neg_vectors: list[list[float]] = []
        pos_vectors: list[list[float]] = []
        for who in contributors:
            under_neg = [0.0] * width
            under_pos = [0.0] * width
            rates = local_rates[who]
            for task, verdict in partitioned[who]:
                reported = 1 if verdict else 0
                under_neg[index[task]] += math.log(rates[0][reported])
                under_pos[index[task]] += math.log(rates[1][reported])
            neg_vectors.append(under_neg)
            pos_vectors.append(under_pos)

        round_seed = seed + 2 * step
        neg_view = secure_sum(neg_vectors, seed=round_seed, min_participants=min_participants)
        pos_view = secure_sum(pos_vectors, seed=round_seed + 1, min_participants=min_participants)

        log_prior_pos = math.log(prevalence)
        log_prior_neg = math.log(1.0 - prevalence)
        moved = 0.0
        for task in tasks:
            # An anchored task's posterior is asserted, not estimated, so it is not a
            # free parameter and EM must not update it. Skipping it here rather than
            # overwriting and re-clamping is also what makes convergence reachable: the
            # clamp would otherwise restore the authority's value every iteration while
            # the movement check kept reporting the gap EM wanted to open, so `moved`
            # never fell below the tolerance and every anchored run burned the full
            # iteration cap. `logcheck` caught that as a non-convergence warning.
            if task in anchored:
                continue
            i = index[task]
            updated = _stable_posterior(
                log_prior_pos + pos_view.total[i], log_prior_neg + neg_view.total[i]
            )
            moved = max(moved, abs(updated - posterior[task]))
            posterior[task] = updated

        if moved < tolerance:
            converged = True
            break

    if not converged:
        get_logger().warning(
            "inference.federated_em_did_not_converge",
            extra={
                "event": "inference.federated_em_did_not_converge",
                "iterations": iterations_run,
                "max_iters": max_iters,
                "tasks": len(tasks),
                "workers": len(contributors),
            },
        )
    record_routine(
        "inference.federated_dawid_skene",
        prevalence,
        converged=converged,
        iterations=iterations_run,
        tasks=len(tasks),
        workers=len(contributors),
        anchors=len(anchored),
    )
    return FederatedDawidSkene(
        posterior,
        prevalence,
        iterations_run,
        converged,
        len(contributors),
        readership,
        anchored,
    )


# --------------------------------------------------------------------- GLAD -----
#
# Dawid-Skene blames only the annotator; Whitehill et al. (NIPS 2009) estimate labeler
# ability, item difficulty and the true label jointly. Both explanations are live here
# and they coincide: the objectively hardest routine items are exactly the ones a
# two-of-three reviewer gets wrong, so the estimator is asked to separate two hypotheses
# that predict identical data. Implemented so that can be measured rather than asserted.

#: Starts at the prior mean. A labeler assumed adversarial at initialisation converges
#: to an inverted solution that fits equally well and is not the one meant.
INITIAL_ABILITY = 1.0
INITIAL_LOG_DIFFICULTY = 1.0

#: Section 3.1: "we used Gaussian priors (mu = 1, sigma = 1) for alpha. For beta ... we
#: re-parameterized beta = e^beta' and imposed a Gaussian prior (mu = 1, sigma = 1) on
#: beta'." Not optional: without them this is an unregularised MLE with unbounded
#: parameters, and the resulting divergence was once published here as a property of
#: the method. `log_difficulty` IS their beta', since difficulty is exp(log_difficulty).
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
            # Log-prior gradient, -(x - mu) / sigma^2. Without it the residual never
            # reaches zero as the sigmoid saturates, so the parameters climb forever.
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
# Singer et al. (arXiv:2607.24622, 2026) on GLAD's single ability per annotator: it
# "prevents them from distinguishing majority-class competence from minority-class
# competence." Their model makes both terms class-specific,
#
#     P(labels correctly | true class k) = sigmoid(ability[r][k] - difficulty[i][k])
#
# which is the one shape GLAD cannot represent and exactly the shape of a two-of-three
# reviewer: right on the significant class, wrong only on routine items at the boundary.
# Finding 17 hedged that a better estimator might resist the confound; this is the
# strongest candidate, so the hedge can be measured instead.
#
# Implemented from the model definition, not adapted from their code.

#: Class-level means, from which deviations are measured: their decomposition
#: `alpha[r][k] = mu_a[k] + g[r][k]`, `beta[i][k] = mu_b[k] + h[i][k]`.
INITIAL_CLASS_ABILITY = 1.0
INITIAL_CLASS_DIFFICULTY = 0.0

#: Identifiability, and not optional. Their constraint is `sum_i h_{i,k} = sum_r
#: g_{r,k} = 0` plus Gaussian priors on the deviations. Without both, the model is
#: translation-invariant per class and EM wanders between mirror-image solutions --
#: an early run here scored 1.000 on one composition and 0.717 on another from
#: near-identical parameters, which is label switching rather than a result.
CC_RASCH_DEVIATION_SIGMA = 1.0

#: Their `lambda_mu` ridge. The means are identified once the deviations are centred,
#: so this only keeps them finite.
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
            # Penalise the DEVIATION from each class mean, not the parameter: their
            # `R(theta)`. Penalising the parameter shrinks every annotator to zero.
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

        # Centring, after the ascent. The model depends only on the DIFFERENCE
        # `ability - difficulty`, so the gauge shift must move both the same way. A
        # first version moved them oppositely, which changes every sigmoid and is no
        # reparameterisation at all; the log-likelihood fell between iterations, which
        # EM cannot do, and that is how it was found.
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
