"""The canonical truth-inference estimator, and where it stops working."""

import math
import random
import statistics

import pytest

from pharos.inference import (
    FederatedDawidSkene,
    _stable_posterior,
    agreement_with,
    dawid_skene,
    federated_dawid_skene,
    partition_by_contributor,
    weighted_targets,
)
from pharos.secagg import CohortTooSmallError


def test_empty_input_infers_nothing():
    estimate = dawid_skene([])
    assert estimate.posterior == {}
    assert estimate.labels() == {}
    assert estimate.converged
    assert estimate.reliability("nobody") == 0.0


def test_unanimous_contributors_are_recovered_exactly():
    rows = [(f"T-{i}", w, i % 2 == 0) for i in range(20) for w in ("a", "b", "c")]
    estimate = dawid_skene(rows)
    assert estimate.converged
    assert all(estimate.labels()[f"T-{i}"] == (i % 2 == 0) for i in range(20))
    for w in ("a", "b", "c"):
        assert estimate.reliability(w) > 0.9


def test_one_liar_among_many_is_identified_without_ground_truth():
    """The property that makes Dawid-Skene worth having: it finds the bad annotator."""
    rows = [
        (f"T-{i}", w, i % 2 == 0) for i in range(40) for w in ("good1", "good2", "good3", "good4")
    ]
    rows += [(f"T-{i}", "liar", i % 2 != 0) for i in range(40)]

    estimate = dawid_skene(rows)
    assert estimate.reliability("liar") < 0.2
    assert min(estimate.reliability(w) for w in ("good1", "good2", "good3", "good4")) > 0.8
    # And the truth is recovered despite the liar.
    truth = {f"T-{i}": i % 2 == 0 for i in range(40)}
    assert agreement_with(estimate.labels(), truth) == 1.0


def test_a_wrong_majority_is_learned_as_the_truth():
    """The failure that matters, and the reason finding 12's claim survives.

    EM started from the majority vote converges to the majority's standard. When most
    contributors are wrong the estimator concludes the correct minority are the
    unreliable ones, which is the opposite of the intended behaviour and is invisible
    without ground truth.
    """
    rows = [
        (f"T-{i}", w, i % 2 != 0)
        for i in range(40)
        for w in ("wrong1", "wrong2", "wrong3", "wrong4", "wrong5")
    ]
    rows += [(f"T-{i}", w, i % 2 == 0) for i in range(40) for w in ("right1", "right2")]

    estimate = dawid_skene(rows)
    truth_map = {f"T-{i}": i % 2 == 0 for i in range(40)}
    assert agreement_with(estimate.labels(), truth_map) == 0.0, "inverted, as predicted"
    # And it rates the wrong majority as the reliable ones.
    assert estimate.reliability("wrong1") > estimate.reliability("right1")


def test_weighted_targets_keeps_only_the_estimated_reliable():
    rows = [(f"T-{i}", w, i % 2 == 0) for i in range(30) for w in ("good1", "good2", "good3")]
    rows += [(f"T-{i}", "liar", i % 2 != 0) for i in range(30)]

    estimate = dawid_skene(rows)
    kept = weighted_targets(rows, estimate, floor=0.5)
    assert kept, "the reliable contributors must survive the filter"
    assert all(who != "liar" for _, who, _ in kept)
    assert len(kept) < len(rows)


def test_agreement_ignores_tasks_without_truth():
    assert agreement_with({}, {}) == 0.0
    assert agreement_with({"a": True}, {"b": False}) == 0.0
    assert agreement_with({"a": True, "b": False}, {"a": True}) == 1.0


def test_iterations_are_bounded():
    rows = [(f"T-{i}", w, bool(i % 3)) for i in range(10) for w in ("a", "b")]
    capped = dawid_skene(rows, max_iters=2, tolerance=0.0)
    assert capped.iterations <= 2
    assert not capped.converged


# ---------------------------------------------------------------- GLAD -----


def test_glad_recovers_a_liar_on_uniform_items():
    """The control case: with no difficulty structure, GLAD must behave like DS."""
    from pharos.inference import glad

    rows = [(f"T-{i}", w, i % 2 == 0) for i in range(40) for w in ("g1", "g2", "g3", "g4")]
    rows += [(f"T-{i}", "liar", i % 2 != 0) for i in range(40)]

    estimate = glad(rows)
    assert estimate.converged
    truth = {f"T-{i}": i % 2 == 0 for i in range(40)}
    assert agreement_with(estimate.labels(), truth) == 1.0
    assert estimate.ability["liar"] < 0 < estimate.ability["g1"]


def test_glad_finds_no_difficulty_when_labelers_agree():
    """A unanimous fleet has nothing to explain, so difficulty must stay flat.

    This is the control the confound measurement rests on. If GLAD invented structure
    here, the structure it reports under a wrong standard would prove nothing.
    """
    from pharos.inference import glad

    rows = [(f"T-{i}", w, i % 3 == 0) for i in range(30) for w in ("a", "b", "c", "d", "e")]
    estimate = glad(rows)
    spread = [estimate.difficulty(f"T-{i}") for i in range(30)]
    assert max(spread) / min(spread) < 1.05, "no disagreement, so no difficulty structure"


def test_glad_blames_the_item_when_a_subgroup_is_systematically_wrong():
    """The finding: a wrong subgroup manufactures apparent item difficulty.

    Three labelers invert a fixed, identifiable subset of items. Nothing about those
    items is intrinsically harder -- the other six labelers resolve them perfectly --
    yet GLAD assigns them elevated difficulty, because "these items are hard" fits the
    data as well as "those three are wrong".
    """
    from pharos.inference import glad

    contested = {f"T-{i}" for i in range(0, 30, 3)}
    rows = [
        (f"T-{i}", w, i % 2 == 0) for i in range(30) for w in ("r1", "r2", "r3", "r4", "r5", "r6")
    ]
    rows += [
        (f"T-{i}", w, (i % 2 != 0) if f"T-{i}" in contested else (i % 2 == 0))
        for i in range(30)
        for w in ("w1", "w2", "w3")
    ]

    estimate = glad(rows)
    hard = statistics_mean([estimate.difficulty(t) for t in contested])
    easy = statistics_mean(
        [estimate.difficulty(f"T-{i}") for i in range(30) if f"T-{i}" not in contested]
    )
    assert hard > easy * 1.5, (
        "the contested items should read as harder, though nothing made them so"
    )


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values)


# ------------------------------------------------- priors and convergence -----


def test_glad_carries_the_priors_its_paper_specifies():
    """The regulariser whose absence looked like the method failing.

    Whitehill et al. section 3.1: "In our implementation we used Gaussian priors
    (mu = 1, sigma = 1) for alpha. For beta, we need a prior that does not generate
    negative values. To do so we re-parameterized beta = e^beta' and imposed a
    Gaussian prior (mu = 1, sigma = 1) on beta'."

    Without those terms the residual never reaches zero as the sigmoid saturates, so
    the parameters climb without bound and EM never settles. This repository published
    that divergence as a property of GLAD before checking the paper for a prior.
    """
    from pharos import inference

    assert inference.PRIOR_MEAN == 1.0
    assert inference.PRIOR_SIGMA == 1.0
    # log_difficulty IS their beta', since difficulty is reported as exp(log_difficulty),
    # so it starts at the same prior mean rather than at zero.
    assert inference.INITIAL_LOG_DIFFICULTY == inference.PRIOR_MEAN
    assert inference.INITIAL_ABILITY == inference.PRIOR_MEAN


def test_glad_converges_and_its_parameters_stay_bounded():
    """The property the priors buy, asserted rather than assumed.

    An unregularised fit on this data grew mean difficulty from 47.8 to 3580.8 as the
    iteration cap rose from 100 to 3000. Raising the cap must now change nothing,
    because the fit has actually settled.
    """
    from pharos.inference import glad

    # A fleet where a third hold a wrong standard: the composition that diverged.
    contributions: list[tuple[str, str, bool]] = []
    for item in range(60):
        near_boundary = item % 3 == 0
        for who in range(9):
            wrong = who < 3
            contributions.append((f"T-{item}", f"w{who}", near_boundary and wrong))

    at_100 = glad(contributions, max_iters=100)
    at_1000 = glad(contributions, max_iters=1000)

    assert at_100.converged, "the fit must settle within the default cap"
    assert at_1000.converged
    assert at_100.iterations == at_1000.iterations, "a settled fit cannot use a raised cap"

    # Bounded, and identical, rather than growing with the budget.
    for task in at_100.log_difficulty:
        assert at_100.difficulty(task) == pytest.approx(at_1000.difficulty(task))
        assert at_100.difficulty(task) < 100.0, "difficulty is climbing without bound again"
    for who in at_100.ability:
        assert abs(at_100.ability[who]) < 100.0


def test_cc_rasch_conditions_ability_on_the_class():
    """The one thing a class-conditional model buys that GLAD structurally cannot.

    Singer et al. (arXiv:2607.24622, 2026) on GLAD and Rasch: a single ability
    parameter per annotator "prevents them from distinguishing majority-class
    competence from minority-class competence". A reviewer who is right on one class
    and wrong on the other must therefore show two different abilities here.
    """
    from pharos.inference import cc_rasch

    # Two groups. Everyone agrees on the significant class; one group is wrong on the
    # routine class only, which is the shape of a two-of-three standard.
    contributions: list[tuple[str, str, bool]] = []
    for item in range(40):
        significant = item % 2 == 0
        for who in range(6):
            skewed = who < 2
            said = True if significant else (skewed and item % 4 == 1)
            contributions.append((f"T-{item}", f"w{who}", said))

    estimate = cc_rasch(contributions)
    assert set(estimate.ability) == {f"w{i}" for i in range(6)}
    for by_class in estimate.ability.values():
        assert set(by_class) == {True, False}
    for by_class in estimate.difficulty.values():
        assert set(by_class) == {True, False}

    # The skewed reviewers must score lower on the routine class than the others,
    # which is exactly the discrimination GLAD's single number cannot make.
    skewed = statistics.mean(estimate.ability[f"w{i}"][False] for i in range(2))
    sound = statistics.mean(estimate.ability[f"w{i}"][False] for i in range(2, 6))
    assert sound > skewed, "class-conditional ability did not localise a class-specific error"

    assert estimate.mean_ability("w0") == pytest.approx(
        (estimate.ability["w0"][True] + estimate.ability["w0"][False]) / 2
    )


def test_cc_rasch_is_empty_safe_and_reports_convergence():
    from pharos.inference import cc_rasch

    empty = cc_rasch([])
    assert empty.labels() == {}
    assert empty.converged and empty.iterations == 0
    assert empty.mean_ability("nobody") == 0.0


def _observed_loglik_glad(contributions, estimate) -> float:
    """log p(observed labels) under a fitted GLAD, marginalising the true label."""
    from pharos.inference import _sigmoid

    by_task: dict[str, list[tuple[str, bool]]] = {}
    for task, who, verdict in contributions:
        by_task.setdefault(task, []).append((who, verdict))
    total = 0.0
    for task, rows in by_task.items():
        inv = math.exp(-estimate.log_difficulty[task])
        log_pos = log_neg = 0.0
        for who, verdict in rows:
            p = min(max(_sigmoid(estimate.ability[who] * inv), 1e-9), 1 - 1e-9)
            log_pos += math.log(p if verdict else 1 - p)
            log_neg += math.log((1 - p) if verdict else p)
        top = max(log_pos, log_neg)
        total += top + math.log(math.exp(log_pos - top) + math.exp(log_neg - top))
    return total


def _observed_loglik_cc_rasch(contributions, estimate) -> float:
    """The same quantity for CC-Rasch, whose parameters are indexed by class."""
    from pharos.inference import _sigmoid

    by_task: dict[str, list[tuple[str, bool]]] = {}
    for task, who, verdict in contributions:
        by_task.setdefault(task, []).append((who, verdict))
    total = 0.0
    for task, rows in by_task.items():
        log_pos = log_neg = 0.0
        for who, verdict in rows:
            for klass in (True, False):
                p = _sigmoid(estimate.ability[who][klass] - estimate.difficulty[task][klass])
                p = min(max(p, 1e-9), 1 - 1e-9)
                term = math.log(p if verdict == klass else 1 - p)
                if klass:
                    log_pos += term
                else:
                    log_neg += term
        top = max(log_pos, log_neg)
        total += top + math.log(math.exp(log_pos - top) + math.exp(log_neg - top))
    return total


def _confounded_fleet(n_items: int = 40, n_wrong: int = 3, fleet: int = 9):
    """A fleet where some reviewers err only on near-boundary routine items."""
    rows: list[tuple[str, str, bool]] = []
    for item in range(n_items):
        near_boundary = item % 3 == 0
        for who in range(fleet):
            wrong = who < n_wrong
            rows.append((f"T-{item}", f"w{who}", near_boundary and wrong))
    return rows


def test_em_objective_never_decreases():
    """EM increases the observed-data likelihood every iteration. If it does not, the
    implementation is not EM, whatever the code looks like.

    This is the check that caught both estimator bugs in this module, each of which
    survived reading the code. GLAD was missing the Gaussian priors its paper
    specifies, so the parameters climbed without bound. CC-Rasch's centring step moved
    ability and difficulty in opposite directions, which is not a gauge transformation
    at all -- the model depends on their difference -- and its likelihood fell between
    iterations.
    """
    from pharos.inference import cc_rasch, glad

    contributions = _confounded_fleet()

    for fit, loglik in ((glad, _observed_loglik_glad), (cc_rasch, _observed_loglik_cc_rasch)):
        previous = None
        for cap in range(1, 22, 2):
            value = loglik(contributions, fit(contributions, max_iters=cap))
            if previous is not None:
                assert value >= previous - 1e-6, (
                    f"{fit.__name__} log-likelihood fell from {previous:.4f} to {value:.4f} "
                    f"between iteration caps; EM cannot do that"
                )
            previous = value


def test_cc_rasch_centring_is_a_gauge_transformation():
    """Adding a constant to ability and difficulty together must change nothing.

    The model is `sigmoid(ability - difficulty)`, so this is the invariance the
    centring constraint exists to remove -- and the reason the shift has to move both
    in the same direction. Asserting the invariance directly is what makes the fix
    permanent rather than incidental.
    """
    from pharos.inference import _sigmoid, cc_rasch

    estimate = cc_rasch(_confounded_fleet())
    assert estimate.converged

    for who, by_class in estimate.ability.items():
        for klass, value in by_class.items():
            task = next(iter(estimate.difficulty))
            plain = _sigmoid(value - estimate.difficulty[task][klass])
            shifted = _sigmoid((value + 2.5) - (estimate.difficulty[task][klass] + 2.5))
            assert plain == pytest.approx(shifted), f"{who}/{klass} is not gauge invariant"

    # And the gauge is actually fixed: mean ability sits at the reference per class.
    from pharos.inference import INITIAL_CLASS_ABILITY

    for klass in (True, False):
        mean = statistics.mean(v[klass] for v in estimate.ability.values())
        assert mean == pytest.approx(INITIAL_CLASS_ABILITY, abs=1e-6), (
            "centring did not fix the free translation"
        )


# ------------------------------------------------- Dawid-Skene under a sum -----


def _fleet_rows(n_wrong: int, tasks: int = 60, workers: int = 9):
    """A fleet where `n_wrong` contributors invert the truth on every task."""
    truth = {f"T-{i}": i % 3 == 0 for i in range(tasks)}
    rows = []
    for w in range(workers):
        wrong = w < n_wrong
        for task, value in truth.items():
            rows.append((task, f"w{w}", (not value) if wrong else value))
    return rows, truth


def test_partition_regroups_without_losing_a_row():
    rows = [("T-1", "a", True), ("T-2", "a", False), ("T-1", "b", True)]
    partitioned = partition_by_contributor(rows)
    assert partitioned == {"a": [("T-1", True), ("T-2", False)], "b": [("T-1", True)]}
    assert sum(len(v) for v in partitioned.values()) == len(rows)


def test_federated_reproduces_the_centralized_estimator():
    """The port must not change the answer, or nothing downstream is comparable."""
    rows, _ = _fleet_rows(n_wrong=3)
    central = dawid_skene(rows)
    federated = federated_dawid_skene(partition_by_contributor(rows), seed=17)

    assert federated.converged == central.converged
    for task in central.posterior:
        assert federated.posterior[task] == pytest.approx(central.posterior[task], abs=1e-9)
    assert federated.labels() == central.labels()


def test_federated_inherits_the_cliff_rather_than_escaping_it():
    """Finding 18's prediction: moving the computation does not move the failure."""
    minority_rows, truth = _fleet_rows(n_wrong=4)
    majority_rows, _ = _fleet_rows(n_wrong=5)

    minority = federated_dawid_skene(partition_by_contributor(minority_rows), seed=17)
    majority = federated_dawid_skene(partition_by_contributor(majority_rows), seed=17)

    assert agreement_with(minority.labels(), truth) == 1.0
    # Past the majority crossing the estimator ratifies the wrong standard, and does
    # so with no sign of trouble: it converges just as happily.
    assert agreement_with(majority.labels(), truth) == 0.0
    assert majority.converged


def test_the_result_carries_no_per_contributor_reliability():
    """The server never learns a confusion matrix, so it must not report one."""
    rows, _ = _fleet_rows(n_wrong=2)
    federated = federated_dawid_skene(partition_by_contributor(rows), seed=17)
    assert not hasattr(federated, "error_rates")
    assert not hasattr(federated, "reliability")


def test_readership_counts_what_the_server_can_actually_see():
    """A per-task mean needs its denominator, so the denominator is disclosed."""
    rows = [("T-1", "a", True), ("T-1", "b", True), ("T-1", "c", False), ("T-2", "a", True)]
    federated = federated_dawid_skene(partition_by_contributor(rows), seed=17)
    assert federated.readership == {"T-1": 3, "T-2": 1}


def test_anchors_are_clamped_and_never_revised():
    """The authority's ruling is exogenous: EM may not talk it out of one."""
    rows, truth = _fleet_rows(n_wrong=5)
    anchored = sorted(truth)[:20]
    anchors = {task: truth[task] for task in anchored}

    federated = federated_dawid_skene(partition_by_contributor(rows), seed=17, anchors=anchors)
    assert federated.anchored == frozenset(anchored)
    for task in anchored:
        assert federated.labels()[task] == truth[task]


def test_anchors_repair_a_bare_majority_on_tasks_they_did_not_rule_on():
    """Finding 19's mechanism, scored only where the authority stayed silent."""
    rows, truth = _fleet_rows(n_wrong=5)
    anchors = {task: truth[task] for task in sorted(truth)[:20]}

    without = federated_dawid_skene(partition_by_contributor(rows), seed=17)
    with_authority = federated_dawid_skene(partition_by_contributor(rows), seed=17, anchors=anchors)

    unruled = {t: v for t, v in truth.items() if t not in anchors}
    before = agreement_with({t: v for t, v in without.labels().items() if t in unruled}, unruled)
    after = agreement_with(
        {t: v for t, v in with_authority.labels().items() if t in unruled}, unruled
    )
    assert before == 0.0
    assert after == 1.0


def test_empty_input_infers_nothing_federated():
    federated = federated_dawid_skene({}, seed=1)
    assert federated.posterior == {}
    assert federated.labels() == {}
    assert federated.readership == {}
    assert federated.converged


def test_a_fleet_below_the_aggregation_floor_is_refused():
    """Two contributors cannot be aggregated, and the estimator must not pretend."""
    rows = [("T-1", "a", True), ("T-1", "b", False)]
    with pytest.raises(CohortTooSmallError):
        federated_dawid_skene(partition_by_contributor(rows), seed=1)


def test_stable_posterior_does_not_overflow_in_either_regime():
    """Both branches used to use the overflowing form for their own regime.

    `math.exp` raises rather than saturating, so `|log_pos - log_neg| > 709.8` crashed
    -- reachable from the shipped CLI, which exposes `--fleet`. The two forms are
    algebraically identical, which is why a hand-check of the algebra could not tell
    them apart and did not.
    """
    assert _stable_posterior(-800.0, 0.0) == 0.0
    assert _stable_posterior(0.0, -800.0) == 1.0
    assert _stable_posterior(-1e308, 0.0) == 0.0
    assert _stable_posterior(0.0, -1e308) == 1.0
    assert _stable_posterior(0.0, 0.0) == 0.5


def test_stable_posterior_matches_a_log_sum_exp_reference():
    """Correctness, not merely absence of a crash."""

    def reference(log_pos: float, log_neg: float) -> float:
        shift = max(log_pos, log_neg)
        pos, neg = math.exp(log_pos - shift), math.exp(log_neg - shift)
        return pos / (pos + neg)

    rng = random.Random(0)
    worst = max(
        abs(_stable_posterior(lp, ln) - reference(lp, ln))
        for lp, ln in ((rng.uniform(-5000, 0), rng.uniform(-5000, 0)) for _ in range(20000))
    )
    assert worst == 0.0


def test_the_federated_result_type_pins_its_fields():
    """`hasattr` on a slots dataclass is False for any name, so it tests nothing.

    The earlier check asserted `not hasattr(result, "error_rates")` -- which also
    passes for `erorr_rates`, and would pass for a future `per_client_rates` field.
    Pinning the slots is what actually holds the no-per-contributor-data property.
    """
    assert set(FederatedDawidSkene.__slots__) == {
        "posterior",
        "prevalence",
        "iterations",
        "converged",
        "participants",
        "readership",
        "anchored",
    }
