"""The canonical truth-inference estimator, and where it stops working."""

from pharos.inference import agreement_with, dawid_skene, weighted_targets


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
