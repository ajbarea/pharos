"""Finding 22: the trace a shared blind spot leaves after it stops leaving disagreement."""

import json
from pathlib import Path
from typing import Any

import pytest
from measure_channel_bias import (
    ALPHA,
    Detection,
    detect,
    scan,
    stratified_delta,
    verdict_rates,
)

from pharos.generate import GeneratorConfig, generate
from pharos.labels import Compartment
from pharos.tasks import build_triage_tasks


def _artifact() -> dict[str, Any]:
    """The committed measurement. Loaded in one place so the path is stated once."""
    return json.loads(
        (Path(__file__).resolve().parents[1] / "results" / "channel_bias.json").read_text(
            encoding="utf-8"
        )
    )


def test_stratified_delta_conditions_on_difficulty():
    """The whole point: a channel that only correlates with evidence must score zero.

    Tasks carrying the channel here are exactly the high-evidence ones, and their rates
    match the non-carrying tasks *at the same level*. An unconditioned comparison would
    report a large effect; conditioning removes it.
    """
    rates = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    evidence = {"a": 3, "b": 3, "c": 0, "d": 0}
    carries = {"a": True, "b": False, "c": True, "d": False}
    delta, strata = stratified_delta(rates, carries, evidence)
    assert delta == 0.0
    assert strata == 2


def test_stratified_delta_is_negative_when_the_channel_depresses_the_verdict():
    """The direction a blind spot produces, pinned so a sign flip cannot pass."""
    rates = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0}
    evidence = {"a": 3, "b": 3, "c": 3, "d": 3}
    carries = {"a": True, "b": False, "c": True, "d": False}
    delta, _ = stratified_delta(rates, carries, evidence)
    assert delta == -1.0


def test_a_stratum_missing_either_side_is_skipped_not_scored_zero():
    """An absent comparison is not a null result, and averaging it in would dilute."""
    rates = {"a": 1.0, "b": 0.0, "c": 1.0}
    evidence = {"a": 3, "b": 3, "c": 1}
    carries = {"a": True, "b": False, "c": True}  # level 1 has no non-carrying task
    delta, strata = stratified_delta(rates, carries, evidence)
    assert strata == 1
    assert delta == 1.0


def test_detect_returns_none_when_no_stratum_is_comparable():
    rates = {"a": 1.0, "b": 1.0}
    evidence = {"a": 3, "b": 3}
    carries = {"a": True, "b": True}
    assert detect(rates, carries, evidence, permutations=10, seed=1) is None


def test_detect_scores_a_real_depression_and_ignores_a_balanced_one():
    """A one-sided statistic: an elevated rate must not read as the same finding."""
    evidence = {f"t{i}": 3 for i in range(20)}
    carries = {f"t{i}": i < 10 for i in range(20)}

    # 2000 permutations, not 200: the floor on an achievable p-value is 1 / (m + 1),
    # so a 200-permutation null bottoms out at 0.005 and could never clear ALPHA no
    # matter how total the effect. That is a real constraint on the design, pinned in
    # its own test below.
    depressed = {f"t{i}": (0.0 if i < 10 else 1.0) for i in range(20)}
    hit = detect(depressed, carries, evidence, permutations=2000, seed=1)
    assert hit is not None
    assert hit.delta == -1.0
    assert hit.p_value <= ALPHA

    # The mirror image: carrying tasks scored HIGHER. Same magnitude, opposite sign,
    # and it must not be reported as a blind spot.
    elevated = {f"t{i}": (1.0 if i < 10 else 0.0) for i in range(20)}
    miss = detect(elevated, carries, evidence, permutations=2000, seed=1)
    assert miss is not None
    # One-sided: an elevated rate is the opposite of the finding, so it must sit at the
    # far end of the null rather than near its floor.
    assert miss.p_value > 0.5
    assert not miss.detected


def test_no_effect_scores_near_zero():
    """The false-positive control, in miniature."""
    evidence = {f"t{i}": 3 for i in range(20)}
    carries = {f"t{i}": i % 2 == 0 for i in range(20)}
    rates = {f"t{i}": 0.5 for i in range(20)}
    result = detect(rates, carries, evidence, permutations=200, seed=1)
    assert result is not None
    assert not result.detected


def test_the_permutation_budget_bounds_what_can_be_detected_at_all():
    """A p-value cannot go below 1 / (m + 1), so m decides what ALPHA can ever mean.

    This is the constraint that replaces the old z-score's unboundedness, and it is a
    feature: a z of 40 claimed a precision no finite number of shuffles could support.
    But it has to be respected when choosing the budget. At 200 permutations the
    smallest reachable p is about 0.005, so a detection threshold of 0.001 is
    unreachable and every channel would read as undetected no matter how total the
    effect. The measurement uses 4200 for exactly this reason.
    """
    evidence = {f"t{i}": 3 for i in range(20)}
    carries = {f"t{i}": i < 10 for i in range(20)}
    total = {f"t{i}": (0.0 if i < 10 else 1.0) for i in range(20)}

    starved = detect(total, carries, evidence, permutations=200, seed=1)
    assert starved is not None
    assert starved.p_value == pytest.approx(1 / 201)
    assert not starved.detected, "a 200-permutation null cannot resolve ALPHA = 0.001"

    ample = detect(total, carries, evidence, permutations=2000, seed=1)
    assert ample is not None
    assert ample.p_value == pytest.approx(1 / 2001)
    assert ample.detected, "the same total effect, with a budget that can resolve it"


def test_verdict_rates_read_only_the_aggregate():
    """Deployability: the statistic must not need a per-analyst stream.

    `verdict_rates` takes the partitioned contributions only to build the same per-task
    sums the aggregator already holds under finding 18's protocol, and returns one
    number per task with no contributor distinguishable in it.
    """
    partitioned = {
        "a": [("T-1", True), ("T-2", False)],
        "b": [("T-1", True), ("T-2", False)],
        "c": [("T-1", False), ("T-2", False)],
    }
    rates = verdict_rates(partitioned)
    assert rates["T-1"] == pytest.approx(2 / 3)
    assert rates["T-2"] == 0.0
    assert set(rates) == {"T-1", "T-2"}


def test_detection_threshold_is_the_gates_own_convention():
    """Three sigma's one-sided tail, not a number chosen after seeing the effect."""
    assert ALPHA == 0.001


def test_detection_serializes_every_field():
    d = Detection(
        channel="PARTNER",
        delta=-0.3,
        null_mean=0.0,
        p_value=0.0002,
        extreme=0,
        permutations=4200,
        detected=True,
        strata=4,
    )
    assert set(d.as_dict()) == {
        "channel",
        "delta",
        "null_mean",
        "p_value",
        "extreme",
        "permutations",
        "detected",
        "strata",
    }


def test_scan_tests_every_channel_so_false_positives_are_visible():
    """One channel reported in isolation cannot be judged; the others are the control.

    A detector that fires on the true channel is only interesting if it stays quiet on
    the rest, so `scan` returns all of them and the finding shows the whole row.
    """
    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=40)))
    # An unbiased fleet: every contributor reports the world's own rule, so no channel
    # should score. Built directly rather than through the analyst policies, because
    # what is under test is the statistic and not the fleet machinery.
    partitioned = {f"a{i}": [(t.task_id, t.significant) for t in tasks] for i in range(3)}
    found = scan(tasks, partitioned, permutations=50, seed=1)

    assert {d.channel for d in found} <= {c.value for c in Compartment}
    assert found, "scan returned nothing; every channel was skipped"
    # Truth carries no channel dependence, so nothing may be detected.
    assert not [d for d in found if d.detected]


def test_the_effect_is_linear_in_the_share_and_the_p_value_saturates():
    """Which measure reports extent, and which cannot.

    The gap is linear in how many analysts are blind: each one withholds the same
    verdicts on the same tasks, so nine of them move the stratified gap nine times as
    far as one. That makes `delta` the thing to read for *extent*.

    A p-value cannot do that job and must not be asked to. Once an effect is
    comfortably significant the p-value sits on its own floor, 1 / (m + 1), and stays
    there no matter how much larger the effect gets -- so equal p-values across shares
    are a property of the floor rather than evidence that the shares are alike. This
    used to be reported as a z-score, which did vary, and its invariance across shares
    was published as a finding. That invariance was an artifact of a noiseless fleet.
    """
    payload = _artifact()
    blind = payload["blind_channel"]
    scored = {
        entry["n_blind"]: next(d for d in entry["detections"] if d["channel"] == blind)
        for entry in payload["sweep"]
        if entry["n_blind"] > 0 and entry["slip_rate"] == 0.0
    }
    assert len(scored) >= 3, "too few non-zero shares to test the scaling"

    fleet = payload["fleet"]
    largest = max(scored)
    for n_blind, detection in scored.items():
        expected = scored[largest]["delta"] * n_blind / largest
        assert detection["delta"] == pytest.approx(expected, abs=1e-3), (
            f"the gap at {n_blind} blind is not proportional to the share; "
            "extent is read off this number, so its linearity is the claim"
        )
        assert n_blind <= fleet

    # And the floor is real: no p-value anywhere in the artifact may sit below it.
    floor = 1.0 / (payload["permutations"] + 1)
    for entry in payload["sweep"]:
        for detection in entry["detections"]:
            # Relative tolerance, because p is serialized to three significant
            # figures and the floor is not a round number.
            assert detection["p_value"] >= floor * (1 - 1e-3), (
                f"p = {detection['p_value']} is below the achievable floor {floor}; "
                "a permutation p-value cannot be smaller than one draw in m + 1"
            )


def test_the_healthy_fleet_and_the_other_channels_are_both_reported():
    """A detector is only interesting beside its controls, so both must be present."""
    payload = _artifact()
    assert payload["controls_clean"]
    zero = next(e for e in payload["sweep"] if e["n_blind"] == 0)
    assert not [d for d in zero["detections"] if d["detected"]], (
        "a channel fired with nobody blind; the false-positive control failed"
    )
    for entry in payload["sweep"]:
        fired = {d["channel"] for d in entry["detections"] if d["detected"]}
        assert fired <= {payload["blind_channel"]}, (
            f"a channel other than the blinded one fired at {entry['n_blind']} blind: {fired}"
        )


def test_the_noiseless_control_returns_a_real_answer_rather_than_a_special_case():
    """The controls used to pass because they could not do anything else.

    Both control fleets are noiseless, so every analyst is identical and deterministic,
    each task's verdict rate is a function of its evidence stratum alone, and every
    permutation of the channel labels returns the observed gap exactly. Under the old
    z-score the null's standard deviation was zero, z was undefined, and the code
    divided that case into a `0.0` that read as a clean pass. The finding called the
    threshold control "the load-bearing one" and said it could have voided the result.
    It could not.

    A permutation p-value has no such hole. Every draw is at least as extreme, so
    b = m and p = 1.0 -- the correct answer, arrived at by construction. So the test is
    no longer "is at least one control informative"; it is that the noiseless controls
    report exactly 1.0, and that the ones with noise, which genuinely could fire, do
    not.
    """
    payload = _artifact()

    controls = {entry["slip_rate"]: entry["detections"] for entry in payload["threshold_control"]}
    assert controls, "no threshold control in the artifact"

    for detection in controls.get(0.0, []):
        assert detection["p_value"] == 1.0, (
            "a noiseless control did not return p = 1.0; every permutation of a "
            "deterministic fleet returns the observed gap, so b = m by construction"
        )

    noisy = [d for slip, found in controls.items() if slip > 0 for d in found]
    assert noisy, "no control was run on a fleet that could actually produce a null"
    assert not [d for d in noisy if d["detected"]], (
        "the threshold control fired: the statistic is reading generic error rather "
        "than channel bias, and the finding is void"
    )


def test_the_invariance_is_a_property_of_a_noiseless_fleet_and_says_so():
    """The exact invariance needs a condition the finding did not originally state.

    z is a ratio of the observed gap to the null's spread, and both are linear in the
    blind share -- but linear *through the origin* only when the baseline gap is exactly
    zero, which requires a fleet with no verdict noise. Give the sighted analysts this
    repo's own `inattentive` slip rate and the invariance is gone, while every sentence
    of the derivation stays true. So the sweep measures more than one noise level, and
    the claim is allowed to be stated only for the level that supports it.
    """
    payload = _artifact()

    levels = {entry["slip_rate"] for entry in payload["sweep"]}
    assert levels != {0.0}, (
        "the sweep only measures a noiseless fleet, so it cannot tell whether the "
        "invariance is a property of the statistic or of the idealization"
    )
    assert 0.0 in levels, "the noiseless reference column is how this was first reported"


def test_the_sweep_reaches_the_shares_the_finding_makes_claims_about():
    """The finding's useful claim is early detection, so the sweep has to test early.

    It did not. `SHARES` ran (0, 5, 7, 9), whose lowest non-zero entry is already the
    majority, while the write-up claimed a house style is catchable *before* it becomes
    one and tied that to finding 16. A share the sweep never runs cannot support a
    sentence about that share, and this asserts the sweep still reaches below the
    majority so the claim cannot quietly lose its evidence again.
    """
    from measure_channel_bias import SHARES

    fleet = 9
    below_majority = [n for n in SHARES if 0 < n <= fleet // 2]
    assert below_majority, (
        f"SHARES={SHARES} tests no share below a majority of {fleet}; the early-detection "
        "claim would rest on nothing"
    )
    assert 1 in SHARES, "the single-blind-analyst claim needs a single blind analyst"
