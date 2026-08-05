"""Finding 22: the trace a shared blind spot leaves after it stops leaving disagreement."""

import pytest
from measure_channel_bias import (
    DETECTION_Z,
    Detection,
    detect,
    scan,
    stratified_delta,
    verdict_rates,
)

from pharos.generate import GeneratorConfig, generate
from pharos.labels import Compartment
from pharos.tasks import build_triage_tasks


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
    assert detect(rates, carries, evidence, trials=10, seed=1) is None


def test_detect_scores_a_real_depression_and_ignores_a_balanced_one():
    """A one-sided statistic: an elevated rate must not read as the same finding."""
    evidence = {f"t{i}": 3 for i in range(20)}
    carries = {f"t{i}": i < 10 for i in range(20)}

    depressed = {f"t{i}": (0.0 if i < 10 else 1.0) for i in range(20)}
    hit = detect(depressed, carries, evidence, trials=200, seed=1)
    assert hit is not None
    assert hit.delta == -1.0
    assert hit.z > DETECTION_Z

    # The mirror image: carrying tasks scored HIGHER. Same magnitude, opposite sign,
    # and it must not be reported as a blind spot.
    elevated = {f"t{i}": (1.0 if i < 10 else 0.0) for i in range(20)}
    miss = detect(elevated, carries, evidence, trials=200, seed=1)
    assert miss is not None
    assert miss.z < 0
    assert not miss.detected


def test_no_effect_scores_near_zero():
    """The false-positive control, in miniature."""
    evidence = {f"t{i}": 3 for i in range(20)}
    carries = {f"t{i}": i % 2 == 0 for i in range(20)}
    rates = {f"t{i}": 0.5 for i in range(20)}
    result = detect(rates, carries, evidence, trials=200, seed=1)
    assert result is not None
    assert not result.detected


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
    """Three sigma, not a number chosen after seeing the effect."""
    assert DETECTION_Z == 3.0


def test_detection_serializes_every_field():
    d = Detection(
        channel="PARTNER",
        delta=-0.3,
        null_mean=0.0,
        null_sd=0.04,
        z=7.5,
        detected=True,
        strata=4,
    )
    assert set(d.as_dict()) == {
        "channel",
        "delta",
        "null_mean",
        "null_sd",
        "z",
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
    found = scan(tasks, partitioned, trials=50, seed=1)

    assert {d.channel for d in found} <= {c.value for c in Compartment}
    assert found, "scan returned nothing; every channel was skipped"
    # Truth carries no channel dependence, so nothing may be detected.
    assert not [d for d in found if d.detected]
