"""Each check must fire on the failure it was written for, and stay quiet otherwise.

A validity warning that cries wolf gets ignored, which is worse than not having it,
so the negative cases matter as much as the positive ones.
"""

import logging

import pytest

from pharos.validity import (
    HIGH_UNPARSED_RATE,
    SMALL_N,
    ValidityReport,
    check_classification,
    check_sample_size,
)


def concerns_of(**kwargs) -> str:
    return " | ".join(check_classification(**kwargs).concerns)


# --- the failure this module was written for --------------------------------


def test_small_n_is_flagged_with_the_cost_of_one_instance():
    """Two findings were published from n=8 and did not reproduce. The warning
    quantifies why: at that size one flipped task moves accuracy 12.5 points."""
    report = check_classification(tp=2, fp=2, tn=2, fn=2, label="n8")
    assert not report.quotable
    joined = " ".join(report.concerns)
    assert "n=8" in joined
    assert "12.5 points" in joined


def test_a_healthy_measurement_is_quotable():
    """No concern means no caveat. This is the case that must stay silent."""
    report = check_classification(tp=40, fp=8, tn=42, fn=10, label="healthy")
    assert report.quotable, report.concerns
    assert report.n == 100


# --- degenerate predictions -------------------------------------------------


def test_predicting_one_class_for_everything_is_flagged():
    """Precision, recall, and F1 are all computable here and all meaningless."""
    text = concerns_of(tp=30, fp=70, tn=0, fn=0, label="all-positive")
    assert "every prediction was positive" in text


def test_predicting_all_negative_is_flagged():
    text = concerns_of(tp=0, fp=0, tn=70, fn=30, label="all-negative")
    assert "every prediction was negative" in text


def test_over_escalation_is_named_specifically():
    """Recall 1.000 with more false than true positives is the exact pattern the
    six-model sweep found in every model, so it gets its own message."""
    text = concerns_of(tp=20, fp=35, tn=5, fn=0, label="escalator")
    assert "recall is 1.000" in text
    assert "escalates indiscriminately" in text


def test_perfect_recall_alone_is_not_flagged_as_over_escalation():
    """A model that catches everything AND is precise is not over-escalating."""
    text = concerns_of(tp=40, fp=3, tn=50, fn=0, label="good")
    assert "escalates indiscriminately" not in text


# --- unparsed answers -------------------------------------------------------


def test_a_high_unparsed_rate_is_flagged():
    text = concerns_of(tp=20, fp=10, tn=20, fn=10, unparsed=20, label="mute")
    assert "unparsable" in text


def test_a_few_unparsed_answers_are_tolerated():
    below = int(100 * HIGH_UNPARSED_RATE) - 1
    text = concerns_of(tp=25, fp=25, tn=25, fn=25, unparsed=below, label="mostly-fine")
    assert "unparsable" not in text


# --- class balance and the majority floor -----------------------------------


def test_a_skewed_evaluation_set_is_flagged_with_its_floor():
    text = concerns_of(tp=5, fp=5, tn=85, fn=5, label="skewed")
    assert "majority floor" in text


def test_failing_to_beat_the_majority_floor_is_flagged():
    """The sweep's central result: no model cleared this. It should be automatic."""
    text = concerns_of(tp=10, fp=25, tn=15, fn=0, label="below-floor")
    assert "does not beat the majority floor" in text


def test_beating_the_floor_is_not_flagged():
    text = concerns_of(tp=45, fp=5, tn=45, fn=5, label="above-floor")
    assert "majority floor" not in text


# --- edges ------------------------------------------------------------------


def test_an_empty_measurement_is_reported_rather_than_dividing_by_zero():
    report = check_classification(tp=0, fp=0, tn=0, fn=0, label="empty")
    assert report.n == 0
    assert not report.quotable
    assert "no instances" in " ".join(report.concerns)


def test_check_sample_size_alone():
    assert check_sample_size(SMALL_N + 1, label="big").quotable
    assert not check_sample_size(5, label="small").quotable


def test_a_report_serialises_for_an_artifact():
    payload = check_classification(tp=2, fp=2, tn=2, fn=2).as_dict()
    assert set(payload) == {"n", "quotable", "concerns"}
    assert isinstance(payload["concerns"], list)


def test_reports_are_immutable():
    """A report describes a run that already happened; nothing should edit it after
    the fact. The assignment below is deliberate, hence the checker pragma."""
    report = ValidityReport(n=10, concerns=("x",))
    with pytest.raises(AttributeError):
        report.n = 20  # ty: ignore[invalid-assignment]


# --- the warnings actually reach the log ------------------------------------


def test_concerns_are_logged_not_only_returned(caplog):
    """A caller that ignores the return value must still leave a trace in the run's
    own output, because that is where a reader looks after the fact."""
    with caplog.at_level(logging.WARNING, logger="pharos"):
        check_classification(tp=1, fp=1, tn=1, fn=1, label="tiny")
    events = [r.__dict__.get("event") for r in caplog.records]
    assert "validity.small_n" in events


def test_a_clean_measurement_logs_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="pharos"):
        check_classification(tp=40, fp=8, tn=42, fn=10, label="healthy")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_perfect_score_is_flagged_rather_than_celebrated() -> None:
    """The adapter run scored 60/60. That is the shape of a leak and also the shape
    of a genuinely learnable rule, and the number alone cannot tell them apart."""
    report = check_classification(tp=20, fp=0, tn=40, fn=0, label="saturated")

    assert not report.quotable
    joined = " ".join(report.concerns)
    assert "saturated" in joined
    # The rule of three: zero errors in 60 still permits a 5% true error rate.
    assert "5.0%" in joined


def test_one_error_is_enough_to_clear_the_saturation_flag() -> None:
    report = check_classification(tp=20, fp=1, tn=39, fn=0, label="near-perfect")

    assert not any("saturated" in c for c in report.concerns)


# The three tests below were written from a mutation run: each kills a mutant that the
# whole suite -- not just this file -- let through. Coverage had every one of these
# lines at 100%, which is the distinction the paper draws. A line that runs is not a
# check that bites, and the flags are the instrument this project claims for others.


def test_unparsed_answers_count_toward_the_sample_size() -> None:
    """`total = scored + unparsed`, and the sign is load-bearing.

    Inverting it to a subtraction passed the entire suite. It is the same defect the
    self-audit already records once, where a sample size was wired to the number of
    decode regimes rather than the number of tasks: the number stays plausible, so
    only an arithmetic assertion can see it.
    """
    report = check_classification(tp=10, fp=5, tn=10, fn=5, unparsed=10, label="mixed")

    assert report.n == 40  # 30 scored + 10 unparsed, not 20
    # Under the subtraction the sample is 20, which is below SMALL_N, so the small-n
    # flag would fire on a measurement that does not warrant it.
    assert not any("is below" in c for c in report.concerns)


def test_the_small_n_boundary_is_exclusive() -> None:
    """Exactly SMALL_N is not below SMALL_N.

    `<` to `<=` survived the suite. An off-by-one here does not corrupt a number, it
    attaches or withholds the caveat that says whether the number may be quoted, which
    is the only thing this module exists to decide.
    """
    at = check_classification(tp=15, fp=0, tn=15, fn=0, label="at")
    assert at.n == SMALL_N
    assert not any("is below" in c for c in at.concerns)

    below = check_classification(tp=14, fp=0, tn=15, fn=0, label="below")
    assert below.n == SMALL_N - 1
    assert any("is below" in c for c in below.concerns)


def test_the_unparsed_rate_is_taken_against_the_full_sample() -> None:
    """The denominator of the unparsed rate is `total`, so it moves with the same sign.

    Pinned separately because the two uses of `total` could be corrected apart, and a
    rate reported against the wrong denominator is how finding 20's censoring error
    happened in the first place.
    """
    report = check_classification(tp=40, fp=20, tn=15, fn=10, unparsed=15, label="rate")

    assert report.n == 100  # 85 scored + 15 unparsed
    assert any("15/100" in c for c in report.concerns)
