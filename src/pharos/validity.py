"""Conditions under which a measured number should not be quoted.

Every check here exists because something in this project's history went wrong in
exactly the way it detects. They are warnings rather than errors on purpose: a run
that trips one still produced a real number, and suppressing it would be worse than
publishing it. What must not happen is quoting it without noticing.

- **Small n.** Two findings were first measured at n=8, written up with strong
  severity claims, and did not reproduce at n=24 and n=40. Nothing about those runs
  looked wrong; the artifacts were well-formed and internally consistent.
- **Degenerate predictions.** A model answering one class for everything yields a
  confusion matrix that still computes a precision, a recall, and an F1. Recall
  1.000 at precision 0.4 is a real pattern in this benchmark and worth surfacing
  rather than rediscovering.
- **Unparsed answers.** How often a model fails to answer at all is a result, and it
  quietly caps every other number in the same run.
- **Class imbalance.** An evaluation set skewed enough that the majority floor
  exceeds the score being celebrated.
- **Ceiling and floor collisions.** A score at or below the majority baseline is not
  evidence of capability, whatever its absolute value.

Nothing here changes a measurement. It only says what a reader would need to know
before repeating one.
"""

from dataclasses import dataclass, field

from pharos.telemetry import get_logger

#: Below this, a difference of a few points is one or two flipped instances. Chosen
#: from the n=8 failure: at that size a single task moves accuracy by 12.5 points.
SMALL_N = 30

#: An unparsed rate above this means the score is describing a different population
#: than the one requested.
HIGH_UNPARSED_RATE = 0.10

#: Above this, one class dominates enough that the majority floor is the number
#: worth beating, not chance.
SKEWED_PREVALENCE = 0.70

# A perfect score is not automatically a good one. With zero observed errors the
# sample bounds the true error rate only loosely: the rule of three puts the
# one-sided 95% upper bound at 3/n, so 60 flawless answers are still consistent with
# a 5% error rate. A saturated measurement also has no headroom, which means it can
# no longer rank anything above it -- worth saying out loud before it is reported as
# a headline.
SATURATED = 1.0


@dataclass(frozen=True, slots=True)
class ValidityReport:
    """What is wrong with a measurement, if anything."""

    n: int
    concerns: tuple[str, ...] = field(default=())

    @property
    def quotable(self) -> bool:
        """Whether the number can be quoted without a caveat attached."""
        return not self.concerns

    def as_dict(self) -> dict[str, object]:
        return {"n": self.n, "quotable": self.quotable, "concerns": list(self.concerns)}


def check_classification(
    *,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    unparsed: int = 0,
    label: str = "measurement",
) -> ValidityReport:
    """Inspect a confusion matrix for the conditions that make a score misleading.

    Emits one structured warning per concern, so a run's own log says which of its
    numbers need a caveat rather than leaving it to whoever reads the JSON later.
    """
    logger = get_logger()
    scored = tp + fp + tn + fn
    total = scored + unparsed
    concerns: list[str] = []

    if total == 0:
        concerns.append("no instances were scored at all")
        logger.warning("validity.empty", extra={"event": "validity.empty", "label": label, "n": 0})
        return ValidityReport(n=0, concerns=tuple(concerns))

    if total < SMALL_N:
        concerns.append(
            f"n={total} is below {SMALL_N}: one flipped instance moves accuracy by "
            f"{100 / total:.1f} points, so differences smaller than that are noise"
        )
        logger.warning(
            "validity.small_n",
            extra={
                "event": "validity.small_n",
                "label": label,
                "n": total,
                "points_per_instance": round(100 / total, 2),
            },
        )

    if unparsed and unparsed / total > HIGH_UNPARSED_RATE:
        concerns.append(
            f"{unparsed}/{total} answers were unparsable ({unparsed / total:.0%}); "
            "every other number describes only the remainder"
        )
        logger.warning(
            "validity.high_unparsed",
            extra={
                "event": "validity.high_unparsed",
                "label": label,
                "unparsed": unparsed,
                "n": total,
                "rate": round(unparsed / total, 4),
            },
        )

    positives = tp + fn
    if scored:
        prevalence = positives / scored
        majority = max(prevalence, 1 - prevalence)
        if majority > SKEWED_PREVALENCE:
            concerns.append(
                f"class prevalence {prevalence:.2f} gives a majority floor of "
                f"{majority:.3f}; compare scores to that, not to 0.5"
            )
            logger.warning(
                "validity.skewed_classes",
                extra={
                    "event": "validity.skewed_classes",
                    "label": label,
                    "prevalence": round(prevalence, 4),
                    "majority_floor": round(majority, 4),
                },
            )

        accuracy = (tp + tn) / scored
        if accuracy <= majority:
            concerns.append(
                f"accuracy {accuracy:.3f} does not beat the majority floor "
                f"{majority:.3f}: this is not evidence of capability"
            )
            logger.warning(
                "validity.below_majority",
                extra={
                    "event": "validity.below_majority",
                    "label": label,
                    "accuracy": round(accuracy, 4),
                    "majority_floor": round(majority, 4),
                },
            )

    if scored and tp + tn == scored and scored > 0:
        bound = 3 / scored
        concerns.append(
            f"score is saturated (no errors in {scored}): the rule of three bounds "
            f"the true error rate only at {bound:.1%}, and a ceiling cannot rank "
            "anything above it. Report n alongside the score"
        )
        logger.warning(
            "validity.saturated",
            extra={
                "event": "validity.saturated",
                "label": label,
                "n": scored,
                "error_rate_upper_95": round(bound, 4),
            },
        )

    if scored and (tp + fp == scored or tn + fn == scored):
        predicted = "positive" if tp + fp == scored else "negative"
        concerns.append(
            f"every prediction was {predicted}: the model is not discriminating, "
            "and precision/recall describe the class balance rather than the model"
        )
        logger.warning(
            "validity.degenerate_predictions",
            extra={
                "event": "validity.degenerate_predictions",
                "label": label,
                "predicted": predicted,
                "n": scored,
            },
        )

    if positives and fn == 0 and fp > tp:
        concerns.append(
            "recall is 1.000 while false positives exceed true positives: the model "
            "escalates indiscriminately, which scores well on recall alone"
        )
        logger.warning(
            "validity.over_escalation",
            extra={
                "event": "validity.over_escalation",
                "label": label,
                "tp": tp,
                "fp": fp,
            },
        )

    report = ValidityReport(n=total, concerns=tuple(concerns))
    logger.info(
        "validity.report",
        extra={"event": "validity.report", "label": label, **report.as_dict()},
    )
    return report


def check_sample_size(n: int, *, label: str = "measurement") -> ValidityReport:
    """The small-n check alone, for measurements with no confusion matrix."""
    concerns: list[str] = []
    if n < SMALL_N:
        concerns.append(f"n={n} is below {SMALL_N}; treat differences as provisional")
        get_logger().warning(
            "validity.small_n",
            extra={"event": "validity.small_n", "label": label, "n": n},
        )
    return ValidityReport(n=n, concerns=tuple(concerns))
