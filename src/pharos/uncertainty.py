"""Intervals for scores whose measurement is itself noisy.

Finding 9 established that a Pharos score taken under the reasoning decode moves
when nothing changes: 10% of tasks disagree with themselves across identical calls
at temperature 0. A single-pass number therefore has two error sources, and quoting
it bare hides both.

- **Between-task.** The usual one. Some events are harder than others, and a
  different sample of 30 would score differently. Shrinks as tasks are added.
- **Within-task.** Run-to-run instability on the *same* task. Does not shrink with
  more tasks at all; only more runs per task reduce it.

Reporting a binomial interval over `n` tasks accounts for the first and silently
assumes the second is zero. That assumption is what this module removes.

**The estimand has to match the deployment, and here it is not the mean.** A common
reflex is to average several runs per task and interval the average. That estimates
the performance of a system that *votes*, and a Pharos fleet does not vote: one edge
agent answers once and the answer is acted on. `single_run` is therefore the headline
quantity, and `consensus` is reported beside it as what voting would buy. Quoting the
consensus number for a single-shot deployment overstates it, and the gap between the
two is a real design input rather than a rounding detail.

**Cluster bootstrap, because runs within a task are correlated.** Resampling
individual `(task, run)` pairs would treat three calls on one task as three
independent observations and produce an interval far too narrow. Resampling whole
tasks with all their runs attached is the standard correction.

The intervals here are percentile intervals and are honest about being a floor
rather than a guarantee. Published analysis of LLM measurement error finds that
standard intervals under-cover because they omit variation from prompt phrasing,
judge choice, and model version, and that the under-coverage worsens with more data.
Nothing in this module addresses those; it quantifies the two sources Pharos can
resample and leaves the rest named in the docs.
"""

import math
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

#: Enough that the percentile endpoints are stable to about a thousandth, which is
#: finer than any Pharos measurement resolves. More costs time and buys noise.
DEFAULT_RESAMPLES = 2000

#: A trial is correct, incorrect, or unparsable. The third is not a wrong answer and
#: must not be scored as one, so it is carried separately rather than coerced.
Outcome = bool | None


@dataclass(frozen=True, slots=True)
class Trial:
    """One task's outcome on one run."""

    task_id: str
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate and the range resampling puts around it."""

    point: float
    low: float
    high: float
    level: float
    resamples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def covers(self, value: float) -> bool:
        """Whether `value` sits inside the interval.

        The question to ask before claiming two conditions differ: if each covers
        the other's point estimate, the difference is not resolved by this data.
        """
        return self.low <= value <= self.high

    def as_dict(self) -> dict[str, object]:
        return {
            "point": round(self.point, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "level": self.level,
            "resamples": self.resamples,
        }


@dataclass(frozen=True, slots=True)
class VarianceSplit:
    """Where a score's noise comes from, and therefore what would reduce it."""

    between_task: float
    within_task: float

    @property
    def total(self) -> float:
        return self.between_task + self.within_task

    @property
    def within_share(self) -> float:
        """Share of variance that adding tasks cannot reduce.

        The actionable number. High means the measurement is limited by run-to-run
        instability and wants more repeats; low means it is limited by sample size
        and wants more tasks. Reporting a score without knowing which is how an
        experiment ends up spending its budget on the axis that does not help.
        """
        return self.within_task / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "between_task": round(self.between_task, 6),
            "within_task": round(self.within_task, 6),
            "within_share": round(self.within_share, 4),
        }


def _by_task(trials: Iterable[Trial]) -> dict[str, list[Outcome]]:
    grouped: dict[str, list[Outcome]] = {}
    for trial in trials:
        grouped.setdefault(trial.task_id, []).append(trial.outcome)
    return grouped


def scored(outcomes: Iterable[Outcome]) -> list[bool]:
    """Outcomes that are answers, dropping the unparsable ones."""
    return [o for o in outcomes if o is not None]


def single_run_rate(trials: Iterable[Trial]) -> float:
    """Expected correctness of one run drawn at random.

    Every `(task, run)` pair weighted equally, which is what a deployment answering
    once per task actually experiences.
    """
    answers = scored(t.outcome for t in trials)
    return sum(answers) / len(answers) if answers else 0.0


def consensus_rate(trials: Iterable[Trial]) -> float:
    """Correctness of a majority vote over each task's runs.

    What repeated sampling would buy a system able to afford it. Ties count as
    incorrect: a fleet that cannot decide has not produced a verdict, and scoring a
    tie as a coin flip would credit it for half of one.
    """
    grouped = _by_task(trials)
    if not grouped:
        return 0.0
    correct = 0
    for outcomes in grouped.values():
        answers = scored(outcomes)
        if answers and sum(answers) * 2 > len(answers):
            correct += 1
    return correct / len(grouped)


def variance_split(trials: Iterable[Trial]) -> VarianceSplit:
    """Between-task and within-task variance of a binary outcome.

    Within-task uses each task's own Bernoulli variance across its runs, averaged.
    A task answered identically every time contributes zero, which is the property
    that makes the split readable: it is exactly the instability finding 9 measured.
    """
    grouped = _by_task(trials)
    per_task = []
    within = []
    for outcomes in grouped.values():
        answers = scored(outcomes)
        if not answers:
            continue
        rate = sum(answers) / len(answers)
        per_task.append(rate)
        within.append(rate * (1.0 - rate))
    if not per_task:
        return VarianceSplit(0.0, 0.0)
    between = statistics.pvariance(per_task) if len(per_task) > 1 else 0.0
    return VarianceSplit(between, statistics.fmean(within))


def cluster_bootstrap(
    trials: Sequence[Trial],
    statistic: Callable[[Sequence[Trial]], float] = single_run_rate,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 7,
) -> Interval:
    """A percentile interval, resampling whole tasks rather than individual runs.

    Runs of the same task are correlated -- a task the model always gets right
    contributes k identical successes -- so resampling `(task, run)` pairs would
    count them as k independent observations and shrink the interval by roughly
    sqrt(k) for no reason. Clusters are the tasks.

    Deterministic from `seed`, so an interval is reproducible from the artifact
    rather than being a different number on every read.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be within (0, 1), got {level}")
    grouped = _by_task(trials)
    task_ids = sorted(grouped)
    if len(task_ids) < 2:
        point = statistic(list(trials))
        return Interval(point, point, point, level, 0)

    rng = random.Random(seed)
    n = len(task_ids)
    draws: list[float] = []
    for _ in range(resamples):
        picked: list[Trial] = []
        for _ in range(n):
            task_id = task_ids[rng.randrange(n)]
            picked.extend(Trial(task_id, o) for o in grouped[task_id])
        draws.append(statistic(picked))

    draws.sort()
    tail = (1.0 - level) / 2.0
    low = draws[max(0, int(tail * resamples) - 1)]
    high = draws[min(resamples - 1, int((1.0 - tail) * resamples))]
    return Interval(statistic(list(trials)), low, high, level, resamples)


@dataclass(frozen=True, slots=True)
class Measurement:
    """A score reported the way finding 9 says it has to be.

    Carries both estimands rather than choosing for the reader: `single_run` is what
    an agent answering once achieves, `consensus` what voting over the same runs
    would achieve, and the gap between them is the value of repeating.
    """

    label: str
    n_tasks: int
    repeats: int
    unparsed: int
    single_run: Interval
    consensus: float
    variance: VarianceSplit

    @property
    def consensus_gain(self) -> float:
        """What voting buys over answering once. Negative means it costs."""
        return self.consensus - self.single_run.point

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "n_tasks": self.n_tasks,
            "repeats": self.repeats,
            "unparsed": self.unparsed,
            "single_run": self.single_run.as_dict(),
            "consensus": round(self.consensus, 4),
            "consensus_gain": round(self.consensus_gain, 4),
            "variance": self.variance.as_dict(),
        }


def summarize(
    trials: Sequence[Trial],
    *,
    label: str,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 7,
) -> Measurement:
    """Everything that should accompany a model-dependent Pharos number."""
    grouped = _by_task(trials)
    repeats = max((len(v) for v in grouped.values()), default=0)
    return Measurement(
        label=label,
        n_tasks=len(grouped),
        repeats=repeats,
        unparsed=sum(1 for t in trials if t.outcome is None),
        single_run=cluster_bootstrap(
            trials, single_run_rate, resamples=resamples, level=level, seed=seed
        ),
        consensus=consensus_rate(trials),
        variance=variance_split(trials),
    )


def resolves(first: Interval, second: Interval) -> bool:
    """Whether two intervals separate their point estimates.

    False when either covers the other's point. For two intervals of equal
    half-width `h` this is satisfied once the gap exceeds roughly `h`.

    **This is the weaker of the two criteria here, not the stronger one.** An earlier
    version of this docstring called it conservative, which is backwards: the question
    "do two conditions differ" is properly asked of an interval on their *difference*,
    and that interval is wider than either input. Use `resolves_difference` for a
    claim that two conditions differ. Keep this one for the weaker and still useful
    statement that a reported ordering is not an artifact of one interval alone.
    """
    return not (first.covers(second.point) or second.covers(first.point))


def resolves_difference(first: Interval, second: Interval) -> bool:
    """Whether the *difference* between two conditions is resolved.

    The gap must exceed the half-width of an interval on the difference, which under
    independence is the root of the sum of squares rather than either half-width or
    their sum. Compared with the alternatives:

        gap > h                  what `resolves` implements; too permissive
        gap > sqrt(h1^2 + h2^2)  this; an interval on the difference
        gap > h1 + h2            never wrong, and needlessly strict

    Independence is assumed and is the conservative assumption when conditions share
    an evaluation set, since positive correlation across shared tasks would narrow the
    difference interval. A paired bootstrap over per-task outcomes would be tighter
    and is what this should become once measurements persist per-task rows; until they
    do, this bounds the claim rather than overstating it.
    """
    gap = abs(first.point - second.point)
    spread = math.sqrt((first.width / 2.0) ** 2 + (second.width / 2.0) ** 2)
    return gap > spread
