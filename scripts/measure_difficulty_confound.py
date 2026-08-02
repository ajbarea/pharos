#!/usr/bin/env python3
"""Whether item difficulty and a wrong standard can be told apart. They cannot.

Finding 12 showed that agreement-based estimators fail once a wrong standard holds the
majority. The natural objection is that those estimators were too simple: Dawid-Skene
attributes every disagreement to the annotator, and the obvious missing term is the
*item*, since some cases are genuinely near the boundary and everyone struggles there.
Whitehill et al. (NIPS 2009) add exactly that, estimating labeler ability, item
difficulty and the true label jointly.

This corpus is unusually well suited to testing whether that helps, because it has a
real difficulty structure that was built for a different reason. The significant class
is a conjunction of three facts, and three of the ten background patterns carry *two*
of those three. Those routine items sit one fact from the boundary, and they are also
exactly the items a reviewer holding a two-of-three standard gets wrong. The two
explanations for disagreement predict identical data.

The measurement is the control that separates them. Difficulty is estimated on fleets
where nobody holds a wrong standard, and on fleets where some do. If the near-boundary
items are intrinsically hard, they should look hard in both. They do not: under a
correct fleet the estimated difficulty is flat across the corpus, because the correct
rule resolves a two-of-three item unambiguously. All of the apparent difficulty is
manufactured by the reviewers.

That is worse than the estimator merely failing. It fails by relabelling a wrong
standard as a property of the data, and the two diagnoses call for opposite actions:
clarify the guidance and accept lower accuracy on hard cases, or retrain the reviewers
who hold the wrong rule.

The ability estimate then names the wrong people, which is worse still. GLAD scores
each reviewer as well as each item, and that score is what a supervisor would act on.
Below the majority it is right, rating the wrong-standard reviewers 1.13 against 7.31.
Above it, it inverts to 8.63 against 0.27, so the same field read the same way sends
the supervisor to retrain the reviewers who are correct. Agreement at least drops to
0.717 and announces that something is wrong; the ability column stays confident.

**Only converged rows are quoted.** GLAD is an unregularised MLE whose ability and
log-difficulty are unbounded, so where the posteriors never settle the parameters keep
climbing and the estimate reports where the ascent was interrupted. On the random-slip
composition that is what happens: raising the cap from 100 to 3000 iterations grows its
mean difficulty from 47.8 to 3580.8 while the spread wanders between 1.6 and 5.2. Its
magnitude is therefore withheld and only the *position* of its peak is used, which is
stable at overlap 0 throughout. The three rows carrying the finding converge in 1, 4
and 7 iterations; `carries_the_claim` marks them, and this script exits non-zero if any
of them stops doing so. Slow convergence is a documented property of the method rather
than a defect here -- Zheng et al., PVLDB 10(5):541-552, 2017, group GLAD with the
slowest methods tested "because they solve an optimization function in each iteration",
and report separately that difficulty-modelling methods do not improve quality.

Needs no model and no network.

    uv run python scripts/measure_difficulty_confound.py --out results/difficulty.json
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from pharos.analyst import Action, AnalystPolicy, Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.inference import agreement_with, dawid_skene, glad
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.validity import check_sample_size
from pharos.world import SIGNIFICANT_PATTERN

SEED = 7
EVENTS = 200
FLEET = 9

#: The wrong standard, matching findings 12 and 16 so the three are comparable.
WRONG_THRESHOLD = 2

#: Fleet compositions to estimate difficulty under. The first is the control and is
#: the whole point: without it, a difficulty estimate has nothing to be compared to.
COMPOSITIONS: tuple[tuple[str, int, float], ...] = (
    ("correct fleet (control)", 0, 0.0),
    ("correct, 15% random slip", 0, 0.15),
    ("3 of 9 wrong standard", 3, 0.0),
    ("5 of 9 wrong standard", 5, 0.0),
)


def signature_overlap(task: TriageTask) -> int:
    """How many of the three significant facts a task carries.

    The corpus's own difficulty scale, and not one invented for this measurement: an
    item carrying two of three sits one fact from the boundary. Three means significant
    by definition, so only the routine items at two are genuinely near it.
    """
    facts: set[str] = set()
    for report in task.sources:
        facts |= set(report.fact_ids)
    return len(facts & SIGNIFICANT_PATTERN)


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """Estimated difficulty under one fleet composition, by true overlap.

    Keyword-only, like the other measurement rows: several fields here are rates in
    the same range and a positional construction would survive reordering.
    """

    composition: str
    n_wrong: int
    slip_rate: float
    by_overlap: dict[int, float]
    wrong_ability: float | None
    right_ability: float
    dawid_skene_agreement: float
    glad_agreement: float
    #: Whether the EM fit this row reports actually reached a fixed point. GLAD is an
    #: unregularised MLE and its parameters are unbounded, so a row that stopped at
    #: `max_iters` reports where the ascent happened to be, not where it settled. Only
    #: a converged row's magnitudes may be quoted.
    converged: bool
    iterations: int

    @property
    def difficulty_spread(self) -> float:
        """Hardest overlap band over the easiest. One means no structure at all."""
        values = [v for v in self.by_overlap.values() if v > 0]
        return max(values) / min(values) if len(values) > 1 else 1.0

    @property
    def ability_ratio(self) -> float | None:
        """How much abler the estimator thinks the wrong-standard reviewers are.

        Above one means it rates them higher than the reviewers who are correct. None
        where there are no wrong-standard reviewers to compare against, which is not
        a ratio of one: a fleet with nobody wrong has no answer to this question.
        """
        if self.wrong_ability is None or self.right_ability <= 0:
            return None
        return self.wrong_ability / self.right_ability

    @property
    def ability_is_inverted(self) -> bool:
        """Whether acting on the ability score would retrain the correct reviewers."""
        ratio = self.ability_ratio
        return ratio is not None and ratio > 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "composition": self.composition,
            "n_wrong": self.n_wrong,
            "slip_rate": self.slip_rate,
            "difficulty_by_overlap": {str(k): round(v, 4) for k, v in self.by_overlap.items()},
            "difficulty_spread": round(self.difficulty_spread, 3),
            "wrong_ability": None if self.wrong_ability is None else round(self.wrong_ability, 4),
            "right_ability": round(self.right_ability, 4),
            "ability_ratio": None if self.ability_ratio is None else round(self.ability_ratio, 4),
            "ability_is_inverted": self.ability_is_inverted,
            "dawid_skene_agreement": round(self.dawid_skene_agreement, 4),
            "glad_agreement": round(self.glad_agreement, 4),
            "converged": self.converged,
            "iterations": self.iterations,
            "quotable": self.converged,
        }


def carries_the_claim(row: "Row") -> bool:
    """Whether the published finding quotes this row's magnitudes.

    The control and the two wrong-standard rows do: the finding is the contrast
    between a flat control and a spread that appears only when reviewers hold the
    wrong rule. The random-slip row is a foil, and only the *position* of its peak is
    used, which is why it is allowed not to converge.
    """
    return row.n_wrong > 0 or row.slip_rate == 0.0


def build_fleet(n_wrong: int, slip: float) -> tuple[AnalystPolicy, ...]:
    right = [AnalystPolicy(f"right-{i}", slip_rate=slip) for i in range(FLEET - n_wrong)]
    wrong = [
        AnalystPolicy(f"wrong-{i}", escalation_threshold=WRONG_THRESHOLD) for i in range(n_wrong)
    ]
    return tuple(right + wrong)


def collect(
    policies: tuple[AnalystPolicy, ...], tasks, proposals, *, seed: int
) -> list[tuple[str, str, bool]]:
    rows: list[tuple[str, str, bool]] = []
    for policy in policies:
        for task in tasks:
            decision = policy.review(task, proposals[task.task_id], seed=seed)
            if decision.action is Action.ACCEPT:
                verdict: bool | None = proposals[task.task_id].verdict
            elif decision.action is Action.REVISE:
                verdict = decision.corrected_verdict
            else:
                verdict = None
            if verdict is not None:
                rows.append((task.task_id, policy.name, verdict))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}
    overlap = {t.task_id: signature_overlap(t) for t in tasks}
    bands = sorted(set(overlap.values()))

    counts = {k: sum(1 for v in overlap.values() if v == k) for k in bands}
    print(f"{len(tasks)} tasks; items by signature-fact overlap: {counts}")
    print("overlap 2 is a routine item one fact from the boundary, and is exactly where")
    print(f"a {WRONG_THRESHOLD}-of-3 reviewer errs\n")
    header = "".join(f"{f'ovl={k}':>9}" for k in bands)
    print(f"  {'fleet composition':<28}{header}{'spread':>9}")
    print("  " + "-" * (28 + 9 * len(bands) + 9))

    rows: list[Row] = []
    for name, n_wrong, slip in COMPOSITIONS:
        contributions = collect(build_fleet(n_wrong, slip), tasks, proposals, seed=SEED)
        estimate = glad(contributions)
        ds = dawid_skene(contributions)

        by_overlap = {}
        for band in bands:
            ids = [t for t, o in overlap.items() if o == band and t in estimate.log_difficulty]
            if ids:
                by_overlap[band] = statistics.mean(estimate.difficulty(t) for t in ids)

        wrong_names = [p for p in estimate.ability if p.startswith("wrong-")]
        row = Row(
            composition=name,
            n_wrong=n_wrong,
            slip_rate=slip,
            by_overlap=by_overlap,
            wrong_ability=(
                statistics.mean(estimate.ability[p] for p in wrong_names) if wrong_names else None
            ),
            right_ability=statistics.mean(
                v for p, v in estimate.ability.items() if p.startswith("right-")
            ),
            dawid_skene_agreement=agreement_with(ds.labels(), truth),
            glad_agreement=agreement_with(estimate.labels(), truth),
            converged=estimate.converged,
            iterations=estimate.iterations,
        )
        rows.append(row)
        cells = "".join(f"{by_overlap.get(k, float('nan')):>9.3f}" for k in bands)
        mark = "" if row.converged else "  (did not converge)"
        print(f"  {name:<28}{cells}{row.difficulty_spread:>9.1f}{mark}")
        record(
            "difficulty.composition",
            row.difficulty_spread,
            composition=name,
            n_wrong=n_wrong,
            slip_rate=slip,
            glad_agreement=round(row.glad_agreement, 4),
            converged=row.converged,
            iterations=row.iterations,
        )

    # Every number below is read off converged rows only. GLAD is an unregularised
    # MLE whose ability and log-difficulty are unbounded, so when the posteriors do
    # not settle the parameters keep climbing and the estimate reports where the
    # ascent was interrupted. On the random-slip row here the mean difficulty grows
    # from 48 to 1290 between 100 and 1000 iterations while the spread wanders
    # between 1.6 and 4.5, so its magnitude is a function of `max_iters` rather than
    # of the data. Zheng et al. (PVLDB 2017) report the same slowness as a
    # characteristic of GLAD across many real datasets.
    stalled = [r for r in rows if not r.converged]
    settled = [r for r in rows if r.converged]
    if stalled:
        print()
        print(
            "NOT CONVERGED, and therefore not quoted: "
            + ", ".join(f"'{r.composition}' ({r.iterations} iterations)" for r in stalled)
        )
        print("Their magnitudes are where the ascent stopped, not where it settled.")

    # The ability column, which is the one a supervisor would actually act on: it
    # answers "whom do I retrain". Difficulty being manufactured is the headline, but
    # this is the part that sends someone to the wrong person.
    inverted = [r for r in settled if r.ability_is_inverted]
    if any(r.wrong_ability is not None for r in settled):
        print()
        print(f"  {'fleet composition':<28}{'wrong':>10}{'correct':>10}   ability estimate")
        print("  " + "-" * 66)
        for row in settled:
            if row.wrong_ability is None:
                continue
            verdict = "INVERTED: the wrong rule scores higher" if row.ability_is_inverted else "ok"
            print(
                f"  {row.composition:<28}{row.wrong_ability:>10.2f}"
                f"{row.right_ability:>10.2f}   {verdict}"
            )

    if inverted:
        worst_inversion = max(inverted, key=lambda r: r.ability_ratio or 0.0)
        print()
        print(
            "The ability estimate tracks the majority rather than the truth. Acting on it\n"
            f"under '{worst_inversion.composition}' retrains the correct reviewers and\n"
            "certifies the wrong standard as the expert one."
        )
        get_logger().warning(
            "difficulty.ability_inverted",
            extra={
                "event": "difficulty.ability_inverted",
                "composition": worst_inversion.composition,
                "wrong_ability": round(worst_inversion.wrong_ability or 0.0, 4),
                "right_ability": round(worst_inversion.right_ability, 4),
                "ratio": round(worst_inversion.ability_ratio or 0.0, 3),
            },
        )

    control = rows[0]
    worst = max(settled, key=lambda r: r.difficulty_spread)
    print()
    print(
        f"Control spread {control.difficulty_spread:.1f} against "
        f"{worst.difficulty_spread:.1f} for '{worst.composition}'."
    )
    if control.difficulty_spread < 1.5 <= worst.difficulty_spread:
        print(
            "A correct fleet finds no difficulty structure at all, so the structure the\n"
            "other rows report is manufactured by the reviewers rather than carried by\n"
            "the items. The estimator relabels a wrong standard as a property of the data."
        )
        get_logger().warning(
            "difficulty.manufactured_by_reviewers",
            extra={
                "event": "difficulty.manufactured_by_reviewers",
                "control_spread": round(control.difficulty_spread, 3),
                "worst_spread": round(worst.difficulty_spread, 3),
                "worst_composition": worst.composition,
            },
        )

    report = {
        "provenance": run_provenance(seed=SEED),
        "events": args.events,
        "fleet": FLEET,
        "wrong_threshold": WRONG_THRESHOLD,
        "items_by_overlap": {str(k): v for k, v in counts.items()},
        "rows": [r.as_dict() for r in rows],
        "converged_rows": [r.composition for r in settled],
        "unconverged_rows": [r.composition for r in stalled],
        "validity": check_sample_size(len(tasks), label="difficulty confound").as_dict(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    # The rows the published finding quotes have to have settled. A test asserts this
    # of the committed artifact, but CI runs this script without `--out`, so without
    # the check here a load-bearing row could start stalling and CI would report only
    # a warning that logcheck has been told to expect. Failing here makes the guard
    # live rather than retrospective.
    stalled_load_bearing = [r.composition for r in rows if carries_the_claim(r) and not r.converged]
    if stalled_load_bearing:
        print(
            "\nFAIL: these rows carry the published claim and did not converge:\n  "
            + "\n  ".join(stalled_load_bearing)
            + "\nTheir magnitudes cannot be quoted, so the finding no longer stands as"
            " written."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
