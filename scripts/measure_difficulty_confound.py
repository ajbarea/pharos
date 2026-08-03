#!/usr/bin/env python3
"""Whether item difficulty and a wrong standard can be told apart. They cannot.

Finding 12 showed agreement-based estimators fail once a wrong standard holds the
majority. The obvious objection is that they blame only the annotator; Whitehill et al.
(NIPS 2009) add an item-difficulty term. This corpus can test that, because its
near-boundary items -- routine cases carrying two of the three significant facts -- are
exactly the items a two-of-three reviewer gets wrong. "Hard item" and "wrong reviewer"
predict identical data.

The control decides it. Under a correct fleet the estimated difficulty is flat, because
the correct rule resolves a two-of-three item unambiguously, so every unit of structure
the other rows report was manufactured by the reviewers. Worse, GLAD's per-reviewer
ability -- the field a supervisor would act on to decide whom to retrain -- inverts at
the majority.

Class-conditioning does not rescue it. Singer et al. (arXiv:2607.24622, 2026) note that
a single ability per annotator "prevents them from distinguishing majority-class
competence from minority-class competence", which is precisely a two-of-three reviewer.
Their CC-Rasch conditions both terms on the class and is run here as the strongest form
of the objection. It agrees with Dawid-Skene and GLAD to three decimals everywhere,
0.717 included; its class-conditional diagnostic separates the groups below the majority
and reaches zero at it. Zheng et al. (PVLDB 10(5):541-552, 2017) report the same
conclusion from a benchmark that was not looking for it.

`carries_the_claim` marks the rows whose magnitudes are quoted, and this script exits
non-zero if one of them stops converging in either estimator.

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
from pharos.inference import agreement_with, cc_rasch, dawid_skene, glad
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


def compositions(fleet: int) -> tuple[tuple[str, int, float], ...]:
    """Fleet compositions to estimate difficulty under, derived from the fleet size.

    The first is the control and is the whole point: without it, a difficulty estimate
    has nothing to be compared to. The other two are a clear minority holding the wrong
    standard and a bare majority holding it, which is the crossing every finding here
    turns on.

    Derived rather than hardcoded because fleet size is a researcher degree of freedom
    with no principled single value, and a conclusion that moves with it is a property
    of the choice rather than of the system. These reproduce the previous literals
    exactly at the default of 9 -- a third is 3, a bare majority is 5.
    """
    return (
        ("correct fleet (control)", 0, 0.0),
        ("correct, 15% random slip", 0, 0.15),
        (f"{round(fleet / 3)} of {fleet} wrong standard", round(fleet / 3), 0.0),
        (f"{fleet // 2 + 1} of {fleet} wrong standard", fleet // 2 + 1, 0.0),
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
    #: Whether the EM fit this row reports actually reached a fixed point. A row that
    #: stopped at `max_iters` reports where the ascent happened to be, not where it
    #: settled, so only a converged row's magnitudes may be quoted.
    converged: bool
    iterations: int
    #: The class-conditional estimator, which is the strongest answer available to
    #: "a better model would separate these". Singer et al. (arXiv:2607.24622, 2026)
    #: note that GLAD's single ability per annotator "prevents them from
    #: distinguishing majority-class competence from minority-class competence" --
    #: exactly the shape of a two-of-three reviewer, who is right on the significant
    #: class and wrong only on routine items near the boundary.
    cc_rasch_agreement: float
    cc_rasch_converged: bool
    #: Ability on the ROUTINE class specifically, where the wrong standard errs. This
    #: is the diagnostic a class-conditional model buys that GLAD cannot express.
    cc_wrong_routine: float | None
    cc_right_routine: float

    @property
    def cc_routine_separation(self) -> float | None:
        """How far routine-class ability separates correct reviewers from wrong ones.

        A difference rather than a ratio, because CC-Rasch's abilities are logits:
        `sigmoid(ability - difficulty)`. Differences on that scale are the meaningful
        comparison, and a ratio is not even well defined here -- a reviewer who is
        worse than chance on a class scores negative, which is exactly what the
        wrong-standard reviewers do below the majority.

        Above zero means the model can see the error where it actually is. It goes to
        zero at precisely the composition where every estimator fails.
        """
        if self.cc_wrong_routine is None:
            return None
        return self.cc_right_routine - self.cc_wrong_routine

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
            "cc_rasch_agreement": round(self.cc_rasch_agreement, 4),
            "cc_rasch_converged": self.cc_rasch_converged,
            "cc_wrong_routine_ability": (
                None if self.cc_wrong_routine is None else round(self.cc_wrong_routine, 4)
            ),
            "cc_right_routine_ability": round(self.cc_right_routine, 4),
            "cc_routine_separation": (
                None if self.cc_routine_separation is None else round(self.cc_routine_separation, 4)
            ),
        }


def carries_the_claim(row: "Row") -> bool:
    """Whether the published finding quotes this row's magnitudes.

    The control and the two wrong-standard rows do: the finding is the contrast
    between a flat control and a spread that appears only when reviewers hold the
    wrong rule. The random-slip row is a foil, and only the *position* of its peak is
    used, which is why it is allowed not to converge.
    """
    return row.n_wrong > 0 or row.slip_rate == 0.0


def build_fleet(n_wrong: int, slip: float, fleet: int = FLEET) -> tuple[AnalystPolicy, ...]:
    right = [AnalystPolicy(f"right-{i}", slip_rate=slip) for i in range(fleet - n_wrong)]
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
    parser.add_argument("--fleet", type=int, default=FLEET)
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
    for name, n_wrong, slip in compositions(args.fleet):
        contributions = collect(build_fleet(n_wrong, slip, args.fleet), tasks, proposals, seed=SEED)
        estimate = glad(contributions)
        ds = dawid_skene(contributions)
        ccr = cc_rasch(contributions)

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
            cc_rasch_agreement=agreement_with(ccr.labels(), truth),
            cc_rasch_converged=ccr.converged,
            cc_wrong_routine=(
                statistics.mean(ccr.ability[p][False] for p in wrong_names) if wrong_names else None
            ),
            cc_right_routine=statistics.mean(
                v[False] for p, v in ccr.ability.items() if p.startswith("right-")
            ),
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

    # Does class-conditioning rescue it? The obvious objection to everything above is
    # that GLAD gives each reviewer one ability number, and a two-of-three reviewer is
    # not globally unreliable -- they are exactly right on the significant class and
    # wrong only on routine items at the boundary. CC-Rasch models ability per class,
    # which is the shape that objection asks for.
    print()
    print(f"  {'fleet composition':<28}{'D-S':>7}{'GLAD':>7}{'CC-R':>7}{'routine gap':>13}")
    print("  " + "-" * 62)
    for row in rows:
        sep = row.cc_routine_separation
        mark = "" if row.cc_rasch_converged else "  (CC-R stalled)"
        shown = "--" if sep is None else f"{sep:+.2f}"
        print(
            f"  {row.composition:<28}{row.dawid_skene_agreement:>7.3f}"
            f"{row.glad_agreement:>7.3f}{row.cc_rasch_agreement:>7.3f}{shown:>13}{mark}"
        )

    graded = [r for r in settled if r.cc_routine_separation is not None and r.cc_rasch_converged]
    if graded:
        best = max(graded, key=lambda r: r.cc_routine_separation or 0.0)
        worst_sep = min(graded, key=lambda r: r.cc_routine_separation or 0.0)
        print()
        print(
            "Class-conditioning buys a real diagnostic below the majority and loses it "
            "above.\n"
            f"Routine-class ability separates correct from wrong reviewers by "
            f"{best.cc_routine_separation:+.2f} logits\nunder '{best.composition}', and by "
            f"{worst_sep.cc_routine_separation:+.2f} under '{worst_sep.composition}', where the\n"
            "agreement matches Dawid-Skene and GLAD exactly. Modelling the class does not\n"
            "rescue the estimate; it relocates where the failure is visible."
        )
        record(
            "difficulty.class_conditional_separation",
            worst_sep.cc_routine_separation or 0.0,
            best_composition=best.composition,
            best_separation=round(best.cc_routine_separation or 0.0, 4),
            worst_composition=worst_sep.composition,
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
        "fleet": args.fleet,
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
    # Both estimators, on both counts. CC-Rasch's numbers are quoted for the same three
    # rows GLAD's are, so a stall there disqualifies just as much. The random-slip row
    # is a foil for both and is allowed to stall in either, which is why the expected
    # `inference.cc_rasch_did_not_converge` warning is safe to whitelist in logcheck.
    stalled_load_bearing = [
        f"{r.composition} (GLAD)" for r in rows if carries_the_claim(r) and not r.converged
    ] + [
        f"{r.composition} (CC-Rasch)"
        for r in rows
        if carries_the_claim(r) and not r.cc_rasch_converged
    ]
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
