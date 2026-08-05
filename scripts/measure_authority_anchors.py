#!/usr/bin/env python3
"""What an authority of record costs, in audited items per round.

Finding 18 moved the reliability estimate under secure aggregation and the cliff did
not move with it: past five of nine analysts holding the wrong standard, agreement
still falls from 1.000 to 0.660. That was the predicted outcome and it localises the
problem. The cliff is not a leak and not an artifact of pooling; it is
non-identifiability. Dawid-Skene's parameters are only identified up to a relabelling
of the latent class, and the tie-break the literature relies on is diagonal dominance
-- FedDS (Dong, Zhu, Shang and Xue, *Inf. Sci.* 745:123425, 2026) assumes exactly that
in its Eq. (16). A fleet whose majority holds the wrong standard is the fleet where
that assumption is false, so the estimator settles on the wrong labelling and reports
no distress while doing it.

No estimator escapes that from the data alone, which is finding 17's result from the
other direction: modelling item difficulty converts a wrong standard into a hard item
rather than separating them. What breaks a relabelling degeneracy is not a better
estimator but an **exogenous** label -- a task whose true disposition is asserted by an
authority rather than inferred from the fleet. That is the "authority of record" the
build order has owed since step 3, and this measures its price.

**The scoring rule is the whole methodology here.** An anchored task's label is handed
over, so counting it would measure how many answers the authority supplied rather than
what they bought. Every agreement below is computed **only over unanchored tasks**, so
an anchor earns its place solely by what it fixes elsewhere. Getting this wrong would
manufacture a curve that rises with the anchor count by construction.

Needs no model and no network.

    uv run python scripts/measure_authority_anchors.py --out results/authority_anchors.json
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random

from measure_secure_reliability import (
    MASK_SEED,
    contributions_for,
    fleet_of,
)

from pharos.analyst import Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.inference import (
    agreement_with,
    federated_dawid_skene,
    partition_by_contributor,
)
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200
FLEET = 9

#: Anchor counts swept. Dense at the low end because the first question is whether a
#: handful suffices, and carried out to 180 of 200 because at a supermajority it does
#: not: an earlier sweep stopping at 50 reported "not reached" for three compositions,
#: which is a bound on the experiment rather than a price. A budget the sweep never
#: reaches is not a measurement, so the range runs until every repairable composition
#: is repaired and the one that never repairs can be named as such.
ANCHOR_COUNTS = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 100, 120, 150, 180)

#: Which fleet compositions to price. Below the cliff there is nothing to repair, so
#: the sweep runs from the crossing upward and carries one pre-cliff row as a control.
#: Nine is the unanimity control and is expected never to repair: with every analyst
#: holding the wrong standard there is no disagreement to estimate from, and an
#: authority that has to rule on every task has not been assisted by a fleet at all.
COMPOSITIONS = (4, 5, 6, 7, 9)

#: Agreement counted as repaired. The pre-cliff level is 1.000 and the post-cliff
#: level is 0.660; anything at or above this is nearer the first than the second by a
#: wide margin, and the exact threshold is reported so a reader can move it.
REPAIRED = 0.95

#: Seeds for choosing which tasks the authority rules on. Distinct from the corpus seed
#: so the anchor draw cannot correlate with corpus structure by accident.
#:
#: Plural, and that is the point. This measured one draw, and one draw is not a price.
#: Making the draw nested across budgets -- a change to *which* items a given seed
#: picks, not to the method -- moved the five-wrong threshold from 5 anchors to 30 and
#: turned the nine-wrong threshold from "180" into "not reached", with nothing else
#: touched. A number that mobile under a re-draw has to be reported with its spread,
#: so the sweep runs every seed and the headline is the median over them.
#:
#: Odd count on purpose: the median is then an observed draw rather than an average of
#: two, which matters because the quantity is censored -- a draw that never repairs has
#: no finite threshold to average.
ANCHOR_SEEDS = tuple(909 + i for i in range(21))

#: The draw whose full agreement grid is printed and published. One seed's grid is what
#: a reader can follow; the spread is what the claim rests on.
ANCHOR_SEED = ANCHOR_SEEDS[0]


def choose_anchors(task_ids: Sequence[str], count: int, *, seed: int) -> tuple[str, ...]:
    """Which tasks the authority rules on, drawn without regard to difficulty.

    A uniform draw on purpose. An authority that audited the *hardest* items would
    look better and would be assuming the thing in question: knowing which items are
    hard is knowing where the fleet is wrong, which is what the estimate was supposed
    to establish. Uniform is the honest floor, and a targeted policy can only beat it.

    Nested across budgets, which `random.sample` is not. One shuffled order is drawn
    and sliced, so the draw at budget `b` is a subset of the draw at any larger budget
    -- the same property `select` gives the targeted policies by construction, and the
    reason `tests/test_audit_policy.py` could only assert nesting for those. A fresh
    `sample` per budget returns a uniform subset of the right size, so each cell was
    individually honest, but the sweep then moved two things at once: a threshold read
    off it could sit where a *different set of items* was audited rather than where
    more of them were. Slicing one order isolates the budget, and each prefix is still
    a uniform random subset of its size.
    """
    if not count:
        return ()
    order = list(task_ids)
    # `sample` raised on an oversized count; a slice would quietly return fewer and
    # report a budget that was never spent, so the loudness is kept explicitly.
    if count > len(order):
        raise ValueError(f"anchor budget {count} exceeds the {len(order)} available tasks")
    Random(seed).shuffle(order)  # noqa: S311
    return tuple(sorted(order[:count]))


@dataclass(frozen=True, slots=True, kw_only=True)
class ThresholdSpread:
    """What the audited-items price looks like across draws, rather than in one.

    `reached` is the censoring: a draw that never repairs within the sweep has no
    finite threshold, and averaging it in at the sweep's maximum would report the
    experiment's edge as a measurement. So the median is taken over the ordered draws
    with unreached ones sorted last, which yields `None` when more than half of them
    never repair -- the honest answer being "usually not reached", not a number.
    """

    n_wrong: int
    seeds: int
    reached: int
    median: int | None
    lowest: int | None
    highest: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "n_wrong": self.n_wrong,
            "seeds": self.seeds,
            "reached": self.reached,
            "median": self.median,
            "lowest": self.lowest,
            "highest": self.highest,
        }


def summarize_thresholds(n_wrong: int, thresholds: Sequence[int | None]) -> ThresholdSpread:
    """Median and range of a censored threshold, over one draw per seed."""
    if not thresholds:
        raise ValueError("no draws to summarize")
    reached = sorted(t for t in thresholds if t is not None)
    # Unreached sort after every finite value, so the median lands on `None` exactly
    # when at least half the draws failed to repair.
    ordered: list[int | None] = [*reached, *([None] * (len(thresholds) - len(reached)))]
    return ThresholdSpread(
        n_wrong=n_wrong,
        seeds=len(thresholds),
        reached=len(reached),
        median=ordered[(len(ordered) - 1) // 2],
        lowest=reached[0] if reached else None,
        highest=reached[-1] if reached else None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """One fleet composition at one anchor budget."""

    n_wrong: int
    anchors: int
    anchor_share: float
    scored_tasks: int
    agreement: float
    repaired: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "n_wrong": self.n_wrong,
            "anchors": self.anchors,
            "anchor_share": self.anchor_share,
            "scored_tasks": self.scored_tasks,
            "agreement": self.agreement,
            "repaired": self.repaired,
        }


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
    task_ids = sorted(truth)

    print(f"{len(tasks)} tasks, fleet of {args.fleet}, agreement over UNANCHORED tasks only")
    header = "".join(f"{a:>6}" for a in ANCHOR_COUNTS)
    print(f"  {'wrong':>6}{header}")
    print("  " + "-" * (8 + 6 * len(ANCHOR_COUNTS)))

    rows: list[Row] = []
    breakeven: dict[int, int | None] = {}
    spreads: dict[int, ThresholdSpread] = {}

    for n_wrong in COMPOSITIONS:
        if n_wrong > args.fleet:
            continue
        flat = contributions_for(fleet_of(n_wrong, args.fleet), tasks, proposals, seed=SEED)
        partitioned = partition_by_contributor(flat)
        cells: list[str] = []
        per_seed: list[int | None] = []

        for anchor_seed in ANCHOR_SEEDS:
            primary = anchor_seed == ANCHOR_SEED
            first_repaired: int | None = None

            for count in ANCHOR_COUNTS:
                anchored = choose_anchors(task_ids, count, seed=anchor_seed)
                anchors = {t: truth[t] for t in anchored}
                estimate = federated_dawid_skene(partitioned, seed=MASK_SEED, anchors=anchors)

                # The whole methodology: an anchored task's label was supplied, so it is
                # removed from the denominator. What remains is what the anchor bought.
                scored = {t: v for t, v in truth.items() if t not in anchors}
                labels = {t: v for t, v in estimate.labels().items() if t not in anchors}
                agreement = round(agreement_with(labels, scored), 4)
                repaired = agreement >= REPAIRED
                if repaired and first_repaired is None:
                    first_repaired = count

                if not primary:
                    continue
                rows.append(
                    Row(
                        n_wrong=n_wrong,
                        anchors=count,
                        anchor_share=round(count / len(tasks), 4),
                        scored_tasks=len(labels),
                        agreement=agreement,
                        repaired=repaired,
                    )
                )
                cells.append(f"{agreement:>6.3f}")
                record(
                    "authority.cell",
                    agreement,
                    n_wrong=n_wrong,
                    anchors=count,
                    scored=len(labels),
                )

            per_seed.append(first_repaired)
            if primary:
                breakeven[n_wrong] = first_repaired

        spreads[n_wrong] = summarize_thresholds(n_wrong, per_seed)
        print(f"  {n_wrong:>6}{''.join(cells)}")

    print()
    print(f"  {'wrong':>6}  anchors needed to reach {REPAIRED:.2f}, over {len(ANCHOR_SEEDS)} draws")
    print("  " + "-" * 56)
    needed: dict[int, int | None] = {}
    for n_wrong, spread in spreads.items():
        needed[n_wrong] = spread.median
        censored = spread.seeds - spread.reached
        tail = f"  [{spread.lowest}-{spread.highest}]" if spread.reached else ""
        tail += f"  ({censored} of {spread.seeds} never reached)" if censored else ""
        if spread.median is None:
            print(f"  {n_wrong:>6}  not reached within {max(ANCHOR_COUNTS)}{tail}")
        else:
            share = spread.median / len(tasks)
            print(f"  {n_wrong:>6}  median {spread.median} of {len(tasks)} ({share:.1%}){tail}")

    unrepaired = [n for n, c in needed.items() if c is None]
    if unrepaired:
        # Loud, because a composition no affordable authority repairs is the honest
        # limit of this mechanism and is the thing a proposal must not overstate.
        get_logger().warning(
            "authority.not_repaired",
            extra={
                "event": "authority.not_repaired",
                "compositions": unrepaired,
                "max_anchors": max(ANCHOR_COUNTS),
            },
        )

    report = {
        "provenance": run_provenance(seed=SEED),
        "fleet": args.fleet,
        "events": args.events,
        "anchor_seed": ANCHOR_SEED,
        "anchor_seeds": list(ANCHOR_SEEDS),
        "mask_seed": MASK_SEED,
        "repaired_threshold": REPAIRED,
        "anchor_counts": list(ANCHOR_COUNTS),
        "compositions": list(COMPOSITIONS),
        # One seed's agreement grid, the one a reader can follow cell by cell. The
        # claim does not rest on it; `anchors_needed` is the median over every draw.
        "grid": [r.as_dict() for r in rows],
        "grid_anchor_seed": ANCHOR_SEED,
        "anchors_needed": needed,
        "anchors_needed_primary_draw": breakeven,
        "anchors_needed_spread": {str(n): s.as_dict() for n, s in spreads.items()},
        # The corpus is 200 but the estimator only covers the ~97 tasks some
        # contributor reported on, and anchoring shrinks that further -- to 8 at a
        # budget of 180. Scoring validity on len(tasks) reported quotable: true for
        # prices resting on a handful of tasks, which is what this gate exists to stop.
        "validity": check_sample_size(
            min((r.scored_tasks for r in rows), default=0), label="authority anchors"
        ).as_dict(),
        "scored_tasks_min": min((r.scored_tasks for r in rows), default=0),
        "scored_tasks_at_threshold": {
            str(n): next(
                (r.scored_tasks for r in rows if r.n_wrong == n and r.anchors == count),
                None,
            )
            for n, count in needed.items()
            if count is not None
        },
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
