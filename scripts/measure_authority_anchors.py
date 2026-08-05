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

#: Seed for choosing which tasks the authority rules on. Distinct from the corpus seed
#: so the anchor draw cannot correlate with corpus structure by accident.
ANCHOR_SEED = 909


def choose_anchors(task_ids: Sequence[str], count: int, *, seed: int) -> tuple[str, ...]:
    """Which tasks the authority rules on, drawn without regard to difficulty.

    A uniform draw on purpose. An authority that audited the *hardest* items would
    look better and would be assuming the thing in question: knowing which items are
    hard is knowing where the fleet is wrong, which is what the estimate was supposed
    to establish. Uniform is the honest floor, and a targeted policy can only beat it.
    """
    rng = Random(seed)  # noqa: S311
    return tuple(sorted(rng.sample(list(task_ids), count))) if count else ()


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

    for n_wrong in COMPOSITIONS:
        if n_wrong > args.fleet:
            continue
        flat = contributions_for(fleet_of(n_wrong, args.fleet), tasks, proposals, seed=SEED)
        partitioned = partition_by_contributor(flat)
        cells: list[str] = []
        first_repaired: int | None = None

        for count in ANCHOR_COUNTS:
            anchored = choose_anchors(task_ids, count, seed=ANCHOR_SEED)
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

        breakeven[n_wrong] = first_repaired
        print(f"  {n_wrong:>6}{''.join(cells)}")

    print()
    print(f"  {'wrong':>6}  anchors needed to reach {REPAIRED:.2f}")
    print("  " + "-" * 40)
    for n_wrong, count in breakeven.items():
        if count is None:
            print(f"  {n_wrong:>6}  not reached within {max(ANCHOR_COUNTS)}")
        else:
            print(
                f"  {n_wrong:>6}  {count} of {len(tasks)} ({count / len(tasks):.1%} of the round)"
            )

    unrepaired = [n for n, c in breakeven.items() if c is None]
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
        "mask_seed": MASK_SEED,
        "repaired_threshold": REPAIRED,
        "anchor_counts": list(ANCHOR_COUNTS),
        "compositions": list(COMPOSITIONS),
        "grid": [r.as_dict() for r in rows],
        "anchors_needed": breakeven,
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
            for n, count in breakeven.items()
            if count is not None
        },
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
