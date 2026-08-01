#!/usr/bin/env python3
"""Summarise the triage sweep across models, against the surface baseline.

A per-model F1 means nothing on its own here. Ground truth is content-defined, so a
probe reading no words at all already scores well above chance, and the corpus has a
majority class. Both floors are printed alongside every score, because a model that
looks respectable against 0.5 may be doing nothing at all against the real ones.

**Every accuracy now carries a cluster-bootstrap interval**, computed from the
per-task rows already in the committed artifacts. No model is called; this is a
re-reading of measurements that exist. Two claims in the paper depend on it and were
previously asserted without one:

- *No model clears the majority floor.* An interval says whether that survives 40
  tasks, or whether the sweep merely failed to detect a model that does.
- *We claim no ordering between models.* That was stated as a judgement about noise.
  Overlapping intervals make it a measurement.

These artifacts are single-pass, so the interval is between-task only. Finding 9
measured this decode at zero self-disagreement in thirty tasks, so the missing
within-task term is small here -- unlike the reasoning conditions, where it is not.

    python scripts/compare_models.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pharos.uncertainty import Interval, Trial, cluster_bootstrap, resolves

RESULTS = Path(__file__).resolve().parents[1] / "results"


@dataclass(frozen=True, slots=True)
class ModelRow:
    """One model's sweep result, with the interval computed from its own task rows."""

    key: str
    tag: str
    n: int
    acc: float
    majority: float
    baseline: float
    precision: float
    recall: float
    f1: float
    unparsed: int
    interval: Interval | None

    @property
    def clears_floor(self) -> bool:
        return self.acc > self.majority

    @property
    def interval_reaches_floor(self) -> bool:
        """Point below the floor, interval above it: this sample does not resolve it."""
        return self.interval is not None and self.interval.high > self.majority >= self.acc


def main() -> int:
    rows: list[ModelRow] = []
    for path in sorted(RESULTS.glob("triage_lift-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance", {})
        # One trial per task, correct when the verdict matched the world. Rows with
        # a null verdict carry None so an unparsable answer is dropped rather than
        # scored as wrong.
        trials = [
            Trial(
                r["task_id"],
                None if r.get("verdict") is None else r["verdict"] == r["truth"],
            )
            for r in payload.get("rows", [])
        ]
        rows.append(
            ModelRow(
                key=provenance.get("model_key") or path.stem.removeprefix("triage_lift-"),
                tag=payload.get("model", "?"),
                n=payload.get("n_tasks", 0),
                acc=payload.get("accuracy", 0.0),
                majority=payload.get("majority_accuracy", 0.0),
                baseline=payload.get("surface_baseline", 0.0),
                precision=payload.get("precision", 0.0),
                recall=payload.get("recall", 0.0),
                f1=payload.get("f1", 0.0),
                unparsed=payload.get("unparsed", 0),
                interval=cluster_bootstrap(trials) if trials else None,
            )
        )

    if not rows:
        print(f"no sweep artifacts in {RESULTS}. Run scripts/sweep_models.sh first.")
        return 1

    header = (
        f"{'MODEL':<14} {'N':>3} {'ACC':>6} {'95% INTERVAL':>16} {'MAJ':>6} "
        f"{'P':>6} {'R':>6} {'F1':>6} {'UNPARSED':>9}"
    )
    print(header)
    print("-" * len(header))
    ordered = sorted(rows, key=lambda r: -r.f1)
    for row in ordered:
        span = f"[{row.interval.low:.3f}, {row.interval.high:.3f}]" if row.interval else "n/a"
        print(
            f"{row.key:<14} {row.n:>3} {row.acc:>6.3f} {span:>16} "
            f"{row.majority:>6.3f} {row.precision:>6.3f} {row.recall:>6.3f} "
            f"{row.f1:>6.3f} {row.unparsed:>9}"
        )

    baselines = {row.baseline for row in rows}
    print()
    print(
        f"surface baseline (shape alone, reading nothing): {', '.join(f'{b:.4f}' for b in baselines)}"
    )
    print("Report every score against the majority and the surface baseline, never against 0.5.")
    print()
    beat = [r.key for r in rows if r.clears_floor]
    print(
        f"models beating the majority-class floor on accuracy: {', '.join(beat) if beat else 'NONE'}"
    )

    # Whether the floor claim survives the interval. A point estimate below the
    # floor with an interval reaching above it has not shown the model fails to
    # clear it, only that this sample did not show that it does.
    reaching = [r.key for r in rows if r.interval_reaches_floor]
    if reaching:
        print(
            f"\nmodels whose interval REACHES the floor: {', '.join(reaching)}\n"
            "  The point estimates sit below it; at this sample size the data does not\n"
            "  exclude their clearing it. State the claim as unresolved for these."
        )
    else:
        print("\nno model's interval reaches the majority floor: the claim is resolved.")

    # And whether any ordering between models survives. The paper declines to rank
    # them; this is that decision, measured rather than asserted.
    separated = [
        (a.key, b.key)
        for i, a in enumerate(ordered)
        for b in ordered[i + 1 :]
        if a.interval is not None and b.interval is not None and resolves(a.interval, b.interval)
    ]
    print()
    comparisons = len(ordered) * (len(ordered) - 1) // 2
    if separated:
        pairs = ", ".join(f"{a} > {b}" for a, b in separated)
        expected = comparisons * 0.05
        print(f"model pairs the intervals DO separate: {pairs}")
        print(
            f"  Read with care. This is {comparisons} pairwise comparisons at 95%, which\n"
            f"  produces about {expected:.1f} separations by chance alone. {len(separated)} observed is\n"
            "  inside that, so the conservative reading -- no ranking claimed -- stands."
            if len(separated) <= expected + 1
            else f"  {comparisons} comparisons at 95% would yield about {expected:.1f} by chance; "
            f"{len(separated)} is more than that."
        )
    else:
        print(
            "no pair of models is separated by its intervals. The table is sorted for\n"
            "readability and supports no ranking whatever."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
