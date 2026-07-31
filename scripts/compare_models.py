#!/usr/bin/env python3
"""Summarise the triage sweep across models, against the surface baseline.

A per-model F1 means nothing on its own here. Ground truth is content-defined, so a
probe reading no words at all already scores well above chance, and the corpus has a
majority class. Both floors are printed alongside every score, because a model that
looks respectable against 0.5 may be doing nothing at all against the real ones.

    python scripts/compare_models.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    rows = []
    for path in sorted(RESULTS.glob("triage_lift-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance", {})
        rows.append(
            {
                "key": provenance.get("model_key") or path.stem.removeprefix("triage_lift-"),
                "tag": payload.get("model", "?"),
                "n": payload.get("n_tasks", 0),
                "acc": payload.get("accuracy", 0.0),
                "majority": payload.get("majority_accuracy", 0.0),
                "baseline": payload.get("surface_baseline", 0.0),
                "precision": payload.get("precision", 0.0),
                "recall": payload.get("recall", 0.0),
                "f1": payload.get("f1", 0.0),
                "unparsed": payload.get("unparsed", 0),
            }
        )

    if not rows:
        print(f"no sweep artifacts in {RESULTS}. Run scripts/sweep_models.sh first.")
        return 1

    header = (
        f"{'MODEL':<14} {'N':>3} {'ACC':>6} {'MAJ':>6} {'P':>6} {'R':>6} {'F1':>6} {'UNPARSED':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: -r["f1"]):
        print(
            f"{row['key']:<14} {row['n']:>3} {row['acc']:>6.3f} {row['majority']:>6.3f} "
            f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f} "
            f"{row['unparsed']:>9}"
        )

    baselines = {row["baseline"] for row in rows}
    print()
    print(
        f"surface baseline (shape alone, reading nothing): {', '.join(f'{b:.4f}' for b in baselines)}"
    )
    print("Report every score against the majority and the surface baseline, never against 0.5.")
    print()
    beat = [r["key"] for r in rows if r["acc"] > r["majority"]]
    print(
        f"models beating the majority-class floor on accuracy: {', '.join(beat) if beat else 'NONE'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
