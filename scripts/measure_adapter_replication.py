#!/usr/bin/env python3
"""How much does an adapter move when you train it again?

Finding 10 rests on comparing gaps: a teacher who never slips is inherited at a median
of 0.0008, one who does is improved upon at 0.034. Neither number means anything without
knowing how far a *rerun of the same configuration* moves, and nothing here measured
that -- every adapter was trained once.

Four of them were trained twice, by accident. The four named teachers are exact
parameter twins of four grid points: `any-one` is `t1s0`, `two-of-three` is `t2s0`,
`by-the-book` is `t3s0`, `inattentive` is `t3s0.15`. Once the named four were re-run on
the corrected corpus they became replicates of grid points on the same corpus, with the
same targets, the same hyperparameters, and the same held-out split -- differing only in
being a different job on a different day.

Two kinds of pair come out of that, and conflating them would understate the precision
of the better one:

- **Identical targets.** A teacher with slip rate 0 is a deterministic function of the
  corpus, so both runs train on exactly the same labels. What is left is training
  nondeterminism alone: GPU kernel scheduling, cuDNN algorithm selection, and the
  order of a shuffled loader.
- **Redrawn targets.** A slipping teacher draws its mistakes stochastically, so the two
  runs see different labels for the same events. That pair measures the whole procedure
  and is not a replicate of training.

Four pairs is a floor rather than an estimate, and this reports it as one: it says how
far two runs *were* observed to move, not how far they could.

    uv run python scripts/measure_adapter_replication.py --out results/adapter_replication.json
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from pharos.provenance import run_provenance
from pharos.telemetry import get_logger, record

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOG = get_logger()

#: Named teacher to the grid point with identical parameters. Verified rather than
#: assumed: `_load_pair` refuses a pair whose threshold, slip rate, hyperparameters or
#: data configuration differ, because a mislabelled twin would report a replication
#: bound that is really a difference between two experiments.
TWINS = (
    ("any-one", "t1s0"),
    ("two-of-three", "t2s0"),
    ("by-the-book", "t3s0"),
    ("inattentive", "t3s0.15"),
)

#: Fields of `data` that must agree for two runs to be the same experiment. `eval_seed`
#: and `cross_corpus` matter most: a cross-corpus run trains on more tuples and
#: evaluates on a different corpus, so pairing one with a same-corpus run would compare
#: two designs and call the difference noise.
DATA_KEYS = ("n_train", "n_eval", "eval_seed", "cross_corpus")


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / f"review_adapter-{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pair(named: str, grid: str) -> dict[str, Any] | None:
    """One replicate pair, or None when either half is missing or they do not match."""
    a, b = _load(named), _load(grid)
    if a is None or b is None:
        return None

    mismatches = []
    if a["hyperparameters"] != b["hyperparameters"]:
        mismatches.append("hyperparameters")
    if {k: a["data"][k] for k in DATA_KEYS} != {k: b["data"][k] for k in DATA_KEYS}:
        mismatches.append("data")
    mismatches.extend(
        f"teacher.{field}"
        for field in ("escalation_threshold", "slip_rate")
        if a["teacher"][field] != b["teacher"][field]
    )
    if mismatches:
        LOG.warning(
            "replication.not_a_twin",
            extra={"event": "replication.not_a_twin", "pair": f"{named}/{grid}", "on": mismatches},
        )
        return None

    #: Deterministic teachers hand both runs the same labels; slipping ones do not, and
    #: the difference decides which of the two questions this pair answers.
    targets_identical = (
        a["teacher"]["train_target_agreement"] == b["teacher"]["train_target_agreement"]
    )
    return {
        "named": named,
        "grid": grid,
        "slip_rate": a["teacher"]["slip_rate"],
        "targets_identical": targets_identical,
        "delta_vs_world": abs(a["adapter"]["accuracy"] - b["adapter"]["accuracy"]),
        "delta_vs_teacher": abs(
            a["adapter_vs_teacher"]["accuracy"] - b["adapter_vs_teacher"]["accuracy"]
        ),
        "target_agreement_delta": abs(
            a["teacher"]["train_target_agreement"] - b["teacher"]["train_target_agreement"]
        ),
    }


def _bound(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The largest observed movement across a set of pairs, on either scoring."""
    if not pairs:
        return {"n_pairs": 0}
    deltas = [d for p in pairs for d in (p["delta_vs_world"], p["delta_vs_teacher"])]
    return {
        "n_pairs": len(pairs),
        "max": round(max(deltas), 4),
        "median": round(statistics.median(deltas), 4),
        "exactly_identical": sum(1 for p in pairs if p["delta_vs_world"] == 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pairs = [p for p in (_pair(n, g) for n, g in TWINS) if p is not None]
    if not pairs:
        raise SystemExit(
            "no replicate pairs. Both the named teachers and the grid must be present on\n"
            "  the same corpus: `sbatch cluster/review-adapter.sbatch` and\n"
            "  `sbatch --array=0-23%4 cluster/review-adapter.sbatch`, then sync them back."
        )

    same = [p for p in pairs if p["targets_identical"]]
    redrawn = [p for p in pairs if not p["targets_identical"]]
    training_only = _bound(same)
    whole_procedure = _bound(redrawn)

    for p in pairs:
        record("replication.delta", p["delta_vs_world"], teacher=p["named"])

    payload = {
        "pairs": pairs,
        "summary": {
            #: The number finding 10's gaps have to be read against. A gap smaller than
            #: this is indistinguishable from a rerun.
            "training_only": training_only,
            "whole_procedure_including_target_draw": whole_procedure,
        },
        "provenance": run_provenance(n_pairs=len(pairs)),
    }

    print(f"{len(pairs)} replicate pairs")
    for p in pairs:
        kind = "same targets" if p["targets_identical"] else "targets redrawn"
        print(
            f"  {p['named']:14s} vs {p['grid']:9s} {kind:16s} "
            f"world {p['delta_vs_world']:.4f}  teacher {p['delta_vs_teacher']:.4f}"
        )
    if training_only["n_pairs"]:
        print(
            f"\n  training nondeterminism alone: max {training_only['max']:.4f} over "
            f"{training_only['n_pairs']} pairs, {training_only['exactly_identical']} of them "
            "reproducing exactly"
        )
    if whole_procedure["n_pairs"]:
        print(
            f"  with the teacher's targets redrawn: max {whole_procedure['max']:.4f} over "
            f"{whole_procedure['n_pairs']} pair(s)"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
