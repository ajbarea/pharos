#!/usr/bin/env python3
"""Whether the fleet findings survive the fleet size nobody chose on principle.

Findings 12, 16 and 17 were all measured at nine analysts. Nothing selected nine: it
was a module constant in two of the three scripts, so the parameter those findings turn
on could not be varied without editing source. This sweeps it and commits the sweep,
because a conclusion that moves with an arbitrary choice is a property of the choice.

Reported as a multiverse rather than a single specification, after Linde et al.
(arXiv:2605.19745, 2026). Their result is that sweeping the defensible choices mostly
surfaces *computational failures that otherwise go unreported*, which is what happened
twice here: Dawid-Skene's agreement with GLAD turns out to hold only at small fleets,
and the correlated-fleet understatement turns out to grow without bound.

Three questions, one per finding:

- **12** does the consensus cliff stay at the majority crossing, and at the same value?
- **16** does independence understate the wrong-majority probability by more or less as
  the fleet grows?
- **17** does GLAD's ability inversion survive, and do the three estimators still agree?

Needs no model and no network. Slow rather than heavy: every cell is a fresh EM fit.

    uv run python scripts/measure_fleet_sensitivity.py --out results/fleet_sensitivity.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pharos.provenance import run_provenance
from pharos.telemetry import get_logger, progress, record

ROOT = Path(__file__).resolve().parent.parent
LOG = get_logger()

#: Fleet sizes to sweep. Five is the smallest fleet with a meaningful majority, nine is
#: what every committed artifact used, and fifty-one is past the point where a cross-silo
#: deployment would stop adding analysts. Odd throughout so "majority" needs no tie rule.
FLEETS = (5, 9, 15, 25, 51)


def _run(script: str, fleet: int, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """One measurement at one fleet size, read back from a temporary artifact.

    Shelling out rather than importing: each script owns its own defaults, argument
    parsing and validity checks, and re-implementing that here would let this sweep and
    the committed artifacts drift apart. The temporary file is the point -- a
    sensitivity sweep must not overwrite `results/`.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = Path(fh.name)
    try:
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(ROOT / "scripts" / script),
                "--fleet",
                str(fleet),
                "--out",
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        if completed.returncode != 0 or not out.stat().st_size:
            raise SystemExit(f"{script} failed at --fleet {fleet}:\n{completed.stderr[-800:]}")
        return json.loads(out.read_text(encoding="utf-8"))
    finally:
        out.unlink(missing_ok=True)


def consensus_row(fleet: int) -> dict[str, Any]:
    """Finding 12: where the cliff lands, and what it lands on."""
    payload = _run("measure_consensus_reliability.py", fleet)
    cliff = payload["cliff_at"]
    at = next(r for r in payload["grid"] if r["n_wrong"] == cliff)
    before = next(r for r in payload["grid"] if r["n_wrong"] == cliff - 1)
    return {
        "fleet": fleet,
        "majority": fleet // 2 + 1,
        "cliff_at": cliff,
        "cliff_is_at_majority": cliff == fleet // 2 + 1,
        "consensus_before": before["consensus"],
        "consensus_at": at["consensus"],
        "dawid_skene_at": at["dawid_skene"],
        #: Whether reliability weighting bought anything at the crossing. One extra
        #: contributor of margin, not an escape -- see the finding.
        "dawid_skene_survives_crossing": at["dawid_skene"] > at["consensus"],
    }


def difficulty_row(fleet: int) -> dict[str, Any]:
    """Finding 17: whether the estimators agree, and whether ability still inverts."""
    payload = _run("measure_difficulty_confound.py", fleet)
    majority = fleet // 2 + 1
    at = next(r for r in payload["rows"] if r["n_wrong"] == majority)
    return {
        "fleet": fleet,
        "majority": majority,
        "dawid_skene": at["dawid_skene_agreement"],
        "glad": at["glad_agreement"],
        "cc_rasch": at["cc_rasch_agreement"],
        "cc_rasch_converged": at["cc_rasch_converged"],
        "ability_is_inverted": at["ability_is_inverted"],
        #: The claim finding 17 actually makes. GLAD failing at every size is what
        #: carries it; the three estimators agreeing was an artifact of fleet nine.
        "glad_recovers_truth": at["glad_agreement"] >= 0.999,
        "all_three_agree": (
            abs(at["dawid_skene_agreement"] - at["glad_agreement"]) < 1e-3
            and abs(at["glad_agreement"] - at["cc_rasch_agreement"]) < 1e-3
        ),
    }


def correlated_row(fleet: int, draws: int) -> dict[str, Any]:
    """Finding 16: how far independence understates the wrong-majority probability."""
    payload = _run("measure_correlated_fleets.py", fleet, ("--draws", str(draws)))
    cells = {(c["rate"], c["structure"]): c for c in payload["cells"]}
    out: dict[str, Any] = {"fleet": fleet, "by_rate": []}
    for rate in sorted({r for r, _ in cells}):
        independent = cells[(rate, "independent")]["wrong_majority_rate"]
        culture = cells[(rate, "one culture")]["wrong_majority_rate"]
        out["by_rate"].append(
            {
                "rate": rate,
                "independent": independent,
                "one_culture": culture,
                #: None rather than infinity when independence rounds to zero: the
                #: ratio is genuinely unbounded there and a sentinel float would be
                #: quoted as a number by something downstream.
                "understatement": round(culture / independent, 1) if independent else None,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleets", type=int, nargs="+", default=list(FLEETS))
    parser.add_argument("--draws", type=int, default=60)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    consensus, difficulty, correlated = [], [], []
    for fleet in args.fleets:
        progress("fleet_sensitivity.fleet", fleet=fleet)
        print(f">>> fleet {fleet}")
        consensus.append(consensus_row(fleet))
        difficulty.append(difficulty_row(fleet))
        correlated.append(correlated_row(fleet, args.draws))

    cliff_invariant = all(r["cliff_is_at_majority"] for r in consensus)
    cliff_values = {r["consensus_at"] for r in consensus}
    inversion_invariant = all(r["ability_is_inverted"] for r in difficulty)
    glad_never_recovers = not any(r["glad_recovers_truth"] for r in difficulty)
    agreement_sizes = [r["fleet"] for r in difficulty if r["all_three_agree"]]

    for row in consensus:
        record("fleet_sensitivity.cliff", float(row["cliff_at"]), fleet=row["fleet"])

    payload = {
        "fleets": args.fleets,
        "consensus": consensus,
        "difficulty": difficulty,
        "correlated": correlated,
        "invariants": {
            "cliff_is_always_at_majority": cliff_invariant,
            "cliff_value_is_constant": len(cliff_values) == 1,
            "cliff_value": sorted(cliff_values),
            "glad_ability_always_inverts": inversion_invariant,
            "glad_never_recovers_truth": glad_never_recovers,
            "fleets_where_all_three_estimators_agree": agreement_sizes,
        },
        "provenance": run_provenance(fleets=args.fleets, draws=args.draws),
    }

    print()
    print(f"cliff at the majority crossing at every fleet: {cliff_invariant}")
    print(f"cliff value constant across fleets:            {sorted(cliff_values)}")
    print(f"GLAD ability inverts at every fleet:           {inversion_invariant}")
    print(f"GLAD never recovers the truth:                 {glad_never_recovers}")
    print(f"all three estimators agree only at fleets:     {agreement_sizes}")

    if not cliff_invariant or not inversion_invariant:
        LOG.warning(
            "fleet_sensitivity.finding_moved",
            extra={
                "event": "fleet_sensitivity.finding_moved",
                "cliff_invariant": cliff_invariant,
                "inversion_invariant": inversion_invariant,
            },
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
