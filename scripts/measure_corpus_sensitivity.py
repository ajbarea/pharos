#!/usr/bin/env python3
"""Whether findings 20 to 23 survive the corpus draw nobody chose on principle.

`measure_governance_sensitivity.py` swept the fleet size and found the compositions were
hard-coded. Finding 24 then swept the corpus draw on the crossing scan alone and found
that its own headline was a single-draw artifact: the committed seed is favourable at
fleets of fifteen and twenty-five, and the crossing that looked like a fixed share is a
distribution spanning 0.520 to 0.600.

That leaves the obvious exposure. Findings 20 through 23 are measured by four scripts
that each hard-coded `SEED = 7` and did not accept a seed at all, so the corpus was not a
sweepable dimension of this project's headline governance results. This sweeps it.

Three things the first run found before it measured a policy, all of the same kind as the
fleet defect and all invisible from the committed artifact:

- The **auditable pool is not a constant.** `measure_audit_policy` documents "only 97 of
  200 tasks" as though it were a property of the design; over eight draws it ranges 83 to
  99. Its budget ladder topped out at a hard-coded 95, so the script exited non-zero on
  four of eight draws rather than reporting anything. The ladder is truncated to the pool
  now, and both forms are published.
- **Finding 21's experiment is not constructible on every corpus.** It needs a blind
  channel orthogonal to item difficulty, and refuses to run when the two are entangled.
  At some draws they are. That refusal is correct and is recorded here as data rather
  than treated as a failure -- a draw where the negative control cannot be built is a
  draw that says nothing about the finding, which is different from one that contradicts
  it.
- Every rate below therefore carries its own denominator: draws where the experiment
  could be built, not draws attempted.

Needs no model and no network. Slow: a fresh EM fit per policy per composition per
budget, per draw, plus a permutation test.

    uv run python scripts/measure_corpus_sensitivity.py --out results/corpus_sensitivity.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from measure_channel_bias import ALPHA

from pharos.provenance import run_provenance
from pharos.telemetry import get_logger, progress, record

LOG = get_logger()

#: The draws swept, matching `measure_governance_sensitivity.DRAWS` so the fleet
#: multiverse and the corpus multiverse can be read against each other. 7 is the
#: committed seed and is included rather than held out: the point is to see where it
#: sits in the distribution, and a sweep that excluded it could not say.
DRAWS = (1, 7, 11, 23, 101, 202, 303, 404)

#: Fleet size. Nine only, deliberately. This sweep varies the corpus and holds the fleet
#: at the size every published governance number was measured at; the fleet dimension is
#: `measure_governance_sensitivity.py`'s. Crossing the two would be 32 cells of several
#: minutes each and would confound which dimension moved a result.
FLEET = 9

#: Permutations for the channel detector. The committed count, for the reason
#: `measure_governance_sensitivity` gives: a p-value floors at 1/(m+1), and a sweep that
#: reduced it would test a weaker instrument than the finding it is checking.
PERMUTATIONS = 4200


def run_at(script: str, seed: int, extra: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """One measurement script at one corpus draw, or None if it refused to run.

    A refusal is not a failure here. `measure_blind_spot.py` exits non-zero when the
    channel it needs is entangled with item difficulty, which is a precondition of
    finding 21's design rather than a bug, and a draw that cannot host the experiment
    must be excluded from the denominator rather than counted against the finding.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(Path(__file__).parent / script),
                "--seed",
                str(seed),
                "--out",
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not out.exists():
            LOG.warning(
                "corpus_sensitivity.draw_refused",
                extra={
                    "event": "corpus_sensitivity.draw_refused",
                    "script": script,
                    "seed": seed,
                    "reason": result.stderr.strip().splitlines()[-1][:200]
                    if result.stderr.strip()
                    else "no stderr",
                },
            )
            return None
        return json.loads(out.read_text(encoding="utf-8"))


def _better(policy: int | None, baseline: int | None) -> bool:
    """Whether `policy` repairs at a smaller budget. None means never repaired."""
    if policy is None:
        return False
    return baseline is None or policy < baseline


def audit_row(seed: int) -> dict[str, Any] | None:
    """Findings 19 and 20 at one draw: the price, the bound, and the pool."""
    payload = run_at("measure_audit_policy.py", seed)
    if payload is None:
        return None
    thresholds = payload["thresholds"]
    cells = []
    for composition in payload["compositions"]:
        key = str(composition)
        margin = thresholds["margin"].get(key)
        oracle = thresholds["oracle"].get(key)
        uniform = thresholds["uniform"].get(key)
        cells.append(
            {
                "n_wrong": composition,
                "margin": margin,
                "uniform_median": uniform,
                "oracle": oracle,
                "margin_beats_uniform": _better(margin, uniform),
                "margin_ties_oracle": margin == oracle,
                #: Neither repaired at any budget the ladder reached. Separated from a
                #: loss, because "no policy helps here" and "this policy lost" are
                #: different results and only the second is about selection.
                "nothing_repaired": margin is None and uniform is None,
            }
        )
    return {
        "seed": seed,
        "auditable_pool": payload["auditable_pool"],
        "budgets": payload["budgets"],
        "budgets_requested": payload.get("budgets_requested", payload["budgets"]),
        "ladder_truncated": len(payload["budgets"])
        < len(payload.get("budgets_requested", payload["budgets"])),
        "best_deployable": payload["best_deployable"],
        "cells": cells,
    }


def blind_row(seed: int) -> dict[str, Any] | None:
    """Findings 21 and 23 at one draw, or None where the channel is not constructible."""
    payload = run_at("measure_blind_spot.py", seed)
    if payload is None:
        return None
    unanimous = str(payload["fleet"])
    rates = payload["audit_hit_rate"][unanimous]
    deployable = [p for p in payload.get("deployable", ()) if p in rates]
    at_budget = {
        row["policy"]: row
        for row in payload["grid"]
        if row["n_blind"] == payload["fleet"] and row["budget"] == 20
    }
    return {
        "seed": seed,
        "disagreement_policies_at_chance": all(rates[p] <= 0.25 for p in deployable),
        "oracle_finds_all": rates.get("oracle") == 1.0,
        "channel_ties_oracle": rates.get("channel") == rates.get("oracle"),
        "channel_hit_rate": rates.get("channel"),
        "channel_corrected": at_budget.get("channel", {}).get("corrected"),
        "oracle_corrected": at_budget.get("oracle", {}).get("corrected"),
    }


def channel_row(seed: int) -> dict[str, Any] | None:
    """Finding 22 at one draw: is the blinded channel detected, controls still silent?"""
    payload = run_at("measure_channel_bias.py", seed, ("--permutations", str(PERMUTATIONS)))
    if payload is None:
        return None
    blind_channel = payload["blind_channel"]
    unanimous = max(payload["shares"])
    at_unanimity = [c for c in payload["sweep"] if c["n_blind"] == unanimous]
    blinded = [d for c in at_unanimity for d in c["detections"] if d["channel"] == blind_channel]
    others = [d for c in at_unanimity for d in c["detections"] if d["channel"] != blind_channel]
    return {
        "seed": seed,
        "blinded_detected_at_every_noise_level": bool(blinded)
        and all(d["detected"] for d in blinded),
        "no_other_channel_detected": not any(d["detected"] for d in others),
        "controls_clean": payload["controls_clean"],
    }


def rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """A count with its denominator, never a bare fraction."""
    return {"held": sum(1 for r in rows if r[key]), "of": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, nargs="+", default=list(DRAWS))
    parser.add_argument("--skip-channel", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    floor = 1.0 / (PERMUTATIONS + 1)
    if floor >= ALPHA:
        raise SystemExit(
            f"{PERMUTATIONS} permutations floor the p-value at {floor:.5f}, at or above "
            f"alpha {ALPHA}; finding 22 would read as moved when nothing had moved"
        )

    audit, blind, channel = [], [], []
    for seed in args.draws:
        progress("corpus_sensitivity.draw", seed=seed)
        print(f">>> draw {seed}")
        if (row := audit_row(seed)) is not None:
            audit.append(row)
        if (row := blind_row(seed)) is not None:
            blind.append(row)
        if not args.skip_channel and (row := channel_row(seed)) is not None:
            channel.append(row)

    # Cells where a repair was needed at all. A composition nothing repairs prices no
    # policy, and scoring it would mark selection down for failing a fleet no budget
    # in the ladder reaches.
    priced = [cell for row in audit for cell in row["cells"] if not cell["nothing_repaired"]]
    ties = [c for c in priced if c["margin_ties_oracle"]]
    beats = [c for c in priced if c["margin_beats_uniform"]]

    pools = sorted(row["auditable_pool"] for row in audit)
    for row in audit:
        record("corpus_sensitivity.auditable_pool", row["auditable_pool"], seed=row["seed"])

    by_composition = {}
    for cell in priced:
        entry = by_composition.setdefault(
            cell["n_wrong"], {"n_wrong": cell["n_wrong"], "draws": 0, "beats": 0, "ties": 0}
        )
        entry["draws"] += 1
        entry["beats"] += bool(cell["margin_beats_uniform"])
        entry["ties"] += bool(cell["margin_ties_oracle"])

    payload = {
        "fleet": FLEET,
        "draws": args.draws,
        "permutations": PERMUTATIONS,
        "audit": audit,
        "blind": blind,
        "channel": channel,
        "auditable_pool_range": [pools[0], pools[-1]] if pools else None,
        "by_composition": sorted(by_composition.values(), key=lambda e: e["n_wrong"]),
        "draws_attempted": len(args.draws),
        "draws_hosting_the_blind_spot": len(blind),
        "invariants": {
            #: Finding 20's deep claim, and the one that makes it a bound rather than a
            #: lucky draw. If selection ties the oracle everywhere, no better selection
            #: rule exists on this signal.
            "selection_ties_the_oracle_bound_in_every_draw": len(ties) == len(priced),
            #: Finding 20's headline. Weaker than the bound, and the one that moves.
            "selection_beats_uniform_in_every_draw": len(beats) == len(priced),
            #: Findings 21 and 23, on the draws where the experiment is constructible.
            "disagreement_collapses_at_unanimity_wherever_measurable": bool(blind)
            and all(r["disagreement_policies_at_chance"] for r in blind),
            #: Finding 23, split into the two claims it was making at once. Selecting on
            #: the channel the detector named ties the best any selection rule could do,
            #: which is the result; that the bound happens to sit at 1.00 is a property
            #: of the committed corpus, and at other draws the oracle itself finds only
            #: three quarters. Reporting them together read the second as the first.
            "provenance_ties_the_oracle_bound_in_every_draw": bool(blind)
            and all(r["channel_ties_oracle"] for r in blind),
            "provenance_finds_every_corrupted_item_in_every_draw": bool(blind)
            and all(r["channel_hit_rate"] == 1.0 for r in blind),
            "no_policy_repairs_an_unanchored_label": bool(blind)
            and all(r["channel_corrected"] == 0 and r["oracle_corrected"] == 0 for r in blind),
            #: Finding 22.
            "blinded_channel_detected_and_controls_silent": (
                bool(channel)
                and all(
                    r["blinded_detected_at_every_noise_level"] and r["no_other_channel_detected"]
                    for r in channel
                )
            )
            if channel
            else None,
            #: The pool the audit budget is a fraction of. Documented as "97 of 200" in
            #: the script that measures it; carried here as an invariant so the next
            #: reader sees a range instead.
            "the_auditable_pool_is_a_constant": bool(pools) and pools[0] == pools[-1],
        },
        "provenance": run_provenance(draws=args.draws, fleet=FLEET, permutations=PERMUTATIONS),
    }

    print()
    print(f"  auditable pool over {len(audit)} draws: {pools[0]} to {pools[-1]}" if pools else "")
    print(f"  blind-spot experiment constructible in {len(blind)}/{len(args.draws)} draws")
    print()
    print(f"    {'wrong':>6}{'draws':>7}{'beats uniform':>15}{'ties oracle':>13}")
    print("    " + "-" * 41)
    for entry in payload["by_composition"]:
        print(
            f"    {entry['n_wrong']:>6}{entry['draws']:>7}"
            f"{entry['beats']:>10}/{entry['draws']:<4}{entry['ties']:>8}/{entry['draws']:<4}"
        )
    print()
    for name, value in payload["invariants"].items():
        print(f"{name:<62} {value}")

    moved = {name: value for name, value in payload["invariants"].items() if value is False}
    if moved:
        LOG.warning(
            "corpus_sensitivity.finding_moved",
            extra={"event": "corpus_sensitivity.finding_moved", "moved": sorted(moved)},
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
