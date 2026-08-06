#!/usr/bin/env python3
"""Whether the governance findings survive the fleet size nobody chose on principle.

`measure_fleet_sensitivity.py` asked this of findings 12, 16 and 17 and found two
computational failures. Findings 19 through 23 -- the authority of record, the audit
policy, the blind spot, and the channel detector that answers it -- were measured at
nine analysts and only nine, which is the same exposure. They are also the findings a
proposal leads with, so they are the ones where an arbitrary constant is most expensive.

Reported as a multiverse rather than a single specification, after Linde et al.
(arXiv:2605.19745, 2026), whose point is that sweeping the defensible choices mostly
surfaces computational failures that otherwise go unreported. That is what happened
here too, before a single cell was swept: all four scripts accepted `--fleet` and none
of them scaled its *compositions* with it. The counts were absolute -- `(5, 6, 7)` of
nine -- so at `--fleet 5` the audit policy measured 5-of-5, unanimity, under the label
of a bare majority, printed the two rows it had skipped as "none" beside it, and then
chose a "best deployable policy" from a field in which every entry was unmeasured. The
blind spot labelled its unanimity row "9 of 5". Those are fixed in
`measure_authority_anchors.ladder`, which expresses each composition as a position in
the fleet and reproduces every committed constant exactly at nine.

Four questions, one per finding:

- **19/20** does `margin` still beat a uniform draw above the crossing, and still tie
  the oracle bound?
- **21** does the audit policy still collapse to chance at unanimity?
- **22** is the blinded channel still detected, and are the negative controls still
  silent?
- **23** does the channel policy still find every corrupted item at unanimity, and does
  it still correct nothing?

Needs no model and no network. Slow rather than heavy: every cell is a fresh EM fit,
and the channel detector is a permutation test on top of that.

    uv run python scripts/measure_governance_sensitivity.py --out results/governance_sensitivity.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

from measure_audit_policy import EVENTS, SEED, observe
from measure_authority_anchors import REPAIRED, majority
from measure_channel_bias import ALPHA
from measure_fleet_sensitivity import run_at
from measure_secure_reliability import contributions_for, fleet_of

from pharos.analyst import Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.inference import partition_by_contributor
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, progress, record

LOG = get_logger()

#: Fleet sizes to sweep. Matches `measure_fleet_sensitivity.FLEETS` so the two
#: multiverses can be read against each other, minus 51: the audit sweep fits a fresh
#: EM per policy per composition per budget, and at 51 analysts that is hours for a
#: cell whose answer 25 already gives. Odd throughout so "majority" needs no tie rule.
FLEETS = (5, 9, 15, 25)

#: Permutations for the channel detector inside the sweep. The same count the committed
#: artifact uses, deliberately: a sensitivity sweep that reduced it would be testing a
#: different instrument than the finding it is checking.
#:
#: The first draft cut it to 1200 to buy runtime, and a 300-permutation smoke run showed
#: why that is not a free parameter. A permutation p-value floors at 1/(m+1), so at 300
#: the smallest attainable p is 0.0033 -- above alpha -- and *nothing can be detected at
#: any effect size*. The sweep dutifully reported the finding as moved. At 1200 the floor
#: is 0.00083, which clears alpha by one permutation and no more. At 4200 it is 0.00024
#: and four extreme draws still detect.
SWEEP_PERMUTATIONS = 4200


def cliff_row(fleet: int) -> dict[str, Any]:
    """Where the estimator actually breaks, scanned one analyst at a time.

    Findings 19 through 23 all describe the failure as "a majority holds the wrong
    standard", and at nine analysts that is what it looked like: the estimator recovers
    the truth at 4 of 9 and collapses at 5 of 9, which is the majority. The phrase was
    then carried into every downstream write-up as though the majority were the
    mechanism.

    It is not. Scanning every composition rather than the five the ladder visits puts
    the crossing at a fixed *share* of the fleet, and a bare majority is above that share
    only at small fleets. Nine is small enough; fifteen is not.

    Calls the same `observe` the audit policy uses rather than shelling out, because the
    quantity here is one EM fit per composition and the surrounding script would sweep
    every budget and policy to reach it.
    """
    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=EVENTS)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    grid = []
    for n_wrong in range(1, fleet + 1):
        flat = contributions_for(fleet_of(n_wrong, fleet), tasks, proposals, seed=SEED)
        view = observe(partition_by_contributor(flat))
        wrong = sum(1 for task, p in view.posterior.items() if (p >= 0.5) != truth[task])
        agreement = 1 - wrong / len(view.posterior)
        grid.append(
            {
                "n_wrong": n_wrong,
                "share": round(n_wrong / fleet, 4),
                "agreement": round(agreement, 4),
                "recovers": agreement >= REPAIRED,
            }
        )

    safe = [row for row in grid if row["recovers"]]
    broken = [row for row in grid if not row["recovers"]]
    crossing = majority(fleet)
    return {
        "fleet": fleet,
        "majority": crossing,
        "majority_share": round(crossing / fleet, 4),
        "grid": grid,
        "highest_safe": safe[-1] if safe else None,
        "lowest_broken": broken[0] if broken else None,
        #: The claim the phrase "a majority holds the wrong standard" makes, tested
        #: rather than assumed. True at nine; the sweep exists to find out where else.
        "cliff_is_at_the_majority": bool(broken) and broken[0]["n_wrong"] == crossing,
        #: Whether the estimator survives a bare majority. Plain consensus cannot, by
        #: definition, so wherever this is true the reliability weighting is buying real
        #: margin rather than the "one extra contributor" finding 12 measured at nine.
        "survives_a_bare_majority": any(
            row["n_wrong"] == crossing and row["recovers"] for row in grid
        ),
        #: The depth of the cliff, as opposed to its location. Reported separately
        #: because the two turn out to behave differently across fleets.
        "post_cliff_agreement": broken[0]["agreement"] if broken else None,
    }


def audit_row(fleet: int) -> dict[str, Any]:
    """Findings 19 and 20: does selection still beat a uniform draw, and tie the bound?"""
    payload = run_at("measure_audit_policy.py", fleet)
    thresholds = payload["thresholds"]
    spread = payload["uniform_spread"]
    crossing = majority(fleet)

    # Whether the fleet was broken at all before a single item was audited. A threshold
    # of None conflates two opposite states -- "no budget in the sweep repaired it" and
    # "there was nothing to repair" -- and the sweep hit the second at fifteen analysts
    # while reading it as the first. Taken from the artifact's own budget-zero row so
    # the two cases are separable downstream.
    unbroken = {
        row["n_wrong"]
        for row in payload["grid"]
        if row["budget"] == 0 and row["policy"] == "oracle" and row["remaining_errors"] == 0
    }

    rows = []
    for composition in payload["compositions"]:
        key = str(composition)
        margin = thresholds["margin"].get(key)
        oracle = thresholds["oracle"].get(key)
        uniform = thresholds["uniform"].get(key)
        rows.append(
            {
                "n_wrong": composition,
                "nothing_to_repair": composition in unbroken,
                "is_crossing": composition == crossing,
                "margin": margin,
                "uniform_median": uniform,
                "oracle": oracle,
                #: Both halves of finding 20's claim, per cell. `beats_uniform` is the
                #: headline and `ties_oracle` is what makes it a bound rather than a
                #: lucky draw. A None threshold means the policy never repaired at any
                #: budget, which loses to any finite one and ties only another None.
                "margin_beats_uniform": _better(margin, uniform),
                "margin_ties_oracle": margin == oracle,
            }
        )
    return {
        "fleet": fleet,
        "majority": crossing,
        "compositions": payload["compositions"],
        "auditable_pool": payload["auditable_pool"],
        "best_deployable": payload["best_deployable"],
        "uniform_spread": spread,
        "rows": rows,
    }


def _better(policy: int | None, baseline: int | None) -> bool:
    """Whether `policy` repairs at a smaller budget than `baseline`.

    Censoring is the whole subtlety: a threshold of None means the sweep ran out of
    budget before the fleet was repaired, so it is worse than every finite value and
    not equal to zero. Comparing the raw values with `<` would make None a TypeError on
    one side and, if it were coerced, the best possible score on the other.
    """
    if policy is None:
        return False
    return baseline is None or policy < baseline


def blind_row(fleet: int) -> dict[str, Any]:
    """Findings 21 and 23: the collapse at unanimity, and what provenance recovers."""
    payload = run_at("measure_blind_spot.py", fleet)
    unanimous = str(payload["fleet"])
    rates = payload["audit_hit_rate"][unanimous]

    at_budget = {
        row["policy"]: row
        for row in payload["grid"]
        if row["n_blind"] == payload["fleet"] and row["budget"] == 20
    }
    deployable = [p for p in payload.get("deployable", ()) if p in rates]
    return {
        "fleet": fleet,
        "shares": payload["shares"],
        "hit_rate_at_unanimity": rates,
        #: Finding 21. Every policy that reads disagreement is at chance once there is
        #: none, and the oracle on the same data is not -- which is what makes it a
        #: property of the signal rather than of the corpus.
        "disagreement_policies_at_chance": all(rates[p] <= 0.25 for p in deployable),
        "oracle_finds_all": rates.get("oracle") == 1.0,
        #: Finding 23, both halves. The channel policy ties the bound, and neither it
        #: nor the oracle corrects an unanchored label at a budget that clears every
        #: corrupted item -- so the residual obstacle is not one of selection.
        "channel_ties_oracle": rates.get("channel") == rates.get("oracle"),
        "channel_corrected": at_budget.get("channel", {}).get("corrected"),
        "oracle_corrected": at_budget.get("oracle", {}).get("corrected"),
        "channel_remaining_errors": at_budget.get("channel", {}).get("remaining_errors"),
    }


def channel_row(fleet: int, permutations: int) -> dict[str, Any]:
    """Finding 22: is the blinded channel still detected, and the controls still silent?"""
    # The detector has to be able to fire before its silence means anything. A
    # permutation p-value cannot go below 1/(m+1), so too few permutations put the
    # entire attainable range above alpha and every cell reads "not detected" at any
    # effect size -- a negative result manufactured by the sweep's own settings. This
    # is a hard stop rather than a warning because the artifact would otherwise look
    # exactly like a finding that had failed to replicate.
    floor = 1.0 / (permutations + 1)
    if floor >= ALPHA:
        raise SystemExit(
            f"{permutations} permutations floor the p-value at {floor:.5f}, at or above "
            f"alpha {ALPHA}; no cell could be detected and the sweep would report finding "
            "22 as moved when nothing had moved but this argument"
        )

    payload = run_at("measure_channel_bias.py", fleet, ("--permutations", str(permutations)))
    blind_channel = payload["blind_channel"]
    unanimous = max(payload["shares"])

    # Cells at unanimity, one per noise level: the regime finding 21 says defeats every
    # disagreement policy, and therefore the one where detection has to survive.
    at_unanimity = [cell for cell in payload["sweep"] if cell["n_blind"] == unanimous]
    blinded = [
        d for cell in at_unanimity for d in cell["detections"] if d["channel"] == blind_channel
    ]
    others = [
        d for cell in at_unanimity for d in cell["detections"] if d["channel"] != blind_channel
    ]

    # The lowest blinded share detected at each noise level. Under zero verdict noise
    # the baseline gap is zero and the statistic is degenerate, so a floor read off the
    # noiseless row is an artifact; the noisy rows are the ones that carry the claim.
    floors: dict[str, int | None] = {}
    for slip in sorted({cell["slip_rate"] for cell in payload["sweep"]}):
        fired = [
            cell["n_blind"]
            for cell in payload["sweep"]
            if cell["slip_rate"] == slip and cell["n_blind"]
            for d in cell["detections"]
            if d["channel"] == blind_channel and d["detected"]
        ]
        floors[str(slip)] = min(fired) if fired else None

    return {
        "fleet": fleet,
        "permutations": permutations,
        "p_floor": round(1.0 / (permutations + 1), 6),
        "shares": payload["shares"],
        "detected_at_unanimity": payload["detected_at_unanimity"],
        "controls_clean": payload["controls_clean"],
        #: The two claims that carry finding 22, and they must be reported together: a
        #: detector that fires on the blinded channel and also on every other one has
        #: no power, and one that fires on neither has none either.
        "blinded_detected_at_every_noise_level": bool(blinded)
        and all(d["detected"] for d in blinded),
        "no_other_channel_detected": not any(d["detected"] for d in others),
        "detection_floor_by_slip_rate": floors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleets", type=int, nargs="+", default=list(FLEETS))
    parser.add_argument("--permutations", type=int, default=SWEEP_PERMUTATIONS)
    parser.add_argument("--skip-channel", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cliff, audit, blind, channel = [], [], [], []
    for fleet in args.fleets:
        progress("governance_sensitivity.fleet", fleet=fleet)
        print(f">>> fleet {fleet}")
        cliff.append(cliff_row(fleet))
        audit.append(audit_row(fleet))
        blind.append(blind_row(fleet))
        if not args.skip_channel:
            channel.append(channel_row(fleet, args.permutations))

    # Cells where a repair was actually needed. A composition the estimator handles
    # unaided has no price to compare, and scoring it would mark a policy down for
    # failing to fix a fleet that was never broken -- which is how the first run of this
    # sweep read the fifteen- and twenty-five-analyst crossings.
    priced = [
        row
        for entry in audit
        for row in entry["rows"]
        if row["n_wrong"] >= entry["majority"] and not row["nothing_to_repair"]
    ]
    selection_beats_uniform = bool(priced) and all(r["margin_beats_uniform"] for r in priced)
    selection_ties_oracle = bool(priced) and all(r["margin_ties_oracle"] for r in priced)
    collapse = all(r["disagreement_policies_at_chance"] for r in blind)
    channel_recovers = all(
        r["channel_ties_oracle"] and r["hit_rate_at_unanimity"].get("channel") == 1.0 for r in blind
    )
    no_repair = all(r["channel_corrected"] == 0 and r["oracle_corrected"] == 0 for r in blind)
    detected = all(
        r["blinded_detected_at_every_noise_level"] and r["no_other_channel_detected"]
        for r in channel
    )

    # The share, not the pool: `auditable_pool` is 97 at every fleet swept, so emitting
    # it would be a metric that cannot move. The breaking share is the quantity this
    # sweep exists to find.
    for row in cliff:
        broke = row["lowest_broken"]
        if broke:
            record("governance_sensitivity.cliff_share", broke["share"], fleet=row["fleet"])

    # The crossing share, bracketed by the fleets swept. The cliff sits above every
    # safe share observed and at or below every broken one, so these two numbers bound
    # it without asserting a value no fleet size here can resolve.
    safe_shares = [row["highest_safe"]["share"] for row in cliff if row["highest_safe"]]
    broken_shares = [row["lowest_broken"]["share"] for row in cliff if row["lowest_broken"]]
    cliff_at_majority = [row["fleet"] for row in cliff if row["cliff_is_at_the_majority"]]
    survives_majority = [row["fleet"] for row in cliff if row["survives_a_bare_majority"]]
    depths = sorted({row["post_cliff_agreement"] for row in cliff if row["post_cliff_agreement"]})

    payload = {
        "fleets": args.fleets,
        "repaired_threshold": REPAIRED,
        "cliff": cliff,
        "cliff_bracket": {
            "highest_safe_share": max(safe_shares) if safe_shares else None,
            "lowest_broken_share": min(broken_shares) if broken_shares else None,
            "fleets_where_the_cliff_is_at_the_majority": cliff_at_majority,
            "fleets_surviving_a_bare_majority": survives_majority,
            "post_cliff_agreement": depths,
        },
        "audit": audit,
        "blind": blind,
        "channel": channel,
        "invariants": {
            "selection_beats_uniform_wherever_a_repair_is_needed": selection_beats_uniform,
            "selection_ties_the_oracle_bound": selection_ties_oracle,
            "disagreement_collapses_at_unanimity": collapse,
            "provenance_recovers_every_corrupted_item": channel_recovers,
            "no_policy_repairs_an_unanchored_label": no_repair,
            "blinded_channel_detected_and_controls_silent": detected if channel else None,
            #: The phrase every downstream write-up uses. Recorded as an invariant so
            #: that when it is false the artifact says so rather than leaving the claim
            #: standing on the one fleet where it happened to hold.
            "the_cliff_is_at_the_majority_at_every_fleet": len(cliff_at_majority) == len(cliff),
            #: The depth of the cliff, which does not move even where its location does.
            "post_cliff_agreement_is_constant": len(depths) == 1,
        },
        "provenance": run_provenance(fleets=args.fleets, permutations=args.permutations),
    }

    print()
    print("  where the estimator breaks, by fleet")
    print(f"    {'fleet':>6}{'majority':>10}{'safe to':>9}{'breaks at':>11}{'as share':>10}")
    print("    " + "-" * 46)
    for row in cliff:
        safe = row["highest_safe"]
        broke = row["lowest_broken"]
        safe_at = str(safe["n_wrong"]) if safe else "-"
        broke_at = str(broke["n_wrong"]) if broke else "-"
        broke_share = f"{broke['share']:.3f}" if broke else "-"
        print(
            f"    {row['fleet']:>6}{row['majority']:>10}{safe_at:>9}{broke_at:>11}{broke_share:>10}"
        )
    print()
    print(f"selection beats uniform where repair is needed: {selection_beats_uniform}")
    print(f"selection ties the oracle bound:             {selection_ties_oracle}")
    print(f"disagreement collapses at unanimity:         {collapse}")
    print(f"provenance recovers every corrupted item:    {channel_recovers}")
    print(f"no policy repairs an unanchored label:       {no_repair}")
    if channel:
        print(f"blinded channel detected, controls silent:   {detected}")

    moved = {name: value for name, value in payload["invariants"].items() if value is False}
    if moved:
        LOG.warning(
            "governance_sensitivity.finding_moved",
            extra={"event": "governance_sensitivity.finding_moved", "moved": sorted(moved)},
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
