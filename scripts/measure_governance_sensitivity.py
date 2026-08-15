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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pharos.analyst import Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import ALPHA, REPAIRED, contributions_for, fleet_of, majority, observe
from pharos.governance.sweep import MeasurementFailedError, run_measurement
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

#: Corpus size. Defined here rather than imported from another measurement: it is an
#: experiment parameter, and importing one script's parameter into another is how a change
#: to one silently moves the other.
EVENTS = 200

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

#: Corpus draws for the crossing scan. Finding 24's first version measured the crossing
#: on one corpus per fleet, concluded that fifteen and twenty-five analysts survive a
#: bare majority, and was wrong: the committed seed is favourable at both sizes and six
#: of eight draws break at fifteen. Only the crossing scan sweeps these -- the audit,
#: blind-spot and channel sub-sweeps each shell out to a script of their own and cost
#: minutes per fleet, so multiplying them by eight would put this script in hours.
DRAWS = (1, 7, 11, 23, 101, 202, 303, 404)


def run_at(script: str, fleet: int, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """One measurement at one fleet size, through the shared sweep runner.

    Refuses nothing: a fleet size is always constructible, so a non-zero exit is a bug
    rather than a point that could not be measured. The corpus sweep makes the opposite
    choice, and both now say which they are making.
    """
    payload = run_measurement(script, ["--fleet", str(fleet), *extra], allow_refusal=False)
    if payload is None:  # pragma: no cover -- allow_refusal is False
        raise MeasurementFailedError(f"{script} refused at --fleet {fleet}")
    return payload


def cliff_scan(fleet: int, seed: int) -> list[dict[str, Any]]:
    """Every composition at one fleet size on one corpus draw, scanned one analyst at a time.

    Calls the same `observe` the audit policy uses rather than shelling out, because the
    quantity here is one EM fit per composition and the surrounding script would sweep
    every budget and policy to reach it.
    """
    tasks = build_triage_tasks(generate(GeneratorConfig(seed=seed, n_events=EVENTS)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    grid = []
    for n_wrong in range(1, fleet + 1):
        flat = contributions_for(fleet_of(n_wrong, fleet), tasks, proposals, seed=seed)
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
    return grid


def cliff_row(fleet: int, draws: Sequence[int]) -> dict[str, Any]:
    """Where the estimator actually breaks, over every corpus draw rather than one.

    Findings 19 through 23 all describe the failure as "a majority holds the wrong
    standard", and at nine analysts that is what it looked like: the estimator recovers
    the truth at 4 of 9 and collapses at 5 of 9, which is the majority. The phrase was
    then carried into every downstream write-up as though the majority were the
    mechanism. Scanning every composition rather than the five the ladder visits shows
    it is a *share*, and at nine that share and the majority coincide.

    The draw sweep is the second correction, and it retracts the first one's headline.
    Measured on a single corpus this scan reported the crossing at 0.600 for fifteen
    analysts and 0.560 for twenty-five -- both above a bare majority -- and finding 24
    concluded that a larger fleet survives a bare majority. It does not, reliably: over
    eight draws the crossing at fifteen sits at 0.533 in six of them, and at twenty-five
    it ranges 0.520 to 0.600. The committed seed was favourable at both sizes.

    So the location of the crossing is itself a distribution and is reported as one. A
    single-draw crossing is exactly the quantity this repository has now mis-reported
    three times, and the fix is the same each time: sweep the draw and quote the range.
    """
    crossing = majority(fleet)
    per_draw: list[dict[str, Any]] = []
    for seed in draws:
        grid = cliff_scan(fleet, seed)
        safe = [row for row in grid if row["recovers"]]
        broken = [row for row in grid if not row["recovers"]]
        per_draw.append(
            {
                "draw": seed,
                "grid": grid,
                "highest_safe": safe[-1] if safe else None,
                "lowest_broken": broken[0] if broken else None,
                "cliff_is_at_the_majority": bool(broken) and broken[0]["n_wrong"] == crossing,
                "survives_a_bare_majority": any(
                    row["n_wrong"] == crossing and row["recovers"] for row in grid
                ),
                "post_cliff_agreement": broken[0]["agreement"] if broken else None,
            }
        )

    broken_shares = sorted(d["lowest_broken"]["share"] for d in per_draw if d["lowest_broken"])
    survives = [d["draw"] for d in per_draw if d["survives_a_bare_majority"]]
    depths = sorted({d["post_cliff_agreement"] for d in per_draw if d["post_cliff_agreement"]})
    return {
        "fleet": fleet,
        "majority": crossing,
        "majority_share": round(crossing / fleet, 4),
        "draw_seeds": list(draws),
        "per_draw": per_draw,
        #: The crossing as a distribution over draws, which is the only honest form.
        #: A single value here was finding 24's error.
        "breaking_share_range": [broken_shares[0], broken_shares[-1]] if broken_shares else None,
        "breaking_share_median": broken_shares[len(broken_shares) // 2] if broken_shares else None,
        #: How often the crossing lands exactly on the majority, rather than whether it
        #: did on one corpus.
        "draws_where_the_cliff_is_at_the_majority": sum(
            1 for d in per_draw if d["cliff_is_at_the_majority"]
        ),
        #: The claim finding 24 made and this retracts. At fifteen and twenty-five it is
        #: a minority and a coin-flip of the draws respectively, not a property.
        "draws_surviving_a_bare_majority": len(survives),
        "draws": len(per_draw),
        #: The depth of the cliff, as opposed to its location. Reported separately
        #: because the two behave differently: the location moves with the draw and the
        #: depth does not.
        "post_cliff_agreement": depths,
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
    parser.add_argument("--draws", type=int, nargs="+", default=list(DRAWS))
    parser.add_argument("--permutations", type=int, default=SWEEP_PERMUTATIONS)
    parser.add_argument("--skip-channel", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cliff, audit, blind, channel = [], [], [], []
    for fleet in args.fleets:
        progress("governance_sensitivity.fleet", fleet=fleet)
        print(f">>> fleet {fleet}")
        cliff.append(cliff_row(fleet, args.draws))
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
    # sweep exists to find, and it is a median over draws rather than a value.
    for row in cliff:
        if row["breaking_share_median"] is not None:
            record(
                "governance_sensitivity.cliff_share",
                row["breaking_share_median"],
                fleet=row["fleet"],
            )

    # The crossing share over every fleet and every draw. Reported as a range because it
    # is one: 0.52 to 0.60 across the sweep, and not monotone in fleet size. The earlier
    # bracket -- one safe share below one broken share -- was an artifact of reading a
    # single corpus per fleet, where each fleet contributes exactly one number and any
    # two numbers bracket something.
    observed = sorted(
        d["lowest_broken"]["share"] for row in cliff for d in row["per_draw"] if d["lowest_broken"]
    )
    cliff_at_majority = [
        row["fleet"]
        for row in cliff
        if row["draws_where_the_cliff_is_at_the_majority"] == row["draws"]
    ]
    survives_majority = [row["fleet"] for row in cliff if row["draws_surviving_a_bare_majority"]]
    always_survives = [
        row["fleet"] for row in cliff if row["draws_surviving_a_bare_majority"] == row["draws"]
    ]
    depths = sorted({depth for row in cliff for depth in row["post_cliff_agreement"]})

    payload = {
        "fleets": args.fleets,
        "draws": args.draws,
        "repaired_threshold": REPAIRED,
        "cliff": cliff,
        "cliff_bracket": {
            #: The crossing over every fleet and every draw. A range, because a single
            #: draw per fleet is what turned this into a bracket it never supported.
            "breaking_share_range": [observed[0], observed[-1]] if observed else None,
            "breaking_share_median": observed[len(observed) // 2] if observed else None,
            "fleets_where_the_cliff_is_at_the_majority": cliff_at_majority,
            #: Fleets where *some* draw survives a bare majority, and where *every* draw
            #: does. Finding 24 quoted the first and read it as the second.
            "fleets_surviving_a_bare_majority_in_some_draw": survives_majority,
            "fleets_surviving_a_bare_majority_in_every_draw": always_survives,
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
            #: Finding 24's retracted headline, kept as an invariant so the retraction
            #: is checked rather than remembered. False: at fifteen and twenty-five a
            #: bare majority survives in some draws and not others.
            "a_larger_fleet_reliably_survives_a_bare_majority": always_survives
            == survives_majority,
            #: The depth of the cliff, which does not move even where its location does.
            "post_cliff_agreement_is_constant": len(depths) == 1,
        },
        "provenance": run_provenance(fleets=args.fleets, permutations=args.permutations),
    }

    print()
    print("  where the estimator breaks, by fleet, over every draw")
    print(
        f"    {'fleet':>6}{'majority':>10}{'share median':>14}{'share range':>18}{'maj safe':>10}"
    )
    print("    " + "-" * 58)
    for row in cliff:
        span = row["breaking_share_range"]
        rng = f"{span[0]:.3f} - {span[1]:.3f}" if span else "-"
        med = f"{row['breaking_share_median']:.3f}" if row["breaking_share_median"] else "-"
        safe = f"{row['draws_surviving_a_bare_majority']}/{row['draws']}"
        print(f"    {row['fleet']:>6}{row['majority']:>10}{med:>14}{rng:>18}{safe:>10}")
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
