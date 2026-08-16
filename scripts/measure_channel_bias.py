#!/usr/bin/env python3
"""Whether a shared blind spot leaves any trace once it has stopped leaving disagreement.

Finding 21 ended on a specific and uncomfortable note. A fleet that unanimously shares
a channel blind spot is corrupted on a slice of the corpus, every deployable audit
policy falls to chance on it, and an oracle on the same data still finds every affected
item --- so the tasks are identifiable in principle and invisible in practice. What a
deployment cannot do, on that evidence, is tell which regime it is in. Every signal used
up to that point was built from *disagreement*, and unanimity is precisely the absence
of disagreement.

That framing contains its own way out, and this script tests it. Disagreement is not the
only observable. The fleet's verdict *rate* is one too, and it can be conditioned on
public structure rather than on who said what.

**The statistic is a conditional independence test.** For an unbiased fleet, whether a
task happens to carry a particular channel should tell you nothing about the verdict
once you already know how much evidence the task shows. Writing $V$ for the fleet's
significant-rate, $C$ for whether a task carries the channel, and $E$ for the count of
defining facts visible:

    V  independent of  C  |  E

A shared channel blind spot breaks exactly that: at a fixed evidence level, tasks whose
evidence arrives through the discounted channel are called routine more often than tasks
with the same evidence arriving elsewhere. The effect survives unanimity because it is a
property of the *level* of the fleet's verdict, not of the spread.

Conditioning on $E$ is what makes this a test rather than a correlation. Channels are
not distributed evenly across difficulty in this corpus --- SENSOR sits on nearly every
task carrying any evidence at all --- so an unconditioned comparison would report a
channel effect for any channel, which is the confound finding 21 had to retract a claim
over.

**What the detector may read**, held to finding 20's deployability rule: the per-task
verdict rate the aggregator already holds, and the public structure of the corpus. Not a
per-analyst stream, which secure aggregation does not produce, and not ground truth,
which would make the question circular.

**Significance is a one-sided permutation p-value**, matching the gate's idiom rather
than assuming a parametric form: the channel labels are shuffled *within* each evidence
stratum, which preserves the difficulty distribution and destroys only the channel
association. The p-value is `(b + 1) / (m + 1)` after Phipson and Smyth (2010), so it is
bounded below by `1 / (m + 1)` and never reaches zero. `PERMUTATIONS` therefore decides
what `ALPHA` can mean: a budget too small to resolve the threshold makes every channel
read as undetected regardless of the effect.

**Two negative controls, because a detector that fires on everything detects nothing.**
An unbiased fleet must produce no detection on any channel. A fleet holding a *threshold*
error --- finding 12's wrong standard, which is not channel-linked --- must also produce
none, or the statistic is reading generic error rather than channel bias.

Both controls are run at every noise level, and the reason is that neither can fail at
zero noise. A fleet of identical deterministic analysts fixes each task's verdict rate by
its evidence stratum, so every permutation returns the observed gap, `b = m`, and the
p-value is exactly 1.0 by construction rather than by evidence. Reported as clean for two
days on that basis, and described here as the load-bearing check that could have voided
the finding. Only the noisy levels test anything.

Needs no model and no network.

    uv run python scripts/measure_channel_bias.py --out results/channel_bias.json
"""

import argparse
import json
from pathlib import Path

from pharos.analyst import AnalystPolicy, Proposal, evidence_shown
from pharos.disclosure import DROP_COMPARTMENTS, KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import blind_fleet, contributions_for, ladder
from pharos.governance.channel import (
    ALPHA,
    PERMUTATION_SEED,
    PERMUTATIONS,
    Detection,
    compartment_carriage,
    scan_channels,
    verdict_rates,
)
from pharos.inference import partition_by_contributor
from pharos.labels import Compartment, declassify
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200
FLEET = 9

#: Blind-spot shares to sweep. Nine is the case finding 21 leaves open.
#: Blind shares swept. The low end is not decoration: the claim this finding is
#: actually useful for is that a house style is catchable *before* it becomes the
#: majority, which ties it to finding 16. That claim was published while the sweep
#: started at 5 of 9 -- already the majority -- so it rested on nothing. A share the
#: sweep never runs cannot support a sentence about that share.
CHANNEL_RUNGS = (
    "none",
    "one",
    "two",
    "one-third",
    "below",
    "majority",
    "seven-ninths",
    "unanimous",
)
SHARES = ladder(FLEET, CHANNEL_RUNGS)

#: Verdict noise for the sweep. Zero is the idealized fleet this finding was first
#: measured on and is kept as the reference column, but it is a degenerate case rather
#: than a clean one: with no noise every analyst is identical and deterministic, each
#: task's verdict rate is fixed by its evidence stratum, and both the observed
#: within-stratum gap and every permutation of it are exactly zero. The 0.15 is this
#: repo's own `inattentive` rate from `pharos.analyst`, so it is the fleet the rest of
#: the project already treats as realistic rather than a number chosen here.
NOISE_LEVELS = (0.0, 0.05, 0.15)


def scan(
    tasks: list[TriageTask],
    partitioned: dict[str, list[tuple[str, bool]]],
    *,
    permutations: int,
    seed: int,
) -> list[Detection]:
    """Every channel tested against the same fleet, so false positives are visible."""
    rates = verdict_rates(partitioned)
    by_id = {t.task_id: t for t in tasks}
    evidence = {task: len(evidence_shown(by_id[task])) for task in rates}
    return scan_channels(
        rates,
        compartment_carriage(tasks, rates),
        evidence,
        permutations=permutations,
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    shares = ladder(args.fleet, CHANNEL_RUNGS)

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=args.seed, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth_blind = Compartment.PARTNER

    print(
        f"{len(tasks)} tasks, fleet of {args.fleet}, {args.permutations} permutations, "
        f"detection at p <= {ALPHA} (floor {1 / (args.permutations + 1):.1e})"
    )

    def cell(hit: Detection | None) -> str:
        # No degenerate case to print around any more: a null with no spread yields
        # p = 1.0, which is a real and correct answer rather than a division to dodge.
        return f"{'--':>10}" if hit is None else f"{hit.p_value:>10.4f}"

    sweep: list[tuple[float, int, list[Detection]]] = []
    controls: list[tuple[float, list[Detection]]] = []
    header = "".join(f"{c.value:>10}" for c in Compartment)

    for slip in NOISE_LEVELS:
        print(f"\n  blind-spot sweep at slip rate {slip:.2f}")
        print(f"    {'blind':>6}{header}")
        print("    " + "-" * (6 + 10 * len(Compartment)))
        for n_blind in shares:
            if n_blind > args.fleet:
                continue
            flat = contributions_for(
                blind_fleet(n_blind, args.fleet, slip_rate=slip), tasks, proposals, seed=args.seed
            )
            found = scan(
                tasks,
                partition_by_contributor(flat),
                permutations=args.permutations,
                seed=PERMUTATION_SEED,
            )
            print(
                f"    {n_blind:>6}"
                + "".join(
                    cell(next((d for d in found if d.channel == c.value), None))
                    for c in Compartment
                )
            )
            sweep.append((slip, n_blind, found))
            for d in found:
                record(
                    "channelbias.p_value",
                    d.p_value,
                    n_blind=n_blind,
                    channel=d.channel,
                    slip_rate=slip,
                )

        # Control: a wrong standard that is NOT channel-linked. If this fires, the
        # statistic is reading generic error rather than channel bias and the finding is
        # void. It is run at every noise level because at slip 0 it *cannot* fire: the
        # fleet is deterministic, every permutation of it is identical, and the control
        # was returning "clean" from a null with no variance rather than from evidence.
        threshold_flat = contributions_for(
            tuple(
                AnalystPolicy(
                    f"wrong-{i}",
                    escalation_threshold=2,
                    release_policy=DROP_COMPARTMENTS,
                    slip_rate=slip,
                )
                for i in range(args.fleet)
            ),
            tasks,
            proposals,
            seed=args.seed,
        )
        control = scan(
            tasks,
            partition_by_contributor(threshold_flat),
            permutations=args.permutations,
            seed=PERMUTATION_SEED,
        )
        print(f"    {'thr-ctl':>6}" + "".join(cell(d) for d in control))
        controls.append((slip, control))

    # The headline regime stays the noiseless one so the finding is comparable with how
    # it was first reported, but its controls are known-degenerate there, so the honest
    # controls come from every level.
    unanimous = next(
        (found for slip, share, found in sweep if slip == 0.0 and share == max(shares)), None
    )
    detected_at_unanimity = [d.channel for d in unanimous or [] if d.detected]
    control_hits = [
        f"slip={slip}:{d.channel}" for slip, found in controls for d in found if d.detected
    ]
    clean_at_zero = [
        f"slip={slip}:{d.channel}"
        for slip, share, found in sweep
        if share == 0
        for d in found
        if d.detected
    ]
    print()
    if truth_blind.value in detected_at_unanimity:
        print(f"  DETECTED at unanimity: {truth_blind.value}, where disagreement is zero")
    else:
        print("  NOT detected at unanimity -- the trace does not survive")
    if control_hits or clean_at_zero:
        # Loud: either control firing means the statistic does not mean what the finding
        # would claim it means.
        get_logger().warning(
            "channelbias.control_fired",
            extra={
                "event": "channelbias.control_fired",
                "threshold_control": control_hits,
                "unbiased_control": clean_at_zero,
            },
        )
        print(f"  CONTROL FIRED: threshold={control_hits} unbiased={clean_at_zero}")
    else:
        print(f"  controls clean over {len(controls) * len(Compartment)} cells")

    report = {
        "provenance": run_provenance(seed=args.seed),
        "fleet": args.fleet,
        "events": args.events,
        "permutations": args.permutations,
        "alpha": ALPHA,
        "blind_channel": truth_blind.value,
        "shares": list(shares),
        "noise_levels": list(NOISE_LEVELS),
        "permutation_seed": PERMUTATION_SEED,
        "sweep": [
            {"slip_rate": slip, "n_blind": share, "detections": [d.as_dict() for d in found]}
            for slip, share, found in sweep
        ],
        "threshold_control": [
            {"slip_rate": slip, "detections": [d.as_dict() for d in found]}
            for slip, found in controls
        ],
        "detected_at_unanimity": detected_at_unanimity,
        "controls_clean": not (control_hits or clean_at_zero),
        "validity": check_sample_size(len(tasks), label="channel bias").as_dict(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
