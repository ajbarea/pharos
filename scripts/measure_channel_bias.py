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

**Significance is a permutation null**, matching the gate's idiom rather than assuming a
parametric form: the channel labels are shuffled *within* each evidence stratum, which
preserves the difficulty distribution and destroys only the channel association.

**Two negative controls, because a detector that fires on everything detects nothing.**
An unbiased fleet must produce no detection on any channel. A fleet holding a *threshold*
error --- finding 12's wrong standard, which is not channel-linked --- must also produce
none, or the statistic is reading generic error rather than channel bias.

Needs no model and no network.

    uv run python scripts/measure_channel_bias.py --out results/channel_bias.json
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from random import Random

from measure_audit_policy import observe
from measure_blind_spot import blind_fleet
from measure_secure_reliability import contributions_for

from pharos.analyst import AnalystPolicy, Proposal, evidence_shown
from pharos.disclosure import DROP_COMPARTMENTS, KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
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
SHARES = (0, 1, 2, 3, 4, 5, 7, 9)

#: Permutation trials for the null. Matches the gate's trial count so the two nulls are
#: read the same way.
TRIALS = 200

#: Standard deviations above the permutation null at which a channel is called. Three
#: is the gate's own convention and is deliberately not tuned here -- a threshold picked
#: after seeing the effect is not a threshold.
DETECTION_Z = 3.0

PERMUTATION_SEED = 90210


@dataclass(frozen=True, slots=True)
class Detection:
    """The conditional-independence statistic for one channel, against its null."""

    channel: str
    delta: float
    null_mean: float
    null_sd: float
    z: float
    detected: bool
    strata: int

    def as_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "delta": round(self.delta, 4),
            "null_mean": round(self.null_mean, 4),
            "null_sd": round(self.null_sd, 4),
            "z": round(self.z, 2),
            "detected": self.detected,
            "strata": self.strata,
        }


def verdict_rates(partitioned: dict[str, list[tuple[str, bool]]]) -> dict[str, float]:
    """Each task's share of significant verdicts, as the aggregator already sees it.

    Read from the per-task vote sums of finding 18's protocol. No contributor is
    distinguishable here, which is the point: the statistic must survive the protocol
    that made finding 11's attack impossible.
    """
    view = observe(partitioned)
    return {task: view.votes[task] / view.seen[task] for task in view.seen if view.seen[task] > 0}


def stratified_delta(
    rates: dict[str, float],
    carries: dict[str, bool],
    evidence: dict[str, int],
) -> tuple[float, int]:
    """Mean gap in verdict rate between carrying and non-carrying tasks, within strata.

    Signed so that a *negative* delta means tasks carrying the channel are called
    significant less often than equally-evidenced tasks that do not carry it, which is
    the direction a blind spot produces. Strata with either side empty contribute
    nothing rather than zero: an absent comparison is not a null result.
    """
    gaps: list[float] = []
    used = 0
    for level in sorted(set(evidence.values())):
        with_channel = [rates[t] for t in rates if evidence[t] == level and carries[t]]
        without = [rates[t] for t in rates if evidence[t] == level and not carries[t]]
        if not with_channel or not without:
            continue
        gaps.append(statistics.fmean(with_channel) - statistics.fmean(without))
        used += 1
    return (statistics.fmean(gaps) if gaps else 0.0), used


def detect(
    rates: dict[str, float],
    carries: dict[str, bool],
    evidence: dict[str, int],
    *,
    trials: int,
    seed: int,
) -> Detection | None:
    """The observed stratified gap against a within-stratum permutation null.

    Shuffling channel membership *within* an evidence level preserves how difficulty is
    distributed and destroys only the association being tested, so a channel that merely
    correlates with difficulty cannot score here.
    """
    observed, strata = stratified_delta(rates, carries, evidence)
    if strata == 0:
        return None

    rng = Random(seed)  # noqa: S311
    by_level: dict[int, list[str]] = {}
    for task in rates:
        by_level.setdefault(evidence[task], []).append(task)

    null: list[float] = []
    for _ in range(trials):
        shuffled: dict[str, bool] = {}
        for tasks_at_level in by_level.values():
            flags = [carries[t] for t in tasks_at_level]
            rng.shuffle(flags)
            shuffled.update(dict(zip(tasks_at_level, flags, strict=True)))
        null.append(stratified_delta(rates, shuffled, evidence)[0])

    mean = statistics.fmean(null)
    sd = statistics.pstdev(null)
    # A blind spot depresses the rate on carrying tasks, so the alternative is
    # one-sided. Reporting |z| would let an *elevated* rate read as the same finding.
    z = (mean - observed) / sd if sd else 0.0
    return Detection(
        channel="",
        delta=observed,
        null_mean=mean,
        null_sd=sd,
        z=z,
        detected=z >= DETECTION_Z,
        strata=strata,
    )


def scan(
    tasks: list[TriageTask],
    partitioned: dict[str, list[tuple[str, bool]]],
    *,
    trials: int,
    seed: int,
) -> list[Detection]:
    """Every channel tested against the same fleet, so false positives are visible."""
    rates = verdict_rates(partitioned)
    by_id = {t.task_id: t for t in tasks}
    evidence = {task: len(evidence_shown(by_id[task])) for task in rates}

    found: list[Detection] = []
    for channel in Compartment:
        carries = {
            task: any(channel in r.label.compartments for r in by_id[task].sources)
            for task in rates
        }
        result = detect(rates, carries, evidence, trials=trials, seed=seed)
        if result is not None:
            found.append(
                Detection(
                    channel=channel.value,
                    delta=result.delta,
                    null_mean=result.null_mean,
                    null_sd=result.null_sd,
                    z=result.z,
                    detected=result.detected,
                    strata=result.strata,
                )
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth_blind = Compartment.PARTNER

    print(
        f"{len(tasks)} tasks, fleet of {args.fleet}, {args.trials} permutation trials, "
        f"detection at z >= {DETECTION_Z}"
    )
    print("\n  blind-spot sweep: does the trace survive unanimity?")
    header = "".join(f"{c.value:>10}" for c in Compartment)
    print(f"    {'blind':>6}{header}")
    print("    " + "-" * (6 + 10 * len(Compartment)))

    sweep: list[tuple[int, list[Detection]]] = []
    for n_blind in SHARES:
        if n_blind > args.fleet:
            continue
        flat = contributions_for(blind_fleet(n_blind, args.fleet), tasks, proposals, seed=SEED)
        found = scan(
            tasks,
            partition_by_contributor(flat),
            trials=args.trials,
            seed=PERMUTATION_SEED,
        )
        cells = []
        for channel in Compartment:
            hit = next((d for d in found if d.channel == channel.value), None)
            cells.append(f"{'--' if hit is None else f'{hit.z:.1f}':>10}")
        print(f"    {n_blind:>6}" + "".join(cells))
        sweep.append((n_blind, found))
        for d in found:
            record("channelbias.z", d.z, n_blind=n_blind, channel=d.channel)

    # Control: a wrong standard that is NOT channel-linked. If this fires, the statistic
    # is reading generic error rather than channel bias and the finding is void.
    threshold_flat = contributions_for(
        tuple(
            AnalystPolicy(f"wrong-{i}", escalation_threshold=2, release_policy=DROP_COMPARTMENTS)
            for i in range(args.fleet)
        ),
        tasks,
        proposals,
        seed=SEED,
    )
    control = scan(
        tasks,
        partition_by_contributor(threshold_flat),
        trials=args.trials,
        seed=PERMUTATION_SEED,
    )
    print("\n  control: a fleet-wide THRESHOLD error, which no channel explains")
    print(f"    {'':>6}" + "".join(f"{d.channel:>10}" for d in control))
    print(f"    {'z':>6}" + "".join(f"{d.z:>10.1f}" for d in control))

    unanimous = next((found for share, found in sweep if share == max(SHARES)), None)
    detected_at_unanimity = [d.channel for d in unanimous or [] if d.detected]
    control_hits = [d.channel for d in control if d.detected]
    clean_at_zero = [d.channel for share, found in sweep if share == 0 for d in found if d.detected]

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
        print("  both controls clean: no detection on an unbiased or threshold-wrong fleet")

    report = {
        "provenance": run_provenance(seed=SEED),
        "fleet": args.fleet,
        "events": args.events,
        "trials": args.trials,
        "detection_z": DETECTION_Z,
        "blind_channel": truth_blind.value,
        "shares": list(SHARES),
        "sweep": [
            {"n_blind": share, "detections": [d.as_dict() for d in found]} for share, found in sweep
        ],
        "threshold_control": [d.as_dict() for d in control],
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
