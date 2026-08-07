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
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random

from measure_audit_policy import observe
from measure_authority_anchors import ladder
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

#: Significance level for a detection. The gate's own convention elsewhere in this
#: repo is three sigma, whose one-sided normal tail is 0.00135, so 0.001 is the nearest
#: round threshold at least as strict. It is deliberately not tuned here: a threshold
#: picked after seeing the effect is not a threshold. `PERMUTATIONS` has to be large
#: enough to resolve it, since a permutation p-value cannot go below 1 / (m + 1).
ALPHA = 0.001

#: Permutations in the null. One pooled null rather than several small ones: this drew
#: 21 nulls of 200 and reported the median z, which spends the same budget to estimate
#: the same quantity less precisely. The floor on an achievable p-value is 1 / (m + 1),
#: so this resolves down to 2.4e-4 and can therefore actually decide ALPHA.
PERMUTATIONS = 4200

PERMUTATION_SEED = 90210

#: Verdict noise for the sweep. Zero is the idealized fleet this finding was first
#: measured on and is kept as the reference column, but it is a degenerate case rather
#: than a clean one: with no noise every analyst is identical and deterministic, each
#: task's verdict rate is fixed by its evidence stratum, and both the observed
#: within-stratum gap and every permutation of it are exactly zero. The 0.15 is this
#: repo's own `inattentive` rate from `pharos.analyst`, so it is the fleet the rest of
#: the project already treats as realistic rather than a number chosen here.
NOISE_LEVELS = (0.0, 0.05, 0.15)


@dataclass(frozen=True, slots=True)
class Detection:
    """The conditional-independence statistic for one channel, against its null."""

    channel: str
    delta: float
    null_mean: float
    p_value: float
    extreme: int
    permutations: int
    detected: bool
    strata: int

    def as_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            # The effect, and the thing to read for *extent*: it is linear in the blind
            # share. A p-value cannot report extent, because it saturates at its own
            # floor once the effect is comfortably significant.
            "delta": round(self.delta, 4),
            "null_mean": round(self.null_mean, 4),
            # Not rounded to a fixed number of places: these span several orders of
            # magnitude and 2.4e-4 is the floor, so a fixed rounding would flatten the
            # strong results into one another.
            "p_value": float(f"{self.p_value:.3g}"),
            "extreme": self.extreme,
            "permutations": self.permutations,
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
    permutations: int,
    seed: int,
) -> Detection | None:
    """The observed stratified gap against a within-stratum permutation null.

    Shuffling channel membership *within* an evidence level preserves how difficulty is
    distributed and destroys only the association being tested, so a channel that merely
    correlates with difficulty cannot score here. The statistic itself -- the mean gap
    between carrying and non-carrying tasks within a stratum, averaged across strata --
    is the standard one for this design.

    **Significance is a permutation p-value, not a z-score.** This reported
    `z = (null_mean - observed) / null_sd` until 2026-08-06, which was wrong in three
    ways at once and wrong in the same place each time. A permutation test exists
    precisely so the null's shape need not be assumed; standardizing against its mean
    and standard deviation puts the normality assumption back. It is undefined when the
    null has no spread, which is the case both of this finding's negative controls sit
    in, so the controls "passed" from a division this code special-cased rather than
    from evidence. And it invited a fix in kind: the previous attempt drew the null 21
    times and reported the median z, which spends 4200 permutations to estimate a
    quantity one pooled null of 4200 estimates better.

        p = (b + 1) / (m + 1)

    with `b` the number of permutations at least as extreme as the observed gap. The
    `+1` on both sides is Phipson and Smyth (2010): the permuted draws generate an exact
    discrete null distribution rather than an estimate of a tail probability, so the
    observed value is one of its own draws. The naive `b / m` understates by about `1/m`
    and can report zero, which is a claim no finite number of permutations supports. The
    floor here is `1 / (m + 1)`.

    The degenerate case then needs no handling at all. If every permutation returns the
    observed gap -- a noiseless fleet, where each task's rate is fixed by its evidence
    stratum and there is nothing left to shuffle -- then every draw is at least as
    extreme, `b = m`, and `p = 1.0`. No detection, correctly, and by construction rather
    than by a special case.

    One-sided: a blind spot *depresses* the rate on carrying tasks, so a gap at least as
    extreme is one at least as negative. Reporting a two-sided result would let an
    elevated rate read as the same finding.
    """
    observed, strata = stratified_delta(rates, carries, evidence)
    if strata == 0:
        return None

    by_level: dict[int, list[str]] = {}
    for task in rates:
        by_level.setdefault(evidence[task], []).append(task)

    rng = Random(seed)  # noqa: S311
    null: list[float] = []
    for _ in range(permutations):
        shuffled: dict[str, bool] = {}
        for tasks_at_level in by_level.values():
            flags = [carries[t] for t in tasks_at_level]
            rng.shuffle(flags)
            shuffled.update(dict(zip(tasks_at_level, flags, strict=True)))
        null.append(stratified_delta(rates, shuffled, evidence)[0])

    at_least_as_extreme = sum(1 for gap in null if gap <= observed)
    p_value = (at_least_as_extreme + 1) / (permutations + 1)
    return Detection(
        channel="",
        delta=observed,
        null_mean=statistics.fmean(null),
        p_value=p_value,
        extreme=at_least_as_extreme,
        permutations=permutations,
        detected=p_value <= ALPHA,
        strata=strata,
    )


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

    found: list[Detection] = []
    for channel in Compartment:
        carries = {
            task: any(channel in r.label.compartments for r in by_id[task].sources)
            for task in rates
        }
        result = detect(rates, carries, evidence, permutations=permutations, seed=seed)
        if result is not None:
            # `replace` rather than re-listing the fields: this rebuild used to name
            # every field by hand, so adding one to Detection meant silently dropping it
            # here unless the author remembered both sites.
            found.append(replace(result, channel=channel.value))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    shares = ladder(args.fleet, CHANNEL_RUNGS)

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
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
                blind_fleet(n_blind, args.fleet, slip_rate=slip), tasks, proposals, seed=SEED
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
            seed=SEED,
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
        "provenance": run_provenance(seed=SEED),
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
