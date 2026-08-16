#!/usr/bin/env python3
"""What survives when the shared blind spot follows no partition anybody can name.

This repository's stated open problem, quoted from its own README: finding 22 detects a
blind spot aligned with a **known, public partition** of the corpus, and "a shared error
that follows no observable partition would leave no trace in it either, and nothing here
says how to find one."

That sentence contains two claims and this script separates them. Losing the partition
must cost the *channel scan* everything, because a scan enumerates a small public family
and there is nothing left to enumerate. Whether it costs the **index of dispersion**
anything is a different question, because that statistic never reads a partition at all --
it reads per-task vote sums within an evidence stratum. And whether it costs the
**remedy** anything is a third, because finding 28's working answer withholds by the
channel the detector named, and an unnamed channel cannot be withheld by.

**The construction, matched to finding 21's.** `AnalystPolicy.distrusted_reports` is the
same wrong standard as `blind_compartment` with its handle removed: the blind analysts
decline to credit a set of individual reports rather than a channel. The set is drawn so
the verdicts it corrupts match what PARTNER corrupts on the committed corpus -- 20 of 200,
all on tasks showing all three defining facts -- so the two fleets differ in exactly one
respect, which is whether a scan can name the slice. The draw is uniform within that
stratum and a balance precondition refuses a draw a compartment scan could plausibly name,
with the refusals counted in the artifact.

**Predictions, before the run.**

1. The channel scan is silent on the latent fleet at every share, and fires on the
   channel-keyed fleet at the same shares. Anything else means the two constructions are
   not matched and nothing below is comparable.
2. The index of dispersion is **unchanged** between the two, because it never read the
   partition. If this fails, detection really does depend on a nameable channel and the
   open problem stands as written.
3. Localization fails. `deviation` ranks tasks by their within-stratum residual -- the
   per-task summands of the index -- and the expectation is that it does no better than an
   untargeted draw at unanimity, for finding 21's reason: at unanimity the fleet's
   corrupted tasks look exactly like a fleet applying a different rule, and nothing
   distinguishes them from tasks that are merely unusual.
4. If (3) is wrong and the residual does localize, it does so only while the corrupted
   slice is a minority of its stratum, and the slice sweep will show where it turns.

If (2) holds and (3) holds, the honest answer is that the open problem is half solved: a
deployment can know it has a shared error it cannot find. If (2) and (3) both fail in the
other direction, the problem is closed and the README claim is withdrawn.

**What every policy here may read** is the `ServerObservation` and nothing else: per-task
vote sums, contributor counts, the estimator's posterior, and public evidence counts. The
`carries` map is deliberately empty on the latent fleet, because there is no channel to
supply, which is what makes `channel` unavailable rather than merely bad.

Needs no model and no network.

    uv run python scripts/measure_latent_blindspot.py --out results/latent_blindspot.json
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from pharos.analyst import Proposal, evidence_shown
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import (
    BLIND,
    REFUSED_EXIT,
    REPORT_BUDGET,
    UNIFORM_SEEDS,
    ChannelUnusableError,
    LatentSlice,
    ServerObservation,
    assert_channel_usable,
    blind_fleet,
    compartment_carriage,
    contributions_for,
    dispersion,
    draw_balanced_slice,
    eligible_pool,
    ladder,
    latent_blind_fleet,
    observe,
    observed,
    rates_from,
    scan_channels,
    score,
    select,
)
from pharos.governance.channel import ALPHA, PERMUTATION_SEED, PERMUTATIONS
from pharos.governance.shape import NULL_DRAWS
from pharos.inference import partition_by_contributor
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, progress, record
from pharos.validity import check_sample_size

LOG = get_logger()

SEED = 7
EVENTS = 200
FLEET = 9

#: Shares of the fleet carrying the blind spot. The same rungs finding 22's scan sweeps,
#: so the two detectors are compared on the same fleets rather than on similar ones.
SHARE_RUNGS = ("none", "one", "one-third", "majority", "seven-ninths", "unanimous")

#: Independent slip rates. Zero is the degenerate reference -- a fleet of identical
#: deterministic analysts has no within-stratum variance at all -- and 0.15 is this repo's
#: own `inattentive` rate, the fleet the rest of the project treats as realistic.
SLIP_RATES = (0.0, 0.15)

#: Corrupted-slice sizes for the sweep that prices prediction 4. The eligible pool on the
#: committed corpus is 69 tasks, so this crosses its majority, which is the crossing the
#: two-sided rule is predicted to have and the one-sided rule is not.
SLICE_SIZES = (10, 20, 30, 40, 50, 60)

#: Slice draws the headline grid is repeated over. One draw is one draw: this project has
#: retracted five numbers that were properties of a single sample, and a slice is a sample.
SLICE_SEEDS = (7, 1_000_003, 2_000_003, 3_000_003, 4_000_003)

#: What the grid scores. `channel` is deliberately **absent**, and the absence is the
#: result rather than an omission: finding 28's remedy withholds by the channel a detector
#: named, and on the latent construction there is no channel to name. Scoring it anyway
#: would hand `policy_channel` an empty carriage map, which by its own contract degrades to
#: a flat score -- an arbitrary draw wearing the remedy's name, and a column a reader would
#: reasonably mistake for the remedy having been tried. The finding says it cannot be.
POLICIES_HERE = ("uniform", "margin", "posterior", "consensus", "deviation", "shortfall")
BOUND = "oracle"


@dataclass(frozen=True, slots=True, kw_only=True)
class Cell:
    """One fleet under one construction: what the two detectors said, and what each rule bought."""

    keying: str
    n_blind: int
    slip_rate: float
    errors: int
    converged: bool
    channels_fired: tuple[str, ...]
    best_channel_p: float | None
    dispersion_index: float | None
    dispersion_p: float | None
    precision: dict[str, float]
    risk: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "keying": self.keying,
            "n_blind": self.n_blind,
            "slip_rate": self.slip_rate,
            "errors": self.errors,
            "converged": self.converged,
            "channels_fired": list(self.channels_fired),
            "best_channel_p": self.best_channel_p,
            "dispersion_index": self.dispersion_index,
            "dispersion_p": self.dispersion_p,
            "precision": self.precision,
            "risk": self.risk,
        }


def measure(
    tasks: list[TriageTask],
    proposals: dict[str, Proposal],
    truth: dict[str, bool],
    *,
    keying: str,
    n_blind: int,
    slip_rate: float,
    fleet: int,
    seed: int,
    slice_: LatentSlice | None,
    permutations: int,
    null_draws: int,
) -> Cell:
    """One cell: build the fleet, run both detectors over it, and score every policy.

    `slice_` is `None` for the channel-keyed construction and a drawn slice for the latent
    one. The two paths differ only in which fleet is built and in whether `carries` is
    populated, which is the whole comparison and is why they share this function rather
    than getting one each.
    """
    policies = (
        latent_blind_fleet(n_blind, fleet, slice_, slip_rate=slip_rate)
        if slice_ is not None
        else blind_fleet(n_blind, fleet, slip_rate=slip_rate)
    )
    flat = contributions_for(policies, tasks, proposals, seed=seed)
    partitioned = partition_by_contributor(flat)
    # One fit, three readers. This ran `federated_dawid_skene` three times per cell on the
    # same input at the same seed -- inside `observe`, again for the labels, and again
    # inside `verdict_rates` -- recomputing a deterministic result it already held. Worth
    # about 15s of a 2m58s run measured end to end, which is smaller than it looks like it
    # should be: the permutation nulls dominate, not the fits. Kept for the duplication
    # rather than the seconds. The numbers are unchanged by construction: same seed, same
    # input, same fit.
    view, estimate = observed(partitioned)
    by_id = {t.task_id: t for t in tasks}
    evidence = {task: len(evidence_shown(by_id[task])) for task in view.posterior}
    view = replace(view, evidence=evidence)

    labels = estimate.labels()
    wrong = frozenset(task for task, called in labels.items() if called != truth[task])

    detections = scan_channels(
        rates_from(view),
        compartment_carriage(tasks, view.posterior),
        evidence,
        permutations=permutations,
        seed=PERMUTATION_SEED,
    )
    fired = tuple(sorted(d.channel for d in detections if d.detected))
    best_p = min((d.p_value for d in detections), default=None)
    shape = dispersion(view, evidence, draws=null_draws, seed=PERMUTATION_SEED)

    pool = tuple(sorted(view.posterior))
    precision: dict[str, float] = {}
    risk: dict[str, float] = {}
    for name in (*POLICIES_HERE, BOUND):
        if name == "uniform":
            # A sample, not an exact number, and the summary that goes into every
            # comparison is the *best* of the 21 draws rather than their median. This is
            # not a stylistic choice: finding 28 reported confidence-based abstention as
            # working at unanimity on a gap of 0.006 against the median of the same 21
            # draws, and the claim reversed once the comparison was against the best. A
            # median is not a floor, so a targeted rule beating it has beaten a coin flip
            # half the time. The median is published beside it for the reader who wants to
            # know what a deployment would typically draw.
            cells = [
                score(
                    n_blind=n_blind,
                    slip_rate=slip_rate,
                    policy=name,
                    withheld=select(name, view, truth, REPORT_BUDGET, seed=draw),
                    pool=pool,
                    wrong=wrong,
                )
                for draw in UNIFORM_SEEDS
            ]
            found = sorted(c.precision for c in cells)
            precision[name] = round(max(found), 4)
            precision["uniform_median"] = round(found[len(found) // 2], 4)
            risk[name] = round(min(c.risk for c in cells), 4)
            continue
        cell = score(
            n_blind=n_blind,
            slip_rate=slip_rate,
            policy=name,
            withheld=select(name, view, truth, REPORT_BUDGET, seed=PERMUTATION_SEED),
            pool=pool,
            wrong=wrong,
        )
        precision[name] = cell.precision
        risk[name] = cell.risk

    return Cell(
        keying=keying,
        n_blind=n_blind,
        slip_rate=slip_rate,
        errors=len(wrong),
        converged=estimate.converged,
        channels_fired=fired,
        best_channel_p=None if best_p is None else float(f"{best_p:.3g}"),
        dispersion_index=shape.index,
        dispersion_p=shape.p_value,
        precision=precision,
        risk=risk,
    )


def _views_agree(a: ServerObservation, b: ServerObservation) -> bool:
    """Whether two fleets produced the same aggregate, which the comparison must not."""
    return a.votes == b.votes and a.seen == b.seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    # The same arithmetic `measure_error_shape` and `measure_corpus_sensitivity` assert,
    # and for the same reason: a simulated p-value floors at 1/(m+1), and a floor above
    # ALPHA makes every cell read as undetected however extreme it is.
    for label, draws in (("permutations", args.permutations), ("null draws", args.null_draws)):
        if 1 / (draws + 1) > ALPHA:
            raise SystemExit(
                f"{draws} {label} floor the p-value at {1 / (draws + 1):.2g}, which is above "
                f"alpha {ALPHA}; no cell could be detected however extreme"
            )

    tasks = list(build_tasks(seed=args.seed, events=args.events))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}
    shares = ladder(args.fleet, SHARE_RUNGS)

    try:
        check = assert_channel_usable(tasks)
        # Sized from the guard's own count rather than from a second derivation of it. An
        # earlier version recomputed the affected slice here with a slightly different
        # predicate and published `check.affected` beside slices sized by the copy, so a
        # divergence would have recorded one count while measuring another.
        slices = {
            seed: draw_balanced_slice(tasks, seed=seed, size=check.affected) for seed in SLICE_SEEDS
        }
    except ChannelUnusableError as refusal:
        print(refusal, file=sys.stderr)
        raise SystemExit(REFUSED_EXIT) from refusal

    headline = slices[SLICE_SEEDS[0]]
    print(
        f"{len(tasks)} tasks, fleet of {args.fleet}. Channel-keyed blind spot on "
        f"{BLIND.value} corrupts {check.affected} verdicts; the latent slice is drawn to "
        f"corrupt {len(headline.corrupted)} from a pool of {headline.eligible}, at the "
        f"{headline.carriage_percentile:.0%} point of its own balance null after "
        f"{headline.rejected} refused draw(s)."
    )

    # The two constructions must not produce the same aggregate. If they did, the latent
    # fleet would be the channel fleet under another name and every comparison below would
    # be a fleet compared with itself -- which is exactly the shape of failure this
    # project's blind-spot work has hit twice, both times reporting success.
    def aggregate(slice_: LatentSlice | None) -> ServerObservation:
        policies = (
            latent_blind_fleet(args.fleet, args.fleet, slice_)
            if slice_ is not None
            else blind_fleet(args.fleet, args.fleet)
        )
        return observe(
            partition_by_contributor(contributions_for(policies, tasks, proposals, seed=args.seed))
        )

    if _views_agree(aggregate(None), aggregate(headline)):
        raise SystemExit(
            "the latent and channel-keyed fleets produce an identical aggregate; the "
            "constructions are not distinct and nothing below would be a comparison"
        )

    cells: list[Cell] = []
    total = len(SLIP_RATES) * len(shares) * (1 + len(SLICE_SEEDS))
    for slip in SLIP_RATES:
        for n_blind in shares:
            progress("latent_blindspot.grid", done=len(cells), total=total)
            cells.append(
                measure(
                    tasks,
                    proposals,
                    truth,
                    keying="channel",
                    n_blind=n_blind,
                    slip_rate=slip,
                    fleet=args.fleet,
                    seed=args.seed,
                    slice_=None,
                    permutations=args.permutations,
                    null_draws=args.null_draws,
                )
            )
            cells.extend(
                measure(
                    tasks,
                    proposals,
                    truth,
                    keying=f"latent:{slice_seed}",
                    n_blind=n_blind,
                    slip_rate=slip,
                    fleet=args.fleet,
                    seed=args.seed,
                    slice_=slices[slice_seed],
                    permutations=args.permutations,
                    null_draws=args.null_draws,
                )
                for slice_seed in SLICE_SEEDS
            )

    sweep = slice_sweep(
        tasks,
        proposals,
        truth,
        fleet=args.fleet,
        seed=args.seed,
        null_draws=args.null_draws,
    )
    report = assemble(cells, sweep, slices, shares, args, check.affected)
    render(report, cells, sweep, shares)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def build_tasks(*, seed: int, events: int) -> list[TriageTask]:
    return build_triage_tasks(generate(GeneratorConfig(seed=seed, n_events=events)))


@dataclass(frozen=True, slots=True, kw_only=True)
class DispersionPair:
    """The same fleet under both keyings, so prediction 2 is a subtraction rather than a look."""

    n_blind: int
    slip_rate: float
    slice_: str
    channel_index: float
    latent_index: float
    gap: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_blind": self.n_blind,
            "slip_rate": self.slip_rate,
            "slice": self.slice_,
            "channel_index": self.channel_index,
            "latent_index": self.latent_index,
            "gap": self.gap,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SweepRow:
    """One corrupted-slice size at unanimity, over every slice draw.

    `precision` carries a median and a range per rule rather than a number, because a slice
    is a sample and this sweep used to draw exactly one of them. The structural claims held
    across draws when that was checked; the individual cells moved by up to 0.10, and the
    prose was quoting them as constants. That is the shape of error this project has
    retracted five times, and the fix is to publish the spread rather than to remember.
    """

    size: int
    slip_rate: float
    eligible: int
    rejected: int
    errors: dict[str, float]
    draws: int
    precision: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "slip_rate": self.slip_rate,
            "eligible": self.eligible,
            "rejected": self.rejected,
            "errors": self.errors,
            "draws": self.draws,
            "precision": self.precision,
        }


def _median(values: list[float]) -> float:
    """The median, including the even case, rounded the way every table here rounds.

    One definition, because there were three and they were the same wrong one:
    `sorted(v)[len(v) // 2]` is the *upper* median on an even sample, and it silently
    published 0.15 for `margin` and `posterior` where both are 0.125. That sample is even
    by construction rather than by luck -- the unanimity summary is five slice draws times
    two slip rates -- and the error runs one way, always in the rule's favour, which is the
    direction a reader is least able to discount.
    """
    return round(statistics.median(values), 4)


def _spread(values: list[float]) -> dict[str, float]:
    """Median, min and max of one rule at one cell, across slice draws."""
    ordered = sorted(values)
    return {
        "median": _median(values),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def slice_sweep(
    tasks: list[TriageTask],
    proposals: dict[str, Proposal],
    truth: dict[str, bool],
    *,
    fleet: int,
    seed: int,
    null_draws: int,
) -> list[SweepRow]:
    """How each rule fares as the corrupted slice grows past the majority of its stratum.

    Run at unanimity only, and that is the point rather than a saving: unanimity is where
    every disagreement-reading policy is already at chance, so it isolates what the
    residual rules add. The sweep exists because the two-sided rule's failure mode is a
    prediction about *where* it turns, and a prediction about a crossing is only worth
    making if the sweep crosses it.

    Every cell is measured over all of `SLICE_SEEDS`, the same draws the headline grid
    uses. It ran at one draw first, which made every quoted cell a one-draw number wearing
    a constant's clothes -- the structural claims survive the spread and several of the
    quoted values move by up to a tenth.
    """
    rows: list[SweepRow] = []
    # Which sizes this corpus can host, asked before drawing rather than discovered by
    # catching the refusal. A size larger than the eligible pool is a property of the
    # corpus and is knowable up front; learning it from an exception meant the row simply
    # was not there, and an absent row and a row that was never asked for look identical
    # in the artifact. The committed corpus hosts all six; a smaller one says which it
    # dropped and why.
    pool = len(eligible_pool(tasks))
    unhostable = [size for size in SLICE_SIZES if size > pool]
    if unhostable:
        LOG.warning(
            "latent_blindspot.size_unhostable",
            extra={
                "event": "latent_blindspot.size_unhostable",
                "sizes": unhostable,
                "eligible": pool,
            },
        )
    for size in SLICE_SIZES:
        if size > pool:
            continue
        for slip in SLIP_RATES:
            progress("latent_blindspot.slice_sweep", size=size, slip_rate=slip)
            gathered: dict[str, list[float]] = {}
            errors: list[float] = []
            drawn_at: list[LatentSlice] = []
            for slice_seed in SLICE_SEEDS:
                try:
                    slice_ = draw_balanced_slice(tasks, size=size, seed=slice_seed)
                except ChannelUnusableError as refusal:
                    # Only a refusal this draw owns is skippable. A structural one -- a pool
                    # too small for this size, a slice that flips nothing -- is not fixed by
                    # another seed, so catching it here refused all five draws, emptied
                    # `drawn_at`, and dropped the row through the `continue` below: the
                    # artifact would publish a sweep with no trace of the size it skipped.
                    if refusal.structural:
                        raise
                    # A size this corpus cannot host at this draw is skipped and said so,
                    # not zeroed. Skipping the draw rather than the whole cell keeps a size
                    # that most draws can host in the sweep, with a smaller denominator.
                    LOG.warning(
                        "latent_blindspot.size_refused",
                        extra={
                            "event": "latent_blindspot.size_refused",
                            "size": size,
                            "slice_seed": slice_seed,
                            "reason": str(refusal),
                        },
                    )
                    continue
                cell = measure(
                    tasks,
                    proposals,
                    truth,
                    keying=f"latent:{slice_seed}",
                    n_blind=fleet,
                    slip_rate=slip,
                    fleet=fleet,
                    seed=seed,
                    slice_=slice_,
                    # Neither detector is read in this sweep and both nulls are the
                    # expensive part. One draw each still exercises the same path.
                    permutations=1,
                    null_draws=1,
                )
                drawn_at.append(slice_)
                errors.append(float(cell.errors))
                for rule, value in cell.precision.items():
                    gathered.setdefault(rule, []).append(value)
            if not drawn_at:
                continue
            rows.append(
                SweepRow(
                    size=size,
                    slip_rate=slip,
                    eligible=drawn_at[0].eligible,
                    rejected=sum(d.rejected for d in drawn_at),
                    errors=_spread(errors),
                    draws=len(drawn_at),
                    precision={rule: _spread(vals) for rule, vals in gathered.items()},
                )
            )
    return rows


def assemble(
    cells: list[Cell],
    sweep: list[SweepRow],
    slices: dict[int, LatentSlice],
    shares: tuple[int, ...],
    args: argparse.Namespace,
    affected: int,
) -> dict[str, object]:
    """The artifact, with every prediction answered as a boolean beside its evidence."""
    latent = [c for c in cells if c.keying.startswith("latent")]
    channel = [c for c in cells if c.keying == "channel"]
    unanimous = max(shares)

    # Guarded against an empty pool, like the two dispersion verdicts below. `all()` over
    # nothing is True, and a verdict that is true from zero data is the shape of a control
    # that has quietly stopped running.
    scan_silent = bool(latent) and all(not c.channels_fired for c in latent)
    # Both halves of one control, so both are read the same way. This half was `any` while
    # its mirror above was `all`: there are two unanimity cells, one per slip rate, and a
    # change that killed detectability at 0.15 while leaving 0.0 intact would have passed
    # the control on the surviving cell. That is the same shape as the defect this control
    # was added to close -- half a comparison reporting as a whole one -- one level down.
    at_unanimity_channel = [c for c in channel if c.n_blind >= unanimous]
    scan_fires = bool(at_unanimity_channel) and all(c.channels_fired for c in at_unanimity_channel)

    # Prediction 2, as a number rather than as an impression: the same fleet share and
    # slip under two keyings must give the same index, because the index never read the
    # partition. Compared per cell and the worst gap reported.
    paired: list[DispersionPair] = []
    for chan in channel:
        for lat in latent:
            if (lat.n_blind, lat.slip_rate) != (chan.n_blind, chan.slip_rate):
                continue
            if chan.dispersion_index is None or lat.dispersion_index is None:
                continue
            paired.append(
                DispersionPair(
                    n_blind=chan.n_blind,
                    slip_rate=chan.slip_rate,
                    slice_=lat.keying,
                    channel_index=chan.dispersion_index,
                    latent_index=lat.dispersion_index,
                    gap=round(abs(chan.dispersion_index - lat.dispersion_index), 4),
                )
            )
    worst_index_gap = max((p.gap for p in paired), default=None)

    def at_unanimity(rule: str) -> list[float]:
        return [
            c.precision[rule]
            for c in latent
            if c.n_blind >= unanimous and c.errors and rule in c.precision
        ]

    def beats_uniform(rule: str) -> int:
        return sum(
            1
            for c in latent
            if c.n_blind >= unanimous and c.errors and c.precision[rule] > c.precision["uniform"]
        )

    scored = [c for c in latent if c.n_blind >= unanimous and c.errors]

    def no_better_than_uniform(rule: str) -> list[int]:
        """Sizes where a rule's median draw fails to beat the best untargeted draw's median.

        Compared median to median across slice draws rather than cell to cell, now that
        each cell is a spread. A rule that loses on one draw of five has not turned; a rule
        whose median has crossed has.
        """
        return [
            r.size
            for r in sweep
            if r.errors["median"]
            and r.precision[rule]["median"] <= r.precision["uniform"]["median"]
        ]

    inverted = no_better_than_uniform("deviation")

    # The eligible pool, read off a row rather than recomputed from the corpus. Every row
    # carries the same number by construction -- eligibility is a property of the corpus and
    # the threshold, not of the draw -- and reading it here keeps one definition of "the
    # pool" between the sweep that filtered on it and the artifact that reports it.
    pool = sweep[0].eligible if sweep else 0

    # Prediction 2, operationalized the only way the data supports. Equality is the right
    # test where both fleets are deterministic: at slip 0 the two constructions corrupt the
    # same *number* of tasks in the same stratum, and the index reads how many and never
    # which, so it is identical by construction. Once analysts slip independently the two
    # fleets are different draws, and a per-cell inequality was tried first and failed in
    # 13 of 55 pairs by margins from 0.003 to 0.13 -- scattered both directions, which is
    # what a comparison against noise looks like rather than a reduction.
    #
    # So the question becomes whether the index can tell the constructions apart at all:
    # does the channel-keyed value sit inside the spread that *slice draws alone* produce?
    # A statistic that cannot separate them has lost nothing to the loss of the partition,
    # and that is the claim.
    deterministic = [p for p in paired if p.slip_rate == 0.0]
    identical = all(p.gap == 0.0 for p in deterministic)

    spread: list[dict[str, object]] = []
    for chan in channel:
        draws = [
            p.latent_index
            for p in paired
            if (p.n_blind, p.slip_rate) == (chan.n_blind, chan.slip_rate)
        ]
        if not draws or chan.dispersion_index is None:
            continue
        # A row can only test the claim if the two fleets are actually two fleets. At
        # `n_blind == 0` they are not: `latent_blind_fleet` and `blind_fleet` both build
        # the same tuple of sighted analysts and an empty blind list, so the row compares a
        # fleet with itself and reads as agreement whatever the statistic does. `main`
        # already refuses that comparison through `_views_agree`, but only at full
        # unanimity. Published rather than dropped, because a reader looking for the
        # unblinded reference row should find it; excluded from the verdict below, because
        # a cell that cannot fail is not evidence that nothing failed.
        informative = chan.n_blind > 0
        spread.append(
            {
                "n_blind": chan.n_blind,
                "slip_rate": chan.slip_rate,
                "channel_index": chan.dispersion_index,
                "latent_min": min(draws),
                "latent_max": max(draws),
                "latent_median": _median(draws),
                "channel_inside_latent_spread": min(draws) <= chan.dispersion_index <= max(draws),
                "informative": informative,
            }
        )
    testable = [row for row in spread if row["informative"]]
    indistinguishable = bool(testable) and all(
        row["channel_inside_latent_spread"] for row in testable
    )

    findings = {
        # 1: the constructions are matched, which everything else rests on.
        "channel_scan_silent_on_latent": scan_silent,
        "channel_scan_fires_on_channel_keyed": scan_fires,
        # 2: the index does not need the partition.
        "dispersion_identical_on_deterministic_fleets": bool(deterministic) and identical,
        "dispersion_cannot_tell_the_constructions_apart": indistinguishable,
        # 3 and 4: localization.
        # `bool(scored)` for the same reason: with no unanimity cell carrying an error,
        # `0 == 0` would assert both localization results off no measurement at all.
        "two_sided_residual_localizes_at_unanimity": bool(scored)
        and beats_uniform("deviation") == len(scored),
        "one_sided_residual_localizes_at_unanimity": bool(scored)
        and beats_uniform("shortfall") == len(scored),
        "two_sided_residual_inverts_in_the_sweep": bool(inverted),
        "one_sided_residual_inverts_in_the_sweep": bool(no_better_than_uniform("shortfall")),
    }

    return {
        "provenance": run_provenance(seed=args.seed),
        "fleet": args.fleet,
        "events": args.events,
        "shares": list(shares),
        "slip_rates": list(SLIP_RATES),
        "slice_sizes": list(SLICE_SIZES),
        "slice_seeds": list(SLICE_SEEDS),
        "policies": list(POLICIES_HERE),
        "bound": BOUND,
        "report_budget": REPORT_BUDGET,
        "alpha": ALPHA,
        "permutations": args.permutations,
        "null_draws": args.null_draws,
        "blind_compartment": BLIND.value,
        "channel_affected": affected,
        "slices": {str(seed): s.as_dict() for seed, s in slices.items()},
        "findings": findings,
        "dispersion_pairs": [p.as_dict() for p in paired],
        "dispersion_spread": spread,
        "worst_dispersion_gap": worst_index_gap,
        #: How many of those rows could have falsified the claim, which is fewer than the
        #: table is long. The unblinded row compares a fleet with itself; the deterministic
        #: rows are identical by construction and are asserted separately.
        "dispersion_cells_that_could_falsify": len(testable),
        "unanimity_precision": {
            rule: {
                "median": _median(at_unanimity(rule)) if at_unanimity(rule) else None,
                "min": min(at_unanimity(rule), default=None),
                "max": max(at_unanimity(rule), default=None),
                "draws": len(at_unanimity(rule)),
            }
            for rule in (*POLICIES_HERE, BOUND)
        },
        #: The *sizes* at which the two-sided rule turned, deduplicated, because the field
        #: is named for sizes and `no_better_than_uniform` returns one entry per swept cell
        #: -- two per size, one per slip rate -- so it read `[40, 40, 50, 50, 60]` and a
        #: reader taking the name at its word saw a size listed twice. The per-cell count is
        #: `cells_where_two_sided_is_no_better_than_uniform` below, which is what the prose
        #: quotes; the two answer different questions and now each says which.
        "inverted_sizes": sorted(set(inverted)),
        #: Sizes the sweep asked for and did not run, split by which of the two reasons it
        #: was. Both empty on the committed corpus, and published rather than inferred from
        #: a short table: a row that is missing and a row that was never requested look the
        #: same to a reader, and only one of them is a result.
        #:
        #: Split because the first draft of this field was one list derived by subtracting
        #: the sizes that produced rows, under the name `sizes_the_corpus_cannot_host` --
        #: which is true of a size larger than the pool and false of a size the pool can
        #: host whose every draw was refused as lopsided. One name over two causes, on a
        #: branch whose subject is names that overstate what they cover.
        "sizes_the_corpus_cannot_host": [s for s in SLICE_SIZES if s > pool],
        "sizes_no_balanced_draw_reached": [
            s for s in SLICE_SIZES if s <= pool and s not in {r.size for r in sweep}
        ],
        #: The two counts the findings page quotes in prose. Published rather than left to
        #: be counted off a table by hand: this project has had a hand-typed summary of a
        #: generated table go stale silently, and these are the same shape.
        "swept_cells": len(sweep),
        "cells_where_two_sided_is_no_better_than_uniform": sum(
            1
            for r in sweep
            if r.errors["median"]
            and r.precision["deviation"]["median"] <= r.precision["uniform"]["median"]
        ),
        "cells_where_one_sided_ties_the_bound": sum(
            1
            for r in sweep
            if r.errors["median"]
            and r.precision["shortfall"]["median"] == r.precision[BOUND]["median"]
        ),
        "cells_where_one_sided_is_no_better_than_uniform": sum(
            1
            for r in sweep
            if r.errors["median"]
            and r.precision["shortfall"]["median"] <= r.precision["uniform"]["median"]
        ),
        "unconverged_cells": [
            {"keying": c.keying, "n_blind": c.n_blind, "slip_rate": c.slip_rate}
            for c in cells
            if not c.converged
        ],
        "grid": [c.as_dict() for c in cells],
        "slice_sweep": [r.as_dict() for r in sweep],
        "validity": check_sample_size(args.events, label="latent blind spot").as_dict(),
    }


def render(
    report: dict[str, object],
    cells: list[Cell],
    sweep: list[SweepRow],
    shares: tuple[int, ...],
) -> None:
    """Print the two tables a reader needs and every prediction's verdict."""
    print("\n  detection, by what the blind spot is keyed on")
    print(f"    {'keying':>16}{'blind':>7}{'slip':>7}{'scan':>10}{'index':>9}{'p':>10}")
    for cell in cells:
        if cell.keying not in {"channel", f"latent:{SLICE_SEEDS[0]}"}:
            continue
        fired = ",".join(cell.channels_fired) or "--"
        index = "--" if cell.dispersion_index is None else f"{cell.dispersion_index:.2f}"
        p_value = "--" if cell.dispersion_p is None else f"{cell.dispersion_p:.4f}"
        print(
            f"    {cell.keying:>16}{cell.n_blind:>7}{cell.slip_rate:>7}"
            f"{fired:>10}{index:>9}{p_value:>10}"
        )

    print("\n  localization at unanimity, share of a 20-item withhold that was wrong")
    summary = report["unanimity_precision"]
    if not isinstance(summary, dict):  # pragma: no cover -- assembled two lines above
        raise TypeError("unanimity_precision must be a mapping")
    for rule, stats in summary.items():
        median, low, high = stats["median"], stats["min"], stats["max"]
        if median is None:
            print(f"    {rule:>10}  no scored cell")
            continue
        print(f"    {rule:>10}  median {median:.2f}  range {low:.2f}-{high:.2f}")

    print("\n  as the corrupted slice grows past the majority of its stratum")
    print(
        f"    {'size':>6}{'slip':>7}{'errors':>8}{'uniform':>9}"
        f"{'deviation':>18}{'shortfall':>18}   (median, range over draws)"
    )
    for row in sweep:
        two = row.precision["deviation"]
        one = row.precision["shortfall"]
        print(
            f"    {row.size:>6}{row.slip_rate:>7}{row.errors['median']:>8.0f}"
            f"{row.precision['uniform']['median']:>9.2f}"
            f"{two['median']:>8.2f} ({two['min']:.2f}-{two['max']:.2f})"
            f"{one['median']:>8.2f} ({one['min']:.2f}-{one['max']:.2f})"
        )

    print("\n  predictions, as measured")
    findings = report["findings"]
    if not isinstance(findings, dict):  # pragma: no cover -- assembled two lines above
        raise TypeError("findings must be a mapping")
    for name, value in findings.items():
        print(f"    {name:<50} {value}")

    if not findings["channel_scan_silent_on_latent"]:
        LOG.error(
            "latent_blindspot.scan_not_silent",
            extra={"event": "latent_blindspot.scan_not_silent"},
        )
    if not findings["channel_scan_fires_on_channel_keyed"]:
        # The other half of the same control, and it was missing while the CI step
        # described the pair. A scan that has stopped detecting the channel-keyed blind
        # spot makes "silent on the latent one" worthless -- both fleets read the same
        # because the instrument is dead, not because the construction worked. Silence
        # here would have let that land with CI green, which is finding 27's failure class
        # exactly.
        LOG.error(
            "latent_blindspot.control_did_not_fire",
            extra={"event": "latent_blindspot.control_did_not_fire"},
        )
    if not findings["dispersion_cannot_tell_the_constructions_apart"]:
        LOG.warning(
            "latent_blindspot.dispersion_needs_the_partition",
            extra={
                "event": "latent_blindspot.dispersion_needs_the_partition",
                "worst_gap": report["worst_dispersion_gap"],
            },
        )

    gap = report["worst_dispersion_gap"]
    record(
        "latent_blindspot.dispersion_gap",
        float(gap) if isinstance(gap, int | float) else -1.0,
        shares=len(shares),
        cells=len(cells),
    )


if __name__ == "__main__":
    raise SystemExit(main())
