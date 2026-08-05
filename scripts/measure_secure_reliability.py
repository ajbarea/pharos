#!/usr/bin/env python3
"""Whether the estimate can be computed under aggregation, and what that costs.

This repository has carried one open problem since finding 13. Finding 10 needs
contributor identity to weight a fleet by reliability; finding 11 shows contributor
identity is the leak, recovering an analyst's compartment set at 0.820 from a stream
that reads no content; findings 12 and 13 close both ways of recovering the first
from the second *after* aggregation. What was never tried is the other direction:
compute the estimate **under** aggregation, so the per-analyst stream the attack reads
is never produced at all.

It works, and it is exact rather than approximate, because Dawid-Skene splits along
the seam a secure sum offers. Its M step is per contributor over that contributor's
own reports, so it runs locally and is never transmitted. Its E step needs a product
over contributors per task, which is a sum in logs, and a sum is precisely what secure
aggregation reveals and all that it reveals. `pharos.secagg` supplies the masked sum
and hands back a value with no per-client field on it; `pharos.inference` supplies the
estimator built on that value.

Three things are measured here, and only one of them is good news.

  equivalence  federated against centralized Dawid-Skene, task by task. If moving the
               computation changed the answer, nothing downstream would be comparable
               to findings 12, 16 or 17 and the port would be a different estimator
               wearing the same name.
  the cliff    the same wrong-standard sweep finding 12 ran. The prediction is that
               the cliff does not move at all, because it is a failure of
               identifiability rather than of pooling -- FedDS (Dong et al., Inf. Sci.
               745:123425, 2026) proves Dawid-Skene identifiable only where every
               confusion matrix is diagonally dominant (their Eq. 16), and a fleet
               whose majority holds the wrong standard is exactly the fleet that
               violates it. A prediction stated before the run and reported either way.
  readership   what the aggregate discloses anyway. Majority-vote initialization is a
               per-task mean, a mean needs its denominator, and that denominator is
               the number of analysts who could read the task. Nobody is named. The
               question is what a headcount alone gives away.

Needs no model and no network.

    uv run python scripts/measure_secure_reliability.py --out results/secure_reliability.json
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pharos.analyst import Action, AnalystPolicy, Proposal
from pharos.disclosure import DROP_COMPARTMENTS, KEEP_COMPARTMENTS, DeclassificationPolicy
from pharos.fleet import (
    FLEET_CEILING,
    Clearance,
    assign_fleet,
    candidate_clearances,
    contribute,
)
from pharos.generate import GeneratorConfig, generate
from pharos.inference import (
    agreement_with,
    dawid_skene,
    federated_dawid_skene,
    partition_by_contributor,
)
from pharos.labels import Capacity, Label, declassify, join
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, record
from pharos.validity import check_sample_size

# Matched to finding 12 so the cliff is comparable rather than merely similar.
SEED = 7
EVENTS = 200
FLEET = 9
WRONG_THRESHOLD = 2

# Matched to finding 11, whose attack is the one this protocol is meant to starve.
FLEET_SEED = 11
LINKAGE_FLEET = 200

#: Mask seed for the aggregation rounds. Distinct from the corpus seed so a corpus
#: change cannot silently alter the protocol's randomness as well.
MASK_SEED = 4242

#: Agreement gap that counts as the cliff, matching finding 12's threshold.
CLIFF_GAP = 0.01

#: How far the federated and centralized posteriors may drift before the equivalence
#: claim is broken. Four orders of magnitude above the fixed-point resolution the
#: protocol quantizes at (~2.7e-7 for nine contributors) and far below anything that
#: could move a label.
EQUIVALENCE_BOUND = 1e-3


def fleet_of(n_wrong: int, size: int) -> tuple[AnalystPolicy, ...]:
    """`n_wrong` analysts holding the wrong standard, the rest holding the right one.

    Reproduced from finding 12 rather than imported, because that script defines it
    against its own module-level constant and this one has to sweep the size.
    """
    right = [AnalystPolicy(f"right-{i}") for i in range(size - n_wrong)]
    wrong = [
        AnalystPolicy(f"wrong-{i}", escalation_threshold=WRONG_THRESHOLD) for i in range(n_wrong)
    ]
    return tuple(right + wrong)


def contributions_for(
    policies: Sequence[AnalystPolicy],
    tasks: Sequence[TriageTask],
    proposals: dict[str, Proposal],
    *,
    seed: int,
) -> list[tuple[str, str, bool]]:
    """Flat `(task, contributor, verdict)` rows, exactly as finding 12 builds them.

    A rejection contributes nothing rather than counting as a vote, which is finding
    7's result and is preserved here so the two measurements see the same stream.
    """
    rows: list[tuple[str, str, bool]] = []
    for policy in policies:
        for task in tasks:
            decision = policy.review(task, proposals[task.task_id], seed=seed)
            if decision.action is Action.ACCEPT:
                verdict: bool | None = proposals[task.task_id].verdict
            elif decision.action is Action.REVISE:
                verdict = decision.corrected_verdict
            else:
                verdict = None
            if verdict is not None:
                rows.append((task.task_id, policy.name, verdict))
    return rows


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """One fleet composition under both estimators."""

    n_wrong: int
    share_wrong: float
    central: float
    federated: float
    max_posterior_gap: float
    label_disagreements: int
    central_iterations: int
    federated_iterations: int
    both_converged: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "n_wrong": self.n_wrong,
            "share_wrong": self.share_wrong,
            "central": self.central,
            "federated": self.federated,
            "max_posterior_gap": self.max_posterior_gap,
            "label_disagreements": self.label_disagreements,
            "central_iterations": self.central_iterations,
            "federated_iterations": self.federated_iterations,
            "both_converged": self.both_converged,
        }


def _cliff(values: Sequence[float]) -> int | None:
    """The first composition at which agreement drops by more than `CLIFF_GAP`."""
    for i in range(1, len(values)):
        if values[i] < values[i - 1] - CLIFF_GAP:
            return i
    return None


def sweep_equivalence(
    tasks: Sequence[TriageTask], proposals: dict[str, Proposal], truth: dict[str, bool], size: int
) -> list[Row]:
    """Both estimators across every fleet composition from all-right to all-wrong."""
    rows: list[Row] = []
    for n_wrong in range(size + 1):
        flat = contributions_for(fleet_of(n_wrong, size), tasks, proposals, seed=SEED)
        central = dawid_skene(flat)
        federated = federated_dawid_skene(partition_by_contributor(flat), seed=MASK_SEED)

        shared = sorted(set(central.posterior) & set(federated.posterior))
        gap = max((abs(central.posterior[t] - federated.posterior[t]) for t in shared), default=0.0)
        central_labels, federated_labels = central.labels(), federated.labels()
        disagreements = sum(1 for t in shared if central_labels[t] != federated_labels[t])

        rows.append(
            Row(
                n_wrong=n_wrong,
                share_wrong=round(n_wrong / size, 4),
                central=round(agreement_with(central_labels, truth), 4),
                federated=round(agreement_with(federated_labels, truth), 4),
                max_posterior_gap=gap,
                label_disagreements=disagreements,
                central_iterations=central.iterations,
                federated_iterations=federated.iterations,
                both_converged=central.converged and federated.converged,
            )
        )
    return rows


@dataclass(frozen=True, slots=True)
class Readership:
    """What a per-task headcount discloses about who is read into what."""

    labels_probed: int
    exact_headcounts: int
    compartment_sets_determined: int
    headcounts: dict[str, int]
    truth: dict[str, int]
    prior_expectation: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "labels_probed": self.labels_probed,
            "exact_headcounts": self.exact_headcounts,
            "compartment_sets_determined": self.compartment_sets_determined,
            "headcounts": self.headcounts,
            "truth": self.truth,
            "prior_expectation": self.prior_expectation,
        }


def measure_readership(
    tasks: Sequence[TriageTask],
    fleet: Sequence[Clearance],
    *,
    policy: DeclassificationPolicy,
) -> Readership:
    """What the aggregate's denominators give away, with no per-analyst stream at all.

    The adversary here is finding 11's: it knows the public structure of the corpus,
    including each task's label, and it is denied everything else. Under secure
    aggregation it can no longer see which tasks a pseudonym touched, because there
    are no pseudonyms. It can see how many contributors each task drew, because a
    majority-vote initialization is a mean and a mean needs its denominator.

    A task's readership is decided by the join of its sources, so tasks sharing that
    join share a headcount. Reading one headcount is therefore reading, exactly and
    with no inference, the number of analysts cleared for that join. The measurement
    is how much of the fleet's clearance census that pins down.
    """
    stream = contribute(fleet, tasks, policy=policy, ceiling=FLEET_CEILING)
    partitioned: dict[str, list[tuple[str, bool]]] = {}
    for c in stream:
        partitioned.setdefault(c.analyst_id, []).append((c.task_id, c.verdict))

    estimate = federated_dawid_skene(partitioned, seed=MASK_SEED)
    observed = estimate.readership

    by_task = {t.task_id: t for t in tasks}
    joins: dict[Label, list[str]] = {}
    for task_id in observed:
        task = by_task[task_id]
        joins.setdefault(
            join([r.label for r in task.sources], capacity=Capacity.FREETEXT), []
        ).append(task_id)

    # Keyed on the whole join rather than on its compartment set. Two joins can carry
    # the same compartments at different sensitivities, and they have different
    # headcounts; collapsing them would drop one of the two silently and report a
    # smaller table than was actually read.
    def _key(probe: Label) -> str:
        compartments = ",".join(sorted(c.value for c in probe.compartments)) or "(none)"
        return f"{probe.sensitivity.value} | {compartments}"

    space = candidate_clearances()
    headcounts: dict[str, int] = {}
    truth: dict[str, int] = {}
    prior: dict[str, float] = {}
    exact = 0

    for probe, task_ids in joins.items():
        key = _key(probe)
        cleared = sum(1 for c in fleet if c.label.dominates(probe))
        headcounts[key] = observed[task_ids[0]]
        truth[key] = cleared
        # The adversary's prior: clearances are drawn uniformly from the candidate
        # space, so the expected headcount is the fleet size times the share of that
        # space dominating the probe. Reporting it is what makes "the headcount
        # discloses something" a measurement rather than a restatement of fleet size.
        dominating = sum(
            1 for level, subset in space if Label(level, subset, Capacity.FREETEXT).dominates(probe)
        )
        prior[key] = round(len(fleet) * dominating / len(space), 2)
        if all(observed[t] == cleared for t in task_ids):
            exact += 1

    return Readership(
        labels_probed=len(joins),
        exact_headcounts=exact,
        compartment_sets_determined=len({frozenset(probe.compartments) for probe in joins}),
        headcounts=headcounts,
        truth=truth,
        prior_expectation=prior,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--linkage-fleet", type=int, default=LINKAGE_FLEET)
    parser.add_argument("--fleet-seed", type=int, default=FLEET_SEED)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    print(f"{len(tasks)} tasks, fleet of {args.fleet}, wrong standard = {WRONG_THRESHOLD} of 3")
    print(f"  {'wrong':>6} {'central':>9} {'federated':>11} {'gap':>11} {'labels differ':>14}")
    print("  " + "-" * 56)

    rows = sweep_equivalence(tasks, proposals, truth, args.fleet)
    for r in rows:
        print(
            f"  {r.n_wrong:>6} {r.central:>9.4f} {r.federated:>11.4f} "
            f"{r.max_posterior_gap:>11.2e} {r.label_disagreements:>14}"
        )
        record(
            "secure.composition",
            r.federated,
            n_wrong=r.n_wrong,
            central=r.central,
            gap=r.max_posterior_gap,
        )

    worst_gap = max(r.max_posterior_gap for r in rows)
    disagreements = sum(r.label_disagreements for r in rows)
    central_cliff = _cliff([r.central for r in rows])
    federated_cliff = _cliff([r.federated for r in rows])

    print()
    print(
        f"equivalence: worst posterior gap {worst_gap:.2e} over "
        f"{len(rows)} compositions, {disagreements} label disagreements"
    )
    print(f"cliff: centralized at {central_cliff}, federated at {federated_cliff}")
    if central_cliff != federated_cliff:
        # Loud, because it would mean the port changed the finding rather than moving
        # where it is computed, and every downstream comparison would be void.
        get_logger().warning(
            "secure.cliff_moved",
            extra={
                "event": "secure.cliff_moved",
                "central": central_cliff,
                "federated": federated_cliff,
            },
        )

    # DROP_COMPARTMENTS, matching finding 11's headline condition rather than the
    # fail-closed default. Under the default almost nothing is eligible to leave, the
    # stream collapses to universally-readable tasks, and every headcount is the fleet
    # size -- a stream with no variation to disclose, which would read as a protocol
    # that leaks nothing when what happened is that nothing federated.
    fleet = assign_fleet(args.linkage_fleet, seed=args.fleet_seed)
    readership = measure_readership(tasks, fleet, policy=DROP_COMPARTMENTS)
    print()
    print(
        f"readership: {readership.labels_probed} distinct source-joins in the stream, "
        f"{readership.exact_headcounts} of them with an exactly-correct headcount"
    )
    print(f"  {'sensitivity | compartments':>40} {'read off':>9} {'true':>6} {'prior':>8}")
    print("  " + "-" * 67)
    for key in sorted(readership.headcounts):
        print(
            f"  {key:>40} {readership.headcounts[key]:>9} "
            f"{readership.truth[key]:>6} {readership.prior_expectation[key]:>8.2f}"
        )
    record(
        "secure.readership_exact",
        float(readership.exact_headcounts),
        probed=readership.labels_probed,
    )

    report = {
        "provenance": run_provenance(seed=SEED),
        "fleet": args.fleet,
        "events": args.events,
        "wrong_threshold": WRONG_THRESHOLD,
        "mask_seed": MASK_SEED,
        "grid": [r.as_dict() for r in rows],
        "worst_posterior_gap": worst_gap,
        "label_disagreements": disagreements,
        "central_cliff_at": central_cliff,
        "federated_cliff_at": federated_cliff,
        "linkage_fleet": args.linkage_fleet,
        "fleet_seed": args.fleet_seed,
        "readership": readership.as_dict(),
        "validity": check_sample_size(len(tasks), label="secure reliability").as_dict(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    # The CI step for this script says the two estimators "have to stay within 1e-6 of
    # each other, and nothing else in the suite compares them on the regenerated
    # corpus". That was true of the comment and false of the code: main() printed the
    # gap and returned 0 regardless, so an injected 0.3 divergence exited clean. The
    # one claim a refactor would break silently is now the one that fails the build.
    if worst_gap > EQUIVALENCE_BOUND or disagreements:
        get_logger().error(
            "secure.equivalence_broken",
            extra={
                "event": "secure.equivalence_broken",
                "worst_gap": worst_gap,
                "label_disagreements": disagreements,
                "bound": EQUIVALENCE_BOUND,
            },
        )
        print(
            f"\nFAILED: federated and centralized estimators diverged -- worst gap "
            f"{worst_gap:.2e} against a {EQUIVALENCE_BOUND:.0e} bound, "
            f"{disagreements} label disagreements",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
