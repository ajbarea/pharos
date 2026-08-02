#!/usr/bin/env python3
"""Whether reliability can be estimated without knowing who contributed.

Two findings in this repository point in opposite directions and the conflict is not
rhetorical. Finding 10 shows a learner acquires its teacher's standard exactly, so the
mitigation is annotator-reliability weighting, which requires contributor identity
retained through training. Finding 11 shows contributor identity is what makes a
fleet's contributions attributable to a person, and that pooling contributors is the
only control that removes the leak at no cost in training volume. One wants identity;
the other wants none.

The obvious reconciliation is to estimate reliability *without* identity, from
agreement among contributions on the same task. Several analysts reach the same task,
so a contribution that disagrees with its task's consensus could be down-weighted
without anyone knowing who sent it. This measures whether that works.

It does, and only where it is not needed. Three conditions are compared as the share
of the fleet holding a wrong standard is swept from none to all:

  unweighted   pool everything, the baseline a naive aggregator gets
  oracle       drop contributors whose own agreement with the world is low.
               Requires identity, and is an upper bound rather than a method: it is
               told the answer it would have to estimate.
  consensus    take each task's majority verdict. Requires no identity at all.

The result is a cliff rather than a curve, and the cliff sits exactly where the wrong
standard becomes the majority. That matters here specifically because
`finding 3b` found over-escalation in every model tested and `finding 8` found that
agreement is not correctness: a fleet of model-assisted analysts is precisely the
population in which the wrong standard can hold the majority, and there consensus
ratifies it with full confidence.

Needs no model and no network. Reviewers are parameterised policies and their
verdicts are deterministic given a seed, so this reproduces exactly and runs in CI.

    uv run python scripts/measure_consensus_reliability.py --out results/consensus.json
"""

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pharos.analyst import Action, AnalystPolicy, Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.inference import dawid_skene
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.uncertainty import Trial, summarize
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 200

#: Fleet size. Odd so a majority always exists and the cliff is not blurred by ties.
FLEET = 9

#: The wrong standard the mistaken share of the fleet holds. Two-of-three rather than
#: one-of-three because it is the *closer* error: a control that cannot survive the
#: near-miss has no chance against the gross one.
WRONG_THRESHOLD = 2

#: Contributor agreement below which the oracle drops a contributor. The oracle is a
#: bound, not a proposal: it is handed the per-contributor truth that any real method
#: would have to estimate, which is what makes it the right ceiling to compare against.
ORACLE_FLOOR = 0.75


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _show(value: float | None) -> str:
    """`none` rather than a number, so an empty stream cannot read as a score."""
    return "none" if value is None else f"{value:.4f}"


def targets_by_task(
    policies: Sequence[AnalystPolicy],
    tasks: Sequence[TriageTask],
    proposals: dict[str, Proposal],
    *,
    seed: int,
) -> dict[str, list[tuple[str, bool]]]:
    """Each task's contributed verdicts, as `(contributor, verdict)`.

    A rejection contributes nothing: it leaves the learner no verdict to train on,
    which is finding 7's point and is preserved here rather than counted as a vote.
    """
    grouped: dict[str, list[tuple[str, bool]]] = defaultdict(list)
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
                grouped[task.task_id].append((policy.name, verdict))
    return dict(grouped)


def _agreement(rows: Sequence[tuple[str, str, bool]], truth: dict[str, bool]) -> float | None:
    """Share of contributions matching the world, or None when there are none.

    None rather than 0.0 on an empty stream. A condition that kept no contributions
    has produced no evidence, and scoring that as perfectly wrong reads as a
    measurement when it is an absence. The oracle hits this at the far end of the
    sweep: with every contributor below the floor it drops all of them, and 0.0000
    there would say the identity-based control failed when what happened is that it
    correctly refused everything.
    """
    if not rows:
        return None
    return sum(v == truth[t] for t, _, v in rows) / len(rows)


def fleet_of(n_wrong: int, size: int = FLEET) -> tuple[AnalystPolicy, ...]:
    """`n_wrong` analysts holding the wrong standard, the rest holding the right one."""
    right = [AnalystPolicy(f"right-{i}") for i in range(size - n_wrong)]
    wrong = [
        AnalystPolicy(f"wrong-{i}", escalation_threshold=WRONG_THRESHOLD) for i in range(n_wrong)
    ]
    return tuple(right + wrong)


def conditions(
    grouped: dict[str, list[tuple[str, bool]]], truth: dict[str, bool]
) -> dict[str, list[tuple[str, str, bool]]]:
    """The three target streams an aggregator could train on."""
    flat = [(tid, who, v) for tid, rows in grouped.items() for who, v in rows]

    # A contributor appears in `flat` only by having contributed, so their own stream
    # is never empty and `_agreement` never returns None here. Defaulted rather than
    # asserted so a future caller passing an empty group gets a contributor the oracle
    # drops, not a crash.
    reliability: dict[str, float] = {}
    for who in {w for _, w, _ in flat}:
        own = [r for r in flat if r[1] == who]
        reliability[who] = _agreement(own, truth) or 0.0

    consensus = [
        (tid, "consensus", Counter(v for _, v in rows).most_common(1)[0][0])
        for tid, rows in grouped.items()
    ]
    # The canonical estimator, and the one that decides whether the headline claim
    # means anything. Dawid-Skene infers per-contributor error rates and the true
    # labels jointly, with no ground truth, so unlike the oracle it is a method rather
    # than a bound. An earlier version of this measurement compared only against
    # majority vote, which made "reliability cannot be estimated" a claim about the
    # weakest available estimator instead of about the strongest.
    estimate = dawid_skene(flat)
    inferred = [(tid, "dawid-skene", label) for tid, label in estimate.labels().items()]
    return {
        "unweighted": flat,
        "oracle": [r for r in flat if reliability[r[1]] >= ORACLE_FLOOR],
        "consensus": consensus,
        "dawid_skene": inferred,
    }


@dataclass(frozen=True, slots=True)
class Row:
    """One fleet composition, and what each aggregation strategy got from it.

    A dataclass rather than a dict because the cliff is computed by comparing two of
    these fields, and a comparison against a value typed `object` is how a
    None-versus-float slip gets through unnoticed.
    """

    n_wrong: int
    share_wrong: float
    unweighted: float | None
    oracle: float | None
    consensus: float | None
    dawid_skene: float | None
    consensus_interval: dict[str, object]
    targets: dict[str, int]

    @property
    def consensus_lags_oracle(self) -> bool:
        """Whether consensus fell behind the oracle here.

        False when either side kept nothing: an absent stream is not a lower score,
        and the far end of the sweep has an oracle that correctly refused everything.
        """
        if self.oracle is None or self.consensus is None:
            return False
        return self.consensus < self.oracle - 0.01

    def as_dict(self) -> dict[str, object]:
        return {
            "n_wrong": self.n_wrong,
            "share_wrong": self.share_wrong,
            "unweighted": self.unweighted,
            "oracle": self.oracle,
            "consensus": self.consensus,
            "dawid_skene": self.dawid_skene,
            "consensus_interval": self.consensus_interval,
            "targets": self.targets,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--fleet", type=int, default=FLEET)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=args.events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    print(f"{len(tasks)} tasks, fleet of {args.fleet}, wrong standard = {WRONG_THRESHOLD} of 3")
    print(
        f"  {'wrong':>6} {'unweighted':>12} {'oracle (id)':>13} "
        f"{'consensus':>12} {'Dawid-Skene':>13}"
    )
    print("  " + "-" * 60)

    rows: list[Row] = []
    for n_wrong in range(args.fleet + 1):
        grouped = targets_by_task(fleet_of(n_wrong, args.fleet), tasks, proposals, seed=SEED)
        streams = conditions(grouped, truth)
        scored = {name: _agreement(stream, truth) for name, stream in streams.items()}
        measured = summarize(
            [Trial(t, v == truth[t]) for t, _, v in streams["consensus"]],
            label=f"consensus-{n_wrong}",
        )
        rows.append(
            Row(
                n_wrong=n_wrong,
                share_wrong=round(n_wrong / args.fleet, 4),
                unweighted=_round(scored["unweighted"]),
                oracle=_round(scored["oracle"]),
                consensus=_round(scored["consensus"]),
                dawid_skene=_round(scored["dawid_skene"]),
                consensus_interval=measured.single_run.as_dict(),
                targets={k: len(v) for k, v in streams.items()},
            )
        )
        print(
            f"  {n_wrong:>6} {_show(scored['unweighted']):>12} "
            f"{_show(scored['oracle']):>13} {_show(scored['consensus']):>12} "
            f"{_show(scored['dawid_skene']):>13}"
        )

    # The cliff: the first fleet composition at which consensus stops matching the
    # oracle. Computed rather than eyeballed, because the whole claim is where it sits.
    cliff = next((r.n_wrong for r in rows if r.consensus_lags_oracle), None)
    print()
    if cliff is None:
        print("consensus matched the oracle at every composition tested")
    else:
        before, after = rows[cliff - 1], rows[cliff]
        print(
            f"consensus tracks the oracle until {cliff} of {args.fleet} hold the wrong "
            f"standard, then falls from {_show(before.consensus)} to {_show(after.consensus)}"
        )
        print(
            f"  that is the majority crossing: {cliff}/{args.fleet} = "
            f"{cliff / args.fleet:.3f}, and past it consensus returns the wrong rule"
        )

    report = {
        "provenance": run_provenance(seed=SEED),
        "fleet": args.fleet,
        "events": args.events,
        "wrong_threshold": WRONG_THRESHOLD,
        "oracle_floor": ORACLE_FLOOR,
        "grid": [r.as_dict() for r in rows],
        "cliff_at": cliff,
        "validity": check_sample_size(len(tasks), label="consensus reliability").as_dict(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
