#!/usr/bin/env python3
"""What survives when the label is replaced by a review.

Finding 6 showed the decision rule is learnable by gradient descent from clean
labels. The premise of the whole system is weaker than that: a fleet is supposed to
learn from an analyst's accept, revise, and reject decisions over a proposal, which
is indirect, ambiguous about what was wrong, noisier, and scarcer. This measures how
much supervision is left after that substitution, and how much of it is correct.

The proposals are real. Every verdict reviewed here is one a model actually
produced, read from the committed `results/triage_lift-*.json` artifacts, so the
review stream is grounded in the over-escalation that finding 3b measured across
six models rather than in an invented error pattern. The reviewers are not real and
do not claim to be: they are the parameter grid in `pharos.analyst`, one axis moved
at a time.

Needs no model and no network. The corpus is regenerated from `(seed, config)` and
the verdicts come from disk, so this runs in CI and reproduces exactly.

    uv run python scripts/measure_analyst_review.py --out results/analyst_review.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pharos.analyst import (
    DEFAULT_CEILING,
    DEFAULT_ENSEMBLE,
    KEEP_COMPARTMENTS,
    AgreementReport,
    AnalystPolicy,
    Proposal,
    ReleaseRecovery,
    SupervisionYield,
    action_agreement,
    release_recovery,
    review_all,
    supervision_yield,
)
from pharos.generate import GeneratorConfig, generate
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger
from pharos.validity import check_sample_size

LOG = get_logger()

#: The corpus the committed triage artifacts were measured on. Changing either of
#: these without regenerating those artifacts makes every row below describe a
#: different corpus than the one it names.
SEED = 7
EVENTS = 400
TASKS = 40


class ArtifactMismatchError(RuntimeError):
    """A triage artifact does not describe the corpus regenerated here."""


def load_proposals(path: Path, tasks: list[TriageTask]) -> dict[str, Proposal]:
    """Model verdicts from a triage artifact, as proposals over `tasks`.

    Refuses an artifact whose rows disagree with the regenerated corpus on task id
    or on ground truth. A corpus-defining change that lands without a re-sweep would
    otherwise be invisible: the join still succeeds, every count still computes, and
    the numbers silently describe two different worlds. That failure has happened in
    this project once already, which is why `harvest.py` refuses stale artifacts and
    why this refuses them too.

    A row whose verdict is null is dropped rather than coerced. An unparsable answer
    is not a proposal, and counting it as one class would move every number here.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    truth = {task.task_id: task.significant for task in tasks}
    by_id = {task.task_id: task for task in tasks}

    proposals: dict[str, Proposal] = {}
    for row in payload["rows"]:
        task_id = row["task_id"]
        if task_id not in by_id:
            raise ArtifactMismatchError(
                f"{path.name}: task {task_id} is not in the regenerated corpus"
            )
        if row["truth"] != truth[task_id]:
            raise ArtifactMismatchError(
                f"{path.name}: task {task_id} was measured with truth={row['truth']}, "
                f"corpus at seed {SEED} says {truth[task_id]}. Re-run the sweep."
            )
        if row["verdict"] is None:
            continue
        proposals[task_id] = Proposal(
            task_id=task_id,
            verdict=bool(row["verdict"]),
            # The system proposes under the fail-closed default: drop to OPEN,
            # keep every compartment. That is the proposal a reviewer is given,
            # whatever ruling the reviewer would themselves apply.
            release=declassify(by_id[task_id].label, KEEP_COMPARTMENTS),
        )
    return proposals


@dataclass(frozen=True, slots=True)
class Row:
    """One reviewer's line: the parameters that define them, and what they yielded."""

    policy: AnalystPolicy
    yield_: SupervisionYield
    release: ReleaseRecovery

    def as_dict(self) -> dict[str, object]:
        return {
            "analyst": self.policy.name,
            "escalation_threshold": self.policy.escalation_threshold,
            "drop_compartments": self.policy.release_policy.drop_compartments,
            "slip_rate": self.policy.slip_rate,
            "revision_rate": self.policy.revision_rate,
            "names_grounds": self.policy.names_grounds,
            **self.yield_.as_dict(),
            "release": self.release.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModelReview:
    """Every reviewer's reading of one model's proposals."""

    n_proposals: int
    agreement: AgreementReport
    rows: tuple[Row, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_proposals": self.n_proposals,
            "agreement": self.agreement.as_dict(),
            "analysts": [row.as_dict() for row in self.rows],
        }


def measure(
    tasks: list[TriageTask],
    proposals: dict[str, Proposal],
    ensemble: tuple[AnalystPolicy, ...],
) -> ModelReview:
    """Every reviewer's yield over one model's proposals, plus ensemble agreement."""
    truth = {task.task_id: task.significant for task in tasks}
    decisions = review_all(ensemble, tasks, proposals, seed=SEED)

    rows = tuple(
        Row(
            policy=policy,
            yield_=supervision_yield(mine, truth),
            release=release_recovery(mine, proposals, ceiling=DEFAULT_CEILING),
        )
        for policy in ensemble
        if (mine := [d for d in decisions if d.analyst == policy.name]) is not None
    )
    return ModelReview(len(proposals), action_agreement(decisions), rows)


def _print_table(model_key: str, review: ModelReview) -> None:
    print(f"\n{model_key}  ({review.n_proposals} proposals)")
    header = f"  {'analyst':14} {'acc':>4} {'rev':>4} {'rej':>4}"
    print(f"{header} {'targets':>8} {'correct':>8} {'located':>8} {'release':>8}")
    for row in review.rows:
        counts = row.yield_
        print(
            f"  {row.policy.name:14} {counts.accepted:>4} {counts.revised:>4} "
            f"{counts.rejected:>4} {counts.supervised_share:>8.2f} "
            f"{counts.target_accuracy:>8.2f} {counts.located_share:>8.2f} "
            f"{row.release.recovery_rate:>8.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="directory holding the committed triage_lift-*.json artifacts",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    reports = generate(GeneratorConfig(seed=SEED, n_events=EVENTS))
    tasks = build_triage_tasks(reports, limit=TASKS)

    artifacts = sorted(args.results.glob("triage_lift-*.json"))
    if not artifacts:
        print(f"no triage_lift-*.json under {args.results}; run scripts/sweep_models.sh first")
        return 1

    print(f"{len(tasks)} triage tasks at seed {SEED}, {len(DEFAULT_ENSEMBLE)} reviewers")
    print(f"reviewing proposals from {len(artifacts)} model artifacts")

    blocks: dict[str, object] = {}
    for path in artifacts:
        model_key = path.stem.removeprefix("triage_lift-")
        proposals = load_proposals(path, tasks)
        review = measure(tasks, proposals, DEFAULT_ENSEMBLE)
        blocks[model_key] = review.as_dict()
        _print_table(model_key, review)

    validity = check_sample_size(len(tasks) * len(DEFAULT_ENSEMBLE), label="analyst_review")

    print("\n" + "=" * 74)
    print("targets   share of decisions handing the learner a verdict to train on")
    print("correct   share of those targets that match the world")
    print("located   share of objections that say which of verdict/release was wrong")
    print("release   share of unreleasable proposals whose correction clears the ceiling")
    print("=" * 74)

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(seed=SEED),
                    "seed": SEED,
                    "n_events": EVENTS,
                    "n_tasks": len(tasks),
                    "ceiling": (
                        f"{DEFAULT_CEILING.sensitivity.name}"
                        f"[{','.join(sorted(DEFAULT_CEILING.compartments))}]"
                    ),
                    "validity": validity.as_dict(),
                    "models": blocks,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
