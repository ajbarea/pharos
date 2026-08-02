#!/usr/bin/env python3
"""Does an example block transmit a *standard*, or only fail to transmit a rule?

Finding 5 supplied labelled examples with the rule withheld and reported that they
closed almost none of the gap to the stated-rule ceiling. Its teacher was the world,
and it scored only against the world, so it could not distinguish two very different
readings of that result:

- The example channel carries nothing. Whatever standard the examples encode, the
  model ignores it and answers from its own prior.
- The example channel carries a standard fine, and finding 5 looked like failure
  only because the standard being taught was the correct one, which is far from the
  model's prior of escalating everything.

They differ in where the risk of finding 8 lives. If examples transmit a standard,
then a fleet using in-context personalization inherits a wrong analyst's standard
through the prompt, with no gradient step involved. If they do not, that risk is
specific to gradient learning, and a prompt-based deployment is exposed to a
different failure entirely.

Telling them apart needs a teacher who is *wrong in a known direction* and a second
scoring against that teacher's own answers. Both are what this adds.

**Not yet run for publication, and finding 9 is why.** This uses the reasoning
decode, which disagrees with itself on 10% of tasks across identical calls. The
effect it looks for -- a shift in agreement with the teacher -- is plausibly of that
size, so a single pass per condition cannot separate the two. It needs repeats and a
reported spread before any number from it is quotable. The instrument is here; the
measurement is not done.

Runs against Ollama, one call per (teacher, shot count, task).

    uv run python scripts/measure_teacher_transfer.py --out results/teacher_transfer.json
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from measure_rule_learnability import balanced_shots, build_prompt, parse_verdict

from pharos.analyst import DEFAULT_ENSEMBLE, AnalystPolicy
from pharos.attribute import DEFAULT_ENDPOINT, DEFAULT_MODEL, generate_text
from pharos.generate import GeneratorConfig, generate
from pharos.models import resolve
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, progress
from pharos.validity import check_classification

LOG = get_logger()

#: Teachers spanning target accuracy, from finding 8's grid. `by-the-book` is the
#: control and reproduces finding 5's condition exactly, since its labels are the
#: world's; the other two are wrong in a known direction and by a known amount.
TEACHERS = ("by-the-book", "two-of-three", "any-one")

SEED = 7


def teacher_labels(tasks: list[TriageTask], policy: AnalystPolicy, *, seed: int) -> dict[str, bool]:
    """Each task as `policy` calls it. Keyed exactly as `AnalystPolicy.review` keys it."""
    return {
        t.task_id: policy.verdict_for(t, random.Random(f"{seed}:{policy.name}:{t.task_id}"))
        for t in tasks
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """One (teacher, shot count) condition, scored against both answer keys.

    Keyword-only: five of the seven fields are counts in the same range, so a
    positional construction that drifts out of order scores a different quantity
    without failing. The same shape shipped a red build in
    `measure_consensus_reliability`.
    """

    teacher: str
    shots: int
    n: int
    unparsed: int
    agree_world: int
    agree_teacher: int
    said_significant: int

    @property
    def world_rate(self) -> float:
        scored = self.n - self.unparsed
        return self.agree_world / scored if scored else 0.0

    @property
    def teacher_rate(self) -> float:
        scored = self.n - self.unparsed
        return self.agree_teacher / scored if scored else 0.0

    @property
    def escalation_rate(self) -> float:
        scored = self.n - self.unparsed
        return self.said_significant / scored if scored else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "teacher": self.teacher,
            "shots": self.shots,
            "n": self.n,
            "unparsed": self.unparsed,
            "agreement_with_world": round(self.world_rate, 4),
            "agreement_with_teacher": round(self.teacher_rate, 4),
            "escalation_rate": round(self.escalation_rate, 4),
        }


def run_condition(
    targets: list[TriageTask],
    pool: list[TriageTask],
    labels: dict[str, bool],
    *,
    teacher: str,
    shots: int,
    model: str,
    endpoint: str,
) -> Row:
    """Score one condition against the world and against the teacher.

    The shot block is drawn from `pool`, which shares no event with `targets`, and is
    balanced by the *teacher's* labels rather than the world's.
    """
    block = balanced_shots(pool, shots, labels) if shots else []
    unparsed = agree_world = agree_teacher = said_significant = 0

    for index, task in enumerate(targets):
        answer = generate_text(
            build_prompt(task, block, labels), endpoint=endpoint, model=model, num_predict=320
        )
        verdict = parse_verdict(answer)
        if verdict is None:
            unparsed += 1
        else:
            said_significant += int(verdict)
            agree_world += int(verdict == task.significant)
            agree_teacher += int(verdict == labels[task.task_id])
        if (index + 1) % 10 == 0:
            progress("teacher_transfer.progress", teacher=teacher, shots=shots, done=index + 1)

    return Row(
        teacher=teacher,
        shots=shots,
        n=len(targets),
        unparsed=unparsed,
        agree_world=agree_world,
        agree_teacher=agree_teacher,
        said_significant=said_significant,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--events", type=int, default=400)
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    spec = resolve(args.model)
    reports = generate(GeneratorConfig(seed=SEED, n_events=args.events))
    all_tasks = build_triage_tasks(reports)
    targets = all_tasks[: args.tasks]
    pool = all_tasks[args.tasks :]
    leaked = {t.event_id for t in targets} & {t.event_id for t in pool}
    if leaked:
        raise SystemExit(f"{len(leaked)} events leaked between the shot pool and the targets")

    by_name = {p.name: p for p in DEFAULT_ENSEMBLE}
    print(f"{len(targets)} targets, shot pool {len(pool)}, model {spec.tag}")
    print(f"teachers: {', '.join(TEACHERS)}\n")

    rows: list[Row] = []
    for teacher in TEACHERS:
        policy = by_name[teacher]
        labels = teacher_labels(all_tasks, policy, seed=SEED)
        matches = sum(1 for t in targets if labels[t.task_id] == t.significant)
        print(
            f"{teacher}: needs {policy.escalation_threshold} of 3, agrees with the world on {matches}/{len(targets)} targets"
        )
        for shots in args.shots:
            row = run_condition(
                targets,
                pool,
                labels,
                teacher=teacher,
                shots=shots,
                model=spec.tag,
                endpoint=args.endpoint,
            )
            rows.append(row)
            print(
                f"  {shots:>2} shots  world {row.world_rate:.3f}  teacher {row.teacher_rate:.3f}"
                f"  escalated {row.escalation_rate:.3f}  unparsed {row.unparsed}"
            )
        print()

    print("=" * 74)
    print("world     share of answers matching the generator's own truth")
    print("teacher   share matching the reviewer who labelled the examples")
    print("escalated share of answers that were SIGNIFICANT at all")
    print("=" * 74)

    zero = {r.teacher: r for r in rows if r.shots == 0}
    most = max(args.shots)
    for teacher in TEACHERS:
        with_shots = next((r for r in rows if r.teacher == teacher and r.shots == most), None)
        if with_shots is None or teacher not in zero:
            continue
        moved = with_shots.teacher_rate - zero[teacher].teacher_rate
        print(
            f"\n{teacher}: {most} examples moved agreement with the teacher by "
            f"{moved:+.3f} ({zero[teacher].teacher_rate:.3f} -> {with_shots.teacher_rate:.3f})"
        )

    validity = check_classification(
        tp=0, fp=0, tn=0, fn=0, unparsed=sum(r.unparsed for r in rows), label="teacher_transfer"
    )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(
                        model=spec.tag, model_key=spec.key, endpoint=args.endpoint, seed=SEED
                    ),
                    "seed": SEED,
                    "n_targets": len(targets),
                    "shot_counts": args.shots,
                    "teachers": list(TEACHERS),
                    "validity": validity.as_dict(),
                    "rows": [r.as_dict() for r in rows],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
