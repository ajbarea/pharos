"""Prompt construction for the triage task, and reading a verdict back out.

Four measurements put the same task in front of a model: the in-context rule-learnability
sweep, the multi-model sweep, the decode-stability probe, and the adapter trainer. They
have to share one prompt builder, or a comparison between two of them is a comparison
between two prompt formats.

That was true and the code was still in a script, imported by the other three by path --
including the trainer, which runs on a cluster and so could least afford an import that
depends on a working directory.

The instruction deliberately states no rule. What the model must infer is exactly what
this testbed measures: the standard a watch officer applies, from worked examples rather
than from a specification.
"""

from pharos.tasks import TriageTask

__all__ = [
    "EXAMPLE_PREAMBLE",
    "INSTRUCTION_NO_RULE",
    "balanced_shots",
    "build_prompt",
    "parse_verdict",
    "reports_block",
]


INSTRUCTION_NO_RULE = (
    "You are a maritime watch officer. The numbered reports above all concern one "
    "vessel during one watch window. Decide whether they indicate a SIGNIFICANT event "
    "requiring escalation, or ROUTINE activity.\n\n"
    "Reason briefly about what the reports show, then finish with a line reading "
    "exactly 'VERDICT: SIGNIFICANT' or 'VERDICT: ROUTINE'."
)

EXAMPLE_PREAMBLE = (
    "Here are previous cases with the watch officer's own verdict. Infer the standard "
    "the officer applies, then apply that same standard to the new case.\n"
)


def reports_block(task: TriageTask) -> str:
    return "\n\n".join(f"[{i + 1}] {r.text}" for i, r in enumerate(task.sources))


def build_prompt(
    target: TriageTask, shots: list[TriageTask], labels: dict[str, bool] | None = None
) -> str:
    """The target case, preceded by `shots` worked examples and no stated rule.

    `labels` overrides what verdict each example carries. Omitted, the examples are
    labelled by the world, which is what finding 5 measured. Supplying a reviewer's
    labels asks a different question -- whether the example block transmits a
    *standard* rather than the rule -- and the two must share this function or the
    comparison is between two prompt formats rather than between two teachers.
    """
    parts: list[str] = []
    if shots:
        parts.append(EXAMPLE_PREAMBLE)
        for index, shot in enumerate(shots, start=1):
            call = shot.significant if labels is None else labels[shot.task_id]
            verdict = "SIGNIFICANT" if call else "ROUTINE"
            parts.append(
                f"--- CASE {index} ---\n{reports_block(shot)}\nOFFICER'S VERDICT: {verdict}\n"
            )
        parts.append("--- NEW CASE ---")
    parts.append(reports_block(target))
    parts.append(INSTRUCTION_NO_RULE)
    return "\n\n".join(parts)


def parse_verdict(text: str) -> bool | None:
    upper = text.upper()
    tail = upper.split("VERDICT")[-1] if "VERDICT" in upper else upper
    said_significant = "SIGNIFICANT" in tail
    said_routine = "ROUTINE" in tail
    if said_significant == said_routine:
        return None
    return said_significant


def balanced_shots(
    pool: list[TriageTask], k: int, labels: dict[str, bool] | None = None
) -> list[TriageTask]:
    """`k` examples alternating between classes, so the block teaches the rule not the prior.

    Balanced by `labels` when given, not by the world. A block balanced against the
    world but labelled by a lenient reviewer would be lopsided *as the reviewer sees
    it*, and the model would read the prior rather than the standard -- which is the
    confound this whole comparison exists to avoid.
    """
    call = (lambda t: t.significant) if labels is None else (lambda t: labels[t.task_id])
    positives = [t for t in pool if call(t)]
    negatives = [t for t in pool if not call(t)]
    shots: list[TriageTask] = []
    while len(shots) < k and (positives or negatives):
        if len(shots) % 2 == 0 and positives:
            shots.append(positives.pop(0))
        elif negatives:
            shots.append(negatives.pop(0))
        elif positives:
            shots.append(positives.pop(0))
    return shots[:k]
