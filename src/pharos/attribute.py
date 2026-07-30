"""Attributing a summary to the sources that produced it, and labelling the result.

This is the load-bearing measurement of the whole project. The governed label on a
ledger entry is the join of the labels of every source that fed the turn, so the
label is only as correct as the attribution behind it. Two ways it can be wrong,
and they are not symmetric:

- **Under-attribution leaks.** Miss a source and its label is missing from the
  join, so the entry looks more releasable than it is. This is the failure that
  sends restricted material into a shared adapter.
- **Over-attribution creeps.** Include a source that contributed nothing and its
  label inflates the join, so the entry is needlessly restricted. Enough of this
  and nothing federates.

Attribution here is **exact leave-one-out**: drop one source, regenerate, and see
which assertions disappear. The literature's cheaper estimators exist to
approximate exactly this, so with a small source count it is worth measuring the
ceiling directly rather than measuring an approximation of it.
"""

import json
import urllib.request
from dataclasses import dataclass, field

from pharos.detect import detect_facts
from pharos.labels import Capacity, Label
from pharos.tasks import Task

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
REQUEST_TIMEOUT_S = 900

#: A summary is a prose artifact, so its capacity is FREETEXT. A triage verdict
#: over the same sources would be ENUM, which is what makes it declassifiable
#: where the summary is not.
SUMMARY_CAPACITY = Capacity.FREETEXT


def generate_text(prompt: str, *, endpoint: str, model: str, num_predict: int = 320) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.0, "seed": 7},
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310 -- fixed local Ollama endpoint
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as fh:  # noqa: S310
        return str(json.load(fh).get("response", ""))


@dataclass(frozen=True, slots=True)
class Attribution:
    """What one turn asserted, and which sources a leave-one-out sweep blames."""

    task_id: str
    summary: str
    asserted_facts: frozenset[str]
    attributed_sources: frozenset[int]
    truly_contributing: frozenset[int]
    calls: int
    per_source_lost_facts: dict[int, frozenset[str]] = field(default_factory=dict)

    @property
    def source_recall(self) -> float:
        """Share of truly contributing sources that attribution found."""
        if not self.truly_contributing:
            return 1.0
        return len(self.attributed_sources & self.truly_contributing) / len(self.truly_contributing)

    @property
    def source_precision(self) -> float:
        if not self.attributed_sources:
            return 1.0
        return len(self.attributed_sources & self.truly_contributing) / len(self.attributed_sources)

    def attributed_label(self, task: Task, *, capacity: Capacity = SUMMARY_CAPACITY) -> Label:
        return task.label_over(self.attributed_sources, capacity=capacity)

    def true_label(self, task: Task, *, capacity: Capacity = SUMMARY_CAPACITY) -> Label:
        return task.label_over(self.truly_contributing, capacity=capacity)

    def label_outcome(self, task: Task, *, capacity: Capacity = SUMMARY_CAPACITY) -> str:
        """How the attributed label compares with the true one.

        `exact` means they agree. `creep` means the attributed label is stricter
        than the truth, which costs federation but is safe. `leak` means it is
        laxer, which is the outcome that puts restricted material into a shared
        adapter, and it is the only one that must never occur.
        """
        attributed = self.attributed_label(task, capacity=capacity)
        truth = self.true_label(task, capacity=capacity)
        if attributed == truth:
            return "exact"
        if attributed.dominates(truth):
            return "creep"
        if truth.dominates(attributed):
            return "leak"
        return "incomparable"


def label_by_provenance(
    task: Task, asserted: frozenset[str], *, capacity: Capacity = SUMMARY_CAPACITY
) -> Label:
    """The governed label from content provenance, with no ablation and no model calls.

    This is the mechanism leave-one-out attribution should be replaced by, and the
    reason is measured rather than argued. LOO asks which single source is
    load-bearing, and when a fact is corroborated across several sources the answer
    is *none of them*: drop any one and the fact survives in the others. Measured
    on this corpus, that produced a source recall of 0.62 and a wrong label on half
    of all turns, always in the under-restrictive direction, which is the direction
    that puts restricted material into a shared adapter.

    Content provenance asks a different and answerable question. Given what the
    output asserts, which sources *could* have asserted it? Join their labels. No
    ablation, no surrogate, no Jacobian, and no per-turn model cost beyond one
    detection pass over the output.

    It is conservative by construction. A corroborated fact pulls in every source
    carrying it, so the join can only be at or above the truth, which means the
    error direction is creep and never leak. That asymmetry is the whole point:
    creep costs federation, leak costs the boundary.

    The cost is that it over-restricts when a fact appears in both an open and a
    restricted source and the model in fact read the open one. Distinguishing those
    would need token-level provenance, which no available method supplies, so the
    conservative reading is the tightest safe one.
    """
    return task.label_over(truly_contributing(task, asserted), capacity=capacity)


def truly_contributing(task: Task, asserted: frozenset[str]) -> frozenset[int]:
    """Sources that could have produced any asserted fact.

    A fact appearing in several sources makes every one of them a candidate, which
    is the honest ground truth: without token-level provenance there is no way to
    say which copy the model read, and a label join over all candidates is the
    conservative reading the design needs.
    """
    contributing: set[int] = set()
    for fact_id in asserted:
        contributing |= task.sources_containing(fact_id)
    return frozenset(contributing)


def attribute_leave_one_out(
    task: Task,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    baseline: str | None = None,
) -> Attribution:
    """Attribute a summary by dropping each source in turn and regenerating.

    Costs one call for the baseline plus one per source. A source is attributed
    when removing it costs the summary at least one asserted fact.
    """
    summary = (
        baseline
        if baseline is not None
        else generate_text(task.prompt, endpoint=endpoint, model=model)
    )
    asserted = detect_facts(summary)

    attributed: set[int] = set()
    lost: dict[int, frozenset[str]] = {}
    calls = 1
    for index in range(len(task.sources)):
        ablated = generate_text(task.prompt_without(index), endpoint=endpoint, model=model)
        calls += 1
        missing = asserted - detect_facts(ablated)
        if missing:
            attributed.add(index)
            lost[index] = missing

    return Attribution(
        task_id=task.task_id,
        summary=summary,
        asserted_facts=asserted,
        attributed_sources=frozenset(attributed),
        truly_contributing=truly_contributing(task, asserted),
        calls=calls,
        per_source_lost_facts=lost,
    )
