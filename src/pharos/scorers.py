"""Scoring a triage verdict against a plant signature, and an attribution against its sources.

Two scorers, and the triage one is not accuracy. It reports over- and under-escalation
separately, because they are different failures: finding 3b measures recall at exactly
1.000 in every model tested while precision is what moves, so a single correct/incorrect
number would hide the only variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pharos.telemetry import span

if TYPE_CHECKING:
    from pharos.plants import PlantRegistry
    from pharos.tasks import TriageTask


@dataclass(frozen=True)
class TriageScoreResult:
    """Evaluation result for a triage verdict against ground truth and plant signatures."""

    correct: bool
    over_escalation: bool
    under_escalation: bool
    ground_truth_significant: bool
    predicted_significant: bool
    facts_present: tuple[str, ...]
    facts_required: tuple[str, ...]


@dataclass(frozen=True)
class ProvenanceScoreResult:
    """Evaluation result for source attribution accuracy."""

    total_sources: int
    attributed_sources: int
    recall: float
    exact_match: bool


class SpecialistScorer:
    """Holds the plant registry both scorers read their signatures from."""

    def __init__(self, registry: PlantRegistry | None = None) -> None:
        if registry is None:
            from pharos.plants import default_maritime_registry

            registry = default_maritime_registry()
        self.registry = registry

    def score_triage(
        self,
        task: TriageTask,
        predicted_verdict: bool,
        signature_name: str = "maritime_watch_v1",
    ) -> TriageScoreResult:
        """Scores a triage verdict against plant signature conjunctions."""
        with span(
            "score.triage",
            task_id=task.task_id,
            predicted=predicted_verdict,
            truth=task.significant,
        ):
            sig = self.registry.get_signature(signature_name)
            present_facts = tuple({f for r in task.sources for f in r.fact_ids})
            req_facts = sig.required_facts if sig else ()

            ground_truth = task.significant
            correct = predicted_verdict == ground_truth
            over_escalation = predicted_verdict and not ground_truth
            under_escalation = not predicted_verdict and ground_truth

            return TriageScoreResult(
                correct=correct,
                over_escalation=over_escalation,
                under_escalation=under_escalation,
                ground_truth_significant=ground_truth,
                predicted_significant=predicted_verdict,
                facts_present=present_facts,
                facts_required=req_facts,
            )

    def score_provenance(
        self,
        true_sources: set[str] | tuple[str, ...] | list[str],
        attributed_sources: set[str] | tuple[str, ...] | list[str],
    ) -> ProvenanceScoreResult:
        """Scores source attribution completeness."""
        with span("score.provenance", total=len(true_sources), attributed=len(attributed_sources)):
            true_set = set(true_sources)
            attr_set = set(attributed_sources)

            matched = true_set.intersection(attr_set)
            total = len(true_set)
            recall = len(matched) / total if total > 0 else 1.0
            exact_match = true_set == attr_set

            return ProvenanceScoreResult(
                total_sources=total,
                attributed_sources=len(attr_set),
                recall=recall,
                exact_match=exact_match,
            )
