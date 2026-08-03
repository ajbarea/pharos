"""Adversarial input perturbation module for Pharos.

Provides perturbation engines for stress-testing model triage & scoring resilience.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pharos.tasks import TriageTask


SYNONYM_MAP: dict[str, tuple[str, ...]] = {
    "deviation": ("course offset", "bearing drift", "heading variance"),
    "draft": ("freeboard", "submerged depth", "hull waterline"),
    "unlit": ("darkened ship", "running lights extinguished", "blackout vessel"),
}


@dataclass(frozen=True)
class AdversarialTask:
    """An adversarial variant of a TriageTask with perturbed reports/facts."""

    original_task_id: str
    perturbation_type: str
    perturbed_reports: tuple[str, ...]
    ground_truth_significant: bool


def apply_lexical_substitution(task: TriageTask, seed: int = 42) -> AdversarialTask:
    """Replaces surface fact terms with domain-valid synonyms."""
    rng = random.Random(seed)  # noqa: S311
    new_reports: list[str] = []

    for text in [r.text for r in task.sources]:
        modified = text
        for term, synonyms in SYNONYM_MAP.items():
            if term in modified.lower():
                replacement = rng.choice(synonyms)
                modified = modified.replace(term, replacement)
        new_reports.append(modified)

    return AdversarialTask(
        original_task_id=task.task_id,
        perturbation_type="lexical_substitution",
        perturbed_reports=tuple(new_reports),
        ground_truth_significant=task.significant,
    )


def apply_decoy_injection(task: TriageTask, seed: int = 42) -> AdversarialTask:
    """Injects high-urgency decoy phrases into non-significant tasks to test over-escalation."""
    rng = random.Random(seed)  # noqa: S311
    urgent_decoys = [
        "URGENT: Visual observation of anomalous maneuver.",
        "IMMEDIATE ACTION: Watch officer notes suspicious communication blackout.",
        "FLASH: Secondary sensor corroborates potential tracking anomaly.",
    ]

    new_reports = [r.text for r in task.sources]
    if not task.significant and new_reports:
        idx = rng.randint(0, len(new_reports) - 1)
        decoy = rng.choice(urgent_decoys)
        new_reports[idx] = new_reports[idx] + " " + decoy

    return AdversarialTask(
        original_task_id=task.task_id,
        perturbation_type="decoy_injection",
        perturbed_reports=tuple(new_reports),
        ground_truth_significant=task.significant,
    )
