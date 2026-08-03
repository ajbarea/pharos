"""Provenance-Gated Gradient Router (Abstract 2, RQ1).

Determines whether an analyst's decision record trains:
1. Personal Adapter (Style, Need-to-Know, Personal Priorities) - STAYS LOCAL ON NODE
2. Shared Adapter (Tradecraft, Tool Patterns, Failure Avoidance) - FEDERATES TO FLEET
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pharos.telemetry import record, span

if TYPE_CHECKING:
    from pharos.ledger.record import DecisionRecord


class RoutingTarget(StrEnum):
    PERSONAL_ONLY = "personal_only"
    SHARED_FEDERATED = "shared_federated"
    BOTH = "both"
    DISCARD = "discard"


@dataclass(frozen=True)
class RoutingVerdict:
    """Outcome of routing a decision record to LoRA adapters."""

    target: RoutingTarget
    reason: str
    is_personal_classified: bool
    is_tradecraft_pattern: bool


class ProvenanceRouter:
    """Enforces data provenance boundaries before gradient calculation.

    Checkable in advance, auditable after: decisions carrying RESTRICTED or
    compartmentalized needs-to-know route strictly to PersonalAdapter. General
    tradecraft corrections route to SharedAdapter.
    """

    def route(self, record_data: DecisionRecord) -> RoutingVerdict:
        with span("ledger.route", record_id=record_data.record_id, task_id=record_data.task_id):
            verdict = self._evaluate_routing(record_data)
            record("ledger.routing", 1, target=verdict.target.value, analyst=record_data.analyst_id)
            return verdict

    def _evaluate_routing(self, record: DecisionRecord) -> RoutingVerdict:
        # 1. Check for unparsed or rejected calls without revision
        if record.action == "reject" and record.corrected_verdict is None:
            return RoutingVerdict(
                target=RoutingTarget.DISCARD,
                reason="Bare rejection carries no supervision payload for training",
                is_personal_classified=False,
                is_tradecraft_pattern=False,
            )

        # 2. Check if decision involves RESTRICTED or specific compartment grounds
        is_restricted = "RESTRICTED" in record.proposed_release or (
            record.corrected_release and "RESTRICTED" in record.corrected_release
        )
        has_compartment_reason = "COMPARTMENT_NOT_HELD" in record.reasons

        if is_restricted or has_compartment_reason:
            return RoutingVerdict(
                target=RoutingTarget.PERSONAL_ONLY,
                reason="Carries RESTRICTED sensitivity or compartment need-to-know; personal adapter only",
                is_personal_classified=True,
                is_tradecraft_pattern=False,
            )

        # 3. Check for general tradecraft failure avoidance
        is_tradecraft = record.action in ("accept", "revise") and (
            "verdict" in record.grounds or "release" in record.grounds
        )

        if is_tradecraft:
            return RoutingVerdict(
                target=RoutingTarget.SHARED_FEDERATED,
                reason="General tradecraft pattern; cleared for federated shared adapter",
                is_personal_classified=False,
                is_tradecraft_pattern=True,
            )

        return RoutingVerdict(
            target=RoutingTarget.PERSONAL_ONLY,
            reason="Defaulting to local personal adapter for unclassified feedback",
            is_personal_classified=True,
            is_tradecraft_pattern=False,
        )
