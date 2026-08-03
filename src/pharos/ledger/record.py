"""Decision Ledger: Auditable interaction tuples for edge analyst triage turns.

Captures analyst accept, revise, reject, and escalate decisions as structured,
cryptographically hashable records for local logging and provenance routing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pharos.provenance import run_provenance

if TYPE_CHECKING:
    from pharos.analyst import Decision, Proposal
    from pharos.disclosure import ReleaseDecision
    from pharos.labels import Label


@dataclass(frozen=True)
class DecisionRecord:
    """One auditable decision tuple logged by an edge agent daemon."""

    record_id: str
    task_id: str
    analyst_id: str
    seed: int
    timestamp: str
    proposed_verdict: bool | None
    proposed_release: str
    proposal_disposition: str
    proposal_reason: str
    truth_significant: bool
    action: str
    grounds: tuple[str, ...]
    reasons: tuple[str, ...]
    corrected_verdict: bool | None
    corrected_release: str | None
    provenance_stamp: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        """SHA-256 cryptographic hash of the record's payload for chain integrity."""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Serialize record to JSON string."""
        return json.dumps(asdict(self), indent=2)


def _render_label(label: Label | None) -> str | None:
    if label is None:
        return None
    comps = ",".join(sorted(str(c) for c in label.compartments))
    return f"{label.sensitivity.name}[{comps}]"


def record_from_review(
    task_id: str,
    seed: int,
    proposal: Proposal,
    decision: ReleaseDecision,
    review: Decision,
    truth_significant: bool,
) -> DecisionRecord:
    """Construct an auditable DecisionRecord from a Pharos proposal and analyst review."""
    rec_id = f"REC-{seed}-{task_id}-{review.analyst}"
    return DecisionRecord(
        record_id=rec_id,
        task_id=task_id,
        analyst_id=review.analyst,
        seed=seed,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        proposed_verdict=proposal.verdict,
        proposed_release=_render_label(proposal.release) or "",
        proposal_disposition=decision.disposition.value
        if hasattr(decision.disposition, "value")
        else str(decision.disposition),
        proposal_reason=decision.reason.value
        if hasattr(decision.reason, "value")
        else str(decision.reason),
        truth_significant=truth_significant,
        action=review.action.value if hasattr(review.action, "value") else str(review.action),
        grounds=tuple(str(g) for g in review.grounds),
        reasons=tuple(str(r) for r in review.reasons),
        corrected_verdict=review.corrected_verdict,
        corrected_release=_render_label(review.corrected_release),
        provenance_stamp=run_provenance(),
    )


class DecisionLedger:
    """In-memory or persistent decision ledger for an edge analyst node."""

    def __init__(self) -> None:
        self._records: list[DecisionRecord] = []

    def append(self, record: DecisionRecord) -> None:
        """Append an audited record to the ledger."""
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> DecisionRecord:
        return self._records[index]

    def records_for_analyst(self, analyst_id: str) -> list[DecisionRecord]:
        """Filter records belonging to a specific analyst."""
        return [r for r in self._records if r.analyst_id == analyst_id]

    def export_jsonl(self) -> str:
        """Export ledger as JSON Lines format."""
        return "\n".join(r.to_json() for r in self._records) + ("\n" if self._records else "")
