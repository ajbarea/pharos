"""Tests for DecisionLedger and ProvenanceRouter (Abstract 2, RQ1)."""

from pharos.analyst import Action, Proposal
from pharos.analyst import Decision as AnalystDecision
from pharos.disclosure import Disposition, Reason, ReleaseDecision
from pharos.labels import Capacity, Compartment, Label, Sensitivity
from pharos.ledger import (
    DecisionLedger,
    DecisionRecord,
    ProvenanceRouter,
    RoutingTarget,
    record_from_review,
)


def test_decision_record_creation_and_hashing():
    proposal = Proposal(
        task_id="TR-0001",
        verdict=True,
        release=Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.ENUM),
    )
    decision = ReleaseDecision(
        disposition=Disposition.RELEASE,
        reason=Reason.RELEASABLE,
        released=Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.ENUM),
    )

    review = AnalystDecision(
        task_id="TR-0001",
        analyst="by-the-book",
        action=Action.ACCEPT,
        grounds=frozenset(),
        reasons=frozenset(),
        corrected_verdict=None,
        corrected_release=None,
    )
    record = record_from_review(
        "TR-0001",
        seed=7,
        proposal=proposal,
        decision=decision,
        review=review,
        truth_significant=True,
    )

    assert record.task_id == "TR-0001"
    assert record.analyst_id == "by-the-book"
    assert len(record.digest()) == 64  # SHA-256 length


def test_provenance_router_splits_personal_vs_shared():
    router = ProvenanceRouter()

    # Restricted record must route to PERSONAL_ONLY
    restricted_rec = DecisionRecord(
        record_id="R1",
        task_id="TR-0001",
        analyst_id="a1",
        seed=7,
        timestamp="2026-08-02T12:00:00+00:00",
        proposed_verdict=True,
        proposed_release="RESTRICTED[SENSOR]",
        proposal_disposition="release",
        proposal_reason="RELEASABLE",
        truth_significant=True,
        action="accept",
        grounds=("verdict",),
        reasons=(),
        corrected_verdict=None,
        corrected_release=None,
    )
    verdict1 = router.route(restricted_rec)
    assert verdict1.target == RoutingTarget.PERSONAL_ONLY
    assert verdict1.is_personal_classified is True

    # General tradecraft record must route to SHARED_FEDERATED
    shared_rec = DecisionRecord(
        record_id="R2",
        task_id="TR-0002",
        analyst_id="a2",
        seed=7,
        timestamp="2026-08-02T12:00:00+00:00",
        proposed_verdict=False,
        proposed_release="OPEN[SENSOR]",
        proposal_disposition="release",
        proposal_reason="RELEASABLE",
        truth_significant=False,
        action="revise",
        grounds=("verdict",),
        reasons=(),
        corrected_verdict=False,
        corrected_release=None,
    )
    verdict2 = router.route(shared_rec)
    assert verdict2.target == RoutingTarget.SHARED_FEDERATED
    assert verdict2.is_tradecraft_pattern is True


def test_ledger_append_and_filter():
    ledger = DecisionLedger()
    rec = DecisionRecord(
        record_id="R1",
        task_id="TR-0001",
        analyst_id="analyst_alpha",
        seed=7,
        timestamp="2026-08-02T12:00:00+00:00",
        proposed_verdict=True,
        proposed_release="OPEN[]",
        proposal_disposition="release",
        proposal_reason="RELEASABLE",
        truth_significant=True,
        action="accept",
        grounds=(),
        reasons=(),
        corrected_verdict=None,
        corrected_release=None,
    )
    ledger.append(rec)
    assert len(ledger) == 1
    assert len(ledger.records_for_analyst("analyst_alpha")) == 1
    assert len(ledger.records_for_analyst("analyst_beta")) == 0
