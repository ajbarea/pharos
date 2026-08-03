"""Tests for DecisionLedger and ProvenanceRouter."""

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


def _sample_record() -> DecisionRecord:
    """One fully-populated record, so every field is available to mutate below."""
    label = Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.ENUM)
    return record_from_review(
        "TR-0001",
        seed=7,
        proposal=Proposal(task_id="TR-0001", verdict=True, release=label),
        decision=ReleaseDecision(
            disposition=Disposition.RELEASE, reason=Reason.RELEASABLE, released=label
        ),
        review=AnalystDecision(
            task_id="TR-0001",
            analyst="by-the-book",
            action=Action.ACCEPT,
            grounds=frozenset(),
            reasons=frozenset(),
            corrected_verdict=None,
            corrected_release=None,
        ),
        truth_significant=True,
    )


def test_the_digest_is_stable_for_an_unchanged_record():
    """Tamper-evidence needs both halves; this is the half that says nothing moved."""
    record = _sample_record()
    assert record.digest() == record.digest()


def test_every_field_is_covered_by_the_digest():
    """Change any single field and the digest must change.

    The existing check was `len(record.digest()) == 64`, which is a statement about
    SHA-256's output width and passes for a constant. It cannot fail if `digest()`
    silently stops covering a field -- and a field outside the hash is precisely the
    place a record could be altered without the ledger noticing, which is the one
    property a decision ledger exists to provide.

    Iterating over the dataclass rather than naming fields is deliberate: a field added
    later is covered by this test on the day it is added, without anyone remembering to
    extend it.
    """
    import dataclasses

    record = _sample_record()
    baseline = record.digest()
    mutated_any = False

    for field in dataclasses.fields(record):
        current = getattr(record, field.name)
        if isinstance(current, bool):
            altered = not current
        elif isinstance(current, str):
            altered = current + "-tampered"
        elif isinstance(current, int):
            altered = current + 1
        else:
            continue
        changed = dataclasses.replace(record, **{field.name: altered})
        assert changed.digest() != baseline, (
            f"altering {field.name!r} left the digest unchanged; it is outside the hash"
        )
        mutated_any = True

    assert mutated_any, "no field was mutable; this test exercised nothing"
