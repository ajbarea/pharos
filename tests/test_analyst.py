"""The simulated analyst, its parameters, and the statistics over its decisions."""

import random

import pytest

from pharos.analyst import (
    DEFAULT_CEILING,
    DEFAULT_ENSEMBLE,
    DROP_COMPARTMENTS,
    KEEP_COMPARTMENTS,
    Action,
    AnalystPolicy,
    Decision,
    Ground,
    Proposal,
    action_agreement,
    evidence_shown,
    release_recovery,
    review_all,
    supervision_yield,
    with_name,
)
from pharos.generate import GeneratorConfig, generate
from pharos.labels import Capacity, Compartment, Label, Sensitivity, declassify
from pharos.tasks import build_triage_tasks
from pharos.world import SIGNIFICANT_PATTERN


@pytest.fixture(scope="module")
def tasks():
    reports = generate(GeneratorConfig(seed=7, n_events=120))
    return build_triage_tasks(reports, limit=20)


@pytest.fixture(scope="module")
def proposals(tasks):
    """Every verdict wrong, so no reviewer accepts on verdict grounds by accident."""
    return {
        task.task_id: Proposal(
            task_id=task.task_id,
            verdict=not task.significant,
            release=declassify(task.label, KEEP_COMPARTMENTS),
        )
        for task in tasks
    }


def test_evidence_is_read_from_the_reports_not_the_event(tasks):
    for task in tasks:
        shown = evidence_shown(task)
        assert shown <= SIGNIFICANT_PATTERN
        # The corpus fix behind finding 3 is what makes this hold: a significant
        # task renders all three of its defining facts, never a subset.
        assert task.significant == (shown == SIGNIFICANT_PATTERN)


def test_by_the_book_reproduces_the_worlds_own_rule(tasks):
    policy = AnalystPolicy("strict")
    rng_free = random.Random(0)
    for task in tasks:
        assert policy.verdict_for(task, rng_free) == task.significant


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_lower_thresholds_only_ever_escalate_more(tasks, threshold):
    lenient = AnalystPolicy("lenient", escalation_threshold=threshold)
    strict = AnalystPolicy("strict", escalation_threshold=3)
    rng = random.Random(0)
    for task in tasks:
        if strict.verdict_for(task, rng):
            assert lenient.verdict_for(task, rng)


def test_threshold_and_rates_are_validated():
    with pytest.raises(ValueError, match="escalation_threshold"):
        AnalystPolicy("bad", escalation_threshold=0)
    with pytest.raises(ValueError, match="escalation_threshold"):
        AnalystPolicy("bad", escalation_threshold=4)
    with pytest.raises(ValueError, match="slip_rate"):
        AnalystPolicy("bad", slip_rate=1.5)
    with pytest.raises(ValueError, match="revision_rate"):
        AnalystPolicy("bad", revision_rate=-0.1)


def test_a_correct_proposal_is_accepted(tasks):
    policy = AnalystPolicy("strict")
    releasable = [t for t in tasks if policy.permits(declassify(t.label, KEEP_COMPARTMENTS))]
    assert releasable, "fixture must contain at least one releasable task"
    task = releasable[0]
    proposal = Proposal(task.task_id, task.significant, declassify(task.label, KEEP_COMPARTMENTS))
    decision = policy.review(task, proposal, seed=1)
    assert decision.action is Action.ACCEPT
    assert decision.grounds == frozenset()
    assert decision.corrected_verdict == task.significant
    assert decision.is_supervised


def test_grounds_separate_a_wrong_verdict_from_an_unreleasable_label(tasks):
    policy = AnalystPolicy("strict")
    blocked = [t for t in tasks if not policy.permits(declassify(t.label, KEEP_COMPARTMENTS))]
    assert blocked, "fixture must contain at least one unreleasable task"
    task = blocked[0]

    both = policy.review(
        task,
        Proposal(task.task_id, not task.significant, declassify(task.label, KEEP_COMPARTMENTS)),
        seed=1,
    )
    assert both.grounds == {Ground.VERDICT, Ground.RELEASE}

    release_only = policy.review(
        task,
        Proposal(task.task_id, task.significant, declassify(task.label, KEEP_COMPARTMENTS)),
        seed=1,
    )
    assert release_only.grounds == {Ground.RELEASE}

    open_release = Label(Sensitivity.OPEN, frozenset(), Capacity.ENUM)
    verdict_only = policy.review(
        task, Proposal(task.task_id, not task.significant, open_release), seed=1
    )
    assert verdict_only.grounds == {Ground.VERDICT}


def test_unexplained_objections_disclose_nothing_but_decide_the_same(tasks, proposals):
    named = AnalystPolicy("named")
    silent = AnalystPolicy("silent", names_grounds=False)
    for task in tasks:
        a = named.review(task, proposals[task.task_id], seed=3)
        b = silent.review(task, proposals[task.task_id], seed=3)
        assert a.action is b.action
        assert a.corrected_verdict == b.corrected_verdict
        assert b.grounds == frozenset()


def test_never_revising_leaves_no_correction(tasks, proposals):
    """With escalation off, a reviewer who never revises supplies nothing at all."""
    policy = AnalystPolicy("stonewall", revision_rate=0.0, escalates=False)
    decisions = [policy.review(t, proposals[t.task_id], seed=5) for t in tasks]
    assert all(d.action is Action.REJECT for d in decisions)
    assert all(d.corrected_verdict is None for d in decisions)
    assert all(not d.is_supervised for d in decisions)


def test_escalation_survives_a_reviewer_who_never_revises(tasks, proposals):
    """Routing a case upward is not the same work as writing a correction.

    A reviewer who cannot be bothered to revise can still say "not mine to
    decide", so the revision rate must not gate escalation. Gating it there was
    what made the third door look less useful than it is.
    """
    policy = AnalystPolicy("terse", revision_rate=0.0)
    decisions = [policy.review(t, proposals[t.task_id], seed=5) for t in tasks]
    escalations = [d for d in decisions if d.is_escalation]
    assert escalations, "some proposal must be blocked and authorizable"
    assert all(d.corrected_verdict is not None for d in escalations)


def test_the_revision_draw_does_not_shift_the_verdict_draw(tasks, proposals):
    """Turning grounds off must not silently move the slip draws too.

    Both policies consume the revision draw whether or not they use it, so the two
    axes stay independent. Without that, a sweep over `names_grounds` would also be
    a sweep over which tasks the analyst slipped on, and nothing would attribute.
    """
    named = AnalystPolicy("named", slip_rate=0.3)
    silent = AnalystPolicy("named", slip_rate=0.3, names_grounds=False)
    for task in tasks:
        a = named.review(task, proposals[task.task_id], seed=11)
        b = silent.review(task, proposals[task.task_id], seed=11)
        assert a.corrected_verdict == b.corrected_verdict


def test_decisions_do_not_depend_on_review_order(tasks, proposals):
    policy = AnalystPolicy("strict")
    forward = [policy.review(t, proposals[t.task_id], seed=9) for t in tasks]
    backward = [policy.review(t, proposals[t.task_id], seed=9) for t in reversed(tasks)]
    assert forward == list(reversed(backward))


def test_slipping_flips_the_call_rather_than_abstaining(tasks):
    always = AnalystPolicy("always-slips", slip_rate=1.0)
    rng = random.Random(0)
    for task in tasks:
        assert always.verdict_for(task, rng) is not task.significant


def test_review_all_skips_tasks_without_a_proposal(tasks, proposals):
    partial = dict(list(proposals.items())[:5])
    decisions = review_all(DEFAULT_ENSEMBLE, tasks, partial, seed=1)
    assert len(decisions) == 5 * len(DEFAULT_ENSEMBLE)
    assert {d.task_id for d in decisions} == set(partial)


def test_supervision_yield_scores_targets_against_the_world_not_the_reviewer(tasks, proposals):
    truth = {t.task_id: t.significant for t in tasks}
    strict = AnalystPolicy("strict")
    alarmist = AnalystPolicy("alarmist", escalation_threshold=1)

    strict_yield = supervision_yield(
        [strict.review(t, proposals[t.task_id], seed=2) for t in tasks], truth
    )
    alarmist_yield = supervision_yield(
        [alarmist.review(t, proposals[t.task_id], seed=2) for t in tasks], truth
    )

    assert strict_yield.target_accuracy == 1.0
    # The reviewer is confident either way; only the world decides who was right.
    assert alarmist_yield.supervised_share == 1.0
    assert alarmist_yield.target_accuracy < 1.0


def test_empty_yield_reports_zero_rather_than_dividing():
    empty = supervision_yield([], {})
    assert empty.supervised_share == 0.0
    assert empty.located_share == 0.0
    assert empty.target_accuracy == 0.0
    assert empty.as_dict()["n_decisions"] == 0


def test_release_recovery_counts_only_proposals_that_were_blocked(tasks, proposals):
    keeper = AnalystPolicy("keeper")
    releaser = AnalystPolicy("releaser", release_policy=DROP_COMPARTMENTS)

    blocked = sum(1 for t in tasks if not DEFAULT_CEILING.dominates(proposals[t.task_id].release))
    assert blocked, "fixture must contain at least one unreleasable proposal"

    kept = release_recovery(
        [keeper.review(t, proposals[t.task_id], seed=4) for t in tasks], proposals
    )
    dropped = release_recovery(
        [releaser.review(t, proposals[t.task_id], seed=4) for t in tasks], proposals
    )

    assert kept.objections == dropped.objections == blocked
    assert kept.releasable_after == 0
    assert kept.recovery_rate == 0.0
    assert dropped.recovery_rate == 1.0


def test_release_recovery_ignores_decisions_with_no_proposal():
    stray = Decision("T-9999", "ghost", Action.REJECT)
    assert release_recovery([stray], {}).objections == 0


def test_release_recovery_needs_a_correction_to_recover_anything(tasks, proposals):
    silent = AnalystPolicy("silent", revision_rate=0.0, release_policy=DROP_COMPARTMENTS)
    recovery = release_recovery(
        [silent.review(t, proposals[t.task_id], seed=4) for t in tasks], proposals
    )
    assert recovery.objections > 0
    assert recovery.corrections == 0
    assert recovery.releasable_after == 0


def test_agreement_is_high_when_reviewers_share_a_standard(tasks, proposals):
    twins = (AnalystPolicy("a"), AnalystPolicy("b"), AnalystPolicy("c"))
    report = action_agreement(review_all(twins, tasks, proposals, seed=6))
    assert report.n_raters == 3
    assert report.observed == 1.0


def test_agreement_falls_when_thresholds_differ(tasks, proposals):
    twins = (AnalystPolicy("a"), AnalystPolicy("b"), AnalystPolicy("c"))
    mixed = (
        AnalystPolicy("a"),
        AnalystPolicy("b", escalation_threshold=1),
        AnalystPolicy("c", revision_rate=0.0),
    )
    same = action_agreement(review_all(twins, tasks, proposals, seed=6))
    apart = action_agreement(review_all(mixed, tasks, proposals, seed=6))
    assert apart.kappa < same.kappa


def test_unanimity_reports_kappa_zero_rather_than_perfect(tasks, proposals):
    """Every rater in one category leaves the chance term at 1 and kappa undefined.

    Forced by giving both raters a releasable ceiling, so no proposal is contested
    on release and every decision is the same category.
    """
    open_ceiling = Label(Sensitivity.RESTRICTED, frozenset(Compartment), Capacity.FREETEXT)
    twins = (
        AnalystPolicy("a", release_ceiling=open_ceiling),
        AnalystPolicy("b", release_ceiling=open_ceiling),
    )
    report = action_agreement(review_all(twins, tasks, proposals, seed=6))
    assert report.expected == 1.0
    assert report.kappa == 0.0


def test_agreement_degenerates_safely():
    assert action_agreement([]).n_items == 0
    lone = [Decision("T-0000", "solo", Action.ACCEPT)]
    assert action_agreement(lone).kappa == 0.0


def test_agreement_drops_items_with_a_ragged_panel(tasks, proposals):
    full = review_all((AnalystPolicy("a"), AnalystPolicy("b")), tasks, proposals, seed=6)
    ragged = [*full, Decision(tasks[0].task_id, "c", Action.REJECT)]
    assert action_agreement(ragged).n_items == len(tasks) - 1


def test_default_ensemble_moves_one_axis_at_a_time():
    baseline = DEFAULT_ENSEMBLE[0]
    assert baseline.name == "by-the-book"
    fields = (
        "escalation_threshold",
        "release_policy",
        "slip_rate",
        "revision_rate",
        "names_grounds",
        "escalates",
    )
    for policy in DEFAULT_ENSEMBLE[1:]:
        differing = [f for f in fields if getattr(policy, f) != getattr(baseline, f)]
        assert len(differing) == 1, f"{policy.name} differs on {differing}"


def test_with_name_renames_and_changes_nothing_else():
    original = AnalystPolicy("first", escalation_threshold=2, slip_rate=0.4)
    renamed = with_name(original, "second")
    assert renamed.name == "second"
    assert renamed.escalation_threshold == original.escalation_threshold
    assert renamed.slip_rate == original.slip_rate


def test_decision_serialises_its_label_and_grounds(tasks):
    policy = AnalystPolicy("strict", release_policy=DROP_COMPARTMENTS)
    task = next(t for t in tasks if t.label.compartments)
    decision = policy.review(
        task,
        Proposal(
            task.task_id,
            not task.significant,
            Label(Sensitivity.RESTRICTED, frozenset({Compartment.PARTNER}), Capacity.ENUM),
        ),
        seed=1,
    )
    payload = decision.as_dict()
    assert payload["action"] == "revise"
    assert payload["grounds"] == ["release", "verdict"]
    assert payload["corrected_release"] == "OPEN[]@ENUM"


def test_accepted_decision_serialises_without_a_correction():
    bare = Decision("T-0001", "someone", Action.REJECT)
    payload = bare.as_dict()
    assert payload["corrected_release"] is None
    assert payload["grounds"] == []
