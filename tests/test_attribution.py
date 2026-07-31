"""Detection, task construction, and the labelling mechanism.

The most important test here is `test_leave_one_out_under_attributes_a_corroborated_fact`,
which encodes the finding that ruled out ablation-based labelling. It uses a fake
model so the finding stays verified without needing a running Ollama.
"""

import pytest

from pharos.attribute import (
    Attribution,
    label_by_provenance,
    truly_contributing,
)
from pharos.detect import BOILERPLATE, detect_facts, detector_accuracy, unmatched_terms
from pharos.generate import GeneratorConfig, generate
from pharos.labels import Capacity, Compartment, Label, Sensitivity
from pharos.tasks import build_tasks
from pharos.world import ReportType

REPORTS = generate(GeneratorConfig(seed=5, n_events=200))
TASKS = build_tasks(REPORTS, limit=4)


# --- detection ------------------------------------------------------------


def test_every_fact_has_detector_terms():
    # A fact with no terms is invisible to attribution, which silently understates
    # how many sources fed a turn.
    assert unmatched_terms() == ()


def test_the_detector_is_exact_on_the_corpus():
    accuracy = detector_accuracy(REPORTS)
    assert accuracy.recall == 1.0
    assert accuracy.precision == 1.0
    assert accuracy.f1 == 1.0


def test_boilerplate_does_not_register_as_an_assertion():
    # "general cargo" is a hull type and "flag Panama" a header field; neither is a
    # claim the report made. An earlier detector scored precision 0.84 on exactly this.
    header = "Vessel Report R-1. Kestrel Dawn, general cargo, flag Panama. Operator X."
    assert detect_facts(header) == frozenset()


def test_boilerplate_list_covers_every_channel_name():
    for report_type in ReportType:
        phrase = report_type.value.replace("_", " ").lower()
        assert phrase in BOILERPLATE, f"{phrase} would leak into detection"


def test_detection_finds_a_paraphrase():
    assert "draft_mismatch" in detect_facts("The draft was well under the declared figure.")
    assert "unlit_contact" in detect_facts("A vessel showing no lights approached.")


# --- tasks ----------------------------------------------------------------


def test_tasks_have_the_requested_source_count():
    assert all(len(task.sources) == 8 for task in TASKS)


def test_every_task_covers_one_vessel_only():
    for task in TASKS:
        assert {r.vessel_name for r in task.sources} == {task.vessel_name}


def test_the_prompt_numbers_every_source():
    task = TASKS[0]
    for index in range(len(task.sources)):
        assert f"[{index + 1}]" in task.prompt


def test_dropping_a_source_renumbers_without_a_gap():
    """A gap in the numbering tells the model a source was withheld.

    That changes its behaviour and contaminates the ablation, so the ablated
    prompt must look like a prompt that never had the dropped source.
    """
    task = TASKS[0]
    ablated = task.prompt_without(2)
    assert f"[{len(task.sources)}]" not in ablated
    for index in range(len(task.sources) - 1):
        assert f"[{index + 1}]" in ablated
    assert task.sources[2].text not in ablated


def test_label_over_joins_the_named_sources():
    task = TASKS[0]
    everything = task.label_over(frozenset(range(len(task.sources))), capacity=Capacity.FREETEXT)
    for report in task.sources:
        assert everything.dominates(report.label)


def test_label_over_nothing_is_the_bottom_of_the_lattice():
    label = TASKS[0].label_over(frozenset(), capacity=Capacity.ENUM)
    assert label == Label(Sensitivity.OPEN, frozenset(), Capacity.ENUM)


def test_sources_containing_agrees_with_the_recorded_facts():
    task = TASKS[0]
    for index, report in enumerate(task.sources):
        for fact_id in report.fact_ids:
            assert index in task.sources_containing(fact_id)


def test_build_tasks_skips_vessels_with_too_few_reports():
    # Padding across hulls would leave a summary with no coherent ground truth.
    tasks = build_tasks(REPORTS, sources_per_task=8)
    assert all(len(t.sources) == 8 for t in tasks)


# --- the labelling mechanism ---------------------------------------------


def _task_with_redundancy():
    """A task where some asserted fact is carried by more than one source."""
    for task in build_tasks(REPORTS, limit=40):
        for report in task.sources:
            for fact_id in report.fact_ids:
                if len(task.sources_containing(fact_id)) > 2:
                    return task, fact_id
    pytest.skip("no redundant fact in the sampled tasks")


def test_truly_contributing_includes_every_candidate_source():
    task, fact_id = _task_with_redundancy()
    contributing = truly_contributing(task, frozenset({fact_id}))
    assert contributing == task.sources_containing(fact_id)
    assert len(contributing) > 2


def test_label_by_provenance_is_conservative_over_a_redundant_fact():
    """The join must cover every source that could have supplied the fact.

    Without token-level provenance there is no way to know which copy the model
    read, so covering all of them is the tightest safe reading.
    """
    task, fact_id = _task_with_redundancy()
    label = label_by_provenance(task, frozenset({fact_id}), capacity=Capacity.FREETEXT)
    for index in task.sources_containing(fact_id):
        assert label.dominates(task.sources[index].label)


def test_leave_one_out_under_attributes_a_corroborated_fact():
    """The finding that ruled out ablation-based labelling, as a regression test.

    Leave-one-out asks which single source is load-bearing. A fact carried by
    three sources has none: drop any one and the fact survives in the others, so
    no source is blamed, none of their labels enters the join, and the resulting
    label is laxer than the truth. That is the leak direction.
    """
    task, fact_id = _task_with_redundancy()
    carriers = task.sources_containing(fact_id)
    asserted = frozenset({fact_id})

    # Simulate the sweep: with a redundant fact, no single drop loses it.
    attributed_by_loo: set[int] = set()
    for index in range(len(task.sources)):
        survives = bool(carriers - {index})
        if not survives:
            attributed_by_loo.add(index)

    loo = Attribution(
        task_id=task.task_id,
        summary="(simulated)",
        asserted_facts=asserted,
        attributed_sources=frozenset(attributed_by_loo),
        truly_contributing=truly_contributing(task, asserted),
        calls=1 + len(task.sources),
    )
    assert loo.attributed_sources == frozenset(), "a redundant fact should blame nobody"
    assert loo.source_recall == 0.0
    assert loo.label_outcome(task) in {"leak", "incomparable"}


def test_label_outcome_names_the_three_directions():
    task = TASKS[0]
    truth = truly_contributing(task, frozenset(task.sources[0].fact_ids))

    exact = Attribution("t", "s", frozenset(), truth, truth, 0)
    assert exact.label_outcome(task) == "exact"

    everything = frozenset(range(len(task.sources)))
    over = Attribution("t", "s", frozenset(), everything, truth, 0)
    assert over.label_outcome(task) in {"exact", "creep"}

    under = Attribution("t", "s", frozenset(), frozenset(), truth, 0)
    assert under.label_outcome(task) in {"exact", "leak", "incomparable"}


def test_recall_and_precision_are_one_when_nothing_contributed():
    empty = Attribution("t", "s", frozenset(), frozenset(), frozenset(), 0)
    assert empty.source_recall == 1.0
    assert empty.source_precision == 1.0


def test_capacity_decides_whether_a_join_can_ever_be_released():
    """The bimodal result, in miniature.

    The same sources produce the same join, and only the output's capacity decides
    whether it may cross. Prose cannot; a verdict can, and only once a policy
    allows compartments to be shed.
    """
    from pharos.labels import DeclassificationPolicy, shared_eligible

    task = TASKS[0]
    sources = frozenset(range(len(task.sources)))
    ceiling = Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)

    prose = task.label_over(sources, capacity=Capacity.FREETEXT)
    verdict = task.label_over(sources, capacity=Capacity.ENUM)

    assert not shared_eligible(prose, ceiling, DeclassificationPolicy())
    assert not shared_eligible(verdict, ceiling, DeclassificationPolicy())
    assert shared_eligible(verdict, ceiling, DeclassificationPolicy(drop_compartments=True))


# --- Task accessors that carry governance decisions -------------------------
# TriageTask.label is the join that decides whether a verdict may federate, so it
# gets tested against the lattice directly rather than trusted. The web explorer
# used to recompute this inline; it now calls this property, and these tests are
# what keep the two from diverging again.


def test_a_verdicts_label_is_the_join_of_its_sources_at_enum_capacity():
    from pharos.labels import Capacity, join
    from pharos.tasks import build_triage_tasks

    reports = generate(GeneratorConfig(seed=4, n_events=60))
    task = build_triage_tasks(reports)[0]
    expected = join([r.label for r in task.sources], capacity=Capacity.ENUM)
    assert task.label == expected
    # ENUM specifically: a one-word verdict is the low-capacity case that policy may
    # declassify. FREETEXT here would make every verdict unreleasable.
    assert task.label.capacity is Capacity.ENUM


def test_a_verdicts_label_dominates_every_source_label():
    """Conservative by construction: the join must be readable by nobody less cleared."""
    from pharos.tasks import build_triage_tasks

    reports = generate(GeneratorConfig(seed=4, n_events=60))
    for task in build_triage_tasks(reports, limit=6):
        for report in task.sources:
            assert task.label.sensitivity >= report.label.sensitivity
            assert report.label.compartments <= task.label.compartments


def test_source_labels_matches_the_sources():
    """`source_labels` is on the summarization Task, which is what attribution uses."""
    from pharos.tasks import build_tasks

    reports = generate(GeneratorConfig(seed=4, n_events=120))
    task = build_tasks(reports, limit=1)[0]
    assert task.source_labels == tuple(r.label for r in task.sources)


def test_facts_by_source_indexes_every_source():
    from pharos.tasks import build_tasks

    reports = generate(GeneratorConfig(seed=4, n_events=120))
    task = build_tasks(reports, limit=1)[0]
    mapping = task.facts_by_source
    assert set(mapping) == set(range(len(task.sources)))
    for index, facts in mapping.items():
        assert facts == frozenset(task.sources[index].fact_ids)


def test_label_over_a_subset_is_never_more_restrictive_than_the_whole():
    """Monotonicity of the join: attributing fewer sources can only under-protect."""
    from pharos.labels import Capacity
    from pharos.tasks import build_tasks

    reports = generate(GeneratorConfig(seed=4, n_events=120))
    task = build_tasks(reports, limit=1)[0]
    whole = frozenset(range(len(task.sources)))
    subset = frozenset(list(whole)[:2])
    full_label = task.label_over(whole, capacity=Capacity.ENUM)
    part_label = task.label_over(subset, capacity=Capacity.ENUM)
    assert full_label.sensitivity >= part_label.sensitivity
    assert part_label.compartments <= full_label.compartments


def test_build_triage_tasks_honours_its_limit():
    from pharos.tasks import build_triage_tasks

    reports = generate(GeneratorConfig(seed=4, n_events=120))
    assert len(build_triage_tasks(reports, limit=3)) == 3
    assert len(build_triage_tasks(reports)) > 3


# --- Detector accuracy reporting -------------------------------------------


def test_detector_f1_is_zero_rather_than_undefined_when_it_finds_nothing():
    """A totally failed detector must report 0.0, not raise ZeroDivisionError."""
    from pharos.detect import DetectorAccuracy

    dead = DetectorAccuracy(n_reports=10, recall=0.0, precision=0.0)
    assert dead.f1 == 0.0


def test_detector_accuracy_serialises_with_f1():
    from pharos.detect import DetectorAccuracy

    payload = DetectorAccuracy(n_reports=8, recall=1.0, precision=0.5).as_dict()
    assert payload == {"n_reports": 8, "recall": 1.0, "precision": 0.5, "f1": 0.6667}
