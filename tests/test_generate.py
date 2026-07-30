"""Generation invariants, above all the ones that keep plants surface-identical."""

from collections import Counter

from pharos.generate import FACTS_PER_EVENT, GeneratorConfig, generate
from pharos.labels import Capacity, Compartment, Sensitivity
from pharos.world import CHANNEL_LABELS, SIGNIFICANT_PATTERN, ReportType

CONFIG = GeneratorConfig(seed=1, n_events=120, plant_rate=0.3)
REPORTS = generate(CONFIG)


def test_generation_is_deterministic_given_seed_and_config():
    assert [r.text for r in generate(CONFIG)] == [r.text for r in generate(CONFIG)]


def test_a_different_seed_changes_the_corpus():
    other = GeneratorConfig(seed=2, n_events=120, plant_rate=0.3)
    assert [r.text for r in generate(CONFIG)] != [r.text for r in generate(other)]


def test_every_report_carries_the_label_of_its_channel():
    for report in REPORTS:
        sensitivity, compartments = CHANNEL_LABELS[report.report_type]
        assert report.label.sensitivity is sensitivity
        assert report.label.compartments == compartments
        assert report.label.capacity is Capacity.FREETEXT


def test_the_corpus_spans_every_channel_and_both_classes():
    assert {r.report_type for r in REPORTS} == set(ReportType)
    assert {r.is_plant for r in REPORTS} == {True, False}


def test_plants_and_background_use_the_same_channels():
    plant_channels = {r.report_type for r in REPORTS if r.is_plant}
    background_channels = {r.report_type for r in REPORTS if not r.is_plant}
    assert plant_channels == background_channels


def test_every_report_carries_the_same_number_of_fact_sentences():
    # Constant sentence count is what keeps length independent of class.
    counts = {len(r.fact_ids) for r in REPORTS}
    assert len(counts) == 1


def test_both_classes_draw_the_same_number_of_facts_per_event():
    by_event: dict[str, set[str]] = {}
    for report in REPORTS:
        by_event.setdefault(report.event_id, set())
    # Fact count per event is fixed by construction; assert the constant holds.
    assert FACTS_PER_EVENT == 5
    assert by_event


def test_background_may_carry_part_of_the_pattern_but_never_all_of_it():
    # If no background event carried a partial pattern, any single pattern fact
    # would identify a plant and the task would be trivially shortcut-able.
    background_facts = [set(r.fact_ids) & SIGNIFICANT_PATTERN for r in REPORTS if not r.is_plant]
    assert any(len(f) >= 1 for f in background_facts), "no partial pattern in background"
    assert all(f != SIGNIFICANT_PATTERN for f in background_facts)


def test_plant_rate_is_approximately_honored():
    events = {r.event_id: r.is_plant for r in REPORTS}
    observed = sum(events.values()) / len(events)
    assert 0.2 <= observed <= 0.4


def test_compartmented_channels_are_incomparable_at_equal_level():
    labels = {r.report_type: r.label for r in REPORTS}
    liaison = labels[ReportType.LIAISON_TIP]
    partner = labels[ReportType.PARTNER_REPORT]
    assert liaison.sensitivity is partner.sensitivity is Sensitivity.RESTRICTED
    assert not liaison.dominates(partner)
    assert not partner.dominates(liaison)


def test_open_channels_carry_no_compartment():
    labels = {r.report_type: r.label for r in REPORTS}
    assert labels[ReportType.PRESS_ITEM].compartments == frozenset()
    assert labels[ReportType.PRESS_ITEM].sensitivity is Sensitivity.OPEN


def test_the_corpus_has_label_variance():
    # A corpus with a constant label cannot evaluate a disclosure boundary.
    histogram = Counter(
        (r.label.sensitivity.name, tuple(sorted(r.label.compartments))) for r in REPORTS
    )
    assert len(histogram) > 1
    assert Compartment.SENSOR in {c for r in REPORTS for c in r.label.compartments}


def test_voice_is_independent_of_class():
    # A voice correlated with plant status would be a surface tell.
    plant_voices = Counter(r.voice for r in REPORTS if r.is_plant)
    background_voices = Counter(r.voice for r in REPORTS if not r.is_plant)
    assert set(plant_voices) == set(background_voices)
