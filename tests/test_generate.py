"""Generation invariants, above all the ones that keep plants surface-identical."""

from collections import Counter
from typing import Any

import pytest

from pharos.generate import FACTS_PER_EVENT, GeneratorConfig, Report, generate
from pharos.labels import Capacity, Compartment, Sensitivity
from pharos.tasks import build_triage_tasks
from pharos.world import (
    CHANNEL_LABELS,
    DECOY_PATTERNS,
    FACTS_BY_ID,
    SIGNIFICANT_PATTERN,
    ReportType,
)

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


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 101])
def test_plants_and_background_use_the_same_channels(seed):
    """No channel is available to one class and closed to the other.

    Sized so the invariant is observable rather than lucky. The rarest channel for a
    plant is LIAISON_TIP at about 1.9% of plant reports, so `CONFIG`'s 120 events gave
    it an expected count near 2 and this assertion turned on whether a two-in-a-hundred
    draw happened to land: it passes at seeds 2, 7, 13 and 101 and fails at 1 and 3,
    which is a property of the seed and not of the generator. At 600 events the same
    channel is expected about 11 times and the invariant holds at every seed here.

    What the assertion does *not* say is that the channel mix is class-independent. It
    is not, by construction and by a wide margin -- SENSOR_TRACK is roughly 45% of plant
    reports against 24% of background -- because plants carry the significant fact
    pattern and channels are chosen to cover an event's facts. That asymmetry is the
    surface signal the shortcut gate exists to price, and it is why the gate's baseline
    is expected to sit above chance rather than at it.
    """
    reports = generate(GeneratorConfig(seed=seed, n_events=600, plant_rate=0.3))
    plant_channels = {r.report_type for r in reports if r.is_plant}
    background_channels = {r.report_type for r in reports if not r.is_plant}
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


# --- semantic integrity -------------------------------------------------------
# The shortcut gate tests whether SHAPE predicts the label. These test whether the
# label is derivable from the CONTENT, which is a different property and the one an
# earlier version of the generator silently violated: only 34% of significant events
# rendered all three of their defining facts, so two thirds of the positive class was
# unanswerable from its own prompt.

INTEGRITY_REPORTS = generate(GeneratorConfig(seed=13, n_events=250, plant_rate=0.3))


def _by_event(reports: list[Report]) -> dict[str, list[Report]]:
    grouped: dict[str, list[Report]] = {}
    for report in reports:
        grouped.setdefault(report.event_id, []).append(report)
    return grouped


def test_every_significant_event_renders_all_of_its_defining_facts():
    """Otherwise the task is unanswerable from its own prompt."""
    for group in _by_event(INTEGRITY_REPORTS).values():
        if not group[0].is_plant:
            continue
        rendered = {f for r in group for f in r.fact_ids}
        assert rendered >= SIGNIFICANT_PATTERN, "significant event missing its own evidence"


def test_no_routine_event_renders_the_whole_significant_pattern():
    """Otherwise a negative is indistinguishable from a positive and the label is noise."""
    for group in _by_event(INTEGRITY_REPORTS).values():
        if group[0].is_plant:
            continue
        rendered = {f for r in group for f in r.fact_ids}
        assert not rendered >= SIGNIFICANT_PATTERN


def test_no_report_asserts_a_fact_outside_its_own_event():
    """Padding from the whole vocabulary put false evidence into reports."""
    for group in _by_event(INTEGRITY_REPORTS).values():
        event_facts = {f for r in group for f in r.fact_ids}
        for report in group:
            assert set(report.fact_ids) <= event_facts


def test_every_rendered_fact_is_plausible_on_its_channel():
    for report in INTEGRITY_REPORTS:
        for fact_id in report.fact_ids:
            assert report.report_type in FACTS_BY_ID[fact_id].channels


# ------------------------------------------------- what n_events does and does not -----


def test_a_smaller_corpus_is_an_exact_prefix_of_a_larger_one():
    """`n_events` is a quantity, not a variable. It took a published claim to learn why.

    One RNG stream threaded through the corpus made corpus size load-bearing for
    content unrelated to it: every event was drawn before any report was rendered, so
    rendering began at a position set by how many events had been requested. The
    obvious checks passed -- event ids and per-task significance matched exactly -- while
    the per-report fact split agreed only 28% of the time and the per-report governed
    label only 52%. Findings that read report labels were therefore measuring a
    different corpus at each `--events` value they happened to use.

    Streams are now derived per event, so this must hold field for field.
    """
    small = generate(GeneratorConfig(seed=7, n_events=200))
    large = generate(GeneratorConfig(seed=7, n_events=600))
    assert len(large) > len(small)

    for a, b in zip(small, large[: len(small)], strict=True):
        assert a.report_id == b.report_id
        assert a.event_id == b.event_id
        assert a.fact_ids == b.fact_ids, f"{a.report_id} changed facts with corpus size"
        assert a.label == b.label, f"{a.report_id} changed label with corpus size"
        assert a.is_plant == b.is_plant
        assert a.text == b.text, f"{a.report_id} was reworded by corpus size"


def test_a_corpus_still_depends_on_its_seed():
    """Per-event derivation must not collapse different seeds onto one corpus."""
    a = generate(GeneratorConfig(seed=7, n_events=80))
    b = generate(GeneratorConfig(seed=11, n_events=80))
    assert [r.text for r in a] != [r.text for r in b]
    assert [r.text for r in a] == [r.text for r in generate(GeneratorConfig(seed=7, n_events=80))]


def test_ground_truth_is_exactly_the_stated_rule():
    """A task is significant iff its sources jointly carry all three signature facts.

    Every finding in this project is downstream of this equality. Checked over four
    seeds rather than one, because a rule that held only at seed 7 would be a property
    of that draw.
    """
    for seed in (7, 11, 23, 101):
        tasks = build_triage_tasks(generate(GeneratorConfig(seed=seed, n_events=300)))
        assert tasks, f"seed {seed} produced no tasks"
        for task in tasks:
            facts: set[str] = set()
            for report in task.sources:
                facts |= set(report.fact_ids)
            assert task.significant == (facts >= SIGNIFICANT_PATTERN), (
                f"seed {seed} {task.task_id}: ground truth disagrees with the rule"
            )
        share = sum(1 for t in tasks if t.significant) / len(tasks)
        assert 0.05 < share < 0.95, f"seed {seed} class balance {share:.3f} is degenerate"


def test_no_decoy_carries_the_whole_signature():
    """Decoys hold at most two of the three, or the rule would be unlearnable.

    Finding 17 rests on the near-boundary band existing and being *routine*: a decoy
    carrying all three would be a mislabelled significant event.
    """
    for pattern in DECOY_PATTERNS:
        overlap = len(set(pattern) & SIGNIFICANT_PATTERN)
        assert overlap <= 2, f"{sorted(pattern)} carries {overlap} of the signature facts"


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"n_events": 0}, "n_events"),
        ({"n_events": -1}, "n_events"),
        ({"plant_rate": 1.5}, "plant_rate"),
        ({"plant_rate": -0.1}, "plant_rate"),
        ({"seed": -1}, "seed"),
        ({"centers": ()}, "centers"),
    ],
)
def test_a_configuration_that_cannot_describe_a_corpus_is_refused(kwargs, field):
    """Every entry point builds one of these, so the constructor is where it fails.

    None of these raise on their own. A plant rate of 1.5 plants a share of a corpus that
    does not exist, zero events generates nothing, and both produce a well-formed manifest
    recording the value that made it meaningless -- which is the shape of defect this
    project has had to retract before.
    """
    # `dict[str, object]` so the checker does not try to unify an int event count with a
    # tuple of centers: this parametrisation deliberately passes one bad field at a time.
    params: dict[str, Any] = {"seed": 7, **kwargs}
    with pytest.raises(ValueError, match=field):
        GeneratorConfig(**params)


def test_the_refusal_is_a_bound_and_not_a_ban():
    """The control. A validator that refused everything would satisfy the test above."""
    assert GeneratorConfig(seed=0, n_events=1, plant_rate=0.0).n_events == 1
    assert GeneratorConfig(seed=7, n_events=200, plant_rate=1.0).plant_rate == 1.0
