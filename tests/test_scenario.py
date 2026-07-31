"""A scenario must reproduce the published world exactly, and refuse a broken one.

The first test here is the load-bearing one. `scenarios/maritime-watch.toml` is the
world every published Pharos number was measured on, lifted out of code into data. If
it diverges from `pharos.world` by so much as a rendering, every figure in the paper
silently describes a corpus that no longer exists -- and nothing else in the suite
would notice, because the corpus would still generate, still gate, and still look
entirely reasonable.

The rest test that validation actually refuses the ways a scenario can be broken
while still parsing. Those matter because the shortcut gate cannot catch them: it
asks whether shape predicts the label, not whether the vocabulary is coherent.
"""

from dataclasses import replace

import pytest

from pharos import scenario as scenario_module
from pharos import world
from pharos.labels import Compartment, Sensitivity
from pharos.scenario import (
    CenterSpec,
    ChannelSpec,
    FactSpec,
    OperatorSpec,
    Scenario,
    ScenarioError,
    available,
    load,
    validate,
)

DEFAULT = load()


# --- fidelity to the published world ----------------------------------------


def test_the_default_scenario_is_the_published_world():
    """Field by field, not a spot check. Anything that differs changes what every
    measurement means."""
    assert {c.name: (c.sensitivity, c.compartments) for c in DEFAULT.channels} == {
        str(k): v for k, v in world.CHANNEL_LABELS.items()
    }
    assert list(DEFAULT.voices) == [str(v) for v in world.Voice]
    assert [(c.center_id, c.name, c.voice) for c in DEFAULT.centers] == [
        (c.center_id, c.name, str(c.voice)) for c in world.CENTERS
    ]
    assert [(o.operator_id, o.name) for o in DEFAULT.operators] == [
        (o.operator_id, o.name) for o in world.OPERATORS
    ]
    assert DEFAULT.vessel_names == world.VESSEL_NAMES
    assert DEFAULT.hull_types == world.HULL_TYPES
    assert DEFAULT.flags == world.FLAGS
    assert DEFAULT.significant_pattern == world.SIGNIFICANT_PATTERN
    assert DEFAULT.decoy_patterns == world.DECOY_PATTERNS


def test_every_fact_survives_the_round_trip_including_its_renderings():
    """Renderings are where a silent divergence would hide: the corpus would still
    generate and still gate, just from different text."""
    from_file = [
        (f.fact_id, frozenset(str(c) for c in f.channels), f.renderings) for f in DEFAULT.facts
    ]
    from_code = [
        (f.fact_id, frozenset(str(c) for c in f.channels), f.renderings) for f in world.FACTS
    ]
    assert from_file == from_code


def test_the_lattice_stays_non_trivial():
    """Two channels at equal sensitivity with incomparable compartments is the pair
    that makes this a lattice test rather than a ladder test."""
    restricted = [c for c in DEFAULT.channels if c.sensitivity is Sensitivity.RESTRICTED]
    assert len(restricted) >= 2
    a, b = restricted[0].compartments, restricted[1].compartments
    assert not (a <= b) and not (b <= a), "compartments are comparable; the lattice collapsed"


def test_the_shipped_scenario_is_discoverable():
    assert scenario_module.DEFAULT_SCENARIO in available()


def test_a_scenario_summarises_itself_for_a_manifest():
    summary = DEFAULT.summary()
    assert summary["scenario"] == "maritime-watch"
    # int(...) because summary() is dict[str, object]; the coercion is what makes the
    # comparison well typed rather than accidentally so.
    assert int(summary["label_cells"]) > 1  # ty: ignore[invalid-argument-type]
    assert sorted(summary["significant_pattern"]) == sorted(  # ty: ignore[invalid-argument-type]
        world.SIGNIFICANT_PATTERN
    )


# --- validation refuses corpora that would measure nothing ------------------


_MINIMAL = Scenario(
    name="t",
    description="",
    channels=(
        ChannelSpec("open_wire", Sensitivity.OPEN, frozenset()),
        ChannelSpec("secret_wire", Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR})),
    ),
    voices=("PLAIN",),
    centers=(CenterSpec("C1", "One", "PLAIN"),),
    operators=(OperatorSpec("O1", "Op"),),
    vessel_names=("Ship",),
    hull_types=("hull",),
    flags=("Flag",),
    significant_pattern=frozenset({"a", "b"}),
    decoy_patterns=(frozenset({"a", "c"}),),
    facts=(
        FactSpec("a", frozenset({"open_wire"}), ("a happens.",)),
        FactSpec("b", frozenset({"secret_wire"}), ("b happens.",)),
        FactSpec("c", frozenset({"open_wire"}), ("c happens.",)),
    ),
)


def minimal(**overrides) -> Scenario:
    """The smallest valid scenario, so each test perturbs exactly one thing.

    `replace` rather than splatting a dict of heterogeneous values: the splat gave
    the checker a union of every field type for every field, which is 12 warnings
    saying the same true thing.
    """
    return replace(_MINIMAL, **overrides)


def test_the_minimal_fixture_is_itself_valid():
    """Otherwise every test below would pass for the wrong reason."""
    validate(minimal())


def test_a_significance_rule_naming_a_missing_fact_is_refused():
    """The positive class would be empty and every score would describe nothing."""
    with pytest.raises(ScenarioError, match="do not exist"):
        validate(minimal(significant_pattern=frozenset({"a", "nonexistent"})))


def test_a_single_fact_significance_rule_is_refused():
    """One fact is a keyword match. The task is meant to be a conjunction over
    sources that each carry only part of the picture."""
    with pytest.raises(ScenarioError, match="keyword match"):
        validate(minimal(significant_pattern=frozenset({"a"})))


def test_a_decoy_containing_the_whole_rule_is_refused():
    """Background events carrying it would be significant by definition, so the
    label is a lie and no model could ever be right."""
    with pytest.raises(ScenarioError, match="entire significance rule"):
        validate(minimal(decoy_patterns=(frozenset({"a", "b", "c"}),)))


def test_a_fact_on_an_unknown_channel_is_refused():
    with pytest.raises(ScenarioError, match="unknown channels"):
        validate(
            minimal(
                facts=(FactSpec("a", frozenset({"no_such_channel"}), ("x.",)), *minimal().facts[1:])
            )
        )


def test_a_fact_with_no_renderings_is_refused():
    with pytest.raises(ScenarioError, match="no renderings"):
        validate(minimal(facts=(FactSpec("a", frozenset({"open_wire"}), ()), *minimal().facts[1:])))


def test_a_channel_carrying_nothing_is_refused():
    """It would appear in the label histogram and emit no reports."""
    with pytest.raises(ScenarioError, match="carry no fact"):
        validate(
            minimal(
                channels=(
                    *minimal().channels,
                    ChannelSpec("silent", Sensitivity.INTERNAL, frozenset()),
                )
            )
        )


def test_a_single_label_cell_is_refused():
    """A constant label cannot evaluate a disclosure boundary: every entry is
    governed identically, so the question has no content."""
    flat = (
        ChannelSpec("open_wire", Sensitivity.OPEN, frozenset()),
        ChannelSpec("secret_wire", Sensitivity.OPEN, frozenset()),
    )
    with pytest.raises(ScenarioError, match="one label cell"):
        validate(minimal(channels=flat))


def test_a_centre_using_an_undeclared_voice_is_refused():
    with pytest.raises(ScenarioError, match="undeclared voices"):
        validate(minimal(centers=(CenterSpec("C1", "One", "MYSTERY"),)))


def test_an_empty_scenario_is_refused():
    with pytest.raises(ScenarioError, match="at least one"):
        validate(minimal(channels=()))


# --- loading ----------------------------------------------------------------


def test_an_unknown_scenario_name_lists_what_is_available():
    with pytest.raises(ScenarioError, match="Available"):
        load("no-such-scenario")


def test_an_unknown_sensitivity_names_the_valid_options():
    """A typo in a hand-written scenario should say what was expected."""
    import tomllib

    assert "OPEN" in [s.name for s in Sensitivity]
    with pytest.raises(ScenarioError, match="expected one of"):
        scenario_module._enum(Sensitivity, "TOP_SECRET", "sensitivity")
    assert tomllib  # imported to document that loading is TOML-based


def test_loading_by_explicit_path_works(tmp_path):
    """A collaborator's world can live outside this repository."""
    source = (scenario_module.SCENARIO_DIR / "maritime-watch.toml").read_text(encoding="utf-8")
    target = tmp_path / "copy.toml"
    target.write_text(source, encoding="utf-8")
    assert load(target).name == "maritime-watch"


# --- scenario files are untrusted input by design ---------------------------
# A scenario is meant to be written by someone else, which makes every rendering in
# it attacker-controlled. `str.format` on an attacker-controlled template is a known
# Python injection class: a field name can traverse attributes and indices, reaching
# module globals from inside what looks like a text template. This was confirmed
# exploitable against this codebase before the whitelist existed.


def test_attribute_traversal_in_a_rendering_is_refused():
    """The actual exploit: walk from a slot value into module globals."""
    hostile = "Leak {vessel.__class__.__init__.__globals__[SECRET]}"
    with pytest.raises(ScenarioError, match=r"bare name"):
        scenario_module._check_rendering_is_safe(hostile, "evil")


def test_indexing_in_a_rendering_is_refused():
    with pytest.raises(ScenarioError, match="bare name"):
        scenario_module._check_rendering_is_safe("Value {draft[0]}", "evil")


def test_a_conversion_in_a_rendering_is_refused():
    """`!r` calls repr on the value, which leaks object internals into the corpus."""
    with pytest.raises(ScenarioError, match="conversion"):
        scenario_module._check_rendering_is_safe("Value {draft!r}", "evil")


def test_an_undeclared_placeholder_is_refused_with_the_valid_list():
    with pytest.raises(ScenarioError, match="Available"):
        scenario_module._check_rendering_is_safe("Value {not_a_slot}", "evil")


def test_a_malformed_template_is_refused_rather_than_raising_later():
    """An unbalanced brace would otherwise fail deep inside generation."""
    with pytest.raises(ScenarioError, match="not a valid template"):
        scenario_module._check_rendering_is_safe("Unbalanced {draft", "evil")


def test_legitimate_placeholders_are_accepted():
    """The check must not be so strict that the real vocabulary fails it."""
    scenario_module._check_rendering_is_safe("Draft {draft} m at {time}Z, {count} seen.", "ok")


def test_every_shipped_rendering_passes_the_whitelist():
    """The scenario the paper's numbers come from must itself be clean."""
    for fact in DEFAULT.facts:
        for rendering in fact.renderings:
            scenario_module._check_rendering_is_safe(rendering, fact.fact_id)


def test_validation_rejects_a_scenario_carrying_a_hostile_rendering():
    """End to end: the check is wired into validate(), not merely available."""
    hostile = FactSpec("a", frozenset({"open_wire"}), ("{a.__class__}",))
    with pytest.raises(ScenarioError):
        validate(minimal(facts=(hostile, *minimal().facts[1:])))
