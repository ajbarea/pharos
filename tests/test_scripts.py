"""The pure logic in `scripts/`, which is where every published number comes from.

`src/pharos` sits above 96% coverage while `scripts/` had none, and the asymmetry is
backwards: the scripts are what turn the library into the figures in the paper. A
bug in `parse_verdict` silently miscounts every F1. A bug in `tokenize_masked`
trains the model on the wrong tokens and reports a clean loss curve while doing it.
Neither would raise.

Only model-free logic is tested here. Anything needing Ollama or a GPU belongs in a
cluster job, not the suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from measure_rule_learnability import (  # noqa: E402
    INSTRUCTION_NO_RULE,
    balanced_shots,
    build_prompt,
    parse_verdict,
)
from train_adapter import (  # noqa: E402
    prompt_for,
    tokenize_masked,
    verdict_text,
)
from validate_gate_externally import (  # noqa: E402
    TEXT_FEATURES,
    permutation_null,
    surface_baseline,
    text_surface_features,
)

from pharos.generate import GeneratorConfig, generate  # noqa: E402
from pharos.tasks import build_triage_tasks  # noqa: E402

REPORTS = generate(GeneratorConfig(seed=3, n_events=80))
TASKS = build_triage_tasks(REPORTS)


# --- verdict parsing --------------------------------------------------------
# Every F1 in the paper is a count of these. A parser that silently coerces an
# ambiguous answer to a class shifts the score without any symptom.


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("VERDICT: SIGNIFICANT", True),
        ("VERDICT: ROUTINE", False),
        ("Reasoning here.\n\nVERDICT: SIGNIFICANT", True),
        ("verdict: routine", False),
        ("  VERDICT:SIGNIFICANT  ", True),
    ],
)
def test_a_clear_verdict_parses(answer, expected):
    assert parse_verdict(answer) is expected


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "I am not sure.",
        "VERDICT: SIGNIFICANT or ROUTINE",  # names both after the marker
        "This could be SIGNIFICANT. It could be ROUTINE.",
    ],
)
def test_an_ambiguous_answer_is_unparsed_rather_than_guessed(answer):
    """None, never a coerced class. How often a model fails to answer is a result."""
    assert parse_verdict(answer) is None


def test_only_the_tail_after_the_marker_decides():
    """The prompt asks for reasoning first, so the reasoning routinely contains the
    other class word. Parsing the whole string would score the reasoning."""
    assert parse_verdict("This is not ROUTINE at all.\nVERDICT: SIGNIFICANT") is True
    assert parse_verdict("Might look SIGNIFICANT.\nVERDICT: ROUTINE") is False


# --- example selection ------------------------------------------------------


def test_balanced_shots_alternates_classes():
    """An unbalanced example block teaches the prior instead of the rule, which
    would make a null result uninterpretable."""
    shots = balanced_shots(TASKS, 4)
    assert len(shots) == 4
    positives = sum(1 for s in shots if s.significant)
    assert abs(positives - 2) <= 1


def test_balanced_shots_never_exceeds_k():
    for k in (0, 1, 2, 8):
        assert len(balanced_shots(TASKS, k)) <= k


def test_balanced_shots_degrades_when_a_class_runs_out():
    """Asking for more examples than exist must not hang or duplicate."""
    only_positive = [t for t in TASKS if t.significant][:2]
    shots = balanced_shots(only_positive, 6)
    assert len(shots) <= 2
    assert len(shots) == len({id(s) for s in shots})


# --- prompt construction ----------------------------------------------------


def test_the_zero_shot_prompt_states_no_rule():
    """The whole design of finding 5 depends on the rule being absent."""
    prompt = build_prompt(TASKS[0], [])
    assert INSTRUCTION_NO_RULE in prompt
    for leak in ("three", "conjunction", "all of the following"):
        assert leak not in prompt.lower()


def test_shots_appear_before_the_target_case():
    prompt = build_prompt(TASKS[5], balanced_shots(TASKS, 2))
    assert prompt.index("CASE 1") < prompt.index("NEW CASE")
    assert prompt.count("OFFICER'S VERDICT") == 2


# --- the external probe's features -----------------------------------------


def test_surface_features_match_the_declared_names():
    assert len(text_surface_features("Some text. More text.")) == len(TEXT_FEATURES)


def test_surface_features_read_no_content():
    """Texts of identical SHAPE but different words must be indistinguishable.

    This is the property the whole external-validation result rests on: if the probe
    could read content, a leak would prove nothing about surface form.

    "Identical shape" has to be exact, not approximate. The first version of this
    test used two plausible-looking sentences of different character counts and
    failed, correctly: char_len is a surface feature and it was doing its job. The
    strings below are character-for-character equal in length, word count,
    punctuation, and case, and share no vocabulary.
    """
    a = "Alpha bravo charlie. Delta echo foxtrot."
    b = "Xrays yanks uniform. Zulus golf mikeoot."
    assert len(a) == len(b), "the fixtures themselves must match on shape"
    assert not (set(a.lower().split()) & set(b.lower().split())), "no shared words"
    assert text_surface_features(a) == text_surface_features(b)


def test_surface_features_are_finite_on_degenerate_input():
    """Empty and punctuation-only strings occur in real corpora."""
    for text in ("", ".", "   ", "!!!"):
        values = text_surface_features(text)
        assert all(v == v and abs(v) != float("inf") for v in values), text


def test_length_actually_moves_a_feature():
    short = text_surface_features("Hi.")
    long = text_surface_features("Hi. " * 200)
    assert long[0] > short[0]
    assert long[1] > short[1]


# --- adapter training data --------------------------------------------------
# The masking is the single most dangerous untested thing in the repository: if the
# prompt is not masked out of the loss, the model trains to generate maritime
# reports, the loss curve looks fine, and the experiment silently measures nothing.


def test_completion_text_is_exactly_what_the_parser_accepts():

    assert parse_verdict(verdict_text(True)) is True
    assert parse_verdict(verdict_text(False)) is False


def test_training_prompt_is_identical_to_the_measured_one():
    """A score difference between finding 5 and the adapter must not be a prompt
    difference. Both build from the same function, and this asserts it."""

    assert prompt_for(TASKS[0]) == build_prompt(TASKS[0], [])


def test_tokenize_masks_the_prompt_out_of_the_loss():

    tokenizer = _FakeTokenizer()
    example = {"prompt": "aaa bbb ccc", "completion": "ddd"}
    row = tokenize_masked(example, tokenizer, max_len=64)

    assert len(row["input_ids"]) == len(row["labels"])
    supervised = [label for label in row["labels"] if label != -100]
    masked = [label for label in row["labels"] if label == -100]
    assert masked, "the prompt must be masked"
    assert supervised, "the completion must be supervised"
    # Supervised positions are exactly the tail, and they match the input there.
    assert row["labels"][-len(supervised) :] == row["input_ids"][-len(supervised) :]


def test_truncation_keeps_the_completion_supervised():
    """A long prompt truncates from the left; the completion must survive, or the
    example teaches nothing at all."""

    tokenizer = _FakeTokenizer()
    row = tokenize_masked(
        {"prompt": " ".join(["word"] * 500), "completion": "ddd"}, tokenizer, max_len=16
    )
    assert len(row["input_ids"]) == 16
    assert any(label != -100 for label in row["labels"]), "completion was truncated away"


class _FakeTokenizer:
    """One id per whitespace token. Enough to test masking arithmetic without
    downloading a real tokenizer into the unit suite."""

    eos_token = "<eos>"  # noqa: S105 -- a token string, not a credential
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [hash(t) % 1000 + 1 for t in text.split()]}


# --- the statistics behind the external-validation result -------------------
# These produced a finding the paper now leans on, so the machinery gets tested on
# inputs whose answer is known by construction rather than only on real corpora.


def test_a_perfectly_separable_signal_scores_near_one():
    """Sanity: if shape DOES predict the label, the probe must find it. A probe that
    cannot detect a planted leak would report every corpus as clean."""

    rng = np.random.default_rng(0)
    n = 200
    x = np.vstack([rng.normal(0, 1, (n, 3)), rng.normal(6, 1, (n, 3))])
    y = np.array([0] * n + [1] * n)
    verdict, per_probe = surface_baseline(x, y, folds=4, seed=0)
    assert verdict > 0.95, per_probe
    assert set(per_probe) == {"logistic", "gradient_boosting"}


def test_pure_noise_scores_near_chance():
    """The complement: no relationship must not manufacture one."""

    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (400, 3))
    y = rng.integers(0, 2, 400)
    verdict, _ = surface_baseline(x, y, folds=4, seed=0)
    assert 0.35 < verdict < 0.65, verdict


def test_the_null_centres_on_chance_for_a_real_signal():
    """The permutation null must report what the PROCEDURE does under no
    relationship, even when the unshuffled data has a strong one. If the null
    tracked the signal, every leak would look insignificant."""

    rng = np.random.default_rng(2)
    n = 150
    x = np.vstack([rng.normal(0, 1, (n, 3)), rng.normal(5, 1, (n, 3))])
    y = np.array([0] * n + [1] * n)
    mean, sd, p95 = permutation_null(x, y, trials=8, folds=4, seed=0)
    assert 0.40 < mean < 0.60, f"null mean {mean} is not near chance"
    assert sd >= 0.0
    assert p95 >= mean


def test_the_worst_probe_sets_the_verdict():
    """Pharos's worst-news-wins rule, carried over deliberately. Averaging a strong
    probe against a weak one would hide exactly the leak the gate exists to find."""

    rng = np.random.default_rng(3)
    n = 200
    x = np.vstack([rng.normal(0, 1, (n, 2)), rng.normal(4, 1, (n, 2))])
    y = np.array([0] * n + [1] * n)
    verdict, per_probe = surface_baseline(x, y, folds=4, seed=0)
    furthest = max(per_probe.values(), key=lambda v: abs(v - 0.5))
    assert verdict == furthest


def test_a_single_class_raises_rather_than_returning_chance():
    """Silently returning 0.5 would be a gate that always passes.

    ValueError specifically, not any exception: a blind assertion would also pass if
    the code raised a TypeError from some unrelated refactor, which is exactly the
    kind of test that stops testing anything.
    """
    x = np.random.default_rng(4).normal(0, 1, (50, 3))
    with pytest.raises(ValueError, match="no usable fold"):
        surface_baseline(x, np.zeros(50, dtype=int), folds=4, seed=0)


# --- the policy constants that define finding 2 -----------------------------
# CEILINGS and POLICIES are the grid the federation-eligibility result was measured
# over. They are data rather than logic, which is exactly why nothing was checking
# them: a wrong compartment on a ceiling silently changes a published number and
# raises nothing.


def test_the_aggregator_ceiling_sits_below_the_enclaves_it_serves():
    """Sharing has to be a downgrade, or the eligibility question is vacuous."""
    from measure_triage_lift import CEILING

    from pharos.labels import Sensitivity

    assert CEILING.sensitivity < Sensitivity.RESTRICTED
    assert CEILING.compartments, "a ceiling with no compartments admits everything"


def test_triage_lift_parses_verdicts_the_same_way_as_the_learnability_run():
    """Two scripts scoring the same task must agree on what counts as an answer."""
    from measure_triage_lift import parse_verdict as triage_parse

    for answer in ("VERDICT: SIGNIFICANT", "VERDICT: ROUTINE", "no idea"):
        assert triage_parse(answer) == parse_verdict(answer), answer


def test_the_eligibility_grid_spans_a_real_range_of_ceilings():
    """Finding 2 is a claim about behaviour ACROSS aggregator ceilings. One ceiling,
    or three identical ones, would make the range in the paper meaningless."""
    from measure_federation_eligibility import CEILINGS

    assert len(CEILINGS) >= 3
    labels = [label for _, label in CEILINGS]
    assert len({(la.sensitivity, la.compartments) for la in labels}) == len(labels)
    assert len({la.sensitivity for la in labels}) > 1, "ceilings differ only in name"


def test_the_policy_pair_is_exactly_the_ruling_under_test():
    """The finding is bimodal on ONE ruling: may a low-capacity output shed its
    sources' compartments. The grid must contain both sides of it and differ in
    nothing else, or the comparison is confounded."""
    from measure_federation_eligibility import POLICIES

    assert len(POLICIES) == 2
    by_name = dict(POLICIES)
    keep = next(p for name, p in POLICIES if "keep" in name)
    drop = next(p for name, p in POLICIES if "drop" in name)
    assert keep.drop_compartments is False
    assert drop.drop_compartments is True
    # Everything else must match, or the two rows differ by more than the ruling.
    assert keep.declassifiable == drop.declassifiable
    assert keep.release_floor == drop.release_floor
    assert len(by_name) == 2, "policy names collide"


def test_compare_models_reports_nothing_when_there_is_nothing_to_report(tmp_path, monkeypatch):
    """An empty results directory must exit non-zero rather than print an empty
    table that reads like a finding."""
    import compare_models

    monkeypatch.setattr(compare_models, "RESULTS", tmp_path)
    assert compare_models.main() == 1
