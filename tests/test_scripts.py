"""The pure logic in `scripts/`, which is where every published number comes from.

`src/pharos` sits above 96% coverage while `scripts/` had none, and the asymmetry is
backwards: the scripts are what turn the library into the figures in the paper. A
bug in `parse_verdict` silently miscounts every F1. A bug in `tokenize_masked`
trains the model on the wrong tokens and reports a clean loss curve while doing it.
Neither would raise.

Only model-free logic is tested here. Anything needing Ollama or a GPU belongs in a
cluster job, not the suite.
"""

import pathlib
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


# ---------------------------------------------------------------------------
# train_adapter: the reporting layer
#
# evaluate() is where a bug is most expensive and least visible. It owns the
# confusion matrix behind every number in the adapter table, and every failure mode
# it has is silent: a miscounted cell still produces a plausible F1. It takes the
# model and tokenizer as arguments, so it can be driven by stubs and checked against
# a matrix computed by hand.
# ---------------------------------------------------------------------------


class _GreedyTokenizer:
    """Enough of a tokenizer for evaluate(): call, decode, pad id."""

    pad_token_id = 0

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self._served = 0

    def __call__(self, text, **kwargs):
        class _Batch(dict[str, object]):
            def to(self, _device):
                return self

        return _Batch(input_ids=_FakeIds())

    def decode(self, _tokens, **kwargs) -> str:
        answer = self._answers[self._served]
        self._served += 1
        return answer


class _FakeIds:
    """Stands in for a tensor of token ids; only `.shape[1]` is ever read."""

    shape = (1, 0)

    def __getitem__(self, _index):
        return self


class _FakeModel:
    device = "cpu"

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [_FakeIds()]


@pytest.fixture
def _fake_torch(monkeypatch):
    """evaluate() imports torch only for `no_grad()`.

    torch is a multi-gigabyte CUDA dependency that neither CI nor a laptop
    installs, and skipping the test there would leave the confusion matrix
    untested everywhere it actually runs. A no-op context manager is a faithful
    stand-in: `no_grad` affects autograd bookkeeping, not the arithmetic under test.
    """
    import contextlib
    import sys
    import types

    # SimpleNamespace rather than ModuleType: `import torch` returns whatever
    # sys.modules holds, and attributes set in the constructor are typed, where
    # assigning onto a bare module object is not.
    fake = types.SimpleNamespace(no_grad=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def _evaluate_with(answers, tasks, _fixture):
    from train_adapter import evaluate

    return evaluate(_FakeModel(), _GreedyTokenizer(answers), tasks, label="stub")


def test_evaluate_builds_the_confusion_matrix_it_reports(_fake_torch):
    """Two of each cell, hand-checked, including an unparsable answer."""
    significant = [t for t in TASKS if t.significant][:2]
    routine = [t for t in TASKS if not t.significant][:2]
    tasks = significant + routine

    # SIGNIFICANT on a significant task is a true positive; on a routine one a false
    # positive. ROUTINE on a significant task is a false negative, else a true
    # negative. The fourth answer is deliberately unreadable.
    result = _evaluate_with(
        ["SIGNIFICANT", "ROUTINE", "SIGNIFICANT", "not a verdict at all"], tasks, _fake_torch
    )

    assert (result.tp, result.fn) == (1, 1)
    assert (result.fp, result.unparsed) == (1, 1)
    assert result.n == 4
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_evaluate_counts_an_unparsable_answer_as_neither_class(_fake_torch):
    """The 12-token bug: unparsed answers must not quietly become correct ones."""
    tasks = [t for t in TASKS if t.significant][:3]

    result = _evaluate_with(["garbage", "garbage", "garbage"], tasks, _fake_torch)

    assert result.unparsed == 3
    assert (result.tp, result.fp, result.tn, result.fn) == (0, 0, 0, 0)
    # No prediction was made, so no score may be claimed from one.
    assert result.f1 == 0.0
    assert result.n == 3


def test_evaluate_reports_n_including_unparsed_answers(_fake_torch):
    """`n` is the denominator a reader divides by, so it counts every task asked."""
    tasks = list(TASKS[:5])

    result = _evaluate_with(["SIGNIFICANT", "ROUTINE", "?", "?", "?"], tasks, _fake_torch)

    assert result.n == len(tasks)
    assert result.as_dict()["n"] == len(tasks)


def test_eval_result_serialises_every_field_it_reports(_fake_torch):
    result = _evaluate_with(["SIGNIFICANT"], TASKS[:1], _fake_torch)

    payload = result.as_dict()

    assert set(payload["confusion"]) == {"tp", "fp", "tn", "fn"}
    for key in ("label", "n", "unparsed", "accuracy", "majority_accuracy", "f1"):
        assert key in payload


def test_build_examples_pairs_each_prompt_with_its_verdict():
    from train_adapter import build_examples, prompt_for, verdict_text

    tasks = list(TASKS[:4])

    examples = build_examples(tasks)

    assert len(examples) == len(tasks)
    for example, task in zip(examples, tasks, strict=True):
        assert example["prompt"] == prompt_for(task)
        assert example["completion"] == verdict_text(task.significant)
    # The completion is the label. If prompt and completion were ever transposed the
    # adapter would train on the inverse mapping and still report a clean loss.
    assert all(e["completion"] != e["prompt"] for e in examples)


def test_class_balance_counts_both_classes_and_sums_to_the_whole():
    from train_adapter import _class_balance

    tasks = list(TASKS[:20])

    balance = _class_balance(tasks)

    assert balance["significant"] == sum(1 for t in tasks if t.significant)
    assert balance["routine"] == sum(1 for t in tasks if not t.significant)
    assert balance["significant"] + balance["routine"] == len(tasks)


def test_class_balance_of_nothing_is_zero_not_an_error():
    from train_adapter import _class_balance

    assert _class_balance([]) == {"significant": 0, "routine": 0}


def test_evaluate_surfaces_validity_concerns_rather_than_swallowing_them(_fake_torch, capsys):
    """A below-floor, high-unparsed baseline must announce itself.

    This is the reporting path that flagged the real base model at accuracy 0.346
    against a 0.673 majority floor. If it were silent the number would look like a
    measurement instead of a warning.
    """
    tasks = [t for t in TASKS if t.significant][:4]

    _evaluate_with(["ROUTINE", "ROUTINE", "junk", "junk"], tasks, _fake_torch)

    printed = capsys.readouterr().out
    assert "validity concerns" in printed


# ---------------------------------------------------------------------------
# validate_gate_externally: the positive control
#
# content_baseline exists because its absence nearly published a false finding: a
# loader bug made the label unrelated to the text, the surface probe scored chance,
# and "adversarial filtering removes surface signal" was the confident wrong reading.
# The control is only useful if it can tell those two situations apart, so both are
# constructed here rather than assumed.
# ---------------------------------------------------------------------------


def _separable_corpus(n: int = 40):
    """Text whose words genuinely carry the label."""
    texts = [
        ("harbour patrol sighted a vessel" if i % 2 else "routine dock inspection log")
        for i in range(n)
    ]
    labels = np.array([i % 2 for i in range(n)])
    return texts, labels


def _label_unrelated_corpus(n: int = 40):
    """The HellaSwag bug in miniature: identical text, label assigned around it."""
    texts = ["the same sentence every time" for _ in range(n)]
    labels = np.array([i % 2 for i in range(n)])
    return texts, labels


def test_content_baseline_detects_signal_a_reader_could_use():
    from validate_gate_externally import content_baseline

    texts, labels = _separable_corpus()

    auc = content_baseline(texts, labels, folds=4, seed=0)

    assert auc > 0.9, "text that plainly predicts the label must score well above chance"


def test_content_baseline_scores_chance_when_the_label_is_unrelated_to_the_text():
    """The bug this control exists to catch, reproduced deliberately."""
    from validate_gate_externally import content_baseline

    texts, labels = _label_unrelated_corpus()

    auc = content_baseline(texts, labels, folds=4, seed=0)

    assert auc == pytest.approx(0.5, abs=0.15)


def test_content_baseline_refuses_rather_than_returning_a_number_it_cannot_support():
    from validate_gate_externally import content_baseline

    texts = ["only one class here"] * 6
    labels = np.zeros(6, dtype=int)

    with pytest.raises(ValueError):
        content_baseline(texts, labels, folds=2, seed=0)


def test_corpus_prevalence_is_the_positive_share():
    from validate_gate_externally import Corpus

    corpus = Corpus(name="c", texts=["a", "b", "c", "d"], labels=[1, 1, 0, 0], note="")

    assert corpus.prevalence == pytest.approx(0.5)


def test_corpus_prevalence_of_nothing_does_not_divide_by_zero():
    from validate_gate_externally import Corpus

    assert Corpus(name="c", texts=[], labels=[], note="").prevalence == 0.0


def test_permutation_null_refuses_when_no_trial_was_usable():
    """A null with no trials behind it must raise, not return a tidy zero."""
    from validate_gate_externally import permutation_null

    # One class only: every shuffle is degenerate, so every fold is skipped.
    x = np.random.default_rng(0).normal(size=(12, 3))
    y = np.zeros(12, dtype=int)

    with pytest.raises(ValueError, match="no usable trial"):
        permutation_null(x, y, trials=3, folds=2, seed=0)


# ---------------------------------------------------------------------------
# measure_label_fidelity: the experiment driver
#
# This script had no coverage at all, and it is the one that produces finding 1 --
# the negative result that leave-one-out attribution cannot yield a correct governed
# label. Everything in it except the model call is deterministic, so the model call
# is the only thing that needs standing in for. The Attribution it returns is a
# plain dataclass, so the stub builds a real one rather than a mock, and the outcome
# is then computed by the same code the experiment uses.
# ---------------------------------------------------------------------------


def _run_label_fidelity(monkeypatch, tmp_path, *, attribution_for, argv_extra=()):
    """Drive main() with the model call replaced and argv controlled."""
    import measure_label_fidelity as mlf

    from pharos.attribute import Attribution

    def fake_attribute(task, **_kwargs):
        return Attribution(**attribution_for(task))

    monkeypatch.setattr(mlf, "attribute_leave_one_out", fake_attribute)
    out = tmp_path / "fidelity.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_label_fidelity.py",
            "--tasks",
            "3",
            "--events",
            "120",
            "--out",
            str(out),
            *argv_extra,
        ],
    )
    return mlf.main(), out, Attribution


def _perfect(task):
    """Attribution that recovers exactly the sources that truly contributed."""
    from pharos.attribute import truly_contributing
    from pharos.detect import detect_facts

    asserted = detect_facts(" ".join(s.text for s in task.sources))
    truth = truly_contributing(task, asserted)
    return {
        "task_id": task.task_id,
        "summary": " ".join(s.text for s in task.sources),
        "asserted_facts": asserted,
        "attributed_sources": truth,
        "truly_contributing": truth,
        "calls": len(task.sources) + 1,
    }


def _under_attributing(task):
    """Drops contributing sources, which is the direction that leaks."""
    payload = _perfect(task)
    truth = payload["truly_contributing"]
    payload["attributed_sources"] = frozenset(sorted(truth)[: len(truth) // 2])
    return payload


def test_label_fidelity_runs_end_to_end_and_writes_a_provenanced_artifact(monkeypatch, tmp_path):
    import json

    code, out, _ = _run_label_fidelity(monkeypatch, tmp_path, attribution_for=_perfect)

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    # An artifact with no provenance cannot be traced to the code that made it, and
    # harvest.py in the papers repo refuses to build a table from one.
    assert "provenance" in payload
    assert payload["provenance"]["git_commit"]
    assert len(payload["rows"]) == 3
    assert set(payload["outcomes"]) <= {"exact", "creep", "leak", "incomparable"}
    assert payload["source_recall_mean"] == pytest.approx(1.0)


def test_label_fidelity_records_a_leak_when_attribution_misses_a_source(monkeypatch, tmp_path):
    """Under-attribution deflates the join, which is the error that must never pass."""
    import json

    code, out, _ = _run_label_fidelity(monkeypatch, tmp_path, attribution_for=_under_attributing)

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source_recall_mean"] < 1.0
    # Dropping a contributing source can only lose compartments, so the outcome must
    # never come back "exact" for every task.
    assert payload["outcomes"].get("exact", 0) < len(payload["rows"])


def test_label_fidelity_aborts_when_the_detector_is_too_weak_to_interpret(
    monkeypatch, tmp_path, capsys
):
    """The guard that stops a meaningless number being written at all."""
    import measure_label_fidelity as mlf

    from pharos.detect import DetectorAccuracy

    weak = DetectorAccuracy(n_reports=10, recall=0.1, precision=0.1)
    monkeypatch.setattr(mlf, "detector_accuracy", lambda _reports: weak)
    monkeypatch.setattr(
        sys, "argv", ["measure_label_fidelity.py", "--tasks", "1", "--events", "60"]
    )

    assert mlf.main() == 1
    assert "detector too weak" in capsys.readouterr().out


# --------------------------------------------------------------- analyst review -----


def _triage_artifact(path, rows):
    import json

    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def _seed7_tasks():
    from measure_analyst_review import EVENTS, SEED, TASKS

    reports = generate(GeneratorConfig(seed=SEED, n_events=EVENTS))
    return build_triage_tasks(reports, limit=TASKS)


def test_load_proposals_reads_model_verdicts_as_proposals(tmp_path):
    import measure_analyst_review as mar

    tasks = _seed7_tasks()
    rows = [{"task_id": t.task_id, "truth": t.significant, "verdict": True} for t in tasks]
    proposals = mar.load_proposals(_triage_artifact(tmp_path / "a.json", rows), tasks)

    assert len(proposals) == len(tasks)
    assert all(p.verdict for p in proposals.values())
    # Proposals are released under the fail-closed default, whatever the reviewer
    # would themselves rule.
    assert all(p.release.sensitivity.name == "OPEN" for p in proposals.values())
    assert any(p.release.compartments for p in proposals.values())


def test_load_proposals_drops_an_unparsable_answer_rather_than_coercing_it(tmp_path):
    import measure_analyst_review as mar

    tasks = _seed7_tasks()
    rows = [{"task_id": t.task_id, "truth": t.significant, "verdict": None} for t in tasks]
    rows[0]["verdict"] = False
    proposals = mar.load_proposals(_triage_artifact(tmp_path / "a.json", rows), tasks)

    assert set(proposals) == {tasks[0].task_id}


def test_load_proposals_refuses_an_artifact_measured_on_a_different_corpus(tmp_path):
    """The guard that stops two worlds being joined into one table."""
    import measure_analyst_review as mar

    tasks = _seed7_tasks()
    rows = [{"task_id": t.task_id, "truth": t.significant, "verdict": True} for t in tasks]
    rows[0]["truth"] = not rows[0]["truth"]

    with pytest.raises(mar.ArtifactMismatchError, match="Re-run the sweep"):
        mar.load_proposals(_triage_artifact(tmp_path / "a.json", rows), tasks)


def test_load_proposals_refuses_a_task_the_corpus_does_not_contain(tmp_path):
    import measure_analyst_review as mar

    tasks = _seed7_tasks()
    rows = [{"task_id": "TR-9999", "truth": False, "verdict": True}]

    with pytest.raises(mar.ArtifactMismatchError, match="not in the regenerated corpus"):
        mar.load_proposals(_triage_artifact(tmp_path / "a.json", rows), tasks)


def test_measure_reports_one_row_per_reviewer_with_its_parameters():
    import measure_analyst_review as mar

    from pharos.analyst import DEFAULT_ENSEMBLE

    tasks = _seed7_tasks()
    proposals = {t.task_id: mar.Proposal(t.task_id, not t.significant, t.label) for t in tasks}
    review = mar.measure(tasks, proposals, DEFAULT_ENSEMBLE)
    payload = review.as_dict()

    assert payload["n_proposals"] == len(tasks)
    assert [row.policy.name for row in review.rows] == [p.name for p in DEFAULT_ENSEMBLE]

    # The parameters that produced each row travel with it into the artifact, so a
    # table read years later still says what its reviewers were.
    by_name = {row.policy.name: row.as_dict() for row in review.rows}
    assert by_name["any-one"]["escalation_threshold"] == 1
    assert by_name["releaser"]["drop_compartments"] is True
    assert by_name["unexplained"]["names_grounds"] is False
    assert by_name["unexplained"]["located_share"] == 0.0


# ------------------------------------------------------- static explorer -----


@pytest.fixture(scope="module")
def bundle():
    import build_static_explorer as bse

    from pharos.web import create_app

    return bse, bse.build_bundle(create_app())


def test_bundle_covers_every_frozen_input(bundle):
    bse, payload = bundle

    assert payload["seeds"] == list(bse.SEEDS)
    for seed in bse.SEEDS:
        assert str(seed) in payload["corpus"]
        assert str(seed) in payload["gate"]
        per_seed = payload["review"][str(seed)]
        assert set(per_seed) == {str(i) for i in range(bse.REVIEW_TASKS)}
        assert set(per_seed["0"]) == {"true", "false"}


def test_every_dominance_lookup_the_page_can_make_resolves(bundle):
    """The shim's only failure mode is a key it cannot find, so try them all.

    The page composes a join key from the pair table and the capacity named by
    `capacity_follows`. If any combination is absent the reader gets an error
    instead of an answer, and there are tens of thousands of them -- too many to
    trust to a spot check and cheap enough to check exhaustively.
    """
    from pharos.labels import Capacity

    _, payload = bundle
    dominance = payload["dominance"]
    assert dominance["capacity_follows"] == "item"

    labels, pairs = dominance["labels"], dominance["pairs"]
    capacities = [c.name for c in Capacity]
    assert len(pairs) == 64, "four sensitivities times sixteen compartment subsets"
    assert len(labels) == 64 * len(capacities)

    for holder_key, row in pairs.items():
        assert len(row) == 64
        for item_key, cell in row.items():
            for capacity in capacities:
                assert f"{holder_key}|{capacity}" in labels
                assert f"{item_key}|{capacity}" in labels
                assert f"{cell['join']}|{capacity}" in labels, (
                    f"join {cell['join']} at {capacity} is missing; the page would error"
                )


def test_incomparability_survives_the_compaction(bundle):
    """The one property the whole corpus exists to exhibit must reach the page."""
    _, payload = bundle
    pairs = payload["dominance"]["pairs"]

    cell = pairs["INTERNAL|SENSOR"]["INTERNAL|LEGAL"]
    assert cell["incomparable"] is True
    assert cell["holder_may_read_item"] is False
    assert cell["item_may_read_holder"] is False
    assert cell["join"] == "INTERNAL|LEGAL,SENSOR"

    # And a comparable pair, so the test is not satisfied by everything being False.
    ladder = pairs["RESTRICTED|SENSOR"]["OPEN|SENSOR"]
    assert ladder["holder_may_read_item"] is True
    assert ladder["incomparable"] is False


def test_the_builder_refuses_when_an_endpoint_disappears():
    """A renamed route must fail the build, not quietly drop a tab from the page."""
    import build_static_explorer as bse

    class Stub:
        routes = ()

    with pytest.raises(RuntimeError, match="no longer serves"):
        bse.build_bundle(Stub())


def test_compartment_subsets_enumerate_the_whole_lattice():
    import build_static_explorer as bse

    from pharos.labels import Compartment

    subsets = bse._compartment_subsets()
    assert len(subsets) == 2 ** len(Compartment)
    assert [] in subsets
    assert sorted(str(c) for c in Compartment) in subsets
    assert all(s == sorted(s) for s in subsets), "keys must be order-independent"


def test_the_frozen_page_is_byte_identical_to_the_served_page(monkeypatch, tmp_path):
    """One page, two transports.

    The static build is only honest if it ships the same page the live app serves.
    A copy that drifted -- a tweak applied to one and not the other -- would be a
    second frontend wearing the first one's name, which is exactly what the shim
    design exists to avoid.
    """
    import build_static_explorer as bse

    from pharos.web import STATIC

    out = tmp_path / "explorer"
    monkeypatch.setattr(sys, "argv", ["build_static_explorer.py", "--out", str(out)])
    assert bse.main() == 0

    served = (STATIC / "index.html").read_text(encoding="utf-8")
    assert (out / "index.html").read_text(encoding="utf-8") == served
    # And the page has to carry the shim the freeze depends on, or the bundle
    # would ship next to a page that ignores it and silently calls a dead API.
    assert "bundle.json" in served
    assert "capacity_follows" in served

    import json as json_

    frozen = json_.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert frozen["seeds"] == list(bse.SEEDS)


# --------------------------------------------------------- review sweep -----


def _sweep_module():
    import measure_review_sweep as mrs

    return mrs


def test_majority_floor_is_the_larger_class():
    mrs = _sweep_module()
    from pharos.generate import GeneratorConfig, generate
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(
        generate(GeneratorConfig(seed=mrs.SEED, n_events=mrs.EVENTS)), limit=mrs.TASKS
    )
    floor = mrs.majority_floor(tasks)
    share = sum(1 for t in tasks if t.significant) / len(tasks)
    assert floor == max(share, 1 - share)
    assert floor >= 0.5, "a majority floor below half is a counting error"
    assert mrs.majority_floor([]) == 0.0


def test_a_correct_standard_with_no_slip_reproduces_the_world():
    """The control cell. If this is not 1.000 the sweep is measuring something else."""
    mrs = _sweep_module()
    from pharos.generate import GeneratorConfig, generate
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(
        generate(GeneratorConfig(seed=mrs.SEED, n_events=mrs.EVENTS)), limit=mrs.TASKS
    )
    proposals = {t.task_id: mrs.Proposal(t.task_id, not t.significant, t.label) for t in tasks}
    cells = mrs.sweep(tasks, proposals)
    control = next(c for c in cells if c.threshold == 3 and c.slip_rate == 0.0)
    assert control.mean == 1.0
    assert control.sd == 0.0, "a reviewer who never slips cannot vary across review seeds"


def test_the_most_wrong_standard_never_clears_the_floor():
    """What survived the corpus correction of 2026-08-03, asserted over the whole grid.

    This test used to assert that *no* wrong standard cleared the majority floor at any
    carefulness, and its docstring said that a cell clearing it would mean the finding
    needed rewriting. On the corrected corpus four cells cleared it -- the two-of-three
    standard at slips 0.00 to 0.15, topping out at 0.750 against a floor of 0.650 -- so
    the finding was rewritten and this assertion narrowed to the part that held.

    The narrowing is deliberate and it is not a weakening to make a red test green: the
    surviving claim is the one finding 8 actually rests on, which is the *exchange rate*
    between systematic and random error, and that is asserted below rather than dropped.
    A one-step standard error is still not reachable by any amount of carelessness at the
    correct standard, which is the qualitative result; what changed is that one step no
    longer takes a reviewer below the floor outright.
    """
    mrs = _sweep_module()
    from pharos.generate import GeneratorConfig, generate
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(
        generate(GeneratorConfig(seed=mrs.SEED, n_events=mrs.EVENTS)), limit=mrs.TASKS
    )
    floor = mrs.majority_floor(tasks)
    proposals = {t.task_id: mrs.Proposal(t.task_id, not t.significant, t.label) for t in tasks}
    cells = mrs.sweep(tasks, proposals)

    worst = min(mrs.THRESHOLDS)
    over = [(c.slip_rate, c.mean) for c in cells if c.threshold == worst and c.mean >= floor]
    assert over == [], f"the most wrong standard cleared the floor: {over}"


def test_systematic_error_costs_more_than_any_measured_carelessness():
    """Finding 8's actual claim: the exchange rate is lopsided, and by how much.

    A reviewer holding the *correct* standard and slipping at the highest rate measured
    still produces better targets than a reviewer holding the most wrong standard and
    never slipping at all. That is the asymmetry the finding is about, and unlike the
    floor comparison it does not depend on where the majority floor happens to sit.
    """
    mrs = _sweep_module()
    from pharos.generate import GeneratorConfig, generate
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(
        generate(GeneratorConfig(seed=mrs.SEED, n_events=mrs.EVENTS)), limit=mrs.TASKS
    )
    proposals = {t.task_id: mrs.Proposal(t.task_id, not t.significant, t.label) for t in tasks}
    cells = mrs.sweep(tasks, proposals)

    correct, worst = max(mrs.THRESHOLDS), min(mrs.THRESHOLDS)
    sloppiest = max((c for c in cells if c.threshold == correct), key=lambda c: c.slip_rate)
    careful_but_wrong = next(c for c in cells if c.threshold == worst and c.slip_rate == 0.0)
    assert sloppiest.mean > careful_but_wrong.mean, (
        f"correct standard at slip {sloppiest.slip_rate} scored {sloppiest.mean}, "
        f"not better than the most wrong standard at no slip ({careful_but_wrong.mean})"
    )


def test_the_grid_covers_every_standard():
    mrs = _sweep_module()
    from pharos.world import SIGNIFICANT_PATTERN

    assert set(mrs.THRESHOLDS) == set(range(1, len(SIGNIFICANT_PATTERN) + 1))
    assert 0.0 in mrs.SLIP_RATES, "the no-slip control has to be in the grid"
    assert len(mrs.REVIEW_SEEDS) > 1, "a spread needs more than one seed"


def test_equivalent_slip_is_a_measured_cell_not_an_interpolation():
    mrs = _sweep_module()

    cells = [
        mrs.Cell(3, 0.0, (1.0,)),
        mrs.Cell(3, 0.2, (0.8,)),
        mrs.Cell(3, 0.4, (0.5,)),
        mrs.Cell(2, 0.0, (0.6,)),
    ]
    # 0.6 is first undercut by the 0.4 cell, and 0.4 is on the grid.
    assert mrs.equivalent_slip(cells, 2, 0.625) == 0.4
    # A standard nothing on the grid reaches reports None rather than guessing.
    assert mrs.equivalent_slip([*cells, mrs.Cell(1, 0.0, (0.1,))], 1, 0.625) is None
    assert mrs.equivalent_slip(cells, 99, 0.625) is None


# --------------------------------------------------- review-taught adapter ----


def test_review_targets_match_the_reviewers_own_verdicts():
    """The teacher's targets must be the same calls it makes when reviewing.

    Both paths key the rng on `(seed, name, task_id)`. If they diverged, a teacher
    would slip on different tasks than the reviewer of the same name, and finding 8's
    target accuracy would not describe the stream the adapter actually trained on.
    """
    import random as _random

    import train_adapter as ta

    from pharos.analyst import AnalystPolicy, Proposal
    from pharos.disclosure import KEEP_COMPARTMENTS
    from pharos.labels import declassify

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=120)), limit=20)
    policy = AnalystPolicy("inattentive", slip_rate=0.15)

    targets = ta.review_targets(tasks, policy, seed=7)
    for task in tasks:
        proposal = Proposal(
            task.task_id, not task.significant, declassify(task.label, KEEP_COMPARTMENTS)
        )
        decision = policy.review(task, proposal, seed=7)
        expected = policy.verdict_for(task, _random.Random(f"7:{policy.name}:{task.task_id}"))
        assert targets[task.task_id] == expected
        if decision.corrected_verdict is not None and decision.action.value != "accept":
            assert targets[task.task_id] == decision.corrected_verdict


def test_a_correct_teacher_supplies_the_worlds_own_targets():
    import train_adapter as ta

    from pharos.analyst import AnalystPolicy

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=120)), limit=20)
    strict = ta.review_targets(tasks, AnalystPolicy("by-the-book"), seed=7)
    assert strict == ta.world_targets(tasks)


def test_a_wrong_standard_supplies_different_targets():
    import train_adapter as ta

    from pharos.analyst import AnalystPolicy

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=120)), limit=20)
    world = ta.world_targets(tasks)
    lenient = ta.review_targets(tasks, AnalystPolicy("any-one", escalation_threshold=1), seed=7)
    disagreements = [k for k in world if world[k] != lenient[k]]
    assert disagreements, "a one-of-three standard must differ from the world somewhere"
    # And it differs only by escalating: a lenient standard never calls a
    # significant event routine.
    assert all(lenient[k] and not world[k] for k in disagreements)


def test_build_examples_labels_from_the_target_map():
    import train_adapter as ta

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=120)), limit=6)
    flipped = {t.task_id: not t.significant for t in tasks}

    default = ta.build_examples(tasks)
    overridden = ta.build_examples(tasks, flipped)

    assert [e["prompt"] for e in default] == [e["prompt"] for e in overridden]
    for base, other in zip(default, overridden, strict=True):
        assert base["completion"] != other["completion"]
    assert default == ta.build_examples(tasks, ta.world_targets(tasks))


def test_resolve_reviewer_rejects_an_unknown_name():
    import train_adapter as ta

    assert ta.resolve_reviewer("by-the-book").escalation_threshold == 3
    with pytest.raises(SystemExit, match="unknown reviewer"):
        ta.resolve_reviewer("nobody")


# ------------------------------------------------------ decode stability -----


def test_short_parser_refuses_an_answer_saying_both_or_neither():
    import measure_decode_stability as mds

    assert mds.parse_short("SIGNIFICANT") is True
    assert mds.parse_short("routine") is False
    # Saying both is not a verdict, and neither is saying nothing. Coercing either
    # to a class would move every score built on this parser.
    assert mds.parse_short("SIGNIFICANT or ROUTINE, unclear") is None
    assert mds.parse_short("") is None


def test_the_two_regimes_differ_in_the_thing_being_compared():
    """The comparison is meaningless if both regimes decode the same length."""
    import measure_decode_stability as mds

    lengths = {r.num_predict for r in mds.REGIMES}
    assert len(lengths) == len(mds.REGIMES), "regimes must differ in decode length"
    assert {r.reasons for r in mds.REGIMES} == {True, False}
    assert min(lengths) < max(lengths)


def test_stability_reports_a_share_and_survives_an_empty_run():
    import measure_decode_stability as mds

    row = mds.Stability(
        regime="r", num_predict=8, n_tasks=30, repeats=3, unstable=3, unparsed_any=0
    )
    assert row.unstable_share == 0.1
    assert row.as_dict()["unstable_share"] == 0.1
    assert (
        mds.Stability(
            regime="r", num_predict=8, n_tasks=0, repeats=3, unstable=0, unparsed_any=0
        ).unstable_share
        == 0.0
    )


def test_stability_measure_counts_only_self_disagreement(monkeypatch):
    """A task is unstable when its own repeats differ, not when it is merely wrong."""
    import measure_decode_stability as mds

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=120)), limit=4)
    answers = iter(
        [
            "SIGNIFICANT",
            "SIGNIFICANT",
            "SIGNIFICANT",  # stable
            "SIGNIFICANT",
            "ROUTINE",
            "ROUTINE",  # unstable
            "ROUTINE",
            "ROUTINE",
            "ROUTINE",  # stable
            "ROUTINE",
            "ROUTINE",
            "nothing parseable",  # unstable AND unparsed
        ]
    )
    monkeypatch.setattr(mds, "generate_text", lambda *a, **k: next(answers))

    stability, unstable = mds.measure(mds.REGIMES[0], tasks, repeats=3, model="m", endpoint="e")
    assert stability.unstable == 2
    assert len(unstable) == 2
    assert (
        stability.tasks_with_an_unparsed_call
        if False
        else stability.as_dict()["tasks_with_an_unparsed_call"] == 1
    )


# ------------------------------------------------------ teacher transfer -----


def test_teacher_labels_match_the_reviewers_own_calls():
    import measure_teacher_transfer as mtt

    from pharos.analyst import AnalystPolicy

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=120)), limit=20)
    strict = mtt.teacher_labels(tasks, AnalystPolicy("by-the-book"), seed=7)
    assert strict == {t.task_id: t.significant for t in tasks}

    lenient = mtt.teacher_labels(tasks, AnalystPolicy("any-one", escalation_threshold=1), seed=7)
    assert lenient != strict


def test_transfer_row_rates_exclude_unparsed_from_the_denominator():
    """An unparsable answer is not a wrong answer, and must not be scored as one."""
    import measure_teacher_transfer as mtt

    row = mtt.Row(
        teacher="t", shots=8, n=10, unparsed=2, agree_world=4, agree_teacher=8, said_significant=6
    )
    assert row.world_rate == 0.5
    assert row.teacher_rate == 1.0
    assert row.escalation_rate == 0.75
    payload = row.as_dict()
    assert payload["agreement_with_teacher"] == 1.0
    assert payload["n"] == 10 and payload["unparsed"] == 2

    assert (
        mtt.Row(
            teacher="t",
            shots=0,
            n=0,
            unparsed=0,
            agree_world=0,
            agree_teacher=0,
            said_significant=0,
        ).world_rate
        == 0.0
    )


def test_transfer_scores_the_same_decode_against_both_answer_keys(monkeypatch):
    import measure_teacher_transfer as mtt

    from pharos.analyst import AnalystPolicy

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=120)), limit=8)
    targets, pool = tasks[:4], tasks[4:]
    labels = mtt.teacher_labels(tasks, AnalystPolicy("any-one", escalation_threshold=1), seed=7)
    monkeypatch.setattr(mtt, "generate_text", lambda *a, **k: "VERDICT: SIGNIFICANT")

    row = mtt.run_condition(
        targets, pool, labels, teacher="any-one", shots=0, model="m", endpoint="e"
    )
    assert row.n == len(targets)
    assert row.escalation_rate == 1.0
    # Answering SIGNIFICANT everywhere agrees with whichever key says SIGNIFICANT.
    assert row.agree_world == sum(1 for t in targets if t.significant)
    assert row.agree_teacher == sum(1 for t in targets if labels[t.task_id])


def test_the_teacher_list_spans_right_and_wrong_standards():
    import measure_teacher_transfer as mtt

    from pharos.analyst import DEFAULT_ENSEMBLE

    by_name = {p.name: p for p in DEFAULT_ENSEMBLE}
    thresholds = {by_name[t].escalation_threshold for t in mtt.TEACHERS}
    assert 3 in thresholds, "the correct standard is the control and must be present"
    assert len(thresholds) > 1, "a transfer test needs teachers that disagree"


# ------------------------------------------------------- sweep intervals -----


def test_compare_models_builds_trials_that_drop_unparsable_verdicts():
    """An unparsable answer is not a wrong answer, and must not enter the interval."""
    import compare_models as cm

    from pharos.uncertainty import Trial, cluster_bootstrap

    rows = [
        {"task_id": "a", "truth": True, "verdict": True},
        {"task_id": "b", "truth": True, "verdict": False},
        {"task_id": "c", "truth": False, "verdict": None},
    ]
    trials = [
        Trial(str(r["task_id"]), None if r["verdict"] is None else r["verdict"] == r["truth"])
        for r in rows
    ]
    assert [t.outcome for t in trials] == [True, False, None]
    # And the interval is computed over the two answers, not three.
    interval = cluster_bootstrap(trials, resamples=200)
    assert 0.0 <= interval.low <= interval.point <= interval.high <= 1.0
    assert cm.RESULTS.name == "results"


def test_the_floor_claim_is_reported_as_unresolved_when_an_interval_reaches_it():
    """The correction this analysis forced, asserted rather than described.

    A point estimate below the majority floor whose interval reaches above it has
    not shown the model fails to clear the floor. If this stops holding, finding 3b's
    corrected wording is wrong and should fail here first.
    """
    import json

    from pharos.uncertainty import Trial, cluster_bootstrap

    reaching = []
    for path in sorted(cm_results().glob("triage_lift-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            continue
        trials = [
            Trial(r["task_id"], None if r["verdict"] is None else r["verdict"] == r["truth"])
            for r in rows
        ]
        interval = cluster_bootstrap(trials, resamples=400)
        if interval.high > payload["majority_accuracy"] >= interval.point:
            reaching.append(payload["provenance"].get("model_key", path.stem))

    assert reaching, (
        "no model's interval reaches the majority floor; finding 3b's 'unresolved' "
        "wording no longer matches the artifacts"
    )


def cm_results():
    import compare_models as cm

    return cm.RESULTS


# ----------------------------------------------------- cross-corpus eval -----


def test_two_seeds_share_no_rendered_report():
    """The premise of the cross-corpus evaluation, checked rather than assumed.

    Training on one seed and evaluating on another is only the stronger claim if the
    two corpora genuinely share nothing. Event ids collide across seeds by
    construction -- they are positional -- so identity has to be checked on the
    rendered text, which is what the trainer does before it trains.
    """
    a = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=200)), limit=60)
    b = build_triage_tasks(generate(GeneratorConfig(seed=101, n_events=200)), limit=60)

    assert {t.prompt for t in a} & {t.prompt for t in b} == set()
    # Event ids DO collide, which is why the text check exists and the id check
    # would have passed vacuously.
    assert {t.event_id for t in a} & {t.event_id for t in b}


def test_cross_corpus_eval_still_shares_the_rule():
    """Different corpora, same decision rule -- otherwise the transfer test is void."""
    from pharos.analyst import evidence_shown
    from pharos.world import SIGNIFICANT_PATTERN

    for seed in (7, 101, 202):
        tasks = build_triage_tasks(generate(GeneratorConfig(seed=seed, n_events=200)), limit=40)
        for task in tasks:
            assert task.significant == (evidence_shown(task) == SIGNIFICANT_PATTERN)


def test_trainer_exposes_the_eval_seed_flag():
    import train_adapter as ta

    assert hasattr(ta, "world_targets")
    src = pathlib.Path(ta.__file__).read_text(encoding="utf-8")
    assert "--eval-seed" in src
    # And records it in the artifact, so a cross-corpus number cannot be mistaken
    # for a within-corpus one when it is read back later.
    assert '"cross_corpus"' in src


# -------------------------------------------------------- fleet linkage -----


def _linkage_module():
    import measure_fleet_linkage as mfl

    return mfl


def _small_fleet_and_stream():
    from pharos.disclosure import DROP_COMPARTMENTS
    from pharos.fleet import assign_fleet, contribute
    from pharos.generate import GeneratorConfig, generate
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=40)))
    fleet = assign_fleet(20, seed=11)
    return tasks, fleet, contribute(fleet, tasks, policy=DROP_COMPARTMENTS)


def test_recovery_and_anonymity_are_empty_safe():
    mfl = _linkage_module()
    assert mfl.recovery_rate(()) == 0.0
    assert mfl.mean_anonymity(()) == 0.0
    assert mfl.majority_prior(()) == 0.0


def test_majority_prior_is_the_most_common_beat():
    """The floor a linkage attack has to beat, and it is not 1/16 once a fleet is drawn."""
    mfl = _linkage_module()
    from pharos.fleet import assign_fleet

    fleet = assign_fleet(200, seed=11)
    prior = mfl.majority_prior(fleet)
    share = max(
        sum(1 for c in fleet if c.compartments == target) / len(fleet)
        for target in {c.compartments for c in fleet}
    )
    assert prior == pytest.approx(share)
    assert prior >= 1 / mfl.N_COMPARTMENT_SETS


def test_trials_cluster_on_the_analyst_not_the_task():
    """Tasks are shared across the fleet, so a task-clustered resample double-counts people."""
    mfl = _linkage_module()
    from pharos.disclosure import DROP_COMPARTMENTS
    from pharos.fleet import link

    tasks, fleet, stream = _small_fleet_and_stream()
    linkages = link(stream, tasks, fleet, policy=DROP_COMPARTMENTS)
    trials = mfl.trials(linkages)
    assert len(trials) == len(fleet)
    assert len({t.task_id for t in trials}) == len(fleet)


def test_evaluate_prices_a_control_in_retained_volume():
    mfl = _linkage_module()
    from pharos.disclosure import DROP_COMPARTMENTS
    from pharos.fleet import apply_pooling

    tasks, fleet, stream = _small_fleet_and_stream()
    full = mfl.evaluate(fleet, tasks, stream, policy=DROP_COMPARTMENTS, baseline_volume=len(stream))
    assert full["retained_volume"] == 1.0

    pooled = mfl.evaluate(
        fleet, tasks, apply_pooling(stream), policy=DROP_COMPARTMENTS, baseline_volume=len(stream)
    )
    # Pooling costs no volume and takes recovery to zero: the whole point of the row.
    assert pooled["retained_volume"] == 1.0
    assert pooled["recovery"] == 0.0

    # A zero baseline is reported as zero rather than dividing by it.
    assert (
        mfl.evaluate(fleet, tasks, (), policy=DROP_COMPARTMENTS, baseline_volume=0)[
            "retained_volume"
        ]
        == 0.0
    )


# ---------------------------------------------------------- power analysis -----


def _power_module():
    import measure_power as mp

    return mp


def test_a_claim_against_a_constant_costs_one_half_width_not_two():
    """The distinction that flipped a published verdict, so it gets a test.

    A majority floor and a stated ceiling are computed exactly from a generated
    corpus and carry no sampling noise. Charging such a claim two half-widths (the
    two-condition price) reported finding 3b's mistral result as unresolved while
    finding 3b's own interval table showed it clearing outright.
    """
    mp = _power_module()
    against_constant = mp.Claim("x", 40, 0.2, "vs a known floor", True)
    two_conditions = mp.Claim("x", 40, 0.2, "vs another measurement")

    assert against_constant.threshold(0.15) == pytest.approx(0.15)
    assert two_conditions.threshold(0.15) == pytest.approx(0.30)
    # The gap that motivated the correction: resolved against a constant, not
    # against a second sampled condition.
    assert against_constant.effect > against_constant.threshold(0.15)
    assert two_conditions.effect < two_conditions.threshold(0.15)


def test_half_width_shrinks_with_n():
    mp = _power_module()
    wide = mp.half_width(30, 0.5, resamples=200, seed=7)
    narrow = mp.half_width(600, 0.5, resamples=200, seed=7)
    assert narrow < wide
    assert 0.0 < narrow < 0.2


def test_minimum_detectable_is_twice_the_half_width():
    """It is the strict end of the three rules, and the docstring says so."""
    mp = _power_module()
    half = mp.half_width(120, 0.5, resamples=200, seed=7)
    assert mp.minimum_detectable(120, 0.5, resamples=200, seed=7) == pytest.approx(2 * half)


def test_every_claim_carries_a_finding_and_a_description():
    """A claim with no finding behind it is a number nobody can check."""
    mp = _power_module()
    claims = mp._claims()
    assert claims
    for claim in claims:
        assert claim.finding and claim.description
        assert claim.n > 0
        assert 0.0 <= claim.effect <= 1.0
        if claim.rate is not None:
            assert 0.0 < claim.rate < 1.0


def test_every_number_a_claim_quotes_is_in_the_artifact_it_came_from():
    """The claims were hand-typed once, and went stale the next time anything moved.

    They fed a *generated* block, which meant the page rendered a hand-written number
    with all the authority of a measured one. The visible symptom was a claim reading
    `qwen2.5-3b (0.625) clears the majority floor (0.625)` -- a dead heat, reasoned
    about at length in the prose -- while the artifact said 0.775 against 0.650, and
    the 0.625 belonged to a different model in the same table.

    So every numeral a claim quotes has to appear in the artifact behind it. This is
    the property, not the values: it keeps holding when the corpus is re-measured.
    """
    import json as _json
    import re

    mp = _power_module()
    quoted = re.compile(r"\((\d\.\d{3})\)")

    # Every value present anywhere in any committed artifact, rounded the way the
    # descriptions render it. A quoted number absent from this set was typed, not read.
    measured: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            measured.add(f"{float(node):.3f}")

    for path in (SCRIPTS.parent / "results").glob("*.json"):
        walk(_json.loads(path.read_text(encoding="utf-8")))

    # 1.000 is the stated-rule ceiling, a definition rather than a measurement.
    exempt = {"1.000"}
    for claim in mp._claims():
        for value in quoted.findall(claim.description):
            assert value in measured or value in exempt, (
                f"claim {claim.finding!r} quotes {value}, which appears in no artifact: "
                f"{claim.description!r}"
            )


def test_class_balance_is_read_from_the_corpus_not_assumed():
    mp = _power_module()
    from pharos.generate import GeneratorConfig, generate
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=mp.SEED, n_events=60)))
    rate = mp.class_balance(tasks)
    assert rate == pytest.approx(sum(1 for t in tasks if t.significant) / len(tasks))
    assert mp.class_balance([]) == 0.5


def test_saturation_reports_both_regimes():
    """Small corpora disagree about the structure; large ones do not.

    The claim finding 11 rests on is that its headline is a property of the label
    lattice rather than of one corpus draw. That is only true above a size, and the
    grid has to show the size to establish it rather than assert it.
    """
    mfl = _linkage_module()
    from pharos.disclosure import DROP_COMPARTMENTS

    report = mfl.saturation(policy=DROP_COMPARTMENTS, seeds=(1, 7, 101))
    grid = report["grid"]
    assert [r["events"] for r in grid] == list(mfl.SATURATION_EVENTS)

    # The largest size tested must agree across seeds, or the headline is a draw.
    assert grid[-1]["invariant"]
    assert grid[-1]["distinct_structures"] == 1
    assert report["saturates_at"] is not None
    assert report["saturates_at"] <= mfl.SATURATION_EVENTS[-1]

    # Everything at or above the saturation point agrees; that is what the field means.
    assert all(r["invariant"] for r in grid if r["events"] >= report["saturates_at"])


def test_small_corpora_understate_rather_than_overstate():
    """The direction of the small-sample error, which is the part worth trusting."""
    mfl = _linkage_module()
    from pharos.disclosure import DROP_COMPARTMENTS

    report = mfl.saturation(policy=DROP_COMPARTMENTS, seeds=(1, 7, 101))
    grid = {r["events"]: r for r in report["grid"]}
    saturated = max(grid[e]["recoveries"][-1] for e in grid if grid[e]["invariant"])
    smallest = grid[mfl.SATURATION_EVENTS[0]]["recoveries"]
    # No small corpus reports MORE leakage than the saturated structure carries.
    assert max(smallest) <= saturated


# --------------------------------------------------------- docs tables -----


def _docs_module():
    import sync_docs_tables as sdt

    return sdt


def test_power_claims_block_matches_the_artifact():
    sdt = _docs_module()
    import json as _json

    table = sdt.power_claims()
    payload = _json.loads((sdt.RESULTS / "power.json").read_text(encoding="utf-8"))
    claims = payload["claims"]
    # One row per claim, plus a header, a separator, a blank and a tally line.
    assert table.count("\n| ") >= len(claims)
    resolved = sum(1 for c in claims if c["resolved"])
    assert f"**{resolved} of {len(claims)}**" in table
    for claim in claims:
        assert claim["description"] in table


def test_render_rewrites_only_marked_blocks():
    sdt = _docs_module()
    text = (
        "keep me\n\n<!-- BEGIN GENERATED: power-claims -->\nSTALE\n"
        "<!-- END GENERATED: power-claims -->\n\nkeep me too\n"
    )
    updated, names = sdt.render(text)
    assert names == ["power-claims"]
    assert "STALE" not in updated
    assert updated.startswith("keep me\n")
    assert updated.endswith("keep me too\n")
    # Idempotent: rendering an already-current document changes nothing.
    assert sdt.render(updated)[0] == updated


def test_an_unregistered_block_is_an_error_not_a_no_op():
    """A marker with no builder is how a table quietly stops updating."""
    sdt = _docs_module()
    text = "<!-- BEGIN GENERATED: not-a-real-block -->\nx\n<!-- END GENERATED: not-a-real-block -->"
    with pytest.raises(SystemExit):
        sdt.render(text)


def test_unmarked_text_is_left_alone():
    sdt = _docs_module()
    text = "| Finding | n |\n| --- | --- |\n| 5 | 600 |\n"
    updated, names = sdt.render(text)
    assert names == []
    assert updated == text


@pytest.mark.parametrize("name", sorted(_docs_module().BLOCKS))
def test_every_registered_block_builds_a_table_from_its_artifact(name):
    """Each builder was reachable only through `make docs-tables` until now.

    Seven of the eight ran in no test, so a builder that raised on a regenerated
    artifact -- a renamed key, a null where a number was assumed, an empty row list --
    surfaced in CI as a `make` failure with a traceback rather than as a named test.
    They are cheap to call and they read the committed artifacts, which is exactly the
    coupling that goes stale.
    """
    table = _docs_module().BLOCKS[name]()
    lines = table.splitlines()
    # The header is not required to be the first line: a block may open with a caption
    # or a sub-heading above its table, and channel-bias emits one table per noise
    # level. What is required is that a header exists, is followed by a separator, and
    # has rows under it.
    header = next(
        (
            i
            for i, line in enumerate(lines[:-2])
            if line.startswith("| ") and set(lines[i + 1]) <= set("| -")
        ),
        None,
    )
    assert header is not None, f"{name} did not emit a table header with a separator under it"
    # A header and a separator with nothing under them is an empty table, and every
    # builder is documented as refusing to emit one rather than emitting a shell.
    assert any(line.startswith("| ") for line in lines[header + 2 :]), f"{name} emitted no rows"


@pytest.mark.parametrize("name", sorted(_docs_module().BLOCKS))
def test_a_builder_refuses_a_missing_artifact_rather_than_emitting_a_shell(name, monkeypatch):
    """Every builder promises this in its docstring; none of them was checked.

    The failure it prevents is the one this project's tooling has already hit twice: a
    guard that passes because it matched nothing. A builder that returned a bare header
    when its artifact was absent would leave a header and a separator in the published
    doc, and `--check` would call that current.
    """
    sdt = _docs_module()
    # Under ROOT on purpose: `_fail` reports `path.relative_to(ROOT)`, so a directory
    # outside the tree would raise ValueError from the error path itself.
    monkeypatch.setattr(sdt, "RESULTS", sdt.ROOT / "no-such-results-dir")
    with pytest.raises(SystemExit) as raised:
        sdt.BLOCKS[name]()
    assert raised.value.code == 2


def test_no_doc_carries_a_block_without_a_builder():
    """The inverse of `render`'s guard, checked across the whole docs tree.

    `render` refuses an unregistered name when it reaches one, but only for text it is
    handed. This asserts the property over every file the sync would ever walk.
    """
    sdt = _docs_module()
    referenced = {
        match.group("name")
        for path in sorted(sdt.DOCS.rglob("*.md"))
        for match in sdt._OPENER.finditer(path.read_text(encoding="utf-8"))
    }
    assert referenced, "no generated blocks found in docs/; this test went blind"
    assert referenced <= set(sdt.BLOCKS), f"blocks with no builder: {referenced - set(sdt.BLOCKS)}"
    unused = set(sdt.BLOCKS) - referenced
    assert not unused, f"builders no document references: {unused}"


def test_the_committed_docs_match_the_committed_artifacts():
    """`sync_docs_tables.py --check`, as a test rather than only as a CI step.

    The check ran in CI and in `make ci`, which meant a stale table failed the build
    with a diff-less message at a step most local runs skip. Running it here names the
    file and the block, and makes `make test` alone sufficient to catch a number that
    drifted from the artifact it was read off.
    """
    sdt = _docs_module()
    stale = []
    rendered = 0
    for path in sorted(sdt.DOCS.rglob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated, names = sdt.render(original)
        rendered += len(names)
        if names and updated != original:
            stale.append(f"{path.relative_to(sdt.ROOT)} ({', '.join(names)})")
    assert rendered, "no generated block was rendered; the check verified nothing"
    assert not stale, f"stale generated blocks; run `make docs-tables`: {stale}"


# ------------------------------------------------- consensus reliability -----


def _consensus_module():
    import measure_consensus_reliability as mcr

    return mcr


def test_agreement_reports_an_empty_stream_as_no_evidence():
    """None, not 0.0. A condition that kept nothing has not scored zero."""
    mcr = _consensus_module()
    assert mcr._agreement([], {}) is None
    truth = {"T-1": True, "T-2": False}
    assert mcr._agreement([("T-1", "a", True)], truth) == 1.0
    assert mcr._agreement([("T-1", "a", False)], truth) == 0.0
    assert mcr._show(None) == "none"
    assert mcr._show(0.0) == "0.0000"


def test_fleet_of_splits_right_and_wrong():
    mcr = _consensus_module()
    fleet = mcr.fleet_of(4, size=9)
    assert len(fleet) == 9
    wrong = [p for p in fleet if p.escalation_threshold == mcr.WRONG_THRESHOLD]
    assert len(wrong) == 4
    assert len(mcr.fleet_of(0, size=9)) == 9
    assert all(p.escalation_threshold != mcr.WRONG_THRESHOLD for p in mcr.fleet_of(0, size=9))


def _row(mcr, *, n_wrong: int, oracle: float | None, consensus: float | None):
    """A Row with only the two fields the cliff is computed from set meaningfully.

    Named fields, not positions: this test previously built rows positionally and went
    on constructing them silently against a changed field order until `dawid_skene`
    made the arity mismatch loud.
    """
    return mcr.Row(
        n_wrong=n_wrong,
        share_wrong=round(n_wrong / 9, 4),
        unweighted=0.7,
        oracle=oracle,
        consensus=consensus,
        dawid_skene=0.7,
        consensus_interval={},
        targets={},
    )


def test_an_absent_stream_is_not_scored_as_a_lag():
    """The far end of the sweep has an oracle that refused everything, not one that lost."""
    mcr = _consensus_module()
    empty_oracle = _row(mcr, n_wrong=9, oracle=None, consensus=0.7)
    assert not empty_oracle.consensus_lags_oracle

    empty_consensus = _row(mcr, n_wrong=9, oracle=0.7, consensus=None)
    assert not empty_consensus.consensus_lags_oracle

    lagging = _row(mcr, n_wrong=5, oracle=1.0, consensus=0.717)
    assert lagging.consensus_lags_oracle

    matching = _row(mcr, n_wrong=4, oracle=1.0, consensus=1.0)
    assert not matching.consensus_lags_oracle

    # The 0.01 tolerance: a gap this small is rounding, not a cliff.
    within_tolerance = _row(mcr, n_wrong=4, oracle=1.0, consensus=0.995)
    assert not within_tolerance.consensus_lags_oracle


def test_consensus_matches_the_oracle_until_the_wrong_standard_is_the_majority():
    """The finding, at a small size: a cliff at the majority crossing, not a slope."""
    mcr = _consensus_module()
    from pharos.analyst import Proposal
    from pharos.disclosure import KEEP_COMPARTMENTS
    from pharos.generate import GeneratorConfig, generate
    from pharos.labels import declassify
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=mcr.SEED, n_events=60)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    def score(n_wrong: int) -> tuple[float | None, float | None]:
        grouped = mcr.targets_by_task(mcr.fleet_of(n_wrong, 9), tasks, proposals, seed=mcr.SEED)
        streams = mcr.conditions(grouped, truth)
        return (
            mcr._agreement(streams["oracle"], truth),
            mcr._agreement(streams["consensus"], truth),
        )

    # A clear minority of wrong standards: consensus is as good as knowing who is who.
    oracle_minority, consensus_minority = score(2)
    assert consensus_minority == pytest.approx(oracle_minority)

    # A clear majority: consensus ratifies the wrong rule while the oracle does not.
    oracle_majority, consensus_majority = score(7)
    assert oracle_majority is not None and consensus_majority is not None
    assert consensus_majority < oracle_majority


# ------------------------------------------------- tagged aggregation -----


def _tagged_module():
    import measure_tagged_aggregation as mta

    return mta


def test_a_coarse_tag_never_identifies_an_individual():
    """True, and the point is that it is not sufficient. See the aggregate test below."""
    mta = _tagged_module()
    from pharos.disclosure import DROP_COMPARTMENTS
    from pharos.fleet import assign_fleet, contribute, link
    from pharos.generate import GeneratorConfig, generate
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=mta.SEED, n_events=60)))
    fleet = assign_fleet(60, seed=mta.FLEET_SEED)
    stream = contribute(fleet, tasks, policy=DROP_COMPARTMENTS)

    for scheme in mta.SCHEMES:
        if scheme.name == "per_person":
            continue
        tagged = mta.tag(stream, fleet, scheme, seed=mta.SEED)
        linkages = link(tagged, tasks, fleet, policy=DROP_COMPARTMENTS)
        assert not any(x.exact for x in linkages), scheme.name


def test_a_correlated_tag_leaks_clearance_that_the_per_analyst_metric_misses():
    """The finding. Both schemes name nobody; only one of them keeps the secret."""
    mta = _tagged_module()
    from pharos.fleet import assign_fleet

    fleet = assign_fleet(200, seed=mta.FLEET_SEED)
    prior = mta.prior_accuracy(fleet)
    schemes = {s.name: s for s in mta.SCHEMES}

    independent, n_indep = mta.clearance_inference(
        fleet, schemes["tier_independent"], seed=mta.SEED
    )
    correlated, n_corr = mta.clearance_inference(fleet, schemes["tier_correlated"], seed=mta.SEED)
    assert n_indep == n_corr == mta.TIERS, "same partition size, different partition"

    assert independent == pytest.approx(prior, abs=0.05), "independent tag must not leak"
    assert correlated > prior + 0.2, "correlated tag must leak clearance"


def test_pooling_gives_exactly_the_prior():
    """One group cannot beat naming the most common level, by construction."""
    mta = _tagged_module()
    from pharos.fleet import assign_fleet

    fleet = assign_fleet(200, seed=mta.FLEET_SEED)
    pooled = next(s for s in mta.SCHEMES if s.name == "pooled")
    inferred, groups = mta.clearance_inference(fleet, pooled, seed=mta.SEED)
    assert groups == 1
    assert inferred == pytest.approx(mta.prior_accuracy(fleet))


def test_scheme_result_flags_the_invisible_case():
    mta = _tagged_module()
    invisible = mta.SchemeResult("x", "", 0.0, 0.82, 0.49, 3)
    assert invisible.leaks_in_aggregate
    assert invisible.invisible_to_per_analyst_metric

    visible = mta.SchemeResult("y", "", 0.205, 1.0, 0.67, 200)
    assert visible.leaks_in_aggregate
    assert not visible.invisible_to_per_analyst_metric, "a named person is not invisible"

    clean = mta.SchemeResult("z", "", 0.0, 0.33, 0.0, 3)
    assert not clean.leaks_in_aggregate
    assert not clean.invisible_to_per_analyst_metric


# ------------------------------------------------------------ edge cost -----


def _edge_module():
    import measure_edge_cost as mec

    return mec


def test_sync_cost_refuses_to_guess_a_payload(tmp_path):
    """No artifact and no override means no number, not an invented one."""
    mec = _edge_module()
    assert mec.sync_cost(tmp_path) is None


def test_sync_cost_records_where_the_counts_came_from(tmp_path):
    mec = _edge_module()
    supplied = mec.sync_cost(tmp_path, trainable=29_933_568, total=3_115_872_256)
    assert supplied is not None
    assert supplied.trainable_share == pytest.approx(0.00961, abs=1e-4)
    # 29.9M params at two bytes each, in MiB.
    assert supplied.bf16_mib == pytest.approx(57.1, abs=0.2)
    assert supplied.fp32_mib == pytest.approx(2 * supplied.bf16_mib)
    assert "command line" in supplied.source

    import json as _json

    (tmp_path / "adapter_learnability.json").write_text(
        _json.dumps({"lora": {"trainable_params": 1_000_000, "total_params": 100_000_000}}),
        encoding="utf-8",
    )
    from_artifact = mec.sync_cost(tmp_path)
    assert from_artifact is not None
    assert from_artifact.source.endswith("adapter_learnability.json")
    assert from_artifact.trainable_share == pytest.approx(0.01)


def test_cold_start_is_kept_apart_from_warm_steady_state():
    """Folding them together hides the largest cost and inflates p95 into nonsense."""
    mec = _edge_module()
    # One slow first call, then a tight warm cluster.
    latency = mec.split_latency([29.75, 0.34, 0.33, 0.35, 0.32, 0.36])
    assert latency.cold_start_s == pytest.approx(29.75)
    assert latency.warm_median_s == pytest.approx(0.34, abs=0.02)
    assert latency.warm_p95_s < 1.0, "the cold call must not leak into the warm p95"
    assert latency.n_warm == 5
    assert latency.decisions_per_hour == pytest.approx(3600 / latency.warm_median_s, rel=1e-6)
    # The wake-up cost expressed in the unit that makes it comparable.
    assert latency.cold_start_ratio > 50


def test_split_latency_needs_more_than_one_call():
    mec = _edge_module()
    with pytest.raises(SystemExit):
        mec.split_latency([1.0])


def test_footprint_is_empty_without_a_server():
    """No server is a missing measurement, not a fleet of zero-byte models."""
    mec = _edge_module()
    assert mec.footprint("http://127.0.0.1:1/api/tags") == []


def test_edge_cost_dicts_round_trip_to_json():
    """Both artifact rows must serialize; a NaN or a stray object here kills the run."""
    mec = _edge_module()
    import json as _json

    sync = mec.sync_cost(pathlib.Path("/nonexistent"), trainable=29_933_568, total=3_115_872_256)
    assert sync is not None
    encoded = _json.loads(_json.dumps(sync.as_dict()))
    assert encoded["trainable_params"] == 29_933_568
    assert encoded["source"]

    latency = mec.split_latency([4.8, 0.34, 0.33, 0.35])
    row = _json.loads(_json.dumps(latency.as_dict()))
    assert row["cold_start_s"] == pytest.approx(4.8)
    assert row["decisions_per_hour"] > 0
    assert row["cold_start_in_warm_decisions"] == pytest.approx(
        4.8 / row["warm_median_s"], rel=0.01
    )


def test_evicting_a_missing_server_is_reported_not_raised():
    """A failed eviction must warn rather than silently produce a warm 'cold start'."""
    mec = _edge_module()
    assert mec.evict("qwen2.5:7b-instruct", endpoint="http://127.0.0.1:1/api/generate") is False


def test_sync_cost_with_zero_total_does_not_divide_by_zero():
    mec = _edge_module()
    degenerate = mec.SyncCost(
        trainable_params=0, total_params=0, bf16_mib=0, fp32_mib=0, source="x"
    )
    assert degenerate.trainable_share == 0.0


# -------------------------------------------------------- privacy budget -----


def _privacy_module():
    import measure_privacy_budget as mpb

    return mpb


def test_privacy_budget_row_serializes():
    mpb = _privacy_module()
    import json as _json

    row = mpb.Row(
        mechanism="participation",
        setting="keep=0.6,fab=0.4",
        recovery=0.065,
        contributions=1200,
        label_noise=0.756,
        budget={"epsilon": 0.41},
    )
    encoded = _json.loads(_json.dumps(row.as_dict()))
    assert encoded["mechanism"] == "participation"
    assert encoded["recovery"] == pytest.approx(0.065)
    assert encoded["budget"]["epsilon"] == pytest.approx(0.41)

    # A value-noise row carries no budget, and that must survive the round trip.
    bare = mpb.Row(
        mechanism="value",
        setting="flip=0.5",
        recovery=0.205,
        contributions=1200,
        label_noise=0.0,
        budget=None,
    )
    assert _json.loads(_json.dumps(bare.as_dict()))["budget"] is None


def test_privacy_budget_grid_is_ordered_and_covers_the_degenerate_case():
    """fabricate=0 must be present: it is subsampling, and its epsilon is infinite."""
    mpb = _privacy_module()
    assert (1.0, 0.0) in mpb.PARTICIPATION
    assert any(fab == 0.0 for _, fab in mpb.PARTICIPATION)
    # A flip of 0.5 destroys the verdict; if the attack survives that it survives all.
    assert 0.5 in mpb.VALUE_FLIPS
    assert 0.0 in mpb.VALUE_FLIPS, "a no-noise control is needed to read the rest against"


def test_correlated_score_fleet_reports_the_majority_flag():
    """The flag drives the exact/measured composition, so it has to be right."""
    import measure_correlated_fleets as mcf

    from pharos.analyst import Proposal
    from pharos.disclosure import KEEP_COMPARTMENTS
    from pharos.generate import GeneratorConfig, generate
    from pharos.labels import declassify
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=mcf.SEED, n_events=30)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}

    from random import Random

    all_right = mcf.draw_fleet(0.0, schools=3, rng=Random(1))
    consensus, ds, crossed = mcf.score_fleet(all_right, tasks, proposals, truth, seed=mcf.SEED)
    assert not crossed
    assert consensus == pytest.approx(1.0)
    assert ds == pytest.approx(1.0)

    all_wrong = mcf.draw_fleet(1.0, schools=3, rng=Random(1))
    consensus_w, _, crossed_w = mcf.score_fleet(all_wrong, tasks, proposals, truth, seed=mcf.SEED)
    assert crossed_w
    assert consensus_w < 1.0


def test_a_marker_pair_that_renders_nothing_is_an_error():
    """The guard's own blind spot, found by hitting it.

    A BEGIN/END pair on consecutive lines does not match the render pattern, so the
    block silently produced no output while `--check` reported everything current.
    Counting rendered blocks cannot detect a block that was never rendered; the
    declared markers have to be counted separately.
    """
    sdt = _docs_module()

    adjacent = "<!-- BEGIN GENERATED: power-claims -->\n<!-- END GENERATED: power-claims -->\n"
    _, names = sdt.render(adjacent)
    assert names == [], "the pattern genuinely cannot match this"
    # But the opener is still declared, which is what the guard keys on.
    assert [m.group("name") for m in sdt._OPENER.finditer(adjacent)] == ["power-claims"]

    mismatched = "<!-- BEGIN GENERATED: power-claims -->\nx\n<!-- END GENERATED: power-clams -->\n"
    _, names = sdt.render(mismatched)
    assert names == []
    assert [m.group("name") for m in sdt._OPENER.finditer(mismatched)] == ["power-claims"]


def test_measurement_health_publishes_the_flag_rather_than_policing_it():
    sdt = _docs_module()

    table = sdt.measurement_health()
    assert "| Artifact | n | Quotable | Why not |" in table
    # The flagged artifacts must appear WITH their reasons, not be omitted.
    assert "**no**" in table
    assert "majority floor" in table
    # And the count line has to agree with the rows it summarises.
    flagged = sum(1 for line in table.splitlines() if line.startswith("| `") and "**no**" in line)
    assert f"**{flagged} of" in table


# --------------------------------------------------- difficulty confound -----


def _difficulty_module():
    import measure_difficulty_confound as mdc

    return mdc


def _difficulty_corpus(n_events: int = 60):
    from pharos.analyst import Proposal
    from pharos.disclosure import KEEP_COMPARTMENTS
    from pharos.generate import GeneratorConfig, generate
    from pharos.labels import declassify
    from pharos.tasks import build_triage_tasks

    mdc = _difficulty_module()
    tasks = build_triage_tasks(generate(GeneratorConfig(seed=mdc.SEED, n_events=n_events)))
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    return mdc, tasks, proposals


def test_signature_overlap_counts_only_the_significant_facts():
    """The corpus's own difficulty scale. Three means significant by definition."""
    from pharos.world import SIGNIFICANT_PATTERN

    mdc, tasks, _ = _difficulty_corpus()
    for task in tasks:
        overlap = mdc.signature_overlap(task)
        assert 0 <= overlap <= len(SIGNIFICANT_PATTERN)
        # The definition the whole finding rests on: a full signature IS significance.
        if overlap == len(SIGNIFICANT_PATTERN):
            assert task.significant

    # And the band the finding is about is actually populated, or the control is empty.
    assert any(mdc.signature_overlap(t) == 2 for t in tasks)


def test_build_fleet_splits_wrong_standards_from_slip():
    """The two explanations for disagreement are separate knobs, not one."""
    mdc = _difficulty_module()

    control = mdc.build_fleet(0, 0.0)
    assert len(control) == mdc.FLEET
    assert all(p.escalation_threshold != mdc.WRONG_THRESHOLD for p in control)
    assert all(p.slip_rate == 0.0 for p in control)

    slipping = mdc.build_fleet(0, 0.15)
    assert all(p.slip_rate == 0.15 for p in slipping)
    assert all(p.escalation_threshold != mdc.WRONG_THRESHOLD for p in slipping)

    mixed = mdc.build_fleet(3, 0.0)
    wrong = [p for p in mixed if p.escalation_threshold == mdc.WRONG_THRESHOLD]
    assert len(wrong) == 3
    assert len(mixed) == mdc.FLEET


def test_collect_drops_a_reviewer_who_supplied_no_verdict():
    """A rejection is not a label. Counting it as one would move every number here."""
    from pharos.analyst import Action

    mdc, tasks, proposals = _difficulty_corpus()
    fleet = mdc.build_fleet(3, 0.0)
    rows = mdc.collect(fleet, tasks, proposals, seed=mdc.SEED)

    assert rows, "the fleet has to yield some supervision or the estimate is vacuous"
    assert all(isinstance(v, bool) for _, _, v in rows)
    assert {who for _, who, _ in rows} <= {p.name for p in fleet}

    # Every dropped row must correspond to a decision that carried no verdict.
    kept = {(tid, who) for tid, who, _ in rows}
    for policy in fleet:
        for task in tasks:
            decision = policy.review(task, proposals[task.task_id], seed=mdc.SEED)
            carries = decision.action is Action.ACCEPT or (
                decision.action is Action.REVISE and decision.corrected_verdict is not None
            )
            assert ((task.task_id, policy.name) in kept) == carries


def test_difficulty_spread_is_one_when_there_is_nothing_to_compare():
    """A single band, or none, is not a spread. Reporting a ratio there invents one."""
    mdc = _difficulty_module()

    def row(by_overlap):
        return mdc.Row(
            composition="c",
            n_wrong=0,
            slip_rate=0.0,
            by_overlap=by_overlap,
            wrong_ability=None,
            right_ability=1.0,
            dawid_skene_agreement=1.0,
            glad_agreement=1.0,
            converged=True,
            iterations=4,
            cc_rasch_agreement=1.0,
            cc_rasch_converged=True,
            cc_wrong_routine=None,
            cc_right_routine=1.0,
        )

    assert row({}).difficulty_spread == 1.0
    assert row({2: 0.4}).difficulty_spread == 1.0
    assert row({1: 0.2, 2: 0.6}).difficulty_spread == pytest.approx(3.0)
    # A zero band is excluded rather than dividing by it.
    assert row({0: 0.0, 1: 0.2, 2: 0.6}).difficulty_spread == pytest.approx(3.0)

    payload = row({1: 0.2, 2: 0.6}).as_dict()
    assert payload["difficulty_by_overlap"] == {"1": 0.2, "2": 0.6}
    assert payload["wrong_ability"] is None
    assert payload["difficulty_spread"] == pytest.approx(3.0)


def test_ability_inversion_is_undefined_without_a_wrong_standard_to_compare():
    """A fleet with nobody wrong has no answer here, and that is not a ratio of one."""
    mdc = _difficulty_module()

    def row(wrong, right):
        return mdc.Row(
            composition="c",
            n_wrong=0 if wrong is None else 3,
            slip_rate=0.0,
            by_overlap={},
            wrong_ability=wrong,
            right_ability=right,
            dawid_skene_agreement=1.0,
            glad_agreement=1.0,
            converged=True,
            iterations=4,
            cc_rasch_agreement=1.0,
            cc_rasch_converged=True,
            cc_wrong_routine=None,
            cc_right_routine=1.0,
        )

    control = row(None, 3.97)
    assert control.ability_ratio is None
    assert not control.ability_is_inverted
    assert control.as_dict()["ability_ratio"] is None
    assert control.as_dict()["ability_is_inverted"] is False

    # The minority case: the estimator is right, and must not be flagged.
    minority = row(1.96, 3.98)
    assert minority.ability_ratio == pytest.approx(1.96 / 3.98)
    assert not minority.ability_is_inverted

    # The majority case: the wrong rule scores higher, which is the finding.
    majority = row(3.60, 2.54)
    assert majority.ability_ratio == pytest.approx(3.60 / 2.54)
    assert majority.ability_is_inverted
    assert majority.as_dict()["ability_is_inverted"] is True

    # A zero denominator is not an infinite ratio; it is an unanswered question.
    assert row(1.0, 0.0).ability_ratio is None
    assert not row(1.0, 0.0).ability_is_inverted


def test_the_measured_inversion_appears_in_the_committed_artifact():
    """The finding as published: the ability column flips at the majority crossing."""
    import json

    payload = json.loads(Path("results/difficulty_confound.json").read_text(encoding="utf-8"))
    by_name = {r["composition"]: r for r in payload["rows"]}

    control = by_name["correct fleet (control)"]
    assert control["ability_ratio"] is None, "the control has no wrong standard to rank"
    assert control["difficulty_spread"] == pytest.approx(1.0), "the control must be flat"

    minority = by_name["3 of 9 wrong standard"]
    assert minority["ability_is_inverted"] is False
    assert minority["ability_ratio"] < 1.0

    majority = by_name["5 of 9 wrong standard"]
    assert majority["ability_is_inverted"] is True
    # The direction is the finding; the magnitude is modest and was once reported as
    # 31x from a fit missing the priors Whitehill et al. specify. Assert the direction
    # and a floor, not a number this test would have to be edited to keep true.
    assert majority["ability_ratio"] > 1.0, "the wrong standard must score higher"


def test_every_quoted_row_converged():
    """A magnitude may be quoted only from a converged row.

    This rule was written when the random-slip fit did not converge and its spread of
    4.3 had to be withdrawn. The cause turned out to be a missing regulariser rather
    than the method: with the Gaussian priors Whitehill et al. specify in their section
    3.1, every composition settles in under 40 iterations. The rule outlived its
    occasion on purpose -- it is the check that would have caught the problem sooner,
    and the three rows the finding rests on have to keep converging for it to stand.
    """
    import json

    payload = json.loads(Path("results/difficulty_confound.json").read_text(encoding="utf-8"))
    by_name = {r["composition"]: r for r in payload["rows"]}

    for name in ("correct fleet (control)", "3 of 9 wrong standard", "5 of 9 wrong standard"):
        row = by_name[name]
        assert row["converged"] is True, f"{name} is quoted in docs/findings.md but did not settle"
        assert row["quotable"] is True

    # And the artifact has to say which rows those are, so a consumer need not re-derive it.
    assert set(payload["converged_rows"]) >= {
        "correct fleet (control)",
        "3 of 9 wrong standard",
        "5 of 9 wrong standard",
    }
    assert set(payload["converged_rows"]).isdisjoint(payload["unconverged_rows"])

    # An unconverged row must be marked unquotable rather than dropped: the fact that
    # GLAD does not settle on random slip is itself part of the finding.
    for name in payload["unconverged_rows"]:
        assert by_name[name]["converged"] is False
        assert by_name[name]["quotable"] is False


def test_the_random_slip_row_is_the_only_one_allowed_to_stall():
    """Which rows the convergence guard protects, stated rather than inferred.

    CI runs this measurement without `--out`, so the artifact test above cannot see a
    regression there. The script exits non-zero instead, and it needs to be exact
    about which rows that applies to: too wide and a known-stalling foil fails the
    build, too narrow and a quoted row stops settling in silence.
    """
    mdc = _difficulty_module()

    def row(n_wrong: int, slip: float):
        return mdc.Row(
            composition="c",
            n_wrong=n_wrong,
            slip_rate=slip,
            by_overlap={},
            wrong_ability=None,
            right_ability=1.0,
            dawid_skene_agreement=1.0,
            glad_agreement=1.0,
            converged=False,
            iterations=100,
            cc_rasch_agreement=1.0,
            cc_rasch_converged=True,
            cc_wrong_routine=None,
            cc_right_routine=1.0,
        )

    # Exactly the four compositions the script sweeps.
    assert mdc.carries_the_claim(row(0, 0.0)), "the control is the whole argument"
    assert mdc.carries_the_claim(row(3, 0.0))
    assert mdc.carries_the_claim(row(5, 0.0))
    assert not mdc.carries_the_claim(row(0, 0.15)), "the slip row is a foil, and stalls"

    # And the sweep really is those four, so the classification above is exhaustive.
    # `compositions` derives the grid from the fleet size rather than naming literals,
    # so the default is asserted at the default fleet. A third of nine is three and a
    # bare majority is five, which is exactly the set this test has always pinned.
    default = mdc.compositions(mdc.FLEET)
    assert {(n, s) for _, n, s in default} == {(0, 0.0), (0, 0.15), (3, 0.0), (5, 0.0)}
    assert sum(1 for _, n, s in default if not mdc.carries_the_claim(row(n, s))) == 1

    # And the shape holds at any fleet: one control, one slip foil, a minority and a
    # bare majority, with the majority always at floor(fleet/2)+1.
    for fleet in (5, 15, 25, 51):
        swept = mdc.compositions(fleet)
        assert len(swept) == 4
        assert {n for _, n, s in swept} >= {0, fleet // 2 + 1}
        assert sum(1 for _, n, s in swept if not mdc.carries_the_claim(row(n, s))) == 1


def test_a_correct_fleet_finds_no_difficulty_where_a_wrong_one_does():
    """Finding 17, at a small size: the difficulty is manufactured by the reviewers.

    This is the control that gives the measurement its meaning. If the near-boundary
    items were intrinsically hard they would look hard under a correct fleet too.
    """
    import statistics

    from pharos.inference import glad

    mdc, tasks, proposals = _difficulty_corpus(n_events=120)
    overlap = {t.task_id: mdc.signature_overlap(t) for t in tasks}

    def spread(n_wrong: int) -> float:
        estimate = glad(mdc.collect(mdc.build_fleet(n_wrong, 0.0), tasks, proposals, seed=mdc.SEED))
        bands = {}
        for band in sorted(set(overlap.values())):
            ids = [t for t, o in overlap.items() if o == band and t in estimate.log_difficulty]
            if ids:
                bands[band] = statistics.mean(estimate.difficulty(t) for t in ids)
        values = [v for v in bands.values() if v > 0]
        return max(values) / min(values) if len(values) > 1 else 1.0

    control = spread(0)
    corrupted = spread(5)
    assert control < 1.5, "a correct fleet must find the corpus flat, or there is no control"
    assert corrupted > control, "a wrong-standard fleet must manufacture difficulty"


def test_the_awaiting_rerun_list_drops_an_artifact_once_it_is_assessed():
    """The hand-maintained registry goes stale in one direction, so it is not trusted.

    A rerun lands, the artifact gains its validity block, and the entry saying it has
    not is left behind. That produced a published table listing `decode_stability` as
    both quotable at n=30 and awaiting the rerun that produced the 30. The pending
    list is therefore derived from the artifacts; the registry supplies only the
    reasons, which cannot be derived.
    """
    sdt = _docs_module()
    table = sdt.measurement_health()

    if "needs a rerun rather than an edit:" not in table:
        pending_block = ""
    else:
        pending_block = table.split("needs a rerun rather than an edit:")[-1].split(
            "Exempt, because"
        )[0]

    # Anything with a row in the assessed table must not also be listed as pending.
    assessed = {
        line.split("`")[1]
        for line in table.splitlines()
        if line.startswith("| `") and "Artifact" not in line
    }
    assert assessed, "parsed no assessed rows; the parser broke rather than the table"

    both = sorted(name for name in assessed if f"`{name}`" in pending_block)
    assert not both, (
        f"listed as assessed AND as awaiting its rerun: {both}. One of the two statements is false."
    )

    # And every name still pending has to be a real registry entry, not a typo.
    for line in pending_block.splitlines():
        if line.startswith("- `"):
            assert line.split("`")[1] in sdt.AWAITING_RERUN


def test_a_count_of_zero_is_not_a_measurement_of_zero():
    """The bound that stops "0 differ" being read as "the rate is 0".

    Finding 9's replacement was published from 0 of 30 tasks differing across full
    sweeps. That observation is consistent with any true rate up to 9.5%, which does
    not even exclude the 10% the finding originally reported and retracted. The
    measurement now carries the bound so the claim cannot outrun the sample.
    """
    import measure_decode_stability as mds

    # The all-zero case has a closed form; the solver must agree with it exactly.
    for n in (30, 300, 600):
        assert mds.rate_upper_bound(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)

    # The specific numbers the docstrings and findings.md quote.
    assert mds.rate_upper_bound(0, 30) == pytest.approx(0.095, abs=0.001)
    assert mds.rate_upper_bound(0, 300) == pytest.approx(0.0099, abs=0.001)

    # More observed disagreement can never license a tighter bound.
    bounds = [mds.rate_upper_bound(d, 300) for d in range(12)]
    assert bounds == sorted(bounds)
    assert all(b >= d / 300 for d, b in enumerate(bounds)), "a bound below the point estimate"

    # And a larger sample can never license a looser one at the same count.
    assert mds.rate_upper_bound(0, 600) < mds.rate_upper_bound(0, 300)
    assert mds.rate_upper_bound(0, 0) == 1.0, "no observation rules nothing out"


def test_the_default_task_count_supports_the_claim_made_from_it():
    """The default has to be large enough for the sentence the script prints."""
    import measure_decode_stability as mds

    parser_default = 300
    assert mds.rate_upper_bound(0, parser_default) < 0.01, (
        "the default sample must bound the cross-sweep rate under 1%, or 'the sweep "
        "reproduces' is again a claim the sample cannot carry"
    )
    # The old default, kept here as the thing not to go back to.
    assert mds.rate_upper_bound(0, 30) > 0.09


def test_a_flagged_nested_artifact_names_which_pass_is_flagged():
    """Which evaluation pass failed, not just that one did.

    Adapter artifacts carry validity per pass: the untrained `base`, the trained
    `adapter`, and `adapter_vs_teacher`. Collapsing them anonymously reported
    `review_adapter-by-the-book` as unquotable on the strength of its baseline, while
    the adapter the artifact exists to report scored 0.995. The flag was true of
    something in the file and false of the thing being cited from it.
    """
    sdt = _docs_module()
    table = sdt.measurement_health()

    flagged = [
        line
        for line in table.splitlines()
        if line.startswith("| `") and "**no**" in line and "review_adapter" in line
    ]
    if not flagged:
        pytest.skip("no adapter artifact is currently flagged")

    for line in flagged:
        assert "**base**" in line or "**adapter" in line, (
            f"a flagged nested artifact must name its pass, got: {line[:160]}"
        )

    # And a pass name must be one the artifact actually contains, not invented.
    import json

    for line in flagged:
        stem = line.split("`")[1]
        payload = json.loads(Path(f"results/{stem}.json").read_text(encoding="utf-8"))
        named = line.split("**")[1] if "**" in line.split("|")[3] else None
        passes = {k for k, v in payload.items() if isinstance(v, dict) and "validity" in v}
        assert passes, f"{stem} has no nested validity blocks to name"
        if named and named != "no":
            assert named in passes, f"{stem} flagged an unknown pass {named!r}, has {passes}"


# --- FL Benchmarks & E2E Demo Scripts --------------------------------------

from demo_e2e_system import main as run_e2e_demo  # noqa: E402
from sweep_fl_benchmarks import (  # noqa: E402
    generate_fleet_gradients,
    l2_distance,
    run_fl_benchmark_sweep,
)


def test_sweep_fl_benchmarks_helpers():
    assert l2_distance([1.0, 2.0], [4.0, 6.0]) == 5.0

    true_grad, clients, is_byz = generate_fleet_gradients(
        n_clients=10, dim=20, n_byzantine=2, seed=42
    )
    assert len(true_grad) == 20
    assert len(clients) == 10
    assert sum(is_byz) == 2


def test_sweep_fl_benchmarks_run_sweep():
    results = run_fl_benchmark_sweep(n_clients=5, dim=10, n_rounds=1)
    assert "records" in results
    assert len(results["records"]) > 0


def test_demo_e2e_system_runs(monkeypatch):
    import urllib.request
    from io import BytesIO

    # Mock urllib.request.urlopen for API call inside demo_e2e_system
    def mock_urlopen(url):
        return BytesIO(b'{"status": "ok", "n_reports": 360}')

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    run_e2e_demo()


def test_a_perturbation_actually_perturbs_the_text_it_returns():
    """The property `measure_adversarial_robustness.py` never checked, which is why it went.

    That script reported clean accuracy, lexical robustness and decoy vulnerability as
    1.0, 1.0 and 1.0 at every seed, and could not have reported anything else: it scored
    the ground truth against itself and hardcoded the "tricked" verdict, so no
    perturbation reached a prediction. Its test asserted the record count and no value,
    so the constants passed. Both are deleted.

    What survives is the perturbation itself, which does work, and this asserts the part
    the harness threw away: `perturbed_reports` differs from the input while the ground
    truth does not. An adversarial evaluation worth the name has to feed that text to a
    model and compare verdicts; until one does, there is no robustness number here.
    """
    from pharos.adversarial import apply_decoy_injection, apply_lexical_substitution
    from pharos.generate import GeneratorConfig, generate
    from pharos.tasks import build_triage_tasks

    tasks = build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=60)))
    originals = {t.task_id: tuple(s.text for s in t.sources) for t in tasks}

    decoyed = [apply_decoy_injection(t, seed=7) for t in tasks if not t.significant]
    assert decoyed, "the corpus must contain routine tasks for decoy injection to apply"
    assert any(a.perturbed_reports != originals[a.original_task_id] for a in decoyed), (
        "decoy injection changed no report text"
    )
    assert all(a.ground_truth_significant is False for a in decoyed), (
        "a perturbation must not move the ground truth it is testing against"
    )

    substituted = [apply_lexical_substitution(t, seed=7) for t in tasks]
    assert any(a.perturbed_reports != originals[a.original_task_id] for a in substituted), (
        "lexical substitution matched no term in the whole corpus"
    )
    for a, t in zip(substituted, tasks, strict=True):
        assert a.ground_truth_significant == t.significant


def test_make_results_reproduces_the_sample_sizes_it_overwrites():
    """`make results` must name the sample size of every artifact it regenerates.

    This is the command the README and the harvest guard both point at when an artifact
    is stale, so a reader who follows that advice gets whatever the recipe says. Three
    of its five entries once omitted `--tasks` and inherited a script default that
    disagreed with the committed artifact: federation eligibility ran at 8 against a
    committed 40, rule learnability at 30 against 600, and decode stability at 300
    against 30. Nothing failed. The artifacts came back a different size, carrying
    validity flags the published numbers did not have, and two separate people re-ran
    from this target in one day and silently changed what the findings rested on.

    Asserting the recipe against the artifacts is cheap and catches the whole class:
    a default that drifts, a recipe that loses a flag, or an artifact regenerated at a
    size nobody chose.
    """
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    recipe = re.search(
        r"^results:.*?(?=^\S)", (root / "Makefile").read_text(), re.MULTILINE | re.DOTALL
    )
    assert recipe, "the `results` target moved; this guard is reading the wrong recipe"

    #: Where each artifact records the size the run actually used. They differ because
    #: the measurements differ: an eval reports tasks, a decode probe reports the tasks
    #: it repeated, a fidelity pass reports turns.
    size_field = {
        "label_fidelity.json": ("validity", "n"),
        "federation_eligibility.json": ("n_tasks",),
        "triage_lift.json": ("validity", "n"),
        "learnability.json": ("n_eval",),
        "decode_stability.json": ("validity", "n"),
    }

    checked = 0
    for script, args, out in re.findall(
        r"scripts/(measure_\w+\.py)\s+([^\n]*?)--out results/(\S+)", recipe.group(0)
    ):
        declared = re.search(r"--tasks (\d+)", args)
        assert declared, (
            f"{script} in `make results` does not state --tasks; it would inherit a default"
        )

        artifact = root / "results" / out
        if not artifact.exists():
            continue
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        node = payload
        for key in size_field[out]:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        assert node == int(declared.group(1)), (
            f"`make results` runs {script} at --tasks {declared.group(1)}, but "
            f"results/{out} records {node}. The command cannot regenerate the artifact."
        )
        checked += 1

    assert checked >= 3, (
        f"only {checked} artifacts compared; the guard is not exercising the recipe"
    )


def test_a_measurement_script_writes_its_artifact_only_when_asked():
    """No measurement may write into `results/` as a side effect of being imported or called.

    Two scripts did. `measure_adversarial_robustness.py` wrote unconditionally, so
    `logcheck.py` -- a read-only diagnostic -- mutated the results tree on every run.
    `sweep_fl_benchmarks.py` did the same at the bottom of a long function, and the test
    that exercises its helpers calls it with 5 clients and 1 round: every full test run
    replaced the committed 20-client, 50-round artifact with a toy, and nothing failed,
    because a smaller sweep is still a well-formed sweep. It was caught by noticing the
    artifact dirty after a green suite.

    The rule is mechanical, so assert it mechanically rather than trusting review.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for path in sorted((root / "scripts").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        #: Only writes aimed at `results/` are in scope. `sync_docs_tables.py` writes
        #: `docs/findings.md` unconditionally and should -- rendering a page is its
        #: entire job, and it already has `--check` for the read-only mode. The rule
        #: being enforced is narrower than "no unconditional writes": it is that a
        #: measurement artifact is produced only when someone asks for one.
        if "RESULTS" not in source and "results/" not in source:
            continue
        for match in re.finditer(r"^\s*(\S.*?)\.write_text\(", source, re.MULTILINE):
            target = match.group(1)
            line_no = source[: match.start()].count("\n") + 1
            window = source[: match.start()].splitlines()[-6:]
            guarded = any(re.search(r"\bif\b.*\bout\b", line) for line in window)
            #: No name-based exclusion. A first version skipped any target containing
            #: "out", which skipped `out_file` -- the exact variable the original
            #: defect used. Whether the write is permitted is decided by the guard
            #: above it, never by what the variable is called.
            aims_at_results = "RESULTS" in target or "out_file" in target
            if aims_at_results and not guarded:
                offenders.append(f"{path.name}:{line_no} writes {target} unguarded")

    assert not offenders, "measurement scripts writing to results/ unconditionally: " + "; ".join(
        offenders
    )


def test_every_make_command_the_repository_tells_you_to_run_exists():
    """An instruction naming a target that was never added is a dead end.

    This repository has produced that defect twice. `cluster/setup-env.sh` printed
    "Fix: hf auth login" at a point where `hf` was not installed, and a skip message
    here said to run `make replication` before the target existed. Both are the same
    shape: a human-readable instruction that no build step ever executes, so nothing
    disagrees with it until somebody types it.

    Only unambiguous command contexts count -- backticked, or alone on a line. Prose
    like "make the comparison" is not an instruction, and matching it produced a dozen
    imaginary targets on the first attempt.
    """
    import re

    root = Path(__file__).resolve().parents[1]
    targets = set(re.findall(r"^([a-zA-Z][\w-]*):", (root / "Makefile").read_text(), re.MULTILINE))

    pattern = re.compile(
        r"(?:`make ([a-z][\w-]*)`|^\s*(?:\$ )?make ([a-z][\w-]*)\s*$)", re.MULTILINE
    )
    skip = {".git", ".venv", "site", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
    suffixes = {".py", ".md", ".sh", ".sbatch", ".toml", ".yml", ".yaml"}

    referenced: dict[str, set[str]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or any(s in path.parts for s in skip):
            continue
        if path.suffix not in suffixes and path.name != "Makefile":
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8", errors="ignore")):
            name = match.group(1) or match.group(2)
            referenced.setdefault(name, set()).add(str(path.relative_to(root)))

    assert referenced, "no `make <target>` instructions found; the pattern has gone stale"
    missing = {k: sorted(v) for k, v in referenced.items() if k not in targets}
    assert not missing, f"instructions naming targets that do not exist: {missing}"


def test_secure_readership_block_matches_the_artifact():
    """Every row the aggregate discloses, and the bold that marks it exact."""
    sdt = _docs_module()
    import json as _json

    table = sdt.secure_readership()
    payload = _json.loads((sdt.RESULTS / "secure_reliability.json").read_text(encoding="utf-8"))
    readership = payload["readership"]

    for key, count in readership["headcounts"].items():
        assert key.replace("|", "\\|") in table
        # Exact headcounts are bolded, and the finding's whole claim is that all of
        # them are. A row that stopped being exact would lose its bold here.
        if count == readership["truth"][key]:
            assert f"**{count}**" in table
    assert str(readership["exact_headcounts"]) in table
    assert table.count("\n| ") >= len(readership["headcounts"])


def test_authority_price_block_matches_the_artifact():
    sdt = _docs_module()
    import json as _json

    table = sdt.authority_price()
    payload = _json.loads((sdt.RESULTS / "authority_anchors.json").read_text(encoding="utf-8"))
    spreads = payload["anchors_needed_spread"]

    for key, spread in spreads.items():
        assert f"| {key} |" in table
        if spread["median"] is not None:
            assert f"| {spread['median']} |" in table
        # The range is the reason this table exists; a median published without it is
        # the shape that got retracted.
        if spread["reached"]:
            assert f"{spread['lowest']}" in table and f"{spread['highest']}" in table
        assert f"{spread['reached']} of {spread['seeds']}" in table
    assert f"{payload['repaired_threshold']:.2f}" in table
    assert str(payload["events"]) in table
    # The headline column must be the median over draws, not one draw's crossing.
    assert payload["anchors_needed"] == {k: v["median"] for k, v in spreads.items()}


def test_the_published_grid_names_the_draw_it_came_from():
    """A grid from one seed beside a median over 21 must say which is which."""
    sdt = _docs_module()
    import json as _json

    payload = _json.loads((sdt.RESULTS / "authority_anchors.json").read_text(encoding="utf-8"))
    table = sdt.authority_grid()
    assert str(payload["grid_anchor_seed"]) in table
    assert payload["grid_anchor_seed"] in payload["anchor_seeds"]
    # One bolded cell per composition that repairs: the crossing, not the whole tail.
    for row in table.splitlines():
        if row.startswith("| ") and "**" in row:
            assert row.count("**") == 2, f"more than one crossing marked: {row}"


def test_a_composition_no_budget_repairs_is_named_rather_than_left_blank(tmp_path, monkeypatch):
    """An unreached threshold must read as unreached, not as a missing cell."""
    sdt = _docs_module()
    import json as _json

    payload = {
        "anchors_needed": {"5": 5, "9": None},
        "anchors_needed_spread": {
            "5": {"n_wrong": 5, "seeds": 3, "reached": 3, "median": 5, "lowest": 5, "highest": 5},
            "9": {
                "n_wrong": 9,
                "seeds": 3,
                "reached": 0,
                "median": None,
                "lowest": None,
                "highest": None,
            },
        },
        "events": 200,
        "fleet": 9,
        "anchor_counts": [0, 50],
        "anchor_seeds": [1, 2, 3],
        "repaired_threshold": 0.95,
    }
    (tmp_path / "authority_anchors.json").write_text(_json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sdt, "RESULTS", tmp_path)

    table = sdt.authority_price()
    assert "not reached within 50" in table
    assert "| 5 | 5 | 2.5% | 5 | 3 of 3 |" in table
    # A composition no draw repaired has no range to report, and an em dash is not a
    # zero: "0 of 3" carries the fact, the range column must not invent one.
    assert "| 9 | not reached within 50 | — | — | 0 of 3 |" in table


def test_a_missing_or_empty_artifact_fails_rather_than_emitting_an_empty_table(
    tmp_path, monkeypatch
):
    """A generated block with nothing behind it is how a table silently empties."""
    sdt = _docs_module()
    import json as _json

    monkeypatch.setattr(sdt, "RESULTS", tmp_path)
    # ROOT too: the missing-artifact message names the path relative to it, which is
    # not a relative path at all once RESULTS lives outside the repo.
    monkeypatch.setattr(sdt, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        sdt.authority_price()
    with pytest.raises(SystemExit):
        sdt.secure_readership()

    (tmp_path / "authority_anchors.json").write_text(_json.dumps({}), encoding="utf-8")
    (tmp_path / "secure_reliability.json").write_text(_json.dumps({}), encoding="utf-8")
    with pytest.raises(SystemExit):
        sdt.authority_price()
    with pytest.raises(SystemExit):
        sdt.secure_readership()


def test_the_readme_findings_count_matches_the_findings_page():
    """The one volatile number worth writing down, and therefore worth guarding.

    Most counts should not appear in prose at all -- a test total or a coverage figure
    written into a comment is a number that will disagree with itself within a day, and
    this repository has now watched that happen to both. The README's findings count is
    the exception: it is a headline claim a reader acts on, and deleting it would leave
    "some findings so far", which says nothing.

    So it stays, and this keeps it honest. `docs/findings.md` is the source, and the
    count is the number of `## N.` headings in it.
    """
    import re

    root = SCRIPTS.parent
    headings = re.findall(
        r"^## (\d+)\.", (root / "docs" / "findings.md").read_text(encoding="utf-8"), re.MULTILINE
    )
    assert headings, "no numbered findings in docs/findings.md; the parser has broken"
    measured = len(headings)

    words = {
        20: "Twenty",
        21: "Twenty-one",
        22: "Twenty-two",
        23: "Twenty-three",
        24: "Twenty-four",
        25: "Twenty-five",
        26: "Twenty-six",
        27: "Twenty-seven",
    }
    readme = (root / "README.md").read_text(encoding="utf-8")
    claimed = re.search(r"^(\w[\w-]*) findings so far", readme, re.MULTILINE)
    assert claimed, "the README no longer states a findings count in the expected shape"

    expected = words.get(measured)
    assert expected, (
        f"docs/findings.md has {measured} findings and this test cannot spell that; extend `words`."
    )
    assert claimed.group(1) == expected, (
        f"README says '{claimed.group(1)} findings so far' and docs/findings.md has "
        f"{measured}. Update the README, or the page, so a reader is told the truth."
    )
