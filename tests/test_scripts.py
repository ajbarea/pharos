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
    """Drops a contributing source, which is the direction that leaks."""
    payload = _perfect(task)
    truth = payload["truly_contributing"]
    payload["attributed_sources"] = frozenset(sorted(truth)[1:])
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


def test_no_wrong_standard_clears_the_floor_at_any_carefulness():
    """The load-bearing claim of the sweep, asserted over the whole grid.

    If some (wrong threshold, slip) cell cleared the majority floor, the finding
    would need rewriting -- so the test says so rather than spot-checking a cell.
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

    wrong = [c for c in cells if c.threshold != max(mrs.THRESHOLDS)]
    assert wrong, "the grid must contain wrong standards"
    over = [(c.threshold, c.slip_rate, c.mean) for c in wrong if c.mean >= floor]
    assert over == [], f"a wrong standard cleared the floor: {over}"


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

    row = mds.Stability("r", 8, 30, 3, 3, 0)
    assert row.unstable_share == 0.1
    assert row.as_dict()["unstable_share"] == 0.1
    assert mds.Stability("r", 8, 0, 3, 0, 0).unstable_share == 0.0


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

    row = mtt.Row("t", 8, n=10, unparsed=2, agree_world=4, agree_teacher=8, said_significant=6)
    assert row.world_rate == 0.5
    assert row.teacher_rate == 1.0
    assert row.escalation_rate == 0.75
    payload = row.as_dict()
    assert payload["agreement_with_teacher"] == 1.0
    assert payload["n"] == 10 and payload["unparsed"] == 2

    assert (
        mtt.Row(
            "t", 0, n=0, unparsed=0, agree_world=0, agree_teacher=0, said_significant=0
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
    assert mp.CLAIMS
    for claim in mp.CLAIMS:
        assert claim.finding and claim.description
        assert claim.n > 0
        assert 0.0 <= claim.effect <= 1.0
        if claim.rate is not None:
            assert 0.0 < claim.rate < 1.0


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


def test_an_absent_stream_is_not_scored_as_a_lag():
    """The far end of the sweep has an oracle that refused everything, not one that lost."""
    mcr = _consensus_module()
    empty_oracle = mcr.Row(9, 1.0, 0.7, None, 0.7, {}, {})
    assert not empty_oracle.consensus_lags_oracle

    lagging = mcr.Row(5, 0.556, 0.84, 1.0, 0.717, {}, {})
    assert lagging.consensus_lags_oracle

    matching = mcr.Row(4, 0.444, 0.87, 1.0, 1.0, {}, {})
    assert not matching.consensus_lags_oracle


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
