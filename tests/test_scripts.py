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
