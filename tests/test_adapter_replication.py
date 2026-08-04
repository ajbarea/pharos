"""Guards for the replication floor finding 10 reads its gaps against.

The floor decides whether a gap is a result. Report it too low and an inheritance gap of
0.0008 looks like a measurement rather than a number indistinguishable from zero; too
high and the filtering gap of 0.034 stops being resolvable. Both errors are silent.

The pairing is where it can go wrong. A named teacher is only a replicate of a grid point
if every other thing about the two runs agrees, and one of the four pairs differs in a
way that matters -- a slipping teacher redraws its mistakes, so those two runs never saw
the same labels.
"""

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import measure_adapter_replication

    return measure_adapter_replication


def _artifact(
    *,
    threshold: int = 3,
    slip: float = 0.0,
    target_agreement: float = 0.9,
    world: float = 0.8,
    teacher: float = 0.95,
    n_train: int = 1140,
    lora_rank: int = 16,
) -> dict[str, Any]:
    return {
        "teacher": {
            "reviewer": "x",
            "escalation_threshold": threshold,
            "slip_rate": slip,
            "train_target_agreement": target_agreement,
        },
        "adapter": {"accuracy": world},
        "adapter_vs_teacher": {"accuracy": teacher},
        "hyperparameters": {"lora_rank": lora_rank, "epochs": 3.0},
        "data": {"n_train": n_train, "n_eval": 600, "eval_seed": None, "cross_corpus": False},
    }


def _patch(monkeypatch, mapping: dict[str, dict[str, Any]]):
    module = _module()
    monkeypatch.setattr(module, "_load", lambda name: mapping.get(name))
    return module


def test_a_deterministic_teacher_gives_a_training_only_pair(monkeypatch):
    """Slip 0 means both runs trained on identical labels, so only training varies."""
    module = _patch(
        monkeypatch,
        {
            "a": _artifact(slip=0.0, world=0.80, teacher=0.95),
            "t3s0": _artifact(slip=0.0, world=0.81, teacher=0.95),
        },
    )
    monkeypatch.setattr(module, "TWINS", (("a", "t3s0"),))
    pair = module._pair("a", "t3s0")

    assert pair is not None
    assert pair["targets_identical"] is True
    assert round(pair["delta_vs_world"], 4) == 0.01


def test_a_slipping_teacher_is_not_a_training_only_pair(monkeypatch):
    """Its mistakes are redrawn, so the two runs never saw the same labels.

    Counting this as a training replicate would inflate the training-only floor by
    folding in target variance, which is the one thing that must not be conflated here.
    """
    module = _patch(
        monkeypatch,
        {
            "a": _artifact(slip=0.15, target_agreement=0.855),
            "t3s0.15": _artifact(slip=0.15, target_agreement=0.864),
        },
    )
    pair = module._pair("a", "t3s0.15")

    assert pair is not None
    assert pair["targets_identical"] is False
    assert round(pair["target_agreement_delta"], 4) == 0.009


def test_a_pair_with_different_hyperparameters_is_refused(monkeypatch):
    """Two experiments, not one experiment twice."""
    module = _patch(
        monkeypatch,
        {"a": _artifact(lora_rank=16), "b": _artifact(lora_rank=32)},
    )
    assert module._pair("a", "b") is None


def test_a_cross_corpus_run_is_never_paired_with_a_same_corpus_one(monkeypatch):
    """It trains on more data and evaluates elsewhere; the difference is design."""
    module = _module()
    other = _artifact()
    other["data"] = {**other["data"], "cross_corpus": True, "n_train": 1740}
    monkeypatch.setattr(module, "_load", {"a": _artifact(), "b": other}.get)
    assert module._pair("a", "b") is None


def test_a_pair_with_different_teacher_parameters_is_refused(monkeypatch):
    """A mislabelled twin would report a replication floor that is really a difference."""
    module = _patch(
        monkeypatch,
        {"a": _artifact(threshold=3), "b": _artifact(threshold=2)},
    )
    assert module._pair("a", "b") is None


def test_a_missing_half_yields_no_pair(monkeypatch):
    module = _patch(monkeypatch, {"a": _artifact()})
    assert module._pair("a", "absent") is None


def test_the_bound_is_the_worst_movement_on_either_scoring(monkeypatch):
    """Reporting only the world column would miss a run that moved against its teacher."""
    module = _module()
    pairs = [
        {"delta_vs_world": 0.001, "delta_vs_teacher": 0.040},
        {"delta_vs_world": 0.000, "delta_vs_teacher": 0.000},
    ]
    bound = module._bound(pairs)

    assert bound["max"] == 0.04, "the teacher column is part of the floor"
    assert bound["exactly_identical"] == 1
    assert bound["n_pairs"] == 2


def test_an_empty_set_of_pairs_reports_nothing_rather_than_zero(monkeypatch):
    """A floor of 0.0 from no data would make every gap look resolvable."""
    module = _module()
    assert module._bound([]) == {"n_pairs": 0}


def test_the_committed_artifact_still_supports_the_paper(monkeypatch):
    """The floor the manuscript quotes, checked against the artifact it comes from."""
    import json

    import pytest

    path = ROOT / "results" / "adapter_replication.json"
    if not path.is_file():
        pytest.skip("no adapter_replication.json; run make replication")
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    training = summary["training_only"]

    assert training["n_pairs"] >= 3, "the floor rests on at least three same-target pairs"
    assert training["max"] < 0.034, (
        "the training floor has risen above the filtering gap the manuscript resolves "
        "against it; that gap can no longer be claimed"
    )
    assert training["max"] > 0.0008, (
        "the floor is now below the inheritance gap, which would make 0.0008 a resolvable "
        "difference rather than one indistinguishable from zero. The prose says otherwise."
    )
