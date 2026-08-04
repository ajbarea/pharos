"""Guards for the cross-machine gate comparison.

`scripts/measure_gate_determinism.py` exists to answer one question -- do two machines
score the same corpus to the same bits -- and it answered it by refuting a claim the
README had been making. That makes its comparison logic load-bearing: a false pass here
restores an overstated reproducibility claim, and a false refusal makes the check
unrunnable, which is what its first version did.

`measure()` itself is not exercised here. It runs seven gates over seven generated
corpora, which is minutes, and it is the part with no logic to get wrong.
"""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import measure_gate_determinism

    return measure_gate_determinism


def _artifact(
    baselines: dict[str, tuple[float, str]], commit: str = "abc123def456"
) -> dict[str, Any]:
    """An artifact shaped like the real one. `baselines` maps seed to (value, digest)."""
    return {
        "machine": {"platform": "Linux-test", "python": "3.14.0"},
        "n_events": 400,
        "baselines": {
            seed: {"hex": float(value).hex(), "decimal": value, "corpus_sha256": digest}
            for seed, (value, digest) in baselines.items()
        },
        "provenance": {"git_commit": commit},
    }


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_identical_runs_compare_equal(tmp_path, capsys):
    same = {"1": (0.6378328710578888, "d1"), "7": (0.654478846227649, "d7")}
    left = _write(tmp_path, "a.json", _artifact(same))
    right = _write(tmp_path, "b.json", _artifact(same))

    assert _module().compare(left, right) == 0
    assert "bit-identical on all 2 seeds" in capsys.readouterr().out


def test_a_difference_far_below_display_precision_still_fails(tmp_path, capsys):
    """The reason the artifact stores hex at all.

    These two values print identically at any precision a caption would use, and are
    different floats. Comparing rounded decimals would call this a match, which is
    exactly how the original claim survived as long as it did.
    """
    module = _module()
    a, b = 0.6378328710578888, 0.6378328710578889
    assert f"{a:.10f}" == f"{b:.10f}", "the test's own premise: indistinguishable when rounded"
    assert a != b

    left = _write(tmp_path, "a.json", _artifact({"1": (a, "d1")}))
    right = _write(tmp_path, "b.json", _artifact({"1": (b, "d1")}))

    assert module.compare(left, right) == 1
    assert "1 of 1 seeds differ" in capsys.readouterr().out


def test_differing_corpora_are_refused_rather_than_compared(tmp_path, capsys):
    """Two machines that scored different corpora say nothing about each other."""
    left = _write(tmp_path, "a.json", _artifact({"1": (0.63, "d1")}))
    right = _write(tmp_path, "b.json", _artifact({"1": (0.99, "OTHER")}))

    assert _module().compare(left, right) == 2
    assert "corpora differ" in capsys.readouterr().err


def test_the_same_commit_does_not_excuse_a_different_corpus(tmp_path, capsys):
    """A dirty tree changes the corpus without changing the commit.

    The first version of this guard keyed on the commit, so this case passed it.
    """
    left = _write(tmp_path, "a.json", _artifact({"1": (0.63, "d1")}, commit="same"))
    right = _write(tmp_path, "b.json", _artifact({"1": (0.63, "DIFFERENT")}, commit="same"))

    assert _module().compare(left, right) == 2


def test_a_different_commit_does_not_block_an_identical_corpus(tmp_path, capsys):
    """The failure that made the original guard useless.

    Its first real comparison was refused because the two artifacts differed by the
    commit that added the script. Nothing about the corpus changed, so the comparison
    was valid and the guard refused the workflow its own documentation describes.
    """
    same = {"1": (0.6378328710578888, "d1")}
    left = _write(tmp_path, "a.json", _artifact(same, commit="1111111111"))
    right = _write(tmp_path, "b.json", _artifact(same, commit="2222222222"))

    assert _module().compare(left, right) == 0


def test_a_missing_digest_reports_cannot_tell_not_a_difference(tmp_path, capsys):
    """Two findings with different remedies, and the wrong order conflates them."""
    left = _write(tmp_path, "a.json", _artifact({"1": (0.63, "d1")}))
    stale = _artifact({"1": (0.63, "d1")})
    del stale["baselines"]["1"]["corpus_sha256"]
    right = _write(tmp_path, "b.json", stale)

    assert _module().compare(left, right) == 2
    err = capsys.readouterr().err
    assert "predates the corpus digest" in err
    assert "corpora differ" not in err, "an absent digest is not evidence of a difference"


def test_every_differing_seed_is_reported_not_just_the_first(tmp_path, capsys):
    """One seed differing and all of them differing are different diagnoses."""
    left = _write(
        tmp_path, "a.json", _artifact({"1": (0.10, "d"), "7": (0.20, "d"), "11": (0.30, "d")})
    )
    right = _write(
        tmp_path, "b.json", _artifact({"1": (0.11, "d"), "7": (0.20, "d"), "11": (0.33, "d")})
    )

    assert _module().compare(left, right) == 1
    out = capsys.readouterr().out
    assert "2 of 3 seeds differ" in out
    assert "seed 1:" in out and "seed 11:" in out


def test_the_committed_artifacts_still_show_what_the_prose_reports(tmp_path):
    """The finding itself: the corpus matches, two of seven baselines do not.

    Skipped rather than failed when either artifact is absent, since the cluster half is
    produced by a Slurm job and a fresh clone will not have run it.
    """
    import pytest

    laptop = ROOT / "results" / "gate_determinism.json"
    cluster = ROOT / "results" / "gate_determinism-cluster.json"
    if not (laptop.is_file() and cluster.is_file()):
        pytest.skip("no two-machine pair; run cluster/gate-determinism.sbatch")

    a = json.loads(laptop.read_text(encoding="utf-8"))
    b = json.loads(cluster.read_text(encoding="utf-8"))
    shared = sorted(set(a["baselines"]) & set(b["baselines"]), key=int)

    assert all(
        a["baselines"][s]["corpus_sha256"] == b["baselines"][s]["corpus_sha256"] for s in shared
    ), "the corpus must still be bit-identical across machines; that claim did survive"

    differing = [s for s in shared if a["baselines"][s]["hex"] != b["baselines"][s]["hex"]]
    assert differing, (
        "no baseline differs any more. If the machines now agree the README and the "
        "manuscript both understate the result and should be re-checked, not left."
    )
    worst = max(abs(a["baselines"][s]["decimal"] - b["baselines"][s]["decimal"]) for s in differing)
    assert worst < 1e-3, (
        f"largest cross-machine disagreement is {worst:.2e}; the prose argues these are "
        "negligible against the 0.72 acceptance ceiling, which stops being true here"
    )
