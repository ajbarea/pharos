"""The gate's own provenance stamp: can it actually catch a tree edited mid-run?

This exists because the failure it guards is one this repository already had. On
2026-08-06 `secagg.py` was edited while `make ci` was running; the gate measured a
half-written tree, reported 92.52% with `secagg.py` at 40.58%, and passed. The number
was caught only because it was surprising, which is not a process.

So the tests below construct the failure rather than describing it. A guard nobody has
tripped is a guard nobody knows works, which is the defect class the review pass that
prompted this file spent its time finding.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tree_fingerprint.py"


def _module():
    """The script imported in-process.

    The subprocess tests below are the honest end-to-end check -- they exercise the
    exact invocation the gate uses -- but coverage cannot see into a subprocess, so
    calling the functions directly as well is what keeps this file from reading as
    untested while being thoroughly tested.
    """
    spec = importlib.util.spec_from_file_location("tree_fingerprint", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_settled_tree_fingerprints_the_same_twice():
    """Otherwise every gate run would fail its own check."""
    first = _run()
    second = _run()
    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout.strip() == second.stdout.strip()
    assert len(first.stdout.strip()) == 64


def test_verify_passes_when_nothing_moved(tmp_path):
    stamp = tmp_path / "fp"
    written = _run("--write", str(stamp))
    assert written.returncode == 0
    assert stamp.read_text(encoding="utf-8").strip() == written.stdout.strip()

    verified = _run("--verify", str(stamp))
    assert verified.returncode == 0
    assert "unchanged" in verified.stdout


def test_verify_refuses_when_a_watched_file_changed(tmp_path):
    """The tracked-file case: the one that produced the wrong 92.52%."""
    stamp = tmp_path / "fp"
    _run("--write", str(stamp))

    target = ROOT / "src" / "pharos" / "secagg.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# transient edit\n")
        result = _run("--verify", str(stamp))
    finally:
        target.write_bytes(original)

    assert result.returncode == 2
    assert "changed while the gate was running" in result.stderr
    assert "meaningless" in result.stderr


def test_verify_refuses_when_an_untracked_file_appeared(tmp_path):
    """A new test file is invisible to `git diff`, and is exactly what gets added.

    Fingerprinting only tracked content would miss it, and a test added mid-run
    changes both what is measured and what the coverage denominator is.
    """
    stamp = tmp_path / "fp"
    _run("--write", str(stamp))

    intruder = ROOT / "tests" / "test_zz_transient_probe.py"
    try:
        intruder.write_text("assert True\n", encoding="utf-8")
        result = _run("--verify", str(stamp))
    finally:
        intruder.unlink(missing_ok=True)

    assert result.returncode == 2
    assert "changed while the gate was running" in result.stderr


def test_a_missing_stamp_is_refused_rather_than_waved_through():
    """An absent stamp means the gate cannot say what it measured.

    Treating that as a pass would make the guard removable by deleting a file, which
    is the weakest possible failure mode for a provenance check.
    """
    result = _run("--verify", str(ROOT / "no-such-fingerprint-file"))
    assert result.returncode == 2
    assert "unmeasured" in result.stderr


def test_results_are_deliberately_not_watched():
    """Several gate steps rewrite `results/`, so watching it would fail every run.

    Stated as a test because it is a real limitation rather than an oversight: this
    checks that the code under measurement held still, not that the artifacts did.
    """
    module = _module()

    assert "results" not in module.WATCHED
    assert {"src", "scripts", "tests"} <= set(module.WATCHED)


@pytest.mark.parametrize("flag", ["--write", "--verify"])
def test_both_modes_accept_a_path(tmp_path, flag):
    stamp = tmp_path / "fp"
    _run("--write", str(stamp))
    result = _run(flag, str(stamp))
    assert result.returncode == 0


def test_fingerprint_reacts_to_content_in_process(tmp_path, monkeypatch):
    """Same property as the subprocess tests, reached where coverage can see it."""
    module = _module()

    before = module.fingerprint()
    assert before == module.fingerprint(), "fingerprint is not stable on a settled tree"

    target = ROOT / "src" / "pharos" / "secagg.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# transient\n")
        assert module.fingerprint() != before
    finally:
        target.write_bytes(original)
    assert module.fingerprint() == before, "restoring the file did not restore the hash"


def test_main_writes_verifies_and_refuses(tmp_path, monkeypatch, capsys):
    """The three exit paths of `main`, in process."""
    module = _module()
    stamp = tmp_path / "fp"

    monkeypatch.setattr(sys, "argv", ["tree_fingerprint.py", "--write", str(stamp)])
    assert module.main() == 0
    assert stamp.read_text(encoding="utf-8").strip()

    monkeypatch.setattr(sys, "argv", ["tree_fingerprint.py", "--verify", str(stamp)])
    assert module.main() == 0

    stamp.write_text("not-the-hash\n", encoding="utf-8")
    assert module.main() == 2

    monkeypatch.setattr(sys, "argv", ["tree_fingerprint.py", "--verify", str(tmp_path / "gone")])
    assert module.main() == 2


def test_a_missing_git_is_refused_rather_than_crashing(monkeypatch):
    """Without git there is no stamp, and a gate that cannot stamp has not measured."""
    module = _module()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit, match="git is not on PATH"):
        module.fingerprint()
