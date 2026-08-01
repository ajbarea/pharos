"""Provenance must describe the run without ever being able to break it."""

import os
import shutil
import subprocess

import pytest

from pharos import provenance


def test_code_provenance_names_the_version():
    stamp = provenance.code_provenance()
    assert stamp["pharos_version"]
    assert set(stamp) == {"pharos_version", "git_commit", "git_dirty"}


def test_code_provenance_carries_no_clock():
    """A manifest embeds this, and a manifest has to stay reproducible from its seed."""
    first = provenance.code_provenance()
    second = provenance.code_provenance()
    assert first == second


def test_run_provenance_adds_the_clock_and_the_environment():
    stamp = provenance.run_provenance()
    assert stamp["generated_at"].endswith("+00:00")
    assert stamp["python"]
    assert stamp["platform"]


def test_run_provenance_carries_the_run_inputs():
    stamp = provenance.run_provenance(model="qwen2.5:7b-instruct", seed=7)
    assert stamp["model"] == "qwen2.5:7b-instruct"
    assert stamp["seed"] == 7


def test_git_commit_is_a_short_sha_in_a_checkout():
    commit = provenance.git_commit()
    # None is legitimate (source tarball, no git), but a value must look like a SHA.
    if commit is not None:
        assert len(commit) == provenance._SHA_LENGTH
        assert all(character in "0123456789abcdef" for character in commit)


@pytest.mark.parametrize("failure", [FileNotFoundError("no git"), OSError("boom")])
def test_git_helpers_degrade_to_none_when_git_is_missing(monkeypatch, failure):
    """Provenance is metadata. Failing to collect it must not take down a measurement."""

    def explode(*args, **kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "run", explode)
    assert provenance.git_commit() is None
    assert provenance.git_is_dirty() is None
    # The composed stamps still build, just with the git fields empty.
    assert provenance.code_provenance()["git_commit"] is None
    assert provenance.run_provenance()["git_dirty"] is None


def test_git_helpers_degrade_to_none_on_a_nonzero_exit(monkeypatch):
    """Outside a repo git exits non-zero rather than raising, so cover that path too."""

    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=128, stdout="", stderr="fatal")

    monkeypatch.setattr(subprocess, "run", failed)
    assert provenance.git_commit() is None
    assert provenance.git_is_dirty() is None


def test_dirty_is_true_when_git_reports_changes(monkeypatch):
    def dirty(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=" M README.md\n")

    monkeypatch.setattr(subprocess, "run", dirty)
    assert provenance.git_is_dirty() is True


def test_dirty_is_false_on_a_clean_tree(monkeypatch):
    def clean(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", clean)
    assert provenance.git_is_dirty() is False


def test_git_helpers_degrade_to_none_when_git_is_not_installed(monkeypatch):
    """A source tarball on a machine without git must still produce a stamp."""
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)
    assert provenance.git_commit() is None
    assert provenance.git_is_dirty() is None
    assert provenance.code_provenance()["pharos_version"]


def _init_repo(root):
    """A throwaway git repo with a source file and a measurement artifact."""

    executable = shutil.which("git")
    assert executable, "git is required for this test"

    def git(*args):
        subprocess.run(  # noqa: S603  # fixed argv, executable resolved via which
            [executable, *args],
            cwd=root,
            check=True,
            capture_output=True,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "HOME": str(root),
                "PATH": os.environ.get("PATH", ""),
            },
        )

    (root / "src").mkdir()
    (root / "results").mkdir()
    (root / "src" / "code.py").write_text("x = 1\n", encoding="utf-8")
    (root / "results" / "measurement.json").write_text('{"f1": 0.5}\n', encoding="utf-8")
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-qm", "initial")
    return git


def test_a_rewritten_artifact_does_not_mark_the_code_dirty(tmp_path, monkeypatch):
    """The six-model sweep bug.

    A sweep writes one artifact per model into `results/`. Checking the whole tree
    meant the first write dirtied it for every model after, so one clean checkout
    produced one artifact marked clean and five marked dirty -- reading as five
    measurements whose code could not be reconstructed, when it was identical and
    committed throughout.
    """
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert provenance.git_is_dirty() is False

    (tmp_path / "results" / "measurement.json").write_text('{"f1": 0.9}\n', encoding="utf-8")

    assert provenance.git_is_dirty() is False, "an artifact is an output, not the code"


def test_a_modified_source_file_still_marks_the_code_dirty(tmp_path, monkeypatch):
    """The exclusion must not hide the thing the flag exists to report."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    (tmp_path / "src" / "code.py").write_text("x = 2\n", encoding="utf-8")

    assert provenance.git_is_dirty() is True


# ------------------------------------------------- degradation is audible -----


def test_a_missing_git_is_logged_not_swallowed(monkeypatch, caplog):
    """The traceability module must not lose traceability quietly.

    Degrading to None is correct -- a measurement must not die because it cannot
    name its commit -- but silence about it is not. A run that cannot identify its
    code writes an artifact nothing downstream can verify, and the consumer that
    would catch it treated a missing commit as "nothing to check".
    """
    import logging

    from pharos import provenance

    monkeypatch.setattr(provenance.shutil, "which", lambda _: None)
    with caplog.at_level(logging.WARNING, logger="pharos"):
        stamp = provenance.code_provenance()

    assert stamp["git_commit"] is None
    events = [r.__dict__.get("event") for r in caplog.records]
    assert "provenance.degraded" in events, "a failed git call must say so"
    assert "provenance.unidentifiable" in events, "the cost must be stated once at the top"


def test_a_failing_git_names_its_exit_code(monkeypatch, caplog):
    import logging

    from pharos import provenance

    class Failed:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(provenance.shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(provenance.subprocess, "run", lambda *a, **k: Failed())
    with caplog.at_level(logging.WARNING, logger="pharos"):
        provenance.code_provenance()

    reasons = [r.__dict__.get("reason") for r in caplog.records]
    assert any(r and "128" in r for r in reasons), "the exit code is the diagnostic"


def test_a_healthy_checkout_logs_no_degradation_warning(caplog):
    """The warning must be rare enough to mean something when it fires."""
    import logging

    from pharos import provenance

    with caplog.at_level(logging.WARNING, logger="pharos"):
        stamp = provenance.code_provenance()

    if stamp["git_commit"] is not None:
        events = [r.__dict__.get("event") for r in caplog.records]
        assert "provenance.unidentifiable" not in events
