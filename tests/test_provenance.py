"""Provenance must describe the run without ever being able to break it."""

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
