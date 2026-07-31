"""The registry must never claim a model was run when it was not."""

import urllib.error
import urllib.request

import pytest

from pharos import models


def test_the_default_is_the_reference_model():
    assert models.default_spec().key == models.DEFAULT_KEY
    assert models.default_spec().tag == "qwen2.5:7b-instruct"


def test_only_smoke_tested_models_are_marked_verified():
    """The one claim this module makes that a reader cannot check for themselves."""
    verified = {spec.key for spec in models.REGISTRY.values() if spec.verified}
    assert verified == {"qwen2.5-3b", "qwen2.5-7b", "llama3.2-3b", "llama3.1-8b", "mistral-7b"}, (
        "flip `verified` only after the model has answered a Pharos task, "
        "never on the strength of a model card"
    )


def test_resolve_accepts_a_registry_key():
    assert models.resolve("llama3.2-3b").tag == "llama3.2:3b-instruct-q4_K_M"


def test_resolve_accepts_a_raw_tag_already_in_the_registry():
    assert models.resolve("qwen2.5:7b-instruct").key == "qwen2.5-7b"


def test_resolve_passes_an_unknown_name_through_unverified():
    """The registry must not become a gate on what can be run."""
    spec = models.resolve("some-new-model:70b")
    assert spec.tag == "some-new-model:70b"
    assert spec.verified is False
    assert spec.family == "unknown"


def test_resolve_of_none_is_the_default():
    assert models.resolve(None) == models.default_spec()


def test_every_registry_key_matches_its_spec():
    for key, spec in models.REGISTRY.items():
        assert spec.key == key


def test_installed_degrades_to_empty_when_the_daemon_is_down(monkeypatch):
    def refuse(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert models.installed() == set()


def test_installed_degrades_on_malformed_payload(monkeypatch):
    class Response:
        def read(self):
            return b"not json"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response())
    assert models.installed() == set()


def test_catalog_annotates_installation(monkeypatch):
    monkeypatch.setattr(models, "installed", lambda *a, **k: {"qwen2.5:7b-instruct"})
    rows = models.catalog()
    by_key = {row["key"]: row for row in rows}
    assert by_key["qwen2.5-7b"]["installed"] is True
    assert by_key["llama3.2-3b"]["installed"] is False


def test_catalog_covers_the_whole_registry(monkeypatch):
    monkeypatch.setattr(models, "installed", lambda *a, **k: set())
    assert {row["key"] for row in models.catalog()} == set(models.REGISTRY)


def test_mark_verified_does_not_mutate_the_registry():
    before = models.REGISTRY["llama3.2-3b"].verified
    marked = models.mark_verified("llama3.2-3b")
    assert marked.verified is True
    assert models.REGISTRY["llama3.2-3b"].verified is before


@pytest.mark.parametrize("spec", list(models.REGISTRY.values()))
def test_every_spec_carries_a_usable_note(spec):
    assert spec.note.strip()
    assert spec.approx_vram_gb >= 0


def test_models_command_prints_a_table(monkeypatch, capsys):
    from pharos.cli import main

    monkeypatch.setattr(models, "installed", lambda *a, **k: {"qwen2.5:7b-instruct"})
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "qwen2.5-7b" in out
    # The honesty the registry exists for has to reach the terminal, not just the API.
    assert "candidate" in out
    assert "verified = has answered a Pharos triage task" in out


def test_models_command_emits_json(monkeypatch, capsys):
    import json

    from pharos.cli import main

    monkeypatch.setattr(models, "installed", lambda *a, **k: set())
    assert main(["models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["default"] == models.DEFAULT_KEY
    assert len(payload["models"]) == len(models.REGISTRY)


def test_serve_reports_the_missing_group_rather_than_traceback(monkeypatch, capsys):
    """The explorer is optional, so its absence must be a sentence, not a stack trace."""
    import builtins

    from pharos.web import serve

    real_import = builtins.__import__

    def no_uvicorn(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("no uvicorn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_uvicorn)
    assert serve() == 1
    assert "uv sync --group ui" in capsys.readouterr().out
