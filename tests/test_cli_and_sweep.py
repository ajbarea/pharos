"""The CLI's exit contract, and the ablation sweep against a fake model."""

import json

import pytest

from pharos import attribute as attribute_module
from pharos.attribute import attribute_leave_one_out
from pharos.cli import main
from pharos.generate import GeneratorConfig, generate
from pharos.tasks import build_tasks

REPORTS = generate(GeneratorConfig(seed=5, n_events=120))
TASK = build_tasks(REPORTS, limit=1)[0]


def test_cli_exits_zero_on_a_usable_corpus(capsys):
    code = main(["gate", "--seed", "11", "--events", "150"])
    out = capsys.readouterr().out
    assert "surface baseline" in out
    assert "permutation null" in out
    assert code == 0


def test_cli_writes_a_manifest_when_asked(tmp_path, capsys):
    target = tmp_path / "manifest.json"
    main(["gate", "--seed", "11", "--events", "150", "--out", str(target)])
    capsys.readouterr()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["config"]["seed"] == 11
    assert payload["gate"]["null_trials"] > 0
    assert "surface_baseline" in payload["gate"]


def test_cli_rejects_an_unknown_command():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_the_sweep_costs_one_call_per_source_plus_a_baseline(monkeypatch):
    calls: list[str] = []

    def fake(prompt: str, *, endpoint: str, model: str, num_predict: int = 320) -> str:
        calls.append(prompt)
        # Echo the first source's text, so exactly one drop removes the assertion.
        return TASK.sources[0].text if TASK.sources[0].text in prompt else "nothing asserted."

    monkeypatch.setattr(attribute_module, "generate_text", fake)
    result = attribute_leave_one_out(TASK)
    assert result.calls == 1 + len(TASK.sources)
    assert len(calls) == 1 + len(TASK.sources)


def test_a_source_is_attributed_when_dropping_it_costs_an_assertion(monkeypatch):
    """The sweep's contract: attribution follows lost content, not position."""
    target = TASK.sources[3]

    def fake(prompt: str, *, endpoint: str, model: str, num_predict: int = 320) -> str:
        return target.text if target.text in prompt else "nothing asserted."

    monkeypatch.setattr(attribute_module, "generate_text", fake)
    result = attribute_leave_one_out(TASK)
    assert 3 in result.attributed_sources


def test_a_provided_baseline_skips_the_baseline_call(monkeypatch):
    def fake(prompt: str, *, endpoint: str, model: str, num_predict: int = 320) -> str:
        return "nothing asserted."

    monkeypatch.setattr(attribute_module, "generate_text", fake)
    result = attribute_leave_one_out(TASK, baseline="a summary mentioning draft and freeboard")
    assert "draft_mismatch" in result.asserted_facts
    assert result.calls == 1 + len(TASK.sources)


def test_an_impossible_corpus_is_a_usage_error_not_a_traceback(capsys):
    """`--events 0` is a mistyped flag, and it used to reach the generator.

    `parser.error` exits 2 and prints usage, which is what every other bad flag does. A
    ValueError escaping to the shell would present a mistake in the command line as a
    defect in the generator.
    """
    import pytest as _pytest

    from pharos.cli import main

    with _pytest.raises(SystemExit) as excinfo:
        main(["gate", "--events", "0"])
    assert excinfo.value.code == 2
    assert "n_events" in capsys.readouterr().err
