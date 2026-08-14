"""Guards for the command that reads the logs.

`scripts/logcheck.py` is the thing that notices when a measurement stops reporting
itself. Two ways it can quietly stop working, both of which leave it exiting 0:

- a measurement is renamed and drops out of `SCRIPTS`, so its warnings are never
  collected and its absence is indistinguishable from silence;
- a name in `EXPECTED_WARNINGS` is misspelled, so the "expected warning that did not
  fire" detection can never fire for it, because no such warning exists to go missing.

Neither is hypothetical for a list maintained by hand alongside the code it names.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _logcheck_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import logcheck

    return logcheck


def test_every_listed_script_exists():
    """A renamed measurement must break this, not silently leave the sweep."""
    logcheck = _logcheck_module()
    missing = [s for s in logcheck.SCRIPTS if not (ROOT / "scripts" / s).is_file()]
    assert not missing, f"logcheck names scripts that do not exist: {missing}"


def test_the_model_free_sweep_covers_the_model_free_measurements():
    """The sweep is meant to be everything that needs no model and no network.

    A new measurement of that kind that never joins `SCRIPTS` is exactly the decay
    this command exists to prevent, so the omission has to be deliberate rather than
    accidental: anything skipped is named here with a reason.
    """
    logcheck = _logcheck_module()

    # Measurements that genuinely cannot run in this sweep.
    needs_more_than_cpu = {
        "measure_label_fidelity.py": "calls a model",
        "measure_decode_stability.py": "calls a model",
        "measure_rule_learnability.py": "calls a model",
        "measure_teacher_transfer.py": "calls a model",
        "measure_triage_lift.py": "calls a model",
        "measure_federation_eligibility.py": "deterministic; emits no measurement events",
        # Model-free and CPU-only, so it does not belong with the entries above, but it
        # re-runs three of the swept scripts across five fleet sizes and takes roughly
        # twenty minutes. Logcheck runs inside the blocking `shortcut-gate` job, and a
        # sensitivity sweep is not something a commit needs re-derived to be accepted.
        # `make fleet-sensitivity` runs it, and its own artifact carries the invariants.
        "measure_fleet_sensitivity.py": "re-runs three swept scripts; minutes, not seconds",
        # Same shape as the sweep above and slower: it re-runs four swept scripts
        # across four fleet sizes, and the channel detector fits a permutation test on
        # top of each. Its artifact carries the invariants.
        "measure_governance_sensitivity.py": "re-runs four swept scripts; tens of minutes",
        # All four of those scripts again, across eight corpus draws instead of four
        # fleet sizes, with a permutation test per draw. Slower than the sweep above.
        "measure_corpus_sensitivity.py": "re-runs four swept scripts; tens of minutes",
        # Model-free and CPU-only, but it runs the gate seven times, which CI already
        # does for the gate table, and its result is a *comparison between two machines*
        # rather than a number. One machine's half of it carries no signal on its own,
        # so running it here would spend two minutes to learn nothing.
        "measure_gate_determinism.py": "one half of a two-machine comparison; CI is one machine",
        "measure_edge_cost.py": "reads artifacts produced by GPU jobs",
        "measure_teacher_fleet.py": "reads adapter artifacts produced by cluster jobs",
        "measure_adapter_replication.py": "reads adapter artifacts produced by cluster jobs",
        "train_adapter.py": "needs a CUDA GPU",
        "validate_gate_externally.py": "downloads corpora from the network",
        "sweep_models.sh": "shell driver",
        "sync_cluster.sh": "shell driver",
        "sync_docs_tables.py": "renders docs; not a measurement",
        "compare_models.py": "reads artifacts; not a measurement",
        "build_static_explorer.py": "renders a page; not a measurement",
        "logcheck.py": "this command",
    }

    on_disk = {p.name for p in (ROOT / "scripts").glob("measure_*.py")}
    unaccounted = on_disk - set(logcheck.SCRIPTS) - set(needs_more_than_cpu)
    assert not unaccounted, (
        "measurement scripts neither swept nor explicitly excused: "
        f"{sorted(unaccounted)}. Add to logcheck.SCRIPTS, or name the reason here."
    )


def test_no_expected_warning_is_exempt_from_the_missing_check_by_accident():
    """The missing-warning check used to name its members a second time, and drifted.

    `core_missing` was a second literal set, comments and all, duplicating the members
    of EXPECTED_WARNINGS that this command can actually observe. When
    `authority.not_repaired` was added to the first set and not the second, the finding
    whose entire published limit is that no affordable authority repairs it would have
    stopped announcing that limit while logcheck still exited 0.

    The exemption is a property of the `validity.` family, not a list of names, so this
    asserts the property. Anything outside that family must be checked for absence.
    """
    logcheck = _logcheck_module()

    exempt = {w for w in logcheck.EXPECTED_WARNINGS if w.startswith(logcheck.VALIDITY_PREFIX)}
    checked = logcheck.EXPECTED_WARNINGS - exempt

    assert checked, "every expected warning is exempt, so the missing check tests nothing"
    assert "authority.not_repaired" in checked, (
        "the warning whose omission motivated this test is exempt again"
    )
    # Every exempt name has to earn it by being in the family, not by being forgotten.
    assert all(w.startswith(logcheck.VALIDITY_PREFIX) for w in exempt)


def test_every_expected_warning_has_something_that_emits_it():
    """A typo here disables the missing-warning detection for that name, silently."""
    logcheck = _logcheck_module()

    sources = [
        p.read_text(encoding="utf-8")
        for p in [*(ROOT / "src" / "pharos").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]
        if p.name != "logcheck.py"
    ]
    orphans = [name for name in logcheck.EXPECTED_WARNINGS if not any(name in s for s in sources)]
    assert not orphans, (
        f"logcheck expects warnings nothing emits: {sorted(orphans)}. "
        "Either the emitter was removed, or the name is misspelled here."
    )


def test_run_collects_structured_records_and_ignores_the_rest(tmp_path, monkeypatch):
    """Only JSON lines carrying a `logger` are records. Plain output is not evidence."""
    logcheck = _logcheck_module()

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "probe.py").write_text(
        "import json, sys\n"
        "print('a human-readable table row, not a record')\n"
        "print('{ not valid json')\n"
        "print(json.dumps({'no_logger_key': 1}))\n"
        "print(json.dumps({'logger': 'pharos', 'level': 'INFO', 'event': 'x.ok'}))\n"
        "print(json.dumps({'logger': 'pharos', 'level': 'WARNING', 'event': 'x.bad'}),"
        " file=sys.stderr)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(logcheck, "ROOT", tmp_path)

    records, code = logcheck.run("probe.py", debug=False)
    assert code == 3, "a failing measurement must be reported as failing"
    assert [r["event"] for r in records] == ["x.ok", "x.bad"]
    # stderr is collected too: warnings go there, and dropping it would drop the
    # only records that matter.
    assert any(r["level"] == "WARNING" for r in records)


def test_run_asks_for_debug_records_when_told_to(tmp_path, monkeypatch):
    """`--debug` is the difference between routine metrics appearing and not."""
    logcheck = _logcheck_module()

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "probe.py").write_text(
        "import json, os\n"
        "print(json.dumps({'logger': 'pharos', 'level': 'INFO',"
        " 'event': os.environ['PHAROS_LOG_LEVEL']}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(logcheck, "ROOT", tmp_path)

    quiet, _ = logcheck.run("probe.py", debug=False)
    loud, _ = logcheck.run("probe.py", debug=True)
    assert quiet[0]["event"] == "INFO"
    assert loud[0]["event"] == "DEBUG"


@pytest.mark.slow
def test_logcheck_passes_on_this_repository():
    """The end-to-end assertion: the sweep is green, and green means what it says.

    Slow, because it runs every model-free measurement. It is the only check that the
    expected-warning list still matches what the code actually emits.
    """
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts" / "logcheck.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-2000:]
    assert "no unexpected warnings" in completed.stdout
    assert "EXPECTED WARNINGS THAT DID NOT FIRE" not in completed.stdout


def test_expected_warning_json_is_parseable_as_the_records_it_describes():
    """The record shape logcheck keys on -- `event` or `metric`, plus `level`."""
    logcheck = _logcheck_module()
    line = json.dumps(
        {"logger": "pharos", "level": "WARNING", "event": "consensus.cliff", "n_wrong": 5}
    )
    parsed = json.loads(line)
    assert parsed["event"] in logcheck.EXPECTED_WARNINGS
    assert parsed["level"] in {"WARNING", "ERROR", "CRITICAL"}


def _make_ci_recipe() -> list[str]:
    """The command lines of the Makefile's `ci` target."""
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ci:"))
    body: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        body.append(line.strip())
    return body


def _workflow_runs() -> list[str]:
    """Every `run:` command in the CI workflow, one per line."""
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    return [
        line.strip().removeprefix("run:").strip()
        for line in text.splitlines()
        if line.strip().startswith("run:")
    ]


def _measurements(commands: list[str]) -> list[str]:
    """The measurement scripts a list of shell commands invokes, in order."""
    import re

    found: list[str] = []
    for command in commands:
        found.extend(re.findall(r"scripts/(\w+\.py)", command))
    return found


def test_make_ci_runs_what_the_workflow_runs():
    """`make ci` says "exactly as the workflow does", so it has to be exactly that.

    It drifted: `measure_correlated_fleets.py`, `measure_difficulty_confound.py` and
    `logcheck.py` were in one and not the other, and `measure_correlated_fleets.py`
    was in neither -- finding 16's numbers were published with nothing re-running them
    against a generator change. A local gate that is a subset of the remote one is the
    worst arrangement available, because it passes and means nothing.
    """
    local = _measurements(_make_ci_recipe())
    remote = _measurements(_workflow_runs())

    assert local == remote, (
        "make ci and .github/workflows/ci.yml disagree.\n"
        f"  only in make ci: {[s for s in local if s not in remote]}\n"
        f"  only in workflow: {[s for s in remote if s not in local]}\n"
        f"  make ci order:   {local}\n"
        f"  workflow order:  {remote}"
    )
    assert remote, "parsed no measurement steps out of the workflow; the parser broke"


def test_every_model_free_measurement_reaches_ci():
    """A published measurement nothing re-runs is a number frozen at its first result."""
    remote = set(_measurements(_workflow_runs()))
    logcheck = _logcheck_module()

    # logcheck's own sweep is the definition of "runs anywhere", so CI should run
    # every member of it, either directly or by running logcheck itself.
    missing = set(logcheck.SCRIPTS) - remote
    assert not missing or "logcheck.py" in remote, (
        f"model-free measurements absent from CI and not covered by logcheck: {sorted(missing)}"
    )
