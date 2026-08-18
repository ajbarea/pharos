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
        # Measures the test suite rather than a corpus, so it emits no pharos log events
        # for logcheck to sweep. It also runs the whole suite once per mutation plus a
        # baseline, which is five suites, and it rewrites source files while it does it.
        "measure_guard_mutations.py": "runs the suite five times and edits source; not a corpus measurement",
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


def test_every_scoped_warning_still_names_something_its_script_emits():
    """The same question as above, asked of the list that had nobody asking it.

    `EXPECTED_WARNINGS` is checked in both directions -- unexpected fails, and stopping
    firing fails. `SCRIPT_SCOPED_WARNINGS` had neither, and it is the more dangerous of
    the two: it widens what one named script may say, so a reason that has expired keeps
    excusing that script alone, where nobody is looking.
    """
    logcheck = _logcheck_module()
    assert logcheck.SCRIPT_SCOPED_WARNINGS, "nothing is scoped; this test is now vacuous"
    assert not logcheck.verify_scoped_exemptions()


def test_a_scoped_tolerance_that_stopped_meaning_anything_fails_the_check(monkeypatch):
    """Tripped rather than trusted, which is the property the check itself is about.

    Both halves: an event the named script no longer emits, and a script this command
    does not run. Either makes the entry a tolerance with nothing behind it, sitting
    ready to absorb the next thing that happens to use the name.
    """
    logcheck = _logcheck_module()

    monkeypatch.setattr(
        logcheck,
        "SCRIPT_SCOPED_WARNINGS",
        {"measure_error_shape.py": frozenset({"error_shape.event_that_never_was"})},
    )
    stale = logcheck.verify_scoped_exemptions()
    assert stale and "no longer emits" in stale[0]

    monkeypatch.setattr(
        logcheck, "SCRIPT_SCOPED_WARNINGS", {"measure_not_a_script.py": frozenset({"a.b"})}
    )
    unknown = logcheck.verify_scoped_exemptions()
    assert unknown and "not in SCRIPTS" in unknown[0]


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


def _normalize(command: str) -> str:
    """One shell command in a form the Makefile and the workflow can be compared in.

    Whitespace collapses because the workflow folds long commands over several lines;
    `$$` becomes `$` because make escapes it; quotes go because `"$seed"` and `$$seed`
    are the same loop variable written for two readers; and `;` goes because a loop
    written across three lines separates with newlines where a one-liner needs the
    semicolon, which is a difference in typography rather than in what runs.
    """
    collapsed = " ".join(command.replace("$$", "$").replace('"', "").replace(";", " ").split())
    # A trailing discard of stdout is volume, not behaviour: `make ci` silences the tree
    # stamp because it prints a hash nobody reads at that point, and the workflow does not.
    # Only this exact suffix, so a redirect that sends output somewhere real still counts as
    # a difference between the two.
    for suffix in (" >/dev/null 2>&1", " >/dev/null"):
        if collapsed.endswith(suffix):
            return collapsed[: -len(suffix)]
    return collapsed


def _workflow_runs() -> list[str]:
    """Every `run:` command in the CI workflow, in order.

    Parsed rather than grepped. The line-based version this replaces matched only steps
    carrying a `name:`, because a bare `- run:` does not start its line -- so the
    telemetry assertion, the three lint commands, the folded `pytest` invocation and
    `pip-audit` were all invisible to it. Two of those were missing from `make ci` and
    the check that exists to say so could not see them.
    """
    return [command for _, commands in _workflow_runs_by_job() for command in commands]


def _workflow_runs_by_job() -> list[tuple[str, list[str]]]:
    """`(job name, its run commands in order)`.

    Grouped rather than flattened, because the ordering question is only answerable per
    job: the workflow splits across `lint-and-test` and `shortcut-gate`, which run in
    parallel and have no order relative to each other, while `make ci` is one sequence.
    """
    import yaml

    doc = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    return [
        (name, [_normalize(step["run"]) for step in job.get("steps", []) if "run" in step])
        for name, job in doc["jobs"].items()
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


#: Workflow commands `make ci` is not expected to run, each with the reason. Setup is the
#: only shape that qualifies so far: it prepares the environment the target assumes rather
#: than being a gate the target skips.
WORKFLOW_ONLY = {
    "uv sync --all-groups": "environment setup; `make setup` does this before `make ci`",
}

#: Flags that only produce artifacts for the Codecov upload steps, which have no local
#: counterpart. Dropped before comparing, so the command underneath can still be matched.
UPLOAD_ONLY_FLAGS = {"--cov-report=xml", "--junitxml=junit.xml", "-o", "junit_family=legacy"}


def _strip_upload_flags(command: str) -> str:
    """A command without the flags that exist only to feed the Codecov upload steps."""
    return " ".join(token for token in command.split() if token not in UPLOAD_ONLY_FLAGS)


def test_make_ci_runs_the_workflows_tool_steps_too():
    """The comparison above reads `scripts/`, and the drift was in the steps that do not.

    `uv run python -c "... telemetry.configure() is False"` and `uv run pip-audit` were
    both in `ci.yml` and neither was in `make ci`. Neither names a script, so the
    measurement comparison could not see them, and the target's promise to be "exactly as
    the workflow does" held only for the half of the workflow that happened to be checked.
    """
    local = {_strip_upload_flags(_normalize(c)) for c in _make_ci_recipe()}

    missing = []
    for command in _workflow_runs():
        if _measurements([command]) or command in WORKFLOW_ONLY:
            continue
        stripped = _strip_upload_flags(command)
        if stripped not in local:
            missing.append(stripped)

    assert not missing, (
        "`ci.yml` runs commands `make ci` does not, so the local gate is a subset of the "
        f"remote one and passes meaning less than it claims: {missing}. Add them to the "
        "`ci` target, or name the reason in WORKFLOW_ONLY."
    )


#: Commands whose position in `make ci` deliberately differs from the workflow's, with the
#: reason. Order is asserted below, so a difference that is intended has to say so here.
ORDER_EXEMPT = {
    "uv run python scripts/tree_fingerprint.py --write .gate-fingerprint": (
        "the workflow stamps after the gate seeds, inside `shortcut-gate`; `make ci` stamps "
        "before lint and the suite. In CI the suite runs in a different job and cannot be "
        "covered by this stamp at all, so the local window is deliberately the wider of the "
        "two. Matching the workflow here would mean covering less."
    ),
}


def _first_out_of_order(wanted: list[str], actual: list[str]) -> str | None:
    """The first element of `wanted` that `actual` does not contain in that relative order.

    A subsequence test rather than an equality one: `make ci` legitimately holds commands
    between the workflow's, because the workflow splits the same sequence over two jobs and
    each job carries setup that the local target does not repeat.
    """
    remaining = iter(actual)
    for want in wanted:
        if not any(have == want for have in remaining):
            return want
    return None


def test_make_ci_runs_them_in_the_workflows_order():
    """Membership is not enough: several of these steps are only correct where they sit.

    `pip-audit` runs after the suite so a CVE in a transitive dependency cannot mask a real
    test failure. `tree_fingerprint --verify` is last so it covers every step above it. A
    set comparison calls all of those arrangements equal, and the check above was a set
    comparison, so a step could have moved anywhere without failing anything.
    """
    recipe = [_strip_upload_flags(_normalize(c)) for c in _make_ci_recipe()]

    for job, commands in _workflow_runs_by_job():
        wanted = [
            _strip_upload_flags(c)
            for c in commands
            if c not in WORKFLOW_ONLY and c not in ORDER_EXEMPT
        ]
        stray = _first_out_of_order(wanted, recipe)
        assert stray is None, (
            f"`make ci` does not run `{stray}` in the order `{job}` does. The workflow's "
            "steps have to appear in the target in the same relative order, since several "
            "of them are only correct where they sit. Reorder the `ci` target, or name the "
            "command in ORDER_EXEMPT with the reason its position differs."
        )


def test_make_ci_runs_nothing_the_workflow_does_not():
    """The other direction. A local gate that is a superset passes what CI would fail.

    The pair of checks above only ever asked whether the workflow's commands reached the
    target. A command added to `make ci` alone would sail through both, and the developer
    running it would be gating on something CI never checks -- which reads as a stricter
    local gate right up until it is the reason a green pull request breaks `main`.
    """
    remote = {_strip_upload_flags(c) for c in _workflow_runs()}
    local_only = [
        c
        for c in (_strip_upload_flags(_normalize(c)) for c in _make_ci_recipe())
        if c not in remote and not c.startswith("#")
    ]
    assert not local_only, (
        f"`make ci` runs commands `ci.yml` does not: {local_only}. Either the workflow is "
        "missing a gate the local target has, or the target has grown something the gate "
        "never verifies."
    )


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


def test_scoped_warnings_name_a_script_that_exists_and_emits_them():
    """A scoped expectation is narrower than a global one and fails in the same ways.

    Two failures are possible and both are silent. The key can name a script that no
    longer exists, in which case the expectation applies to nothing; or the warning can
    name an event nothing emits, in which case it silences a name that never fires and
    would keep silencing it if a real emitter appeared under that name later.

    This is deliberately stricter than the global set: a scoped warning must be
    unnecessary globally, or the scoping is decoration over an expectation that already
    applies everywhere.
    """
    logcheck = _logcheck_module()

    sources = [
        p.read_text(encoding="utf-8")
        for p in [*(ROOT / "src" / "pharos").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]
        if p.name != "logcheck.py"
    ]
    for script, names in logcheck.SCRIPT_SCOPED_WARNINGS.items():
        assert (ROOT / "scripts" / script).is_file(), (
            f"logcheck scopes warnings to {script}, which does not exist"
        )
        assert script in logcheck.SCRIPTS, (
            f"{script} carries scoped warnings but logcheck never runs it"
        )
        for name in names:
            assert any(name in s for s in sources), (
                f"logcheck expects {name} from {script}, and nothing emits that event"
            )
            assert name not in logcheck.EXPECTED_WARNINGS, (
                f"{name} is expected globally as well as scoped to {script}; "
                "the scoping has no effect and reads as though it does"
            )
