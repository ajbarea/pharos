"""The governance kit, tested as a library rather than through a measurement.

Extracting this package out of `scripts/` made visible how much of it had only ever been
executed inside a `main()` that coverage excludes. Code that runs but is never asserted
against is the same shape as a guard nobody has seen fail: it works until it does not, and
nothing says which.

These are the pieces a third party would call directly, and the ones whose contracts the
measurements depend on without restating.
"""

import json

import pytest

from pharos.governance import (
    ServerObservation,
    policy_channel,
    policy_consensus,
    policy_margin,
    policy_oracle,
    policy_posterior,
    select,
)
from pharos.governance.fleet import REFUSED_EXIT
from pharos.governance.sweep import MeasurementFailedError, run_measurement

TASKS = ("t0", "t1", "t2", "t3")


def _view(
    *,
    votes: dict[str, float] | None = None,
    posterior: dict[str, float] | None = None,
    evidence: dict[str, int] | None = None,
    carries: dict[str, bool] | None = None,
) -> ServerObservation:
    """A view with the fields spelled out rather than splatted.

    Keyword arguments rather than `**kwargs` so the checker sees each field's type: the
    splatted version typed every value as `float` and quietly accepted an int evidence
    map where a bool channel map belonged.
    """
    return ServerObservation(
        votes=votes or {"t0": 9.0, "t1": 0.0, "t2": 5.0, "t3": 4.0},
        seen=dict.fromkeys(TASKS, 9.0),
        posterior=posterior or {"t0": 0.99, "t1": 0.01, "t2": 0.55, "t3": 0.45},
        evidence=evidence or {},
        carries=carries or {},
    )


def test_the_view_exposes_no_per_analyst_field():
    """The deployability line, asserted on the type rather than left to a docstring.

    A policy that can see who said what is not a policy, it is an oracle wearing one's
    name. This is the check that keeps that true when somebody adds a field.
    """
    assert set(ServerObservation.__slots__) == {
        "votes",
        "seen",
        "posterior",
        "evidence",
        "carries",
    }


def test_margin_is_a_dead_heat_at_zero_and_unanimity_at_one():
    view = _view()
    assert view.margin("t0") == 1.0
    assert view.margin("t1") == 1.0
    assert view.margin("t2") == pytest.approx(1 / 9)
    assert view.margin("unseen") == 1.0, "a task nobody reported on is not a dead heat"


def test_uncertainty_and_its_inversion_order_tasks_oppositely():
    """`margin` and `consensus` are the same signal read in opposite directions, and one
    of them is the prediction the audit measurement exists to test."""
    view = _view()
    split = min(TASKS, key=lambda t: policy_margin(view, {})[t])
    agreed = min(TASKS, key=lambda t: policy_consensus(view, {})[t])
    assert split == "t2", "uncertainty sampling did not pick the most split task"
    assert agreed in ("t0", "t1"), "the inversion did not pick a unanimous task"
    assert split != agreed


def test_the_posterior_policy_reads_the_estimator_not_the_votes():
    view = _view(posterior={"t0": 0.5, "t1": 0.01, "t2": 0.99, "t3": 0.9})
    assert min(TASKS, key=lambda t: policy_posterior(view, {})[t]) == "t0"


def test_the_oracle_is_a_bound_and_needs_the_truth():
    truth = {"t0": False, "t1": False, "t2": True, "t3": True}
    scores = policy_oracle(_view(), truth)
    # Wrong under the estimator's own labels: t0 (posterior 0.99, truth False) and
    # t3 (posterior 0.45 -> False, truth True).
    assert scores["t0"] == 0.0
    assert scores["t3"] == 0.0
    assert scores["t1"] == 1.0


def test_the_channel_policy_refuses_to_select_without_a_detection():
    """No named channel, no licence. Scoring every task equally would degrade the policy
    to an arbitrary draw while still reporting under its own name."""
    scores = policy_channel(_view(), {})
    assert len(set(scores.values())) == 1, "selected by provenance with nothing detected"


def test_the_channel_policy_orders_by_evidence_within_the_carrying_set():
    """Carrying the channel is necessary and not sufficient.

    A blind analyst only flips a verdict where the discounted channel was doing the work,
    which is the high-evidence end. Selecting on provenance alone put half a budget on
    tasks that could not have changed.
    """
    view = _view(
        carries={"t0": True, "t1": True, "t2": False, "t3": False},
        evidence={"t0": 3, "t1": 1, "t2": 3, "t3": 1},
    )
    ranked = sorted(TASKS, key=lambda t: policy_channel(view, {})[t])
    assert ranked[0] == "t0", "deepest carrying task was not selected first"
    assert set(ranked[:2]) == {"t0", "t1"}, "a non-carrying task outranked a carrying one"


def test_the_channel_policy_falls_back_to_margin_without_evidence_counts():
    view = _view(carries={"t0": True, "t1": False, "t2": True, "t3": False})
    ranked = sorted(TASKS, key=lambda t: policy_channel(view, {})[t])
    assert set(ranked[:2]) == {"t0", "t2"}


def test_selection_is_nested_across_budgets():
    """A larger budget must audit a superset of a smaller one, or a sweep over budgets
    confounds "more anchors" with "different anchors"."""
    view = _view(carries=dict.fromkeys(TASKS, True), evidence={"t0": 3, "t1": 2, "t2": 1, "t3": 0})
    small = set(select("channel", view, {}, 2, seed=7))
    large = set(select("channel", view, {}, 3, seed=7))
    assert small < large


def test_a_budget_past_the_pool_is_refused_rather_than_clipped():
    """Clipping would report a threshold at a budget that was never tested."""
    with pytest.raises(ValueError, match="exceeds"):
        select("margin", _view(), {}, 99, seed=7)
    assert select("margin", _view(), {}, 0, seed=7) == ()


def _stub_script(tmp_path, body: str) -> str:
    """Write a throwaway script into the scripts directory the runner resolves."""
    from pharos.governance import sweep

    path = sweep.SCRIPTS / "_tmp_stub_measure.py"
    path.write_text(body, encoding="utf-8")
    return path.name


def test_a_refusal_is_not_a_failure(tmp_path):
    """The distinction the whole sweep rests on.

    A draw that cannot host an experiment says nothing about the finding; a crash says
    something about the code. A runner that cannot tell them apart reports the second as
    the first, and the point drops out of the denominator while the rate above it looks
    unchanged.
    """
    name = _stub_script(
        tmp_path,
        f"import sys\nprint('cannot host', file=sys.stderr)\nraise SystemExit({REFUSED_EXIT})\n",
    )
    try:
        assert run_measurement(name, ["--seed", "1"], allow_refusal=True) is None
        with pytest.raises(MeasurementFailedError, match="exited"):
            run_measurement(name, ["--seed", "1"], allow_refusal=False)
    finally:
        from pharos.governance import sweep

        (sweep.SCRIPTS / name).unlink(missing_ok=True)


def test_a_crash_raises_even_where_refusals_are_allowed(tmp_path):
    """`allow_refusal` excuses one exit code, not every non-zero one."""
    name = _stub_script(tmp_path, "raise SystemExit(1)\n")
    try:
        with pytest.raises(MeasurementFailedError):
            run_measurement(name, [], allow_refusal=True)
    finally:
        from pharos.governance import sweep

        (sweep.SCRIPTS / name).unlink(missing_ok=True)


def test_a_measurement_that_writes_nothing_is_a_failure(tmp_path):
    """Exiting zero without writing an artifact would otherwise parse as an empty result."""
    name = _stub_script(tmp_path, "import sys\nprint('ok')\n")
    try:
        with pytest.raises(MeasurementFailedError):
            run_measurement(name, [], allow_refusal=True)
    finally:
        from pharos.governance import sweep

        (sweep.SCRIPTS / name).unlink(missing_ok=True)


def test_a_successful_measurement_is_parsed_and_its_temporary_file_removed(tmp_path):
    name = _stub_script(
        tmp_path,
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        "open(out, 'w').write(json.dumps({'measured': 1}))\n",
    )
    try:
        from pharos.governance import sweep

        before = set(sweep.SCRIPTS.parent.glob("*.json"))
        assert run_measurement(name, [], allow_refusal=True) == {"measured": 1}
        assert set(sweep.SCRIPTS.parent.glob("*.json")) == before, "left a file behind"
    finally:
        from pharos.governance import sweep

        (sweep.SCRIPTS / name).unlink(missing_ok=True)


def test_the_runner_never_writes_into_results(tmp_path):
    """A sensitivity sweep must not overwrite a committed artifact, and the way it cannot
    is that the caller's `--out` is the runner's temporary file rather than a path the
    caller chose."""
    from pharos.governance import sweep

    name = _stub_script(
        tmp_path,
        "import sys\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        "assert '/results/' not in out, out\n"
        "open(out, 'w').write('{}')\n",
    )
    try:
        assert run_measurement(name, [], allow_refusal=False) == {}
    finally:
        (sweep.SCRIPTS / name).unlink(missing_ok=True)
    assert json.loads((sweep.SCRIPTS.parent / "results" / "power.json").read_text())


def test_no_two_submodules_define_the_same_public_name():
    """Two definitions of one name in one package, and the package re-exports one of them.

    `governance.fleet` carried an `AbstentionCell` describing a population rate under a
    correlation structure, left behind when this package was extracted out of `scripts/`.
    `governance.abstention` carries a live `AbstentionCell` describing one policy at one
    budget. Nothing imported the dead one, so nothing failed; the way it *would* have
    failed is silent, because both are frozen dataclasses and a mistaken import produces
    a `TypeError` about field names rather than about the wrong class.

    Asserted over the package rather than fixed in the one module, because the extraction
    that produced it moved thirty-four names and this is the check that says a later one
    did not land twice.
    """
    import importlib
    import inspect
    import pkgutil

    import pharos.governance as pkg

    owners: dict[str, list[str]] = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        for name, value in vars(module).items():
            if name.startswith("_") or not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            # Only names this module defines. An import of a sibling's name is the
            # sharing this package exists for, and is not a collision.
            if getattr(value, "__module__", None) != module.__name__:
                continue
            owners.setdefault(name, []).append(info.name)

    duplicated = {name: mods for name, mods in owners.items() if len(mods) > 1}
    assert not duplicated, f"defined in more than one governance submodule: {duplicated}"


def test_no_script_restates_a_constant_the_package_already_owns():
    """A second literal for a shared constant, which is how one of them goes stale.

    The guard above asks the question of `pharos.governance`'s own submodules. It cannot
    see the other half of the codebase: a measurement script writing its own `= 2` beside
    a package that exports the same name and the same value. That is not a hypothetical
    shape. It was fixed four separate times before this test existed --- `shape.ALPHA`
    became a re-export, `measure_difficulty_confound` stopped defining `WRONG_THRESHOLD`,
    the untargeted-draw count moved from two captions into the artifact, and a dead
    `LATENT_SLICE` default came out --- and the fourth time is where a guard is cheaper
    than a fifth fix.

    Equal values only, deliberately. A script whose `SEED` differs from something in the
    package is not restating it; a script whose value *agrees* today is the one that can
    silently stop agreeing, and it is also the only case where an import is
    unambiguously the same program.
    """
    import ast
    import importlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    exported: dict[str, object] = {}
    for module in ("pharos.governance", "pharos.analyst", "pharos.tasks", "pharos.inference"):
        loaded = importlib.import_module(module)
        for name in dir(loaded):
            if name.isupper() and not name.startswith("_"):
                exported.setdefault(name, getattr(loaded, name))

    restated: list[str] = []
    for path in sorted((root / "scripts").glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not target.id.isupper():
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                # A name, a call, or an expression: already reading something rather
                # than restating a literal, which is the shape being asked for.
                continue
            if target.id in exported and exported[target.id] == value:
                restated.append(f"{path.name}:{target.id} = {value!r}")

    assert not restated, (
        "scripts restating a constant the package exports with the same value: "
        f"{restated}. Import it, so the two cannot disagree. If the script genuinely "
        "means a different quantity that happens to share a name and a value today, "
        "rename it -- the collision is the problem either way."
    )
