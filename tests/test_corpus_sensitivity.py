"""Findings 20 to 23 against the corpus draw, and the denominators that keeps honest."""

from conftest import artifact


def test_the_seed_is_a_sweepable_dimension_of_every_governance_script():
    """The defect finding 26 exists because of, asserted so it cannot come back.

    All four scripts hard-coded `SEED = 7` and took no `--seed`, so the corpus was not a
    parameter of this project's headline governance results. It is the same shape as the
    `--fleet` defect finding 24 found -- a sweepable dimension held by a constant sized
    for one draw of it -- and this is the guard that a new measurement script does not
    reintroduce it.
    """
    import importlib
    from pathlib import Path

    for name in (
        "measure_audit_policy",
        "measure_blind_spot",
        "measure_channel_bias",
        "measure_authority_anchors",
    ):
        # Read the script's source rather than calling main(): each runs for minutes.
        module = importlib.import_module(name)
        source = module.__file__ or ""
        assert source
        text = Path(source).read_text(encoding="utf-8")
        assert '"--seed"' in text, f"{name} takes no --seed; the corpus is not sweepable"
        assert "seed=SEED" not in text.split("args = parser.parse_args()")[-1], (
            f"{name} still reads the module constant after parsing, so --seed is accepted "
            "and then ignored -- the exact shape of the --fleet defect"
        )


def test_selection_ties_the_oracle_bound_below_the_crossing_and_not_above_it():
    """Finding 20's deep claim, scored without the vacuous tie that first inflated it.

    A policy that ties the bound means no better selection rule exists on this signal.
    The first version of this sweep counted `None == None` -- neither margin nor the
    oracle repairing at any budget -- as a tie, which made the claim hold everywhere. It
    does not: at seven wrong it holds in six draws of eight, and that is where the fleet
    is hardest to repair at all.
    """
    payload = artifact("corpus_sensitivity")
    by_composition = {e["n_wrong"]: e for e in payload["by_composition"]}
    for n_wrong in (5, 6):
        entry = by_composition[n_wrong]
        assert entry["ties"] == entry["draws"], (
            f"{n_wrong} of 9 ties the bound in only {entry['ties']}/{entry['draws']} draws; "
            "the part of finding 20 that survived the corpus sweep was the tie below the "
            "crossing, and it has stopped surviving"
        )
    seven = by_composition[7]
    assert seven["ties"] < seven["draws"], (
        "seven of 9 now ties the bound in every draw, which is stronger than published -- "
        "publish it rather than leaving the weaker claim in place"
    )
    for cell in (c for row in payload["audit"] for c in row["cells"]):
        assert not (cell["margin_ties_oracle"] and cell["margin"] is None), (
            f"draw {cell['n_wrong']}: a tie is recorded where margin never repaired, which "
            "is the scoring defect this test exists for"
        )


def test_the_headline_that_moved_is_recorded_as_moved():
    """Finding 23's 1.00, and finding 20's beats-uniform at seven of nine.

    Both were published unconditionally and neither holds in every draw. Asserted false
    rather than deleted, because an invariant that quietly stops being checked is how the
    number gets requoted.
    """
    payload = artifact("corpus_sensitivity")
    invariants = payload["invariants"]
    assert not invariants["provenance_finds_every_corrupted_item_in_every_draw"], (
        "provenance now finds every corrupted item in every draw, which would restore "
        "finding 23's original headline -- publish it rather than relaxing the retraction"
    )
    assert not invariants["selection_beats_uniform_in_every_draw"], (
        "selection now beats a uniform draw everywhere, a stronger result than published"
    )
    assert not invariants["selection_ties_the_oracle_bound_in_every_draw"], (
        "selection ties the oracle bound in every draw again; that invariant was true only "
        "while a cell where neither policy repaired counted as a tie"
    )
    assert not invariants["the_auditable_pool_is_a_constant"], (
        "the auditable pool is one value across every draw, which would restore the 97 "
        "the script documented as a property of the design"
    )


def test_the_advantage_survives_even_where_the_absolute_score_does_not():
    """What finding 23 should be quoted on, and the reason the retraction is partial.

    The comparison is the finding: provenance recovers most corrupted items while every
    policy reading disagreement is at chance. That holds in every draw. Only the 1.00
    moved.
    """
    payload = artifact("corpus_sensitivity")
    blind = payload["blind"]
    assert blind, "no draw hosted the blind-spot experiment; nothing here is tested"
    for row in blind:
        # The claim is only worth asserting if policies were actually compared. It was
        # first read off a key the blind-spot artifact does not have, so the comparison
        # ran over an empty list and `all()` returned true without measuring anything.
        assert row["policies_read"], (
            f"draw {row['seed']}: no disagreement policy was scored, so the row below is "
            "vacuously true rather than measured"
        )
        assert row["disagreement_policies_at_chance"], (
            f"draw {row['seed']}: a disagreement policy is above chance at unanimity, so "
            "finding 21's premise moved and finding 23 answers a question nobody has"
        )
        assert row["channel_hit_rate"] > 0.5, (
            f"draw {row['seed']}: provenance recovers {row['channel_hit_rate']}, which is "
            "no longer an advantage worth quoting over a policy at 0.25"
        )


def test_the_negative_half_holds_in_every_draw():
    """The open question finding 23 leaves, and the twelve-month deliverable behind it."""
    payload = artifact("corpus_sensitivity")
    assert payload["invariants"]["no_policy_repairs_an_unanchored_label"], (
        "a policy now repairs an unanchored label somewhere; the residual obstacle would "
        "be one of selection after all, which changes what is left to build"
    )


def test_a_draw_that_cannot_host_the_control_is_excluded_not_counted_against():
    """Finding 21's precondition, kept separable from a result.

    The blind-spot corpus needs a channel orthogonal to item difficulty. Where it is not,
    the script refuses, and folding that refusal into the denominator would score the
    finding down for a corpus on which it was never tested.
    """
    payload = artifact("corpus_sensitivity")
    hosted = payload["draws_hosting_the_blind_spot"]
    attempted = payload["draws_attempted"]
    assert hosted == len(payload["blind"]), (
        "the hosted count and the rows disagree, so every rate scored against it is wrong"
    )
    assert hosted < attempted, (
        "every draw hosts the experiment, so the exclusion path is untested here; at seed "
        "23 the PARTNER compartment was entangled with difficulty and the script refused"
    )
    assert hosted >= 3, "too few draws host the experiment for any rate to mean anything"


def test_the_budget_ladder_is_truncated_to_the_pool_it_draws_from():
    """The defect that made this script exit non-zero on half the draws.

    `select` refuses a budget past the auditable pool rather than clipping it, which is
    right -- a threshold reported at an untested budget is worse than a missing row. So
    the ladder has to shrink instead, and the artifact has to say when it did.
    """
    payload = artifact("corpus_sensitivity")
    truncated = [row for row in payload["audit"] if row["ladder_truncated"]]
    assert truncated, (
        "no draw truncated the ladder, so the fix is untested here; four of eight draws "
        "have a pool below the ladder's top rung of 95"
    )
    for row in payload["audit"]:
        assert max(row["budgets"]) <= row["auditable_pool"], (
            f"draw {row['seed']}: a budget exceeds the {row['auditable_pool']} auditable "
            "tasks, which `select` refuses outright"
        )
        assert row["budgets"] == [
            b for b in row["budgets_requested"] if b <= row["auditable_pool"]
        ], (
            "the used ladder is not the requested one filtered by the pool, so it is not "
            "a truncation but a different ladder"
        )


def _stub(monkeypatch, payload):
    """Point the module's `run_at` at a payload instead of a subprocess."""
    import measure_corpus_sensitivity as mod

    monkeypatch.setattr(mod, "run_at", lambda *a, **k: payload)
    return mod


def test_a_policy_that_never_repairs_does_not_beat_one_that_never_repairs():
    """`None` is "never repaired at any budget", not "repaired at budget infinity".

    Ordering it as a number would make two failures look like a win for whichever was
    checked first, which is the shape of the censoring error finding 20's sweep made.
    """
    import measure_corpus_sensitivity as mod

    assert mod._better(20, 45) is True
    assert mod._better(20, None) is True
    assert mod._better(None, 45) is False
    assert mod._better(None, None) is False
    assert mod._better(45, 45) is False


def test_the_audit_row_separates_a_loss_from_nothing_being_repairable(monkeypatch):
    """`nothing_repaired` is a property of the corpus, not of the policies failing on it.

    It is read off the artifact's budget-zero oracle row: a composition the fleet gets
    right before any audit had nothing to repair. Deriving it from "neither margin nor
    uniform repaired" instead conflates that with a draw where both simply lost, and
    silently drops the second out of the denominator.
    """
    mod = _stub(
        monkeypatch,
        {
            "thresholds": {
                "margin": {"5": 20, "6": None},
                "oracle": {"5": 20, "6": None},
                "uniform": {"5": 45, "6": None},
            },
            "compositions": [5, 6],
            "grid": [
                {"n_wrong": 5, "budget": 0, "policy": "oracle", "remaining_errors": 12},
                {"n_wrong": 6, "budget": 0, "policy": "oracle", "remaining_errors": 0},
            ],
            "auditable_pool": 97,
            "budgets": [10, 20],
            "budgets_requested": [10, 20, 95],
            "best_deployable": "margin",
        },
    )
    row = mod.audit_row(7)
    won, unbroken = row["cells"]
    assert (won["margin_beats_uniform"], won["margin_ties_oracle"]) == (True, True)
    assert won["nothing_repaired"] is False
    assert unbroken["nothing_repaired"] is True
    assert unbroken["margin_beats_uniform"] is False
    assert row["ladder_truncated"] is True


def test_a_tie_needs_two_finite_thresholds(monkeypatch):
    """The defect that made finding 26's own headline hold everywhere.

    At seed 202, seven wrong, neither margin nor the oracle repaired at any budget the
    truncated ladder reached. `None == None` scored that as "ties the bound", and one such
    cell was the whole of `selection_ties_the_oracle_bound_in_every_draw`.
    """
    mod = _stub(
        monkeypatch,
        {
            "thresholds": {"margin": {"7": None}, "oracle": {"7": None}, "uniform": {"7": 80}},
            "compositions": [7],
            "grid": [{"n_wrong": 7, "budget": 0, "policy": "oracle", "remaining_errors": 9}],
            "auditable_pool": 83,
            "budgets": [10, 80],
            "budgets_requested": [10, 80],
            "best_deployable": "uniform",
        },
    )
    cell = mod.audit_row(202)["cells"][0]
    assert cell["margin_ties_oracle"] is False, (
        "two failures to repair are being scored as a tie with the bound"
    )
    assert cell["nothing_repaired"] is False, (
        "the fleet was broken here, so this cell is priced and counts against selection"
    )


def test_the_blind_row_does_not_credit_a_tie_with_an_absent_oracle(monkeypatch):
    """`channel_ties_oracle` compares two rates, and a missing oracle is not a tie.

    `dict.get` returning None on both sides would make them equal, which would report a
    tie with the bound on a draw where the bound was never measured.
    """
    mod = _stub(
        monkeypatch,
        {
            "fleet": 9,
            "audit_hit_rate": {"9": {"channel": 0.9, "uniform": 0.1, "margin": 0.15}},
            "deployable": ["uniform", "margin"],
            "grid": [],
        },
    )
    row = mod.blind_row(1)
    assert row["oracle_finds_all"] is False
    assert row["channel_ties_oracle"] is False
    assert row["disagreement_policies_at_chance"] is True
    assert row["channel_corrected"] is None


def test_the_channel_row_reports_no_detections_as_undetected(monkeypatch):
    """`all()` over an empty list is True, which would publish silence as detection."""
    mod = _stub(
        monkeypatch,
        {
            "blind_channel": "PARTNER",
            "shares": [1, 9],
            "sweep": [{"n_blind": 9, "detections": []}],
            "controls_clean": True,
        },
    )
    assert mod.channel_row(1)["blinded_detected_at_every_noise_level"] is False


def _fake_subprocess(monkeypatch, *, returncode=0, stderr="", payload=None):
    """Replace the subprocess call with one that writes `payload` and exits `returncode`."""
    import json as _json
    import subprocess
    from pathlib import Path as _Path

    import measure_corpus_sensitivity as mod

    def run(cmd, **_kwargs):
        if payload is not None:
            _Path(cmd[cmd.index("--out") + 1]).write_text(_json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)

    monkeypatch.setattr(mod.subprocess, "run", run)
    return mod


def test_a_draw_that_refuses_by_design_is_excluded(monkeypatch):
    """The one non-zero exit this sweep is allowed to treat as data.

    `measure_blind_spot` exits REFUSED_EXIT when the blind channel is entangled with item
    difficulty. That draw cannot host the negative control and says nothing about the
    finding, so it leaves the denominator rather than counting against it.
    """
    from measure_blind_spot import REFUSED_EXIT

    mod = _fake_subprocess(monkeypatch, returncode=REFUSED_EXIT, stderr="too entangled")
    assert mod.run_at("measure_blind_spot.py", 23) is None


def test_a_crash_is_not_a_refusal(monkeypatch):
    """The defect this guard exists for, and it nearly shipped.

    Every non-zero exit used to be read as the precondition refusal, so the budget-ladder
    crash this branch fixes would have logged four of eight draws as "refused by design"
    and silently shrunk the denominator under a rate that looked unchanged.
    """
    import pytest

    mod = _fake_subprocess(monkeypatch, returncode=1, stderr="Traceback: KeyError 'grid'")
    with pytest.raises(RuntimeError, match="exited 1"):
        mod.run_at("measure_audit_policy.py", 7)


def test_run_at_returns_the_artifact_the_script_wrote(monkeypatch):
    mod = _fake_subprocess(monkeypatch, payload={"fleet": 9})
    assert mod.run_at("measure_audit_policy.py", 7) == {"fleet": 9}


def test_a_refused_draw_produces_no_row_of_any_kind(monkeypatch):
    """All three builders drop the draw rather than emitting a partial row."""
    from measure_blind_spot import REFUSED_EXIT

    mod = _fake_subprocess(monkeypatch, returncode=REFUSED_EXIT, stderr="refused")
    assert mod.audit_row(23) is None
    assert mod.blind_row(23) is None
    assert mod.channel_row(23) is None


def test_the_blind_row_refuses_a_draw_where_no_policy_was_scored(monkeypatch):
    """The vacuous-truth path, asserted so it cannot come back quietly.

    Reading the policy list from a key the artifact does not have gave an empty list, and
    `all()` over it returned true. An empty list is now an error rather than a pass.
    """
    import pytest

    mod = _fake_subprocess(
        monkeypatch,
        payload={
            "fleet": 9,
            "audit_hit_rate": {"9": {"channel": 0.9, "oracle": 1.0}},
            "grid": [],
        },
    )
    with pytest.raises(RuntimeError, match="finding 21's claim cannot be evaluated"):
        mod.blind_row(1)
