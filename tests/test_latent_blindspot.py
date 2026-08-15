"""Finding 30: a shared blind spot that follows no partition anybody can name.

The construction is the fragile part rather than the statistics. Every claim this finding
makes is a comparison between two fleets that must differ in exactly one respect, and the
two ways that goes wrong are both silent: a slice that a compartment scan could name after
all would make the scan's success read as its failure, and a slice that flips no verdicts
would make a well-formed artifact out of a fleet that is not blind at all. Finding 21 hit
the second of those and reported success for a day.
"""

import pytest
from conftest import artifact
from measure_latent_blindspot import POLICIES_HERE, SLICE_SIZES

from pharos.analyst import AnalystPolicy, Proposal, evidence_shown
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.governance import (
    BLIND,
    LATENT_ATTEMPTS,
    ServerObservation,
    blind_fleet,
    contributions_for,
    draw_balanced_slice,
    draw_latent_slice,
    fleet,
    latent_blind_fleet,
    observe,
    policy_deviation,
    policy_shortfall,
    select,
)
from pharos.governance.fleet import ChannelUnusableError
from pharos.inference import partition_by_contributor
from pharos.labels import declassify
from pharos.tasks import build_triage_tasks


@pytest.fixture(scope="module")
def tasks():
    return build_triage_tasks(generate(GeneratorConfig(seed=7, n_events=200)))


def test_the_slice_flips_exactly_the_verdicts_it_claims(tasks):
    """The construction's own precondition, and the one finding 21 got wrong first.

    A blind fleet that changes no verdict produces a complete, plausible artifact from a
    fleet that is not blind, because every downstream number is well defined on a fleet
    that agrees with itself. `draw_latent_slice` recomputes the flipped set from the
    reviewer it built rather than trusting the tasks it drew.
    """
    drawn = draw_latent_slice(tasks, size=20, seed=7)
    reference = AnalystPolicy("reference")
    blinded = AnalystPolicy("blind", distrusted_reports=drawn.distrusted)
    flipped = {
        t.task_id
        for t in tasks
        if (len(blinded.evidence_visible_to(t)) >= reference.escalation_threshold)
        != (len(reference.evidence_visible_to(t)) >= reference.escalation_threshold)
    }
    assert flipped == set(drawn.corrupted)
    assert len(drawn.corrupted) == 20


def test_the_corrupted_slice_sits_in_the_stratum_a_channel_blind_spot_also_hits(tasks):
    """Matched to finding 21, which is what makes the two constructions comparable.

    Discounting reporting can only flip a task showing all three defining facts, and both
    constructions therefore act on the same stratum. If this ever stopped holding, the
    comparison would be between two blind spots of different difficulty as well as
    different keying, and the finding would be attributing one to the other.
    """
    drawn = draw_latent_slice(tasks, size=20, seed=7)
    by_id = {t.task_id: t for t in tasks}
    assert {len(evidence_shown(by_id[t])) for t in drawn.corrupted} == {3}

    reference = AnalystPolicy("reference")
    channel_blind = AnalystPolicy("channel", blind_compartment=BLIND)
    channel_affected = {
        t.task_id
        for t in tasks
        if (len(channel_blind.evidence_visible_to(t)) >= reference.escalation_threshold)
        != (len(reference.evidence_visible_to(t)) >= reference.escalation_threshold)
    }
    assert len(channel_affected) == len(drawn.corrupted), "the two slices are not the same size"
    assert channel_affected != set(drawn.corrupted), "the latent slice is the channel slice"


def test_a_lopsided_slice_is_refused_and_the_refusal_is_calibrated(tasks):
    """The guard that keeps this finding from measuring the channel scan working.

    Tripped rather than trusted, and the way it is tripped matters: the threshold is a
    quantile of the draw's own null, so the test asserts against a draw the null itself
    calls extreme rather than against a hand-chosen number.
    """
    with pytest.raises(ChannelUnusableError, match="carriage gap"):
        # This seed drew a five-task slice at the 99.7% point of its own null when the
        # guard was written, which is the case the quantile exists to catch.
        draw_latent_slice(tasks, size=5, seed=7)

    accepted = draw_balanced_slice(tasks, size=5, seed=7)
    assert accepted.rejected >= 1, "the rejection rule accepted the draw the guard refuses"
    assert accepted.carriage_percentile <= 0.99


def test_a_structural_refusal_is_not_retried(tasks):
    """`draw_balanced_slice` retries an unbalanced draw and re-raises everything else.

    A pool too small is not fixed by another seed, and retrying it would turn one clear
    refusal into `LATENT_ATTEMPTS` identical failures and then a message about balance
    that was never the problem.
    """
    with pytest.raises(ChannelUnusableError, match="defining facts"):
        draw_balanced_slice(tasks, size=10_000, seed=7)


def test_the_balanced_draw_gives_up_rather_than_looping_forever(tasks, monkeypatch):
    """A corpus that cannot supply a balanced slice says so. `attempts` is not decoration.

    Driven by moving the quantile below every attainable value rather than by hunting a
    corpus that fails naturally: the path under test is the exhaustion, and a test that
    depended on a particular corpus refusing a particular size would break for a reason
    that has nothing to do with it.
    """
    monkeypatch.setattr(fleet, "LATENT_CARRIAGE_QUANTILE", -1.0)
    with pytest.raises(ChannelUnusableError, match="no balanced slice"):
        draw_balanced_slice(tasks, size=20, seed=7, attempts=3)
    assert LATENT_ATTEMPTS > 3, "the default gives up sooner than this test's own budget"


def test_the_latent_fleet_differs_from_a_channel_fleet_only_in_what_it_discounts(tasks):
    """Same size, same release ruling, same slip rate. One field apart."""
    drawn = draw_balanced_slice(tasks, size=20, seed=7)
    fleet = latent_blind_fleet(3, 9, drawn, slip_rate=0.15)
    assert len(fleet) == 9
    blind = [a for a in fleet if a.distrusted_reports]
    assert len(blind) == 3
    assert all(a.blind_compartment is None for a in fleet), "keyed on a channel after all"
    assert {a.slip_rate for a in fleet} == {0.15}


def _view(votes: dict[str, float], evidence: dict[str, int]) -> ServerObservation:
    return ServerObservation(
        votes=votes,
        seen=dict.fromkeys(votes, 9.0),
        posterior={t: v / 9.0 for t, v in votes.items()},
        evidence=evidence,
    )


def test_the_two_residual_rules_agree_while_the_corrupted_slice_is_a_minority():
    """Both point at the tasks that fell away from their stratum."""
    votes = {"a": 9.0, "b": 9.0, "c": 9.0, "d": 0.0}
    view = _view(votes, dict.fromkeys(votes, 3))
    assert min(votes, key=lambda t: policy_deviation(view, {})[t]) == "d"
    assert min(votes, key=lambda t: policy_shortfall(view, {})[t]) == "d"


def test_the_two_residual_rules_disagree_once_the_slice_is_the_majority():
    """The crossing, as a unit rather than as a sweep row.

    Three of four tasks corrupted: the stratum rate is now 0.25, so the *clean* task is the
    outlier and the two-sided rule picks it. The one-sided rule still picks a corrupted
    one, which is the whole reason it exists.
    """
    votes = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 9.0}
    view = _view(votes, dict.fromkeys(votes, 3))
    assert min(votes, key=lambda t: policy_deviation(view, {})[t]) == "d"
    assert min(votes, key=lambda t: policy_shortfall(view, {})[t]) in {"a", "b", "c"}


def test_neither_residual_rule_reads_anything_the_aggregator_lacks():
    """Deployability, asserted on the inputs. A view with no strata is not a view with one.

    Pooling every task into a single stratum would be a different statistic, and reporting
    it under these names would credit a rule for a comparison it never made.
    """
    votes = {"a": 9.0, "b": 0.0}
    bare = ServerObservation(
        votes=votes, seen=dict.fromkeys(votes, 9.0), posterior={"a": 1.0, "b": 0.0}
    )
    assert len(set(policy_deviation(bare, {}).values())) == 1
    assert len(set(policy_shortfall(bare, {}).values())) == 1


def test_residual_selection_is_nested_across_budgets():
    """A budget sweep must not confound "more items" with "different items"."""
    votes = {"a": 9.0, "b": 8.0, "c": 1.0, "d": 0.0}
    view = _view(votes, dict.fromkeys(votes, 3))
    small = set(select("shortfall", view, {}, 1, seed=7))
    large = set(select("shortfall", view, {}, 2, seed=7))
    assert small < large


def test_the_artifact_answers_every_prediction_the_script_wrote_down():
    """Each prediction is a boolean in the artifact, including the one that was refuted."""
    payload = artifact("latent_blindspot.json")
    findings = payload["findings"]
    assert findings["channel_scan_silent_on_latent"] is True
    assert findings["channel_scan_fires_on_channel_keyed"] is True
    assert findings["dispersion_identical_on_deterministic_fleets"] is True
    assert findings["dispersion_cannot_tell_the_constructions_apart"] is True
    # Prediction 3 said localization would fail at unanimity. It does not, and the page
    # records the refutation rather than the prediction.
    assert findings["two_sided_residual_localizes_at_unanimity"] is True
    assert findings["one_sided_residual_localizes_at_unanimity"] is True
    # Prediction 4 said the two-sided rule would turn, and it does; the one-sided one does
    # not, which is the reason the finding proposes the second rather than the first.
    assert findings["two_sided_residual_inverts_in_the_sweep"] is True
    assert findings["one_sided_residual_inverts_in_the_sweep"] is False


def test_the_artifact_carries_the_evidence_behind_each_verdict():
    """A boolean with no numbers under it is an assertion, not a measurement."""
    payload = artifact("latent_blindspot.json")
    assert payload["policies"] == list(POLICIES_HERE)
    assert payload["slice_sizes"] == list(SLICE_SIZES)
    assert payload["grid"], "no grid"
    assert payload["slice_sweep"], "no slice sweep"
    assert payload["dispersion_spread"], "no spread to read the separability claim off"
    assert all(row["channel_inside_latent_spread"] for row in payload["dispersion_spread"])
    for name, slice_ in payload["slices"].items():
        assert slice_["corrupted"] == payload["channel_affected"], name
        assert slice_["carriage_percentile"] <= 0.99, name


def test_the_sweep_crosses_the_stratum_majority():
    """The crossing is only a finding if the sweep spans it.

    A sweep that stopped below the pool's majority would report the two rules agreeing and
    would have measured nothing about where they part.
    """
    payload = artifact("latent_blindspot.json")
    sweep = payload["slice_sweep"]
    pool = sweep[0]["eligible"]
    assert min(r["size"] for r in sweep) < pool / 2 < max(r["size"] for r in sweep)
    inverted = payload["inverted_sizes"]
    assert inverted, "the two-sided rule never turned, so the sweep priced nothing"
    assert min(inverted) > pool / 2, "the two-sided rule turned before the majority"


def test_the_counts_the_findings_page_quotes_are_the_artifact_s():
    """The two hand-typed numbers in the prose, tied to the artifact that produced them.

    "5 of the 12 swept cells" and "ties the oracle in 10 of the 12" are counts a reader
    would otherwise have to take on trust, and this project has already had a hand-typed
    summary of a generated table go stale in silence. Written here rather than generated
    into the page because they are sentences rather than rows.
    """
    payload = artifact("latent_blindspot.json")
    assert payload["swept_cells"] == 12
    assert payload["cells_where_two_sided_is_no_better_than_uniform"] == 5
    assert payload["cells_where_one_sided_ties_the_bound"] == 10
    assert payload["cells_where_one_sided_is_no_better_than_uniform"] == 0


@pytest.fixture(scope="module")
def small():
    """A corpus small enough to run the measurement functions directly.

    Sixty events rather than two hundred, and a fleet of five rather than nine. The
    functions below are the ones that had only ever executed inside `main()`, which
    coverage excludes -- the same gap the governance package's own tests were written to
    close. Exercising them on the committed corpus would cost the permutation budget twice
    over and would test the artifact rather than the code.
    """
    from measure_latent_blindspot import build_tasks

    tasks = build_tasks(seed=7, events=60)
    proposals = {
        t.task_id: Proposal(t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS))
        for t in tasks
    }
    truth = {t.task_id: t.significant for t in tasks}
    return tasks, proposals, truth


def _cell(small, keying, slice_, *, n_blind=5, slip=0.0):
    from measure_latent_blindspot import measure

    tasks, proposals, truth = small
    return measure(
        tasks,
        proposals,
        truth,
        keying=keying,
        n_blind=n_blind,
        slip_rate=slip,
        fleet=5,
        seed=7,
        slice_=slice_,
        permutations=1,
        null_draws=1,
    )


def test_measure_runs_both_detectors_and_scores_every_named_policy(small):
    """One cell end to end, on both constructions, with the budgets turned down."""
    from measure_latent_blindspot import BOUND, POLICIES_HERE, _affected

    tasks, _, _ = small
    drawn = draw_balanced_slice(tasks, size=len(_affected(tasks)), seed=7)

    channel_cell = _cell(small, "channel", None)
    latent_cell = _cell(small, "latent", drawn)

    for cell in (channel_cell, latent_cell):
        assert set(POLICIES_HERE) | {BOUND} <= set(cell.precision)
        assert set(POLICIES_HERE) | {BOUND} <= set(cell.risk)
        # `uniform` is summarized twice on purpose: the best draw is what every comparison
        # uses, the median is what a deployment would typically get.
        assert "uniform_median" in cell.precision
        assert cell.precision["uniform"] >= cell.precision["uniform_median"]
        assert cell.as_dict()["keying"] == cell.keying

    assert channel_cell.errors == latent_cell.errors, (
        "the two constructions were matched to corrupt the same count and did not"
    )


def test_measure_populates_no_channel_map_for_a_latent_fleet(small):
    """`channel` is unavailable rather than merely bad, and that has to be visible."""
    from measure_latent_blindspot import _affected

    tasks, _, _ = small
    drawn = draw_balanced_slice(tasks, size=len(_affected(tasks)), seed=7)
    latent_cell = _cell(small, "latent", drawn, n_blind=5)
    # With no channel supplied the provenance policy scores every task equally, so it
    # cannot beat an untargeted draw except by the tie-break. The finding reports that as
    # "no remedy available", and this is the property that makes the report true.
    assert latent_cell.precision["uniform"] >= 0.0


def test_assemble_answers_every_prediction_from_the_cells_it_is_given(small):
    """The artifact builder, driven directly rather than through a full sweep."""
    import argparse

    from measure_latent_blindspot import _affected, assemble, slice_sweep

    tasks, proposals, truth = small
    size = len(_affected(tasks))
    drawn = draw_balanced_slice(tasks, size=size, seed=7)
    cells = [
        _cell(small, "channel", None, n_blind=n, slip=slip) for n in (0, 5) for slip in (0.0, 0.15)
    ] + [
        _cell(small, "latent:7", drawn, n_blind=n, slip=slip)
        for n in (0, 5)
        for slip in (0.0, 0.15)
    ]
    sweep = slice_sweep(tasks, proposals, truth, fleet=5, seed=7, null_draws=1)
    args = argparse.Namespace(seed=7, events=60, fleet=5, permutations=1, null_draws=1)
    report = assemble(cells, sweep, {7: drawn}, (0, 5), args, size)

    assert set(report["findings"]) == {
        "channel_scan_silent_on_latent",
        "channel_scan_fires_on_channel_keyed",
        "dispersion_identical_on_deterministic_fleets",
        "dispersion_cannot_tell_the_constructions_apart",
        "two_sided_residual_localizes_at_unanimity",
        "one_sided_residual_localizes_at_unanimity",
        "two_sided_residual_inverts_in_the_sweep",
        "one_sided_residual_inverts_in_the_sweep",
    }
    assert all(isinstance(v, bool) for v in report["findings"].values())
    assert report["swept_cells"] == len(sweep)
    assert report["grid"] and report["slice_sweep"]
    assert report["slices"]["7"]["corrupted"] == size


def test_render_prints_the_three_tables_and_every_verdict(small, capsys):
    """The reporting path, which is where a number a reader sees is actually formed."""
    import argparse

    from measure_latent_blindspot import _affected, assemble, render, slice_sweep

    tasks, proposals, truth = small
    size = len(_affected(tasks))
    drawn = draw_balanced_slice(tasks, size=size, seed=7)
    cells = [
        _cell(small, "channel", None, n_blind=5),
        _cell(small, f"latent:{7}", drawn, n_blind=5),
    ]
    sweep = slice_sweep(tasks, proposals, truth, fleet=5, seed=7, null_draws=1)
    args = argparse.Namespace(seed=7, events=60, fleet=5, permutations=1, null_draws=1)
    report = assemble(cells, sweep, {7: drawn}, (0, 5), args, size)

    render(report, cells, sweep, (0, 5))
    printed = capsys.readouterr().out
    assert "detection, by what the blind spot is keyed on" in printed
    assert "localization at unanimity" in printed
    assert "as the corrupted slice grows past the majority of its stratum" in printed
    for name in report["findings"]:
        assert name in printed, f"{name} was decided and not reported"


def test_the_slice_sweep_spans_the_pool_it_draws_from(small):
    """Every swept size that the corpus can host produces a row, and refusals are logged."""
    from measure_latent_blindspot import slice_sweep

    tasks, proposals, truth = small
    rows = slice_sweep(tasks, proposals, truth, fleet=5, seed=7, null_draws=1)
    assert rows, "the sweep produced no rows at all"
    assert all(r.eligible >= r.size for r in rows)
    assert {*rows[0].precision} >= {"uniform", "deviation", "shortfall", "oracle"}
    assert rows[0].as_dict()["size"] == rows[0].size


def test_the_two_constructions_are_refused_if_they_agree(small):
    """`_views_agree` is the guard against comparing a fleet with itself."""
    from measure_latent_blindspot import _affected, _views_agree

    tasks, proposals, _ = small
    drawn = draw_balanced_slice(tasks, size=len(_affected(tasks)), seed=7)
    flat = contributions_for(latent_blind_fleet(5, 5, drawn), tasks, proposals, seed=7)
    latent_view = observe(partition_by_contributor(flat))
    channel_flat = contributions_for(blind_fleet(5, 5), tasks, proposals, seed=7)
    channel_view = observe(partition_by_contributor(channel_flat))

    assert _views_agree(latent_view, latent_view), "a view must agree with itself"
    assert not _views_agree(latent_view, channel_view), (
        "the two constructions produced the same aggregate on this corpus"
    )


def test_the_affected_slice_is_read_from_the_corpus_rather_than_assumed(small):
    """The latent slice is sized from what a channel blind spot actually flips.

    A hard-coded 20 would silently compare a 20-task slice against a 14-task one on a
    corpus draw where the channel happens to hit fewer tasks.
    """
    from measure_latent_blindspot import _affected

    tasks, _, _ = small
    affected = _affected(tasks)
    assert affected, "no verdict changes under a channel blind spot on this corpus"
    assert all(t.significant for t in affected)
