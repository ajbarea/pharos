"""Guards for the grid aggregate behind finding 10.

The manuscript quotes five numbers from `results/teacher_fleet.json` that no other
artifact carries: the residual against the denoised ceiling, how many adapters beat
their teacher's own self-agreement, the by-threshold split that changes sign, and the
adapters with high fidelity that the validity check refuses. Each was computed once in
a shell one-liner while the section was being written, which is exactly how a paper
number stops being reproducible.

These tests pin the arithmetic, not the values: the artifact regenerates from committed
adapter results, so a claim can be re-derived, but only if the derivation is right.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import measure_teacher_fleet

    return measure_teacher_fleet


def _row(threshold: int, slip: float, fidelity: float, teacher: float, adapter: float, **kw):
    return {
        "reviewer": f"t{threshold}s{slip:g}",
        "threshold": threshold,
        "slip_rate": slip,
        "adapter_agrees_with_teacher": fidelity,
        "teacher_agrees_with_world": teacher,
        "adapter_agrees_with_world": adapter,
        "quotable": kw.get("quotable", True),
        "adapter_recall": kw.get("recall", 1.0),
        "adapter_unparsed": 0,
    }


def test_a_perfect_adapter_sits_exactly_on_the_ceiling():
    """The bound the fidelity column has to be read against.

    An adapter that has learned the rule perfectly still disagrees with a teacher
    slipping at rate `s` exactly `s` of the time, because the teacher's labels do. So
    fidelity of `1 - s` is a *perfect* score, not a degraded one, and the residual it
    produces must be zero at every slip rate.
    """
    grid = [_row(3, s, 1 - s, 0.9, 0.9) for s in (0.0, 0.05, 0.2, 0.5)]
    ceiling = _module()._ceiling_analysis(grid)["residual_vs_denoised_ceiling"]
    assert ceiling["median"] == 0.0
    assert ceiling["largest_shortfall"] == 0.0


def test_the_shortfall_is_reported_as_a_positive_distance():
    """`largest_shortfall` answers "how far below", so a sign error would invert it."""
    grid = [_row(3, 0.0, 0.94, 0.9, 0.9), _row(3, 0.0, 0.99, 0.9, 0.9)]
    ceiling = _module()._ceiling_analysis(grid)["residual_vs_denoised_ceiling"]
    assert ceiling["min"] == -0.06
    assert ceiling["largest_shortfall"] == 0.06


def test_exceeding_the_teachers_self_agreement_is_counted_against_the_right_bound():
    """Two draws of a teacher slipping at `s` agree `(1-s)^2 + s^2` of the time.

    At `s = 0.5` that is `0.5`, so a coin-flip teacher agrees with itself half the time
    and the bar is low; at `s = 0` it is `1.0` and cannot be beaten. Comparing against
    `1 - s` instead would make the first trivially true and the second impossible.
    """
    module = _module()
    # s=0.5: retest bound 0.50. Fidelity 0.55 clears it.
    assert (
        module._ceiling_analysis([_row(3, 0.5, 0.55, 0.5, 0.5)])["vs_teacher_self_agreement"][
            "n_exceeding"
        ]
        == 1
    )
    # s=0: retest bound 1.00. Nothing can exceed it, including a perfect adapter.
    assert (
        module._ceiling_analysis([_row(3, 0.0, 1.0, 0.9, 0.9)])["vs_teacher_self_agreement"][
            "n_exceeding"
        ]
        == 0
    )


def test_the_threshold_split_preserves_a_sign_change_the_fleet_median_hides():
    """The reason this split exists rather than a single fleet-wide number.

    One threshold made worse and another improved can average to a positive median, and
    reporting only that would state the opposite of what happened to half the fleet.
    """
    grid = [_row(1, 0.1, 0.9, 0.50, 0.44) for _ in range(4)]  # -0.06 each
    grid += [_row(3, 0.1, 0.9, 0.50, 0.60) for _ in range(4)]  # +0.10 each
    by_threshold = {r["threshold"]: r for r in _module()._by_threshold(grid)}

    assert by_threshold[1]["median_gap"] < 0 < by_threshold[3]["median_gap"]
    assert by_threshold[1]["adapters_better_than_their_teacher"] == 0
    assert by_threshold[3]["adapters_better_than_their_teacher"] == 4


def test_high_fidelity_but_refused_finds_the_governance_case():
    """The claim: a model can reproduce its teacher perfectly and still be unusable."""
    grid = [
        _row(1, 0.0, 1.00, 0.45, 0.47, quotable=False),  # the case
        _row(3, 0.0, 1.00, 0.99, 0.99, quotable=True),  # faithful and fine
        _row(2, 0.5, 0.50, 0.50, 0.33, quotable=False),  # refused, but no fidelity
    ]
    refused = _module()._fidelity_without_usefulness(grid)

    assert [r["reviewer"] for r in refused] == ["t1s0"], (
        "must select on refused AND high fidelity; either alone is a different claim"
    )


def test_high_fidelity_but_refused_is_empty_when_no_such_adapter_exists():
    """An empty answer weakens the claim, so it has to be derivable rather than assumed."""
    grid = [_row(3, 0.0, 1.0, 0.99, 0.99, quotable=True)]
    assert _module()._fidelity_without_usefulness(grid) == []


def test_the_committed_artifact_still_supports_what_the_manuscript_says():
    """The five quoted numbers, checked against the artifact the paper harvests.

    Directional rather than exact: the point is that the artifact still says what the
    prose says, not that a re-measurement can never move a digit.
    """
    import json

    path = ROOT / "results" / "teacher_fleet.json"
    if not path.is_file():
        import pytest

        pytest.skip("no teacher_fleet.json; run `make teacher-fleet`")
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]

    assert summary["dissociation"]["systematic_only"]["median_gap"] < 0.01, (
        "a never-slipping teacher's error must be inherited nearly intact"
    )
    assert summary["dissociation"]["with_random_error"]["median_gap"] > 0.01, (
        "a slipping teacher must be improved upon"
    )
    assert summary["ceiling"]["residual_vs_denoised_ceiling"]["largest_shortfall"] < 0.1, (
        "no adapter should fall far below the ceiling; the prose says never more than 0.078"
    )
    exceeding = summary["ceiling"]["vs_teacher_self_agreement"]
    assert exceeding["n_exceeding"] > exceeding["n"] // 2, (
        "most adapters must beat their teacher's self-agreement; the prose says 20 of 24"
    )
    gaps = [r["median_gap"] for r in summary["by_threshold"]]
    assert min(gaps) < 0 < max(gaps), (
        "the by-threshold split must still change sign; that is the whole paragraph"
    )
    assert summary["high_fidelity_but_refused"], (
        "the governance claim needs at least one high-fidelity refused adapter"
    )
