"""The argument the explorer serves, and the guard that it cannot be typed in."""

import json
import re
from pathlib import Path

import pytest
from conftest import artifact

from pharos import escapes
from pharos.escapes import ESCAPES, RESULTS, TENSION, VERDICTS, Reading, _dig, argument

EVERY = (*TENSION, *ESCAPES)


def test_every_reading_resolves_against_a_committed_artifact():
    """The whole point of the module: no number in the explorer is hand-written.

    A reading that stops resolving degrades to "unmeasured" at runtime rather than
    raising, which is right for a page and wrong for a build. This is where it becomes
    an error, so a renamed artifact field fails here instead of quietly blanking a
    number on a page nobody is watching.
    """
    unresolved = [
        f"{step.key}: {reading.artifact}:{reading.path}"
        for step in EVERY
        for reading in step.readings
        if reading.resolve()["value"] is None
    ]
    assert not unresolved, f"readings that no longer resolve: {unresolved}"


def test_every_named_artifact_exists():
    named = {reading.artifact for step in EVERY for reading in step.readings}
    assert named, "no artifact is named anywhere; the argument would carry no evidence"
    missing = [name for name in named if not (RESULTS / name).exists()]
    assert not missing, f"artifacts named but absent: {missing}"


def test_every_reading_matches_its_artifact_exactly():
    """The drift guard, and the reason this module is allowed to exist at all.

    These numbers already appear in `docs/findings.md` and in two manuscripts. A third
    rendering is only safe while it is derived, so this re-reads each artifact
    independently and compares against what the module served.
    """
    for step in EVERY:
        for reading in step.readings:
            resolved = reading.resolve()
            payload = json.loads((RESULTS / reading.artifact).read_text(encoding="utf-8"))
            assert resolved["raw"] == _dig(payload, reading.path)
            assert resolved["value"] == reading.fmt.format(resolved["raw"])


def test_a_selector_names_its_row_rather_than_counting_to_it():
    """Positional indexing into an artifact list is how a page shows a wrong number.

    `by_clearance_level.3` reads whatever is fourth. Reorder the artifact and the page
    reports a different clearance level's recovery with no sign anything moved. The
    selector form either finds the row it names or fails.
    """
    rows = [{"level": "OPEN", "v": 1}, {"level": "RESTRICTED", "v": 2}]
    assert _dig({"rows": rows}, "rows.level=RESTRICTED.v") == 2
    assert _dig({"rows": rows}, "rows.level=NOPE.v") is None
    assert _dig({"rows": list(reversed(rows))}, "rows.level=RESTRICTED.v") == 2, (
        "a selector must survive a reordering that a positional index would not"
    )


def test_a_missing_artifact_marks_an_exit_unmeasured_rather_than_dropping_it():
    """A list that shortens when evidence goes missing flatters the argument."""
    step = escapes.Escape(
        key="probe",
        objection="?",
        findings=(1,),
        verdict="open",
        outcome="?",
        readings=(Reading("x", "no_such_artifact.json", "a.b"),),
    )
    payload = step.payload()
    assert payload["measured"] is False
    assert payload["readings"][0]["value"] is None
    assert payload["key"] == "probe", "the exit must still be present"


def test_the_unmeasured_list_covers_the_tension_and_not_only_the_exits():
    """It did not, when first written, and reported [] while two readings were broken.

    The tension is three of the fourteen rows the page renders. A completeness field
    that inspects eleven of them is worse than none, because it answers the question
    confidently and wrongly.
    """
    keys = {step.key for step in EVERY}
    served = argument()
    rendered = {row["key"] for row in (*served["tension"], *served["escapes"])}
    assert rendered == keys
    assert not served["unmeasured"], f"unmeasured exits: {served['unmeasured']}"


@pytest.mark.parametrize("step", EVERY, ids=lambda s: s.key)
def test_each_exit_is_well_formed(step):
    assert step.verdict in VERDICTS
    assert step.findings, f"{step.key} cites no finding"
    assert step.objection.endswith((".", "?")), "an objection reads as a sentence"
    # Rendered as HTML in a browser, where `--` is two hyphens rather than a dash.
    assert " -- " not in step.outcome and " -- " not in step.objection


def test_every_finding_cited_exists_on_the_findings_page():
    """A citation to a finding that was renumbered or withdrawn is a dead reference."""
    # `^## (\d+)\.` and not every `## ` heading: the page carries unnumbered sections
    # too ("Status: provisional", "Measurement health"). Same rule the README count
    # guard uses in test_scripts.py, deliberately, because two ways of deciding what
    # counts as a finding is one way too many.
    text = (RESULTS.parent / "docs" / "findings.md").read_text(encoding="utf-8")
    headings = {int(n) for n in re.findall(r"^## (\d+)\.", text, re.MULTILINE)}
    assert headings, "no numbered findings parsed; the guard would pass vacuously"
    cited = {n for step in EVERY for n in step.findings}
    assert cited <= headings, f"cited findings absent from the page: {sorted(cited - headings)}"


def test_the_counts_add_up_to_the_exits_served():
    served = argument()
    assert sum(served["counts"].values()) == len(served["escapes"])
    assert set(served["counts"]) == set(VERDICTS)


def test_the_freeze_carries_the_same_argument_the_server_does(tmp_path, monkeypatch):
    """The offline page is a client of this module, not a second rendering of it.

    Built into `tmp_path` rather than read from `docs/explorer/`, because that
    directory is gitignored and generated by the docs workflow. Asserting against the
    copy on disk would pass locally off a stale build and fail in CI off no build at
    all, which is a guard that reports on the wrong thing in both environments.

    What has to hold is that the *builder* carries the argument. If it stops, the
    published page silently loses a tab while every other check stays green.
    """
    import sys

    import build_static_explorer as bse

    out = tmp_path / "explorer"
    monkeypatch.setattr(sys, "argv", ["build_static_explorer.py", "--out", str(out)])
    assert bse.main() == 0

    frozen = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert "escapes" in frozen, "the freeze dropped the escapes tab's data"
    assert frozen["escapes"] == argument()

    # And the page must actually route the offline call, or the bundle ships beside a
    # tab that calls a dead endpoint and renders empty.
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "'/api/escapes'" in page
    assert "BUNDLE.escapes" in page


def test_the_governance_numbers_the_page_shows_are_the_ones_finding_24_publishes():
    """Cross-checked against the artifact the findings page reads, not against itself."""
    served = {row["key"]: row for row in argument()["escapes"]}["fleetsize"]
    bracket = artifact("governance_sensitivity")["cliff_bracket"]
    values = {reading["label"]: reading["raw"] for reading in served["readings"]}
    assert values["Breaking share, median"] == bracket["breaking_share_median"]
    assert values["Breaking share, lowest"] == bracket["breaking_share_range"][0]
    assert values["Breaking share, highest"] == bracket["breaking_share_range"][1]


def _relative_luminance(hex_colour: str) -> float:
    """WCAG 2.2 relative luminance, from the sRGB definition."""
    raw = hex_colour.lstrip("#")
    channels = []
    for offset in (0, 2, 4):
        value = int(raw[offset : offset + 2], 16) / 255
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    first, second = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def test_every_verdict_colour_is_readable_in_both_themes():
    """WCAG 2.2 AA, 4.5:1, against both surfaces the badges actually sit on.

    The verdict badges render their label in the token colour, so this is text contrast
    and not the 3:1 graphical threshold. Light-theme amber was 4.24:1 when the tab was
    written, which reads fine to me and fails for a reader it was not written for.

    Colour is never the only channel here -- each badge also carries a glyph and the
    verdict spelled out, so this passing is not what makes the tab accessible. It is
    what stops the redundant channel from being the only usable one.
    """
    page = (Path(__file__).resolve().parents[1] / "src/pharos/static/index.html").read_text(
        encoding="utf-8"
    )
    themes = {
        "light": dict(
            re.findall(r"--(\w+): (#[0-9a-f]{6})", page.split("prefers-color-scheme")[0])
        ),
        "dark": dict(re.findall(r"--(\w+): (#[0-9a-f]{6})", page.split("prefers-color-scheme")[1])),
    }
    failures = []
    for name, tokens in themes.items():
        for token in ("ok", "amber", "bad", "accent", "muted", "fg"):
            assert token in tokens, f"{name} theme no longer defines --{token}"
            for surface in ("card", "bg"):
                ratio = _contrast(tokens[token], tokens[surface])
                if ratio < 4.5:
                    failures.append(f"{name}/--{token} on --{surface} = {ratio:.2f}")
    assert not failures, f"below WCAG AA 4.5:1: {failures}"
