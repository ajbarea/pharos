"""The mechanical half of the documentation, checked.

Most of what the docs say is prose and stays prose. Two kinds of claim are not: a
`make <target>` a reader is told to run, and a repository path a reader is told to look
at. Both are exact, both are cheap to verify, and both rot silently -- a renamed target
leaves a page telling somebody to run a command that does not exist, and the page still
builds, still deploys, and still reads fine.

This is deliberately narrow. It does not check prose, claims about behaviour, or numbers;
`sync_docs_tables.py` covers the tables that restate an artifact, and nothing covers the
rest. What it does cover, it covers exhaustively, and the controls below prove the
extraction can fail rather than merely pass.
"""

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Prefixes a backticked string must start with to be read as a repository path. Without
#: this the extractor treats every backticked token as a path and drowns in false
#: positives -- and a check that cries wolf gets deleted, which is a slower way of not
#: having one.
PATH_PREFIXES = ("scripts/", "src/", "tests/", "cluster/", "scenarios/", "results/", "docs/")

#: Paths that name a file in a *sibling* repository rather than this one. Each is
#: exempted with the repository it lives in, because an exemption is a claim too: the
#: alternative is a checker that silently skips anything it cannot find, which is the
#: same as no checker.
SIBLING_PATHS = {
    "docs/research/federated-forge/pharos-testbed.md": "kourai-khryseai",
    "docs/research/federated-forge/future-work.md": "kourai-khryseai",
    "docs/research/federated-forge/index.md": "kourai-khryseai",
}


def _docs() -> list[Path]:
    """Every hand-written page. `docs/explorer/` is generated and is not one."""
    pages = [ROOT / "README.md", ROOT / "RESEARCH.md", *(ROOT / "docs").rglob("*.md")]
    return [p for p in pages if p.exists() and "explorer" not in p.parts]


def _make_targets() -> set[str]:
    return set(re.findall(r"^([a-z][a-z0-9-]*):", (ROOT / "Makefile").read_text(), re.MULTILINE))


def make_targets_named_in(text: str) -> set[str]:
    """Targets a page tells a reader to run, from backticks and from fenced blocks.

    Both forms, because the docs use both and an extractor that read only one would pass
    while half the claims went unchecked. Bare prose is deliberately not matched: "make
    the corpus smaller" is English, and treating it as a target is how this check would
    become noise.
    """
    inline: list[str] = re.findall(r"`make ([a-z][a-z0-9-]*)[^`]*`", text)
    fenced: list[str] = re.findall(r"^\s*make ([a-z][a-z0-9-]*)", text, re.MULTILINE)
    return {*inline, *fenced}


def paths_named_in(text: str) -> set[str]:
    """Repository paths a page points at, from backticked strings only."""
    return {
        path
        for path in re.findall(r"`([\w./*-]+)`", text)
        if path.startswith(PATH_PREFIXES) and not path.endswith("/")
    }


def headings_in(text: str) -> set[str]:
    """The anchors a page defines, slugged the way the docs site and GitHub both slug them.

    Lowercase, punctuation dropped, whitespace to hyphens. Emphasis markers go with the
    punctuation, which matters because several headings here carry a quoted phrase.
    """
    return {
        re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", heading.lower()).strip())
        for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, re.MULTILINE)
    }


def anchor_links_in(text: str) -> set[tuple[str, str]]:
    """`(page, anchor)` for every in-repository link that names a fragment.

    `page` is empty for a same-page link. External links are excluded by requiring the
    target to be a fragment or a relative Markdown file, since an anchor on somebody
    else's site is not this repository's claim to check.
    """
    found: set[tuple[str, str]] = set()
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:")) or "#" not in target:
            continue
        page, _, anchor = target.partition("#")
        if page and not page.endswith(".md"):
            continue
        found.add((page, anchor))
    return found


@pytest.mark.parametrize("page", _docs(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_anchor_the_docs_link_to_exists(page):
    """A retitled heading leaves every link to it pointing at nothing, and the page still
    builds, still deploys, and still reads fine.

    This is the same rot the two checks below catch in commands and paths, in the one
    remaining mechanical claim these pages make. It found four dead links on the day it
    was written -- findings 16, 17, 20 and 21 had each been retitled and the eight links
    into them were never chased -- which is the failure mode rather than a hypothesis
    about one.
    """
    dangling: list[str] = []
    for target, anchor in sorted(anchor_links_in(page.read_text())):
        other = (page.parent / target).resolve() if target else page
        if not other.exists():
            dangling.append(f"{target}#{anchor} (no such page)")
        elif anchor not in headings_in(other.read_text()):
            dangling.append(f"{target}#{anchor}")
    assert not dangling, f"{page.relative_to(ROOT)} links to headings that do not exist: {dangling}"


@pytest.mark.parametrize("page", _docs(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_make_target_the_docs_name_exists(page):
    targets = _make_targets()
    named = make_targets_named_in(page.read_text(encoding="utf-8"))
    missing = sorted(named - targets)
    assert not missing, (
        f"{page.relative_to(ROOT)} tells a reader to run {missing}, and the Makefile has "
        "no such target"
    )


@pytest.mark.parametrize("page", _docs(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_repository_path_the_docs_name_exists(page):
    missing = []
    for path in sorted(paths_named_in(page.read_text(encoding="utf-8"))):
        if path in SIBLING_PATHS:
            continue
        if "*" in path:
            if not list(ROOT.glob(path)):
                missing.append(path)
        elif not (ROOT / path).exists():
            missing.append(path)
    assert not missing, f"{page.relative_to(ROOT)} points at paths that do not exist: {missing}"


def siblings_are_required() -> bool:
    """`PHAROS_REQUIRE_SIBLINGS` read as a flag rather than as a string.

    `os.environ.get(name)` alone makes `0` and `false` mean on, because both are non-empty
    strings. A documented switch that cannot be switched off the obvious way is a switch
    nobody can trust.
    """
    return os.environ.get("PHAROS_REQUIRE_SIBLINGS", "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }


def test_a_sibling_exemption_names_a_repository_that_has_the_file():
    """An exemption that stopped being true is a path nobody checks in either repository.

    Skipped rather than failed when the sibling is not checked out: the main CI job runs
    with only this repository present, and a check that fails on a missing neighbour would
    fail for a reason that has nothing to do with the claim.

    Except where the neighbour was checked out on purpose. `sibling-links.yml` clones it
    two steps before running this, so there a skip means that clone silently produced
    nothing, and the exemption goes unverified in the one place built to verify it.
    `PHAROS_REQUIRE_SIBLINGS` says which situation this is.

    Absent repositories are collected rather than acted on where they are found. Skipping
    inside the loop ends the whole test at the first one, so a second sibling that is not
    checked out would take the first sibling's paths down with it, unasserted -- the
    silent-skip failure this exemption exists to prevent, one repository over.
    """
    absent: list[str] = []
    for path, repo in SIBLING_PATHS.items():
        sibling = ROOT.parent / repo
        if not sibling.is_dir():
            absent.append(repo)
            continue
        assert (sibling / path).exists(), (
            f"{path} is exempted as living in {repo}, and it does not exist there either"
        )

    if not absent:
        return
    named = ", ".join(sorted(set(absent)))
    if siblings_are_required():
        pytest.fail(
            f"{named} was required beside this repository and is not there, so the "
            "exemption was not checked. The checkout produced nothing."
        )
    pytest.skip(f"{named} is not checked out beside this repository")


@pytest.mark.parametrize(
    ("value", "required"),
    [(None, False), ("", False), ("0", False), ("false", False), ("1", True), ("true", True)],
)
def test_the_sibling_requirement_reads_its_variable_as_a_flag(monkeypatch, value, required):
    """`0` has to mean off. It is the first thing anyone tries when turning a switch off."""
    if value is None:
        monkeypatch.delenv("PHAROS_REQUIRE_SIBLINGS", raising=False)
    else:
        monkeypatch.setenv("PHAROS_REQUIRE_SIBLINGS", value)
    assert siblings_are_required() is required


@pytest.mark.parametrize(
    ("sibling_present", "required", "outcome"),
    [(True, True, "passes"), (False, False, "skips"), (False, True, "fails")],
)
def test_the_sibling_guard_reports_what_its_workflow_depends_on(
    monkeypatch, tmp_path, sibling_present, required, outcome
):
    """The three states `sibling-links.yml` rests on, asserted rather than described.

    The scheduled workflow is the only place that sets `PHAROS_REQUIRE_SIBLINGS`, so the
    line turning a skip into a failure never executes in a local or per-PR run. Without
    this, a refactor that removed it -- reordering the guard below the skip, renaming the
    variable, hoisting the directory check out of the loop -- would leave every run here
    green while the scheduled job went back to skipping under a green tick, which is the
    defect that workflow exists to close.
    """
    root = tmp_path / "pharos"
    root.mkdir()
    if sibling_present:
        for path, repo in SIBLING_PATHS.items():
            target = tmp_path / repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", root)
    monkeypatch.setenv("PHAROS_REQUIRE_SIBLINGS", "1" if required else "0")

    if outcome == "passes":
        test_a_sibling_exemption_names_a_repository_that_has_the_file()
        return

    # Both outcomes are caught and the type asserted afterwards, rather than naming only
    # the expected one. A `Skipped` raised inside `pytest.raises(Failed)` is not caught,
    # so it propagates and marks *this* test skipped -- green, and reporting nothing. A
    # guard that fails to fail would have been invisible here, which is the failure this
    # test is about.
    expected = pytest.fail.Exception if outcome == "fails" else pytest.skip.Exception
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as excinfo:
        test_a_sibling_exemption_names_a_repository_that_has_the_file()
    assert type(excinfo.value) is expected, (
        f"expected the guard to {outcome[:-1]}, and it raised {type(excinfo.value).__name__}"
    )


def test_an_absent_sibling_does_not_take_another_repositorys_paths_with_it(monkeypatch, tmp_path):
    """The skip used to end the test where it was raised, which is one repository too early.

    All three exemptions name one neighbour today, so this is the shape of a second one
    being added rather than a live defect. The check that matters is that the repository
    which *is* checked out still gets asserted: here its exempted file is missing, so a
    loop that stopped at the absent neighbour would report a skip and never notice.
    """
    root = tmp_path / "pharos"
    root.mkdir()
    (tmp_path / "present-repo" / "docs").mkdir(parents=True)
    monkeypatch.setattr(sys.modules[__name__], "ROOT", root)
    monkeypatch.setattr(
        sys.modules[__name__],
        "SIBLING_PATHS",
        # Absent first. With the present repository first, the assertion on its missing
        # file fires on iteration one and the absent one is never reached -- so the test
        # passed identically with and without the in-loop skip it exists to forbid.
        {"docs/gone.md": "absent-repo", "docs/kept.md": "present-repo"},
    )
    monkeypatch.setenv("PHAROS_REQUIRE_SIBLINGS", "1")

    # Skip and fail are caught here for the same reason as above: a loop that stops at the
    # absent neighbour raises one of them, and an uncaught `Skipped` would mark this test
    # skipped rather than red.
    with pytest.raises((AssertionError, pytest.fail.Exception, pytest.skip.Exception)) as excinfo:
        test_a_sibling_exemption_names_a_repository_that_has_the_file()
    assert type(excinfo.value) is AssertionError, (
        "the present repository's exemption went unasserted because the absent one ended "
        f"the loop first, raising {type(excinfo.value).__name__}"
    )
    assert "docs/kept.md is exempted as living in" in str(excinfo.value)


#: Enough of the number words to name a finding count, which is what this repository
#: writes them for. Digits are accepted too; the README happens to spell it out.
_UNITS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def spelled_number(word: str) -> int | None:
    """`thirty-one` as 31. None when the word is not a number this understands."""
    word = word.strip().lower()
    if word.isdigit():
        return int(word)
    if word in _UNITS:
        return _UNITS[word]
    if word in _TENS:
        return _TENS[word]
    tens, _, unit = word.partition("-")
    if tens in _TENS and unit in _UNITS:
        return _TENS[tens] + _UNITS[unit]
    return None


def test_the_readme_counts_the_findings_the_findings_page_has():
    """A count in prose is a second copy of something the page already knows.

    The rest of this repository refuses to write one down twice, and the generated blocks
    exist so a table cannot disagree with its artifact. This one number sits in `README.md`
    with nothing recounting it, and it moves every time a finding lands -- which is exactly
    when nobody is looking at the sentence above the link.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    claimed = re.search(r"^([A-Za-z-]+|\d+) findings so far", readme, re.MULTILINE)
    assert claimed, (
        "README no longer says how many findings there are in the form this checks. "
        "Update the pattern here, or drop the count and this test with it."
    )
    count = spelled_number(claimed.group(1))
    assert count is not None, f"README says {claimed.group(1)!r} findings, which is not a number"

    numbered = [
        int(n)
        for n in re.findall(
            r"^## (\d+)\. ", (ROOT / "docs" / "findings.md").read_text("utf-8"), re.MULTILINE
        )
    ]
    assert count == len(numbered), (
        f"README says {count} findings and findings.md has {len(numbered)}. The count is "
        "written in two places, so one of them is now wrong."
    )
    # A removed finding would leave the count right and the numbering holed.
    assert numbered == list(range(1, len(numbered) + 1)), (
        f"findings.md numbers are not 1..{len(numbered)}: {numbered}"
    )


#: Provenance keys a run adds only when they apply, and what their absence says. Anything
#: outside this set and `run_provenance`'s own keys is undocumented in `releasing.md`.
CONDITIONAL_PROVENANCE = {"seed", "model", "model_key", "endpoint"}


def test_the_documented_provenance_block_matches_what_a_run_writes():
    """`releasing.md` prints a provenance object. It printed one that could not occur.

    The block omitted `executable`, which `run_provenance` has put in all 74 committed
    artifacts, and showed `model`, which only appears where a model was actually called.
    A reader writing a consumer against that block would have missed a field that is
    always there and expected one that is usually not.
    """
    import json as _json

    from pharos.provenance import run_provenance

    text = (ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")
    fence = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
    assert fence, "releasing.md no longer shows a JSON provenance block in the form this checks"
    documented = set(_json.loads(fence.group(1)))

    always = set(run_provenance())
    assert always <= documented, (
        f"`run_provenance` writes {sorted(always - documented)} and `releasing.md` does not "
        "show it. A key present in every artifact belongs in the block a reader copies."
    )
    assert documented <= always | CONDITIONAL_PROVENANCE, (
        f"`releasing.md` shows {sorted(documented - always - CONDITIONAL_PROVENANCE)}, which "
        "no run writes unconditionally and which nothing here documents as conditional."
    )


def test_the_extractors_can_fail():
    """The control. Both extractors are regular expressions over prose, and a regular
    expression that matches nothing is indistinguishable from one that found nothing.

    Every pattern above is given a string it must catch and a string it must not, drawn
    from the shapes these pages actually use.
    """
    caught = make_targets_named_in(
        "run `make gate` first, then the fenced form:\n\n    make review\n"
    )
    assert caught == {"gate", "review"}
    assert make_targets_named_in("this does not make a claim about targets") == set()
    assert make_targets_named_in("`make selective-risk` with a comment") == {"selective-risk"}

    assert paths_named_in("see `scripts/measure_selective_risk.py` for the run") == {
        "scripts/measure_selective_risk.py"
    }
    assert paths_named_in("`results/*.json` are tracked") == {"results/*.json"}
    assert paths_named_in("a `dict` is not a path, nor is `uv sync`") == set()

    assert headings_in('## 24. A share, and "a majority" was nine\ntext\n') == {
        "24-a-share-and-a-majority-was-nine"
    }
    assert headings_in("not a heading: # inside a line") == set()
    assert anchor_links_in("[a](#here) and [b](findings.md#there)") == {
        ("", "here"),
        ("findings.md", "there"),
    }
    assert anchor_links_in("[c](https://example.com/x#frag) and [d](findings.md)") == set()


def test_the_package_ships_its_type_information():
    """`ty` gates every annotation in this package, and none of it reaches a consumer
    without this marker.

    PEP 561: a package without `py.typed` is treated as untyped by every downstream
    checker, however thoroughly it is annotated. The repository enforces types on itself
    and shipped none of that guarantee outward, which is the same shape as a guard whose
    result nothing reads.
    """
    assert (ROOT / "src" / "pharos" / "py.typed").is_file()
    # And the wheel has to carry it. `packages = ["src/pharos"]` includes package data by
    # default under hatchling; asserting the file's presence in the source tree is only
    # half the claim, so pin the build configuration that carries it too.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["src/pharos"]' in pyproject


def test_the_repository_says_how_to_cite_it():
    """A public testbed a paper points at needs a citation record, and the record has to
    name the artifact rather than a person's best guess at it."""
    import yaml

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"].startswith("1.2")
    assert citation["authors"], "no authors recorded"
    assert citation["license"] == "MIT", "the citation record disagrees with LICENSE"
    assert (ROOT / "LICENSE").is_file()
    assert citation["repository-code"].endswith("/pharos")


def _script_imports() -> dict[str, list[str]]:
    """Which scripts import which other scripts, by module name."""
    import re

    found: dict[str, list[str]] = {}
    for path in sorted((ROOT / "scripts").glob("*.py")):
        modules = sorted(
            set(
                re.findall(
                    r"^from (measure_\w+|train_adapter) import", path.read_text(), re.MULTILINE
                )
            )
        )
        if modules:
            found[path.name] = modules
    return found


def test_no_script_imports_another():
    """The library boundary, enforced rather than intended.

    Thirty-four names crossed script boundaries before `pharos.governance` and
    `pharos.prompting` existed, two files were each imported by seven others, and one
    import reached past a leading underscore. A boundary nothing checks is a boundary that
    erodes the next time a measurement needs something a neighbour already has.

    There is no allowlist. If a script needs what another script has, the shared thing
    belongs in the package -- which is also the answer for anyone using this testbed to
    measure their own policy, since they cannot import a script at all.
    """
    offenders = _script_imports()
    assert not offenders, (
        f"these scripts import other scripts: {offenders}. Move the shared symbol into "
        "the package rather than importing across experiment files."
    )


def test_no_script_redefines_what_the_package_exports():
    """Two definitions of one idea is the state the extraction removed.

    It happened during that extraction: the audit-scoring functions were moved and the
    originals left behind, so the script kept using its own copy while the package held
    another. Coverage surfaced it; this makes it fail directly.
    """
    import re

    import pharos.governance as governance

    exported = set(governance.__all__)
    clashes = {}
    for path in sorted((ROOT / "scripts").glob("*.py")):
        defined = set(re.findall(r"^(?:def|class) (\w+)", path.read_text(), re.MULTILINE))
        overlap = defined & exported
        if overlap:
            clashes[path.name] = sorted(overlap)
    assert not clashes, (
        f"these scripts redefine names the package exports: {clashes}. Either the script "
        "should import it, or the two are different things and one needs a better name."
    )
