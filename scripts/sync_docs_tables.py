#!/usr/bin/env python3
"""Regenerate the tables in `docs/` that restate a committed artifact.

Most numbers on the docs site are prose written once beside a measurement, and prose
is allowed to be prose. A few are different: they restate an artifact row for row, and
those rot silently the moment the artifact moves. The claim table in `findings.md` did
exactly that twice in one day. It was updated by hand when the power analysis learned
to price a claim against a known constant, then went stale again hours later when two
of its claims were rerun at a larger size, and in between it stated a verdict that
contradicted another section of the same page.

So the fix is the one the paper's tables already use: generate the block, and give CI
a way to fail when the file disagrees with the artifact behind it.

Blocks are delimited by HTML comments, which render as nothing:

    <!-- BEGIN GENERATED: power-claims -->
    ...
    <!-- END GENERATED: power-claims -->

Anything outside a marked block is left alone, so the surrounding argument stays
hand-written where it belongs.

    uv run python scripts/sync_docs_tables.py           # rewrite the blocks
    uv run python scripts/sync_docs_tables.py --check   # fail if any is stale
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"


def _fail(message: str) -> NoReturn:
    """Always raises. Typed `NoReturn` so callers narrow correctly after it."""
    print(f"sync_docs_tables: {message}", file=sys.stderr)
    raise SystemExit(2)


def power_claims() -> str:
    """The claim table: every headline claim against the size it was measured at.

    Verdict wording is derived rather than typed, because the distinction between
    "unresolved at a size we could buy" and "unresolved at any size" is the one a
    reader acts on, and it was previously maintained by hand.
    """
    path = RESULTS / "power.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make power` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = payload.get("claims") or []
    if not claims:
        _fail("power.json carries no claims; refusing to emit an empty table")

    lines = [
        "| Finding | n | Gap it rests on | vs | Verdict | Claim |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for claim in claims:
        against = "constant" if claim["against_constant"] else "condition"
        if claim["resolved"]:
            verdict = "**resolved**"
        elif claim["n_needed"]:
            verdict = f"unresolved (needs n≥{claim['n_needed']})"
        else:
            verdict = "unresolved (needs n>2000)"
        lines.append(
            f"| {claim['finding']} | {claim['n']} | {claim['effect']:.3f} | "
            f"{against} | {verdict} | {claim['description']} |"
        )
    resolved = sum(1 for c in claims if c["resolved"])
    lines.append("")
    lines.append(f"**{resolved} of {len(claims)}** resolve at the size they were run.")
    return "\n".join(lines)


#: Artifacts whose *script* computes validity but whose committed artifact predates
#: that change. A rerun, not a code change, and hours of GPU rather than minutes of
#: editing -- which is why they are listed apart from the exemptions below.
AWAITING_RERUN = {
    "learnability": "measure_rule_learnability.py now records it per shot count",
    "label_fidelity": "measure_label_fidelity.py now records it over the scored turns",
    "decode_stability": "measure_decode_stability.py now records it over the repeated passes",
    "adapter_learnability": "train_adapter.py now records it per evaluation pass",
    **{
        f"review_adapter-{t}{x}": "train_adapter.py now records it per evaluation pass"
        for t in ("by-the-book", "inattentive", "two-of-three", "any-one")
        for x in ("", "-xseed101")
    },
}

#: Artifacts with no sampling-validity question to answer, and why. An exemption is a
#: claim, so each carries its reason and appears on the page rather than being filtered
#: out silently.
NO_SAMPLING_QUESTION = {
    "power": "prices hypothetical evaluation sizes; simulates outcomes rather than measuring any",
    "federation_eligibility": "deterministic over the label lattice; nothing is sampled",
    "external_gate_validation": "carries its own permutation-null statistics per corpus",
    "triage_lift": "superseded by the per-model triage_lift-* artifacts, which are assessed",
    "fleet_sensitivity": "a sweep over a nuisance parameter; reports invariants, samples nothing",
    "teacher_fleet": "aggregates assessed adapter artifacts; adds no measurement of its own",
}


def measurement_health() -> str:
    """Every artifact's validity assessment, published rather than left in a field.

    `pharos.validity` marks a measurement unquotable when it trips a condition that
    makes a score misleading: too few samples, a class floor the score does not clear,
    degenerate predictions, unparsed answers. The flag was computed, warned about on
    the console, written into some artifacts, and then read by nothing. A guard that
    refuses to let prose quote a flagged number would be wrong, because quoting one as
    evidence of *failure* is exactly what the flag licenses and is what finding 3b
    correctly does. So the enforcement is publication instead: the caveats appear on
    the page, generated from the artifacts, and cannot drift away from them.

    Artifacts with no validity block are listed too. An unassessed measurement is a
    different problem from a flagged one and the two should not look alike.
    """
    rows: list[tuple[str, str, str]] = []
    unassessed: list[str] = []
    # `AWAITING_RERUN` is hand-maintained and goes stale in one direction: a rerun
    # lands and the entry saying it has not is left behind, which once listed
    # `decode_stability` as both quotable at n=30 and awaiting the rerun that produced
    # the 30. Derive the pending list from the artifacts; the registry keeps only the
    # reasons, which cannot be derived.
    now_assessed: set[str] = set()
    for path in sorted(RESULTS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validity = payload.get("validity")
        which_pass = ""
        if not isinstance(validity, dict):
            # Adapter artifacts carry the assessment per evaluation pass rather than
            # at the top level, so look one level down before calling it unassessed.
            nested = [
                (name, v["validity"])
                for name, v in payload.items()
                if isinstance(v, dict) and isinstance(v.get("validity"), dict)
            ]
            # And one level down inside a *list*, which is where a per-condition
            # measurement puts it: `learnability` assesses each shot count separately
            # and has no single top-level verdict. Only the dict case was handled, so
            # every such artifact was reported as carrying no assessment at all while
            # its rows each carried one. `learnability` was masked by an
            # `AWAITING_RERUN` entry that has since been satisfied; its replication had
            # no such cover and showed up as an unassessed gap on the page.
            for name, value in payload.items():
                if not isinstance(value, list):
                    continue
                for index, item in enumerate(value):
                    if not isinstance(item, dict):
                        continue
                    row_validity = item.get("validity")
                    if isinstance(row_validity, dict):
                        nested.append((f"{name}[{index}]", row_validity))
            if not nested:
                if path.stem not in NO_SAMPLING_QUESTION and path.stem not in AWAITING_RERUN:
                    unassessed.append(path.stem)
                continue
            # The worst pass decides the flag, but it has to say WHICH pass. Collapsing
            # them anonymously reported `review_adapter-by-the-book` as unquotable on
            # the strength of its untrained baseline, while the adapter that artifact
            # exists to report scored 0.995 and was perfectly quotable. A flag that
            # names the wrong thing is the same defect as a flag on the wrong number.
            name, validity = min(nested, key=lambda item: bool(item[1].get("quotable", True)))
            which_pass = f"**{name}**: "
        concerns = validity.get("concerns") or []
        mark = "yes" if validity.get("quotable") else "**no**"
        note = f"{which_pass}{'; '.join(str(c) for c in concerns)}" if concerns else "-"
        rows.append((path.stem, f"{validity.get('n', '?')}", f"{mark} | {note}"))
        now_assessed.add(path.stem)

    lines = [
        "| Artifact | n | Quotable | Why not |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| `{name}` | {n} | {rest} |" for name, n, rest in rows]
    flagged = sum(1 for _, _, rest in rows if rest.startswith("**no**"))
    lines.append("")
    lines.append(
        f"**{flagged} of {len(rows)}** assessed artifacts are flagged. A flagged number "
        "may still be quoted as evidence that something *failed*, which is what the "
        "flag asserts; it may not be quoted as evidence of capability."
    )
    if unassessed:
        lines.append("")
        lines.append(
            "**Carrying no validity assessment, which is a gap rather than a pass:** "
            + ", ".join(f"`{name}`" for name in sorted(unassessed))
            + "."
        )
    present = {p.stem for p in RESULTS.glob("*.json")}
    pending = sorted(
        (n, w) for n, w in AWAITING_RERUN.items() if n in present and n not in now_assessed
    )
    if pending:
        lines.append("")
        lines.append(
            f"Assessed by their script but not yet in the committed artifact "
            f"({len(pending)} of these), which needs a rerun rather than an edit:"
        )
        lines.append("")
        for name, why in pending:
            lines.append(f"- `{name}` -- {why}")
    exempt = sorted(NO_SAMPLING_QUESTION.items())
    if exempt:
        lines.append("")
        lines.append("Exempt, because there is no sampling question to answer:")
        lines.append("")
        for name, why in exempt:
            missing = "" if name in present else " *(artifact absent)*"
            lines.append(f"- `{name}` -- {why}{missing}")
    return "\n".join(lines)


#: Block name to builder. A block present in a doc but absent here is an error rather
#: than a no-op: a marker with nothing behind it is how a table quietly stops updating.
BLOCKS = {"power-claims": power_claims, "measurement-health": measurement_health}

#: A BEGIN marker on its own. Used to catch pairs the full pattern cannot match --
#: adjacent markers with no body, a mismatched name, a missing END -- because those
#: are silently skipped by `_MARKER` and a block that is never rendered looks exactly
#: like a block that is up to date. This guard exists because that happened: the
#: measurement-health block was added with its markers on consecutive lines, produced
#: no output, and `--check` reported everything current.
_OPENER = re.compile(r"<!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->")

_MARKER = re.compile(
    r"(?P<open><!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<close>\n<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def render(text: str) -> tuple[str, list[str]]:
    """Rewrite every marked block. Returns the new text and the names it rewrote."""
    seen: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        builder = BLOCKS.get(name)
        if builder is None:
            _fail(f"no builder registered for generated block {name!r}")
        seen.append(name)
        return f"{match.group('open')}{builder()}{match.group('close')}"

    return _MARKER.sub(replace, text), seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any generated block differs from its artifact",
    )
    args = parser.parse_args()

    stale: list[str] = []
    malformed: list[str] = []
    checked = 0
    for path in sorted(DOCS.rglob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated, names = render(original)
        # Every BEGIN marker must have been rendered. One that was not is a marker the
        # full pattern could not match, which renders nothing and reports nothing.
        malformed.extend(
            f"{path.relative_to(ROOT)} ({m.group('name')})"
            for m in _OPENER.finditer(original)
            if m.group("name") not in names
        )
        checked += len(names)
        if not names:
            continue
        if args.check:
            if updated != original:
                stale.append(f"{path.relative_to(ROOT)} ({', '.join(names)})")
        elif updated != original:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}: {', '.join(names)}")

    # A guard that verified nothing must not report success. This exact failure --
    # a check passing because it silently matched no files -- has happened twice in
    # this project's tooling, so it is asserted rather than assumed.
    if checked == 0:
        _fail("found no generated blocks in docs/; the markers are missing or renamed")

    if malformed:
        print(
            "generated blocks whose markers did not match and rendered nothing: "
            + "; ".join(malformed),
            file=sys.stderr,
        )
        print(
            "check the BEGIN/END pair: same name, END on its own line, a body between them",
            file=sys.stderr,
        )
        return 1

    if stale:
        print("stale generated blocks: " + "; ".join(stale), file=sys.stderr)
        print("run `uv run python scripts/sync_docs_tables.py` to refresh", file=sys.stderr)
        return 1
    print(f"sync_docs_tables: {checked} generated block(s) current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
