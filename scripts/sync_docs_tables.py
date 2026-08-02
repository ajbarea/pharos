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


#: Block name to builder. A block present in a doc but absent here is an error rather
#: than a no-op: a marker with nothing behind it is how a table quietly stops updating.
BLOCKS = {"power-claims": power_claims}

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
    checked = 0
    for path in sorted(DOCS.rglob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated, names = render(original)
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

    if stale:
        print("stale generated blocks: " + "; ".join(stale), file=sys.stderr)
        print("run `uv run python scripts/sync_docs_tables.py` to refresh", file=sys.stderr)
        return 1
    print(f"sync_docs_tables: {checked} generated block(s) current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
