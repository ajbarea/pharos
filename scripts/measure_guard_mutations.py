#!/usr/bin/env python3
"""Finding 27: ask the guards the question they ask of everything else.

Coverage says a line ran. It cannot say a check would have objected, and those are
different questions -- a guard executes identically whether or not anything verifies
what it decided. Every retraction in this project was a measurement that looked healthy,
so "the code ran" was never the property worth having.

This applies a small set of hand-chosen mutations to the two modules the manuscript
names as instruments, runs the *whole* suite against each, and records whether anything
failed. A mutation that survives is a condition nobody checks.

The mutations are curated rather than generated. A full `mutmut` run over these modules
produces 438 mutants, most of them string literals and log text that nobody should chase;
`[tool.mutmut]` in pyproject.toml records that scope for anyone who wants it. What is
worth committing is the handful that are guards, so that a future change which stops
killing one of them is a visible regression rather than a discovery.

Usage:  python scripts/measure_guard_mutations.py
        python scripts/measure_guard_mutations.py --check   # verdicts only, no write

Costly on purpose: one full test suite per mutation plus a baseline, several minutes
each. It is not in CI's per-push path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharos.governance import REFUSED_EXIT as _REFUSED_EXIT
from pharos.provenance import run_provenance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "guard_mutations.json"

#: The precondition failed: the tree was dirty, or the suite was already red. Distinct
#: from a crash and from a finding, because a mutation result measured against a broken
#: baseline is not a weak result, it is not a result.
#:
#: Imported rather than restated, and the reason is a shared *convention* rather than a
#: consumed contract, which is worth being exact about. `governance.sweep` matches this
#: code to tell a corpus that cannot host an experiment from a crash, and nothing runs
#: this script automatically, so no caller reads the code emitted here. What is shared is
#: the meaning: across this repository, exit 3 says "refused, and that is not a failure".
#: The refusals differ -- there it is a corpus, here it is a dirty tree or an already-red
#: suite -- and a second literal is how one of them drifts to a different number and the
#: convention quietly stops being one.
REFUSED_EXIT = _REFUSED_EXIT


@dataclass(frozen=True)
class Mutation:
    path: str
    old: str
    new: str
    guard: str
    breaks: str


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "src/pharos/validity.py",
        "    total = scored + unparsed",
        "    total = scored - unparsed",
        "the sample size every validity flag is computed against",
        "n is wrong and stays plausible, which is how the decode-regime defect happened",
    ),
    Mutation(
        "src/pharos/validity.py",
        "    if total < SMALL_N:",
        "    if total <= SMALL_N:",
        "the small-n boundary",
        "a measurement at exactly n=30 gains or loses its caveat",
    ),
    Mutation(
        "src/pharos/provenance.py",
        "    if not path:",
        "    if path:",
        "the empty-path guard in the executable sanitiser",
        "`executable` becomes empty in every artifact this project writes",
    ),
    Mutation(
        "src/pharos/provenance.py",
        '    if ".venv" in path:',
        '    if ".venv" not in path:',
        "the venv branch of the executable sanitiser",
        "the recorded path stops being the stable relative form",
    ),
)


#: Ignored while measuring, and only here. `test_guard_mutations.py` asserts that each
#: anchor below appears once in its file -- which is false by construction while that
#: file is mutated, so it would fail against every mutant and report four kills that
#: were really the harness noticing its own edit. A test that cannot pass under the
#: measurement is not evidence about the measurement's subject.
SELF_TEST = "tests/test_guard_mutations.py"


def _suite_fails() -> bool:
    """Whether the full suite objects. Read from the exit status, deliberately.

    The first version of this harness grepped pytest's last two lines for "N failed".
    Under `-q` the final line is "FAILED tests/...", so the count line was never in the
    window and every real failure read as a pass. It produced four false survivals. The
    exit status cannot be misread that way.
    """
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-cov",
            "-p",
            "no:randomly",
            "--ignore",
            SELF_TEST,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode != 0


def _refuse(message: str) -> None:
    print(f"refusing: {message}", file=sys.stderr)
    raise SystemExit(REFUSED_EXIT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="print verdicts, write nothing")
    args = parser.parse_args()

    targets = sorted({m.path for m in MUTATIONS})

    # This script rewrites source files and restores them from an in-memory copy. Against
    # a dirty tree that restore is a silent revert of somebody's uncommitted work, so the
    # precondition is not politeness.
    status = subprocess.run(  # noqa: S603
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *targets],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        _refuse("git could not report the status of the files this script rewrites")
    if status.stdout.strip():
        _refuse(
            "the files this script rewrites have uncommitted changes:\n  "
            + "\n  ".join(status.stdout.strip().splitlines())
            + "\n  Commit or stash them first; restoring would discard them."
        )

    print("baseline: running the suite unmutated ...", flush=True)
    if _suite_fails():
        _refuse("the suite is already failing, so no mutation verdict would mean anything")
    print("  baseline green\n", flush=True)

    originals = {path: (ROOT / path).read_text(encoding="utf-8") for path in targets}
    results = []
    try:
        for mutation in MUTATIONS:
            source = originals[mutation.path]
            if source.count(mutation.old) != 1:
                _refuse(
                    f"anchor for {mutation.guard!r} appears "
                    f"{source.count(mutation.old)} times in {mutation.path}; "
                    "the mutation no longer describes the code"
                )
            print(f"mutating {mutation.guard} ...", flush=True)
            (ROOT / mutation.path).write_text(
                source.replace(mutation.old, mutation.new), encoding="utf-8"
            )
            killed = _suite_fails()
            (ROOT / mutation.path).write_text(source, encoding="utf-8")
            print(f"  {'KILLED' if killed else 'SURVIVED'}\n", flush=True)
            results.append(
                {
                    "path": mutation.path,
                    "guard": mutation.guard,
                    "breaks": mutation.breaks,
                    "from": mutation.old.strip(),
                    "to": mutation.new.strip(),
                    "killed": killed,
                }
            )
    finally:
        # Restore unconditionally: an interrupt in the middle otherwise leaves a mutated
        # guard in the working tree, which is the worst possible artifact of this script.
        for path, text in originals.items():
            (ROOT / path).write_text(text, encoding="utf-8")

    surviving = [r for r in results if not r["killed"]]
    payload = {
        "mutations": len(results),
        "killed": len(results) - len(surviving),
        "surviving": len(surviving),
        "results": results,
        "scope": {
            "modules": targets,
            "excluded": {
                "src/pharos/gate.py": (
                    "in scope on the merits, excluded on cost: its tests take 90 seconds "
                    "against half a second for these two. A budget, not a verdict."
                )
            },
        },
        "provenance": run_provenance(),
    }

    for row in results:
        print(f"{'KILLED  ' if row['killed'] else 'SURVIVED'} {row['guard']}")
    print(f"\n{payload['killed']}/{payload['mutations']} killed")

    if args.check:
        return 1 if surviving else 0

    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
