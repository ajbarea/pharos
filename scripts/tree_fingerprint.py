#!/usr/bin/env python3
"""A hash of the working tree the gate is about to measure, and a check that it held.

A local gate takes about ten minutes. Nothing stops a file being edited during those
ten minutes, and if one is, the run reports a number for a tree that never existed.
That happened on 2026-08-06: `secagg.py` and its tests were edited while `make ci` was
in flight, and the run reported 93.99% -> 92.52% with `secagg.py` at 40.58%, having
measured a half-written tree. The gate passed. Nothing was wrong with the code.

The number was caught only because 92.52% was surprising enough to chase, which is
luck rather than process. A measurement whose provenance nobody checked is exactly the
failure this repository exists to study, so it should not be the one the repository
itself ships.

The fix is the same one used on every published artifact here: stamp the input, and
refuse when the stamp does not match at the end. This is cheaper than isolating the run
in a worktree and catches the same failure -- a poisoned measurement becomes a loud
error instead of a plausible number.

Untracked files are included by path and content, because a new test file is invisible
to `git diff` and is exactly the kind of thing added mid-run.

    uv run python scripts/tree_fingerprint.py                    # print the fingerprint
    uv run python scripts/tree_fingerprint.py --write PATH       # print and save it
    uv run python scripts/tree_fingerprint.py --verify PATH      # exit 2 if it moved
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories whose contents decide what a gate measures. `results/` is deliberately
#: absent: several gate steps rewrite artifacts as they run, so including it would make
#: every run fail its own check. What matters here is the *code*, which no gate step
#: modifies.
WATCHED = ("src", "scripts", "tests", "pyproject.toml", "Makefile")


def _run(*args: str) -> str:
    """One git invocation. Executable resolved rather than looked up by the shell,
    matching `pharos.provenance`: every argument here is a literal from this file."""
    executable = shutil.which("git")
    if executable is None:
        raise SystemExit("tree_fingerprint: git is not on PATH, so the tree cannot be stamped")
    return subprocess.run(  # noqa: S603  # fixed argv, resolved executable
        [executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def fingerprint() -> str:
    """A stable hash over tracked content plus untracked files under WATCHED."""
    digest = hashlib.sha256()

    # Tracked content, including staged and unstaged modifications. `--binary` so a
    # change that is not valid UTF-8 still alters the diff rather than being elided.
    digest.update(_run("diff", "HEAD", "--binary", "--", *WATCHED).encode())
    digest.update(_run("rev-parse", "HEAD").encode())

    # Untracked files are not in any diff, and a new test file added mid-run is the
    # case this is here for.
    untracked = sorted(
        line for line in _run("ls-files", "--others", "--exclude-standard", "--", *WATCHED).split()
    )
    for name in untracked:
        path = ROOT / name
        digest.update(name.encode())
        if path.is_file():
            digest.update(path.read_bytes())

    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="save the fingerprint to this path")
    parser.add_argument("--verify", type=Path, help="fail if the tree no longer matches this file")
    args = parser.parse_args()

    current = fingerprint()

    if args.verify:
        if not args.verify.exists():
            print(
                f"tree_fingerprint: {args.verify} is missing, so the gate cannot show "
                "what it measured; treat this run as unmeasured",
                file=sys.stderr,
            )
            return 2
        expected = args.verify.read_text(encoding="utf-8").strip()
        if expected != current:
            print(
                "tree_fingerprint: the working tree changed while the gate was running, "
                "so this run measured a tree that never existed. Its coverage number and "
                "its pass are both meaningless. Re-run on a settled tree.\n"
                f"  at start: {expected}\n"
                f"  at end:   {current}",
                file=sys.stderr,
            )
            return 2
        print(f"tree_fingerprint: tree unchanged through the gate ({current[:12]})")
        return 0

    if args.write:
        args.write.write_text(current + "\n", encoding="utf-8")
    print(current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
