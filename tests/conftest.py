"""Test-suite setup.

`scripts/` is not a package and is imported by path, so it has to be on `sys.path`
before any test module that measures a script can import one. Doing it here rather than
per-file means a new test does not have to rediscover the dance, and it matches what
`[tool.ty.environment] extra-paths` already tells the type checker.

`tests/test_scripts.py` still performs its own insert. That is now redundant rather
than wrong, and it is left alone because removing it would couple an unrelated file to
this change.

`artifact()` is here for a duller reason and a real one. Nine test files read committed
artifacts out of `results/`, and every one of them had rebuilt the same three lines --
resolve the repo root, join the filename, parse the JSON. Rewriting that helper is how
its return annotation came out as a bare `dict` three separate times in one day, each
caught by `ty` after the fact. One typed helper removes the repetition and the slip
together.
"""

import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@cache
def _load(name: str) -> str:
    path = RESULTS / name
    if not path.exists():
        raise AssertionError(
            f"{path.relative_to(ROOT)} is not committed, so the test asserting against it "
            "cannot say what it checked. Re-run the measurement that writes it."
        )
    return path.read_text(encoding="utf-8")


def artifact(name: str) -> dict[str, Any]:
    """One committed measurement artifact, by filename.

    Raises rather than skipping when the file is absent: a test that quietly passes
    because its evidence is missing is the failure mode this repository spends most of
    its guards on. Text is cached because the suite reads the same few artifacts many
    times, but the parse is not, so a caller mutating the result cannot affect another.
    """
    if not name.endswith(".json"):
        name = f"{name}.json"
    return json.loads(_load(name))
