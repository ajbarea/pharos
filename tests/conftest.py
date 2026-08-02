"""Test-suite setup.

`scripts/` is not a package and is imported by path, so it has to be on `sys.path`
before any test module that measures a script can import one. Doing it here rather than
per-file means a new test does not have to rediscover the dance, and it matches what
`[tool.ty.environment] extra-paths` already tells the type checker.

`tests/test_scripts.py` still performs its own insert. That is now redundant rather
than wrong, and it is left alone because removing it would couple an unrelated file to
this change.
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
