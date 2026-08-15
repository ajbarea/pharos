"""Running a measurement at one point of a sweep, and reading back what it wrote.

A sweep spawns measurement scripts as subprocesses rather than importing them, so each
point gets a clean interpreter and its artifact records its own provenance. Shelling out
is also what keeps a sweep honest: each script owns its defaults, its argument parsing and
its validity checks, and re-implementing any of that here would let the sweep and the
committed artifacts drift apart.

Two sweeps needed this and both wrote it. The copies had the same name, different
signatures, and -- the part that mattered -- different answers to the question that
actually distinguishes them: what a non-zero exit means.

**Refusal is not failure, and that distinction is now a parameter.** A script that cannot
host an experiment on a given corpus exits `REFUSED_EXIT`. A draw that cannot host the
negative control says nothing about the finding, and counting it against the finding would
shrink the denominator while leaving the rate looking unchanged. A sweep over corpus draws
must therefore accept refusals and exclude them; a sweep over fleet sizes has no such
precondition and must treat every non-zero exit as the bug it is. Passing that as
`allow_refusal` makes the choice visible at each call site instead of implicit in which
copy of the function a script happened to import.
"""

import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pharos.governance.fleet import REFUSED_EXIT
from pharos.telemetry import get_logger

__all__ = ["SCRIPTS", "MeasurementFailedError", "run_measurement"]

LOG = get_logger()

#: Where the measurement scripts live, resolved from this file rather than from a
#: caller's working directory, so a sweep run from anywhere finds the same scripts.
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


class MeasurementFailedError(RuntimeError):
    """A measurement exited non-zero for a reason that is not a declared refusal.

    A `RuntimeError` rather than a `SystemExit`, because a library that kills the
    interpreter takes the decision away from a caller that may want to record the failure
    and keep sweeping. The scripts translate it.
    """


def run_measurement(
    script: str,
    args: Sequence[str],
    *,
    allow_refusal: bool,
    cwd: Path | None = None,
) -> dict[str, Any] | None:
    """One measurement, run to a temporary artifact and parsed back.

    Returns `None` only when `allow_refusal` is set and the script exited `REFUSED_EXIT`,
    which is the declared "this input cannot host the experiment" code. Every other
    non-zero exit raises, because a sweep that cannot tell a refusal from a crash reports
    a bug as a property of the input: the point drops out of the denominator and the rate
    above it looks unchanged.

    The temporary file is deliberate. A sensitivity sweep must never overwrite `results/`,
    and an earlier version of one of these that wrote through the caller's `--out` would
    have replaced a committed artifact with a swept one.
    """
    root = cwd or SCRIPTS.parent
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = Path(handle.name)
    try:
        completed = subprocess.run(  # noqa: S603  # fixed argv, this package's own scripts
            [sys.executable, str(SCRIPTS / script), *args, "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
        )
        reason = (
            completed.stderr.strip().splitlines()[-1][:200]
            if completed.stderr.strip()
            else "no stderr"
        )
        if allow_refusal and completed.returncode == REFUSED_EXIT:
            LOG.warning(
                "sweep.point_refused",
                extra={
                    "event": "sweep.point_refused",
                    "script": script,
                    # `argv` rather than `args`: `args` is a reserved LogRecord
                    # attribute, and the stdlib raises rather than shadowing it. The
                    # first refused draw of any sweep would have crashed here.
                    "argv": list(args),
                    "reason": reason,
                },
            )
            return None
        if completed.returncode != 0 or not out.exists() or not out.stat().st_size:
            raise MeasurementFailedError(
                f"{script} {' '.join(args)} exited {completed.returncode}: {reason}"
            )
        return json.loads(out.read_text(encoding="utf-8"))
    finally:
        out.unlink(missing_ok=True)
