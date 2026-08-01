"""Where a number came from: the code state, and the environment that ran it.

A measurement is worth what its provenance is worth. A table in a paper that says
"F1 1.000" is unfalsifiable unless a reader can name the commit that produced it,
so every result Pharos writes carries a stamp identifying the code, and every
corpus manifest carries the commit that generated it.

The split between the two functions here is deliberate.

`code_provenance` is **stable for a given checkout**: version and commit, no clock
and no host. It goes into the corpus manifest, which has to stay reproducible from
`(seed, config)` alone. A timestamp there would make two identical corpora compare
unequal, which is exactly the property the manifest exists to certify.

`run_provenance` adds the clock and the interpreter. It goes into measurement
results, which are records of a *run* rather than of a corpus, and for which "when,
on what" is part of the finding.

Everything degrades to `None` rather than raising. Pharos has to work from a source
tarball with no `.git`, and provenance is metadata: failing to collect it must never
take down the measurement it describes.
"""

import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

from pharos import __version__

#: Long enough to be unambiguous in this repo, short enough to read in a table.
_SHA_LENGTH = 12


def _git(*args: str) -> str | None:
    """Run a git command, or return None when git or the repo is unavailable.

    The executable is resolved through `shutil.which` rather than relying on PATH
    resolution inside the subprocess, and every argument is a literal from this
    module, so nothing user-supplied reaches the command line.
    """
    executable = shutil.which("git")
    if executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603  # fixed argv, resolved executable
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_commit() -> str | None:
    """The short SHA of HEAD, or None outside a git checkout."""
    return _git("rev-parse", f"--short={_SHA_LENGTH}", "HEAD") or None


# `results/` holds measurement outputs, not code. It is excluded from the dirtiness
# check because a script that writes there would otherwise dirty the tree for
# everything that runs after it: a six-model sweep from one clean checkout produced
# one artifact marked clean and five marked dirty, in commit order, purely because
# the first write landed before the second model started. That reads as five
# measurements whose code could not be reconstructed, when the code was identical
# and committed throughout.
_NOT_CODE = (".", ":(exclude)results")


def git_is_dirty() -> bool | None:
    """Whether tracked *code* differs from HEAD, or None outside a git checkout.

    Reported rather than forbidden. A dirty measurement is often the honest state
    of an experiment in progress, and the useful thing is that a reader can see it
    was dirty rather than be told a commit that does not describe the code that ran.
    """
    status = _git("status", "--porcelain", "--untracked-files=no", "--", *_NOT_CODE)
    if status is None:
        return None
    return bool(status)


def code_provenance() -> dict[str, Any]:
    """Version and commit. No clock, so a manifest carrying this stays reproducible."""
    return {
        "pharos_version": __version__,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
    }


def run_provenance(**extra: Any) -> dict[str, Any]:
    """Code provenance plus when and on what, for a measurement result.

    `extra` carries whatever identifies the run's own inputs, typically the model
    and endpoint a script used, so one object answers "what produced this number".
    """
    return {
        **code_provenance(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        **extra,
    }
