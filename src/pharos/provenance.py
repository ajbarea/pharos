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

import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

from pharos import __version__
from pharos.telemetry import get_logger

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
        _degraded("git is not on PATH", args)
        return None
    try:
        completed = subprocess.run(  # noqa: S603  # fixed argv, resolved executable
            [executable, *args],
            capture_output=True,
            text=True,
            # 5s was enough on a laptop and not on a shared filesystem: four array
            # tasks starting together on the cluster each ran `git status` against the
            # same NFS-mounted checkout, and one timed out and stamped its artifact
            # `git_dirty: null`. The tree was clean, but the artifact could no longer
            # say so, which is the one thing this stamp exists to record. Reading a
            # git index is not a 5-second operation on local disk and can be on a
            # contended mount, so the bound is generous rather than tight -- the cost
            # of waiting is a few seconds on a job that trains a LoRA, and the cost of
            # giving up is an artifact whose provenance is unverifiable.
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _degraded(f"git could not be run: {type(exc).__name__}", args)
        return None
    if completed.returncode != 0:
        _degraded(f"git exited {completed.returncode}", args)
        return None
    return completed.stdout.strip()


def _degraded(why: str, args: tuple[str, ...]) -> None:
    """Say that provenance could not be collected, and why.

    Degrading to `None` is correct -- a measurement must not die because it cannot
    name its own commit -- but doing it *silently* is not, and this module is the
    one place where silence is most expensive. A run that cannot identify its code
    produces an artifact nothing downstream can verify, and the consumer that would
    have caught it treats a missing commit as "nothing to check" rather than as
    "cannot check". Both ends of that chain were quiet until this existed.
    """
    get_logger().warning(
        "provenance.degraded",
        extra={
            "event": "provenance.degraded",
            "reason": why,
            "git_args": " ".join(args),
        },
    )


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
    """Version and commit. No clock, so a manifest carrying this stays reproducible.

    Warns once more at the top level when the result is incomplete. The per-call
    warnings above say what failed; this one says what it *cost*, because that is
    the sentence a reader scanning a log needs: an artifact written from this cannot
    be traced to code, and every staleness check over it is vacuous.
    """
    stamp: dict[str, Any] = {
        "pharos_version": __version__,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
    }
    if stamp["git_commit"] is None:
        get_logger().warning(
            "provenance.unidentifiable",
            extra={
                "event": "provenance.unidentifiable",
                "impact": (
                    "artifacts written from this run cannot name the code that "
                    "produced them, and staleness checks over them cannot run"
                ),
            },
        )
    return stamp


def _sanitize_executable(path: str) -> str:
    """Sanitize Python executable path to avoid leaking absolute personal home directory paths."""
    if not path:
        return path
    if ".venv" in path:
        idx = path.find(".venv")
        return path[idx:]
    home = os.path.expanduser("~")  # noqa: PTH111

    if home and path.startswith(home):
        return "~" + path[len(home) :]
    return path


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
        "executable": _sanitize_executable(sys.executable),
        **extra,
    }
