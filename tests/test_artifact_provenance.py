"""No committed artifact may claim it came from a tree that matches no commit.

`git_dirty: true` on a published artifact means the code behind its numbers cannot be
reconstructed from its own provenance stamp. `cluster/README.md` records what that cost
once already: an rsync-based sync left the cluster's HEAD stale while the contents
changed underneath it, so every artifact produced there stamped a commit that had never
contained the code that ran.

The same thing happens far more cheaply and far more often on a laptop. Regenerating an
artifact in the middle of editing is the natural way to work, and it stamps the artifact
dirty; committing it then publishes a number whose provenance points at nothing. That
happened four times in two days here -- `blind_spot`, `secure_reliability`,
`channel_bias` and `power` -- and each was found by reading the files rather than by
anything failing.

So: a dirty artifact has to be named here with a reason, exactly as `test_logcheck`
requires of a measurement script that skips the sweep. The exemptions are real -- an
artifact produced on a GPU or on a second machine cannot be re-stamped by re-running it
locally -- but they have to be *claimed* rather than accumulated.
"""

import json
from pathlib import Path
from typing import Any

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"

#: Artifacts that legitimately carry a dirty stamp, and why. Every one of these is
#: produced somewhere this repository cannot re-run: a CUDA GPU, or the second machine
#: of a two-machine comparison. Re-stamping them means re-running the job that made
#: them, which is a cluster submission rather than a local command.
DIRTY_IS_EXPECTED = {
    "adapter_replication": "compares adapter artifacts produced by GPU cluster jobs",
    "edge_cost": "reads adapter artifacts produced by GPU cluster jobs",
    "gate_determinism": "the cluster half of a two-machine comparison",
    "label_fidelity": "calls a model; regenerating it needs Ollama serving locally",
    "teacher_fleet": "aggregates adapter artifacts produced by GPU cluster jobs",
}


def _artifacts() -> list[tuple[str, dict[str, Any]]]:
    return [
        (path.stem, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(RESULTS.glob("*.json"))
    ]


def test_every_artifact_carries_a_provenance_stamp():
    """An artifact with no stamp is worse than one with a dirty stamp: it makes no claim."""
    unstamped = [
        name for name, payload in _artifacts() if not isinstance(payload.get("provenance"), dict)
    ]
    assert not unstamped, f"artifacts with no provenance block: {unstamped}"


def test_no_model_free_artifact_is_committed_from_a_dirty_tree():
    """The guard. A dirty stamp has to be claimed here, not merely tolerated."""
    dirty = {
        name
        for name, payload in _artifacts()
        if payload["provenance"].get("git_dirty") is True and name not in DIRTY_IS_EXPECTED
    }
    assert not dirty, (
        f"artifacts stamped from a modified working tree: {sorted(dirty)}. "
        "Re-run the measurement on a clean tree so its stamp names the code that "
        "produced it, or add it to DIRTY_IS_EXPECTED with the reason it cannot be."
    )


@pytest.mark.parametrize("name", sorted(DIRTY_IS_EXPECTED))
def test_each_claimed_exemption_is_still_needed(name):
    """An exemption that stopped being true is a guard quietly switched off.

    If one of these is re-run on a clean tree, its entry here should go rather than
    linger and cover a future dirty stamp on the same artifact.
    """
    path = RESULTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name}.json is not committed in this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provenance"].get("git_dirty") is True, (
        f"{name} is exempted from the dirty-artifact check but is no longer dirty. "
        "Remove its entry from DIRTY_IS_EXPECTED so the check covers it again."
    )


def test_every_artifact_names_a_commit():
    """A stamp with no commit cannot be resolved to code at all."""
    nameless = [
        name for name, payload in _artifacts() if not payload["provenance"].get("git_commit")
    ]
    assert not nameless, f"artifacts whose provenance names no commit: {nameless}"
