"""The mutation definitions must keep describing the code they mutate.

`measure_guard_mutations.py` costs a full test suite per mutation, so nothing runs it
often. That is exactly the condition under which its anchors rot: a guard gets reworded,
the anchor stops matching, and the finding quietly measures nothing. These tests are
cheap and check the part that goes stale.
"""

from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    return importlib.import_module("measure_guard_mutations")


def test_every_mutation_anchor_still_matches_exactly_once():
    module = _module()
    for mutation in module.MUTATIONS:
        text = (ROOT / mutation.path).read_text(encoding="utf-8")
        assert text.count(mutation.old) == 1, (
            f"{mutation.guard}: anchor appears {text.count(mutation.old)} times in "
            f"{mutation.path}. The mutation no longer describes the code, so finding 27 "
            "would measure nothing."
        )


def test_every_mutation_actually_changes_the_source():
    """An anchor equal to its replacement is a mutation that cannot fail."""
    for mutation in _module().MUTATIONS:
        assert mutation.old != mutation.new, mutation.guard


def test_the_refusal_code_is_distinguishable_from_a_verdict():
    """0 and 1 are answers; the refusal must not be mistaken for either.

    Same distinction `measure_blind_spot.py` draws: a precondition that failed is not a
    weak result, and a harness that reports it as one manufactures a finding.
    """
    module = _module()
    assert module.REFUSED_EXIT not in (0, 1)


def test_the_harness_excludes_this_file_while_measuring():
    """Otherwise every mutant dies here, and dies for the wrong reason.

    The anchor test above reads the source, which is mutated while the harness runs, so
    it fails by construction against every mutation. Left in scope it would have reported
    4 of 4 killed on the strength of the harness noticing its own edit.
    """
    module = _module()
    assert module.SELF_TEST.endswith("test_guard_mutations.py")
    assert (ROOT / module.SELF_TEST).exists(), (
        "the harness ignores a path that does not exist, so the exclusion is silently "
        "doing nothing and this file is back in scope"
    )


def test_the_scope_names_what_it_leaves_out():
    """The excluded module is named in the artifact, not just omitted from it."""
    source = (ROOT / "scripts" / "measure_guard_mutations.py").read_text(encoding="utf-8")
    assert "gate.py" in source, (
        "gate.py is excluded on cost; a scope that does not say what it excludes reads "
        "as a scope that covers everything"
    )
