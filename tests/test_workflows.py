"""Guards on the workflow files themselves.

Nothing else checks them. A workflow is not imported, not covered, and not exercised by
any other test here, so a step that quietly stops doing what its comment claims is
invisible until someone reads the YAML -- which is how the Pages deploy spent months
passing a `timeout` the action had always clamped away, announcing it in a warning line
on every run.
"""

import pathlib
import re

import pytest
import yaml

#: A full commit sha, and a container image digest. Hex-checked rather than
#: length-checked, so a 40-character branch name is not mistaken for a pin.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: Both extensions, because GitHub accepts both and a guard that reads only one is a
#: guard a new workflow can be added past. A `.yaml` file here would otherwise be checked
#: by nothing while `test_there_are_workflows_to_check` stayed green on the others.
WORKFLOWS = sorted(
    p for ext in ("*.yml", "*.yaml") for p in (ROOT / ".github" / "workflows").glob(ext)
)

#: Checkouts that must keep their credentials, each with the reason. A job that pushes,
#: tags, or fetches another ref during the run needs the token in `.git/config`; nothing
#: here does today, and an entry added without a reason is the thing this guard is for.
CHECKOUTS_NEEDING_CREDENTIALS: dict[tuple[str, str], str] = {}


def _steps(workflow: pathlib.Path):
    """`(job name, step)` for every step in a workflow."""
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            yield job_name, step


def _action_refs(workflow: pathlib.Path):
    """`(job name, uses)` for every action reference, at either level.

    A job may carry `uses:` itself instead of `steps:` -- that is a reusable-workflow
    call, and it executes another repository's workflow with this repository's token.
    `_steps` never descends into one, so a `@main` there was checked by nothing while
    every step-level reference beside it was required to be a full sha.
    """
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    for job_name, job in (doc.get("jobs") or {}).items():
        if "uses" in job:
            yield job_name, job["uses"]
        for step in job.get("steps") or []:
            if "uses" in step:
                yield job_name, step["uses"]


def test_there_are_workflows_to_check():
    """The guards below compare against what they find, so finding nothing must fail."""
    assert WORKFLOWS, "no workflow files were parsed, so every check in this file is vacuous"


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_checkout_drops_a_token_it_does_not_use(workflow):
    """`persist-credentials` defaults to true, so the omission is silent and is the default.

    The token stays in `.git/config` for every later step to read, and the steps that
    follow a checkout here resolve and execute third-party package code. `docs.yml` set
    this from the start; `ci.yml` did not, and the scheduled sibling job repeated the
    omission when it was written by copying the shape of a job rather than its guards --
    which is the recurrence this asserts over the whole directory rather than fixing in
    the two files that had it wrong.
    """
    unhardened = [
        job
        for job, step in _steps(workflow)
        # `or ""` rather than a default: a bare `uses:` key parses to None, and `in None`
        # raises rather than asserting. The pin guard beside this one already handles it.
        if "actions/checkout" in (step.get("uses") or "")
        and (step.get("with") or {}).get("persist-credentials") is not False
        and (workflow.name, job) not in CHECKOUTS_NEEDING_CREDENTIALS
    ]
    assert not unhardened, (
        f"{workflow.name} checks out with credentials persisted in: {unhardened}. Set "
        "`persist-credentials: false`, or name the job in CHECKOUTS_NEEDING_CREDENTIALS "
        "with the git operation that needs the token."
    )


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit(workflow):
    """A tag is a moving target, and moving a tag is how a supply chain attack arrives.

    Already true of every action here. Asserted so that it stays true of the next one,
    since a `@v4` reads as perfectly ordinary in review.
    """
    floating = []
    for _, uses in _action_refs(workflow):
        if not uses or uses.startswith("./"):
            # A path into this repository is versioned by the commit being run. There is
            # nothing to pin it to.
            continue
        _, _, ref = uses.partition("@")
        # No `@` at all is the case an earlier version of this test waved through: a
        # `docker://image:tag` step is a moving target with no ref to check, so an
        # unpinnable `uses` is unpinned rather than exempt. A `docker://` step pinned by
        # digest is the opposite case and has to pass: `sha256:` plus 64 hex is the
        # strongest form GitHub documents for one, and there is no 40-character
        # equivalent to rewrite it into.
        if not _COMMIT_SHA.fullmatch(ref) and not _IMAGE_DIGEST.fullmatch(ref):
            floating.append(uses)
    assert not floating, (
        f"{workflow.name} uses actions pinned to something other than a full commit sha: {floating}"
    )


def test_the_pin_guard_sees_a_reusable_workflow_call(tmp_path):
    """A job-level `uses:` is an action reference too, and it was invisible to the guard.

    No workflow here calls one today, so this is the shape of the next one being added --
    the same shape as the `.yaml` extension and the `docker://` step: an entry point the
    guard did not look at, staying green while the thing it forbids is true.
    """
    caller = tmp_path / "caller.yml"
    caller.write_text(
        "name: caller\n"
        "on: push\n"
        "permissions:\n"
        "  contents: read\n"
        "jobs:\n"
        "  call:\n"
        "    uses: some-org/some-repo/.github/workflows/build.yml@main\n",
        encoding="utf-8",
    )
    assert [uses for _, uses in _action_refs(caller)] == [
        "some-org/some-repo/.github/workflows/build.yml@main"
    ]
    with pytest.raises(AssertionError, match="pinned to something other than a full commit sha"):
        test_every_action_is_pinned_to_a_commit(caller)


@pytest.mark.parametrize(
    ("ref", "floating"),
    [
        ("actions/checkout@" + "a" * 40, False),
        ("docker://alpine@sha256:" + "b" * 64, False),
        ("docker://alpine:3.20", True),
        ("actions/checkout@v4", True),
        # 40 characters, and a branch. Length alone would have read this as a pin.
        ("actions/checkout@" + "z" * 40, True),
    ],
)
def test_the_pin_guard_accepts_a_digest_and_rejects_a_moving_ref(tmp_path, ref, floating):
    """A digest-pinned container step has no 40-character form to be rewritten into.

    Rejecting one told its author to do something impossible, which is how a guard that
    is right about the rule ends up wrong about the case.
    """
    workflow = tmp_path / "probe.yml"
    workflow.write_text(
        "name: probe\non: push\npermissions:\n  contents: read\n"
        f"jobs:\n  probe:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: {ref}\n",
        encoding="utf-8",
    )
    if floating:
        with pytest.raises(AssertionError, match="pinned to something other than"):
            test_every_action_is_pinned_to_a_commit(workflow)
    else:
        test_every_action_is_pinned_to_a_commit(workflow)


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_workflow_states_its_permissions(workflow):
    """An unstated `permissions:` block takes the repository default, which can be write.

    Stating it is what makes the grant reviewable: `contents: read` at the top of a file
    is a claim a reader can check against what the jobs actually do.
    """
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    jobs = doc.get("jobs") or {}
    # `all()` over no jobs is True, so a workflow that parses to nothing would satisfy the
    # assertion below without stating anything.
    assert jobs, f"{workflow.name} declares no jobs, so it was checked for nothing"
    assert "permissions" in doc or all("permissions" in job for job in jobs.values()), (
        f"{workflow.name} states no `permissions:`, at the top level or on every job, so "
        "its token scope is whatever the repository default happens to be"
    )
