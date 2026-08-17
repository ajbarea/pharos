"""Guards on the workflow files themselves.

Nothing else checks them. A workflow is not imported, not covered, and not exercised by
any other test here, so a step that quietly stops doing what its comment claims is
invisible until someone reads the YAML -- which is how the Pages deploy spent months
passing a `timeout` the action had always clamped away, announcing it in a warning line
on every run.
"""

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

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
        if "actions/checkout" in step.get("uses", "")
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
    floating = [
        step["uses"]
        for _, step in _steps(workflow)
        if "uses" in step and "@" in step["uses"] and len(step["uses"].split("@")[1]) != 40
    ]
    assert not floating, (
        f"{workflow.name} uses actions pinned to something other than a full commit sha: {floating}"
    )


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_workflow_states_its_permissions(workflow):
    """An unstated `permissions:` block takes the repository default, which can be write.

    Stating it is what makes the grant reviewable: `contents: read` at the top of a file
    is a claim a reader can check against what the jobs actually do.
    """
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    jobs = doc.get("jobs") or {}
    assert "permissions" in doc or all("permissions" in job for job in jobs.values()), (
        f"{workflow.name} states no `permissions:`, at the top level or on every job, so "
        "its token scope is whatever the repository default happens to be"
    )
