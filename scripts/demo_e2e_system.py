"""Smoke test across the pieces that only meet each other at runtime.

Generator, ledger, router, disclosure, and the DP and attack primitives each have unit
tests. What none of those cover is whether they still compose: a `Report` reaching
`build_triage_tasks`, a review becoming a ledger record, that record routing to an
adapter, and the explorer's API answering with the same vocabulary the lattice defines.
This walks that path once and checks at each step.

It is a test, not a demonstration, and the distinction is why it was rewritten. The
previous version printed a tick per stage unconditionally -- including for the HTTP
probe, which caught its own exception, printed a warning, and then reported
"ALL COMPONENTS FUNCTIONAL" anyway. Output fixed by construction is not verification,
and this repository has had enough of that in one week.

The live server is optional and its absence is not a failure: `pharos.cli serve` is not
running in CI. Every other stage is required, and a failure exits non-zero.

    uv run python scripts/demo_e2e_system.py
"""

import json
import urllib.request

from pharos.analyst import DEFAULT_CEILING, DEFAULT_ENSEMBLE, Proposal
from pharos.disclosure import DEFAULT_RELEASE_POLICY, admit
from pharos.fl import PrivacyBudget, add_gaussian_dp_noise, apply_sign_flip
from pharos.generate import GeneratorConfig, generate
from pharos.labels import Sensitivity
from pharos.ledger import DecisionLedger, ProvenanceRouter, RoutingTarget, record_from_review
from pharos.tasks import build_triage_tasks

EXPLORER_URL = "http://127.0.0.1:8080/api/vocabulary"


def check(condition: object, message: str) -> None:
    """Raise when `condition` is falsy.

    Not `assert`: `python -O` strips assert statements, and a script whose whole
    purpose is to fail on a broken invariant must not become a script that prints
    "ok" under an interpreter flag.
    """
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def main() -> int:
    """Walk the composed path. Returns 0 when every required stage held."""
    print("pharos end-to-end smoke test")

    cfg = GeneratorConfig(seed=7, n_events=40)
    reports = generate(config=cfg)
    tasks = build_triage_tasks(reports, limit=10)
    check(reports, "the generator produced no reports")
    check(tasks, "no triage tasks were built from a non-empty corpus")
    print(f"  generator   {len(reports)} reports over 40 events -> {len(tasks)} triage tasks")

    ledger = DecisionLedger()
    router = ProvenanceRouter()
    routed: dict[RoutingTarget, int] = {}
    analyst_policy = DEFAULT_ENSEMBLE[0]

    for task in tasks:
        proposal = Proposal(task_id=task.task_id, verdict=task.significant, release=task.label)
        decision = admit(proposal.release, DEFAULT_CEILING, DEFAULT_RELEASE_POLICY)
        review = analyst_policy.review(task, proposal, seed=7)
        record = record_from_review(
            task.task_id,
            seed=7,
            proposal=proposal,
            decision=decision,
            review=review,
            truth_significant=task.significant,
        )
        ledger.append(record)
        routed[router.route(record).target] = routed.get(router.route(record).target, 0) + 1

    check(len(ledger) == len(tasks), f"{len(tasks)} reviews produced {len(ledger)} records")
    digest = ledger[0].digest()
    check(len(digest) == 64, f"a SHA-256 digest is 64 hex characters, got {len(digest)}")
    check(routed, "the router assigned no record to any target")
    print(f"  ledger      {len(ledger)} records, first digest {digest[:16]}...")
    for target, count in sorted(routed.items(), key=lambda kv: kv[0].name):
        print(f"  router      {target.name}: {count}")

    # The point of the DP and attack primitives is that they *change* the update. A
    # mechanism that returns its input is the failure this asserts against, and it is
    # not hypothetical: a sign flip with severity 0 and a budget with infinite epsilon
    # both reduce to the identity.
    weights = [0.42, 0.65, -0.12, 0.89, 0.33]
    noisy = add_gaussian_dp_noise(weights, PrivacyBudget(epsilon=1.0, delta=1e-5))
    poisoned = apply_sign_flip(weights, severity=1.0)
    check(noisy != weights, "DP noise left the update unchanged")
    check(poisoned != weights, "the sign flip left the update unchanged")
    check(
        all(a * b <= 0 for a, b in zip(weights, poisoned, strict=True)),
        "a sign flip must invert every coordinate",
    )
    print(f"  dp/attack   noise and sign-flip both altered a {len(weights)}-vector")

    try:
        with urllib.request.urlopen(EXPLORER_URL, timeout=2) as fh:
            vocabulary = json.loads(fh.read().decode("utf-8"))
    except Exception as exc:
        print(f"  explorer    not running, skipped ({type(exc).__name__})")
    else:
        levels = vocabulary["sensitivity"]
        check(
            len(levels) == len(Sensitivity),
            f"the API offers {len(levels)} sensitivity levels, "
            f"the lattice defines {len(Sensitivity)}",
        )
        print(f"  explorer    /api/vocabulary agrees with the lattice on {len(levels)} levels")

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
