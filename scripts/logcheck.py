#!/usr/bin/env python3
"""Run the model-free measurements and summarise what they logged.

Logs are only useful if someone reads them, and a log nobody reads decays without
anyone noticing. This repository's did: the fleet-linkage script once emitted thirty
identical corpus-generation lines and nothing whatsoever about the attack it ran, and
the habit that grew around it was to filter the log stream out entirely.

So this makes reading them one command. It runs every measurement that needs no model
and no network, groups the structured events by level, and prints warnings in full
because those are the ones that mean something is off.

What "clean" looks like here is not silence. The warnings below are *expected*: three
headline claims that their samples cannot resolve, one tagging scheme whose aggregate
leak the per-analyst metric cannot see, and the consensus cliff. Each is a real finding
that the code is correctly reporting. What would be suspicious is a warning that is not
on that list, or the disappearance of one that is.

    uv run python scripts/logcheck.py
    uv run python scripts/logcheck.py --debug   # include routine inner operations
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Measurements that run anywhere: no model server, no GPU, no network.
SCRIPTS = (
    "measure_analyst_review.py",
    "measure_review_sweep.py",
    "measure_power.py",
    "measure_fleet_linkage.py",
    "measure_consensus_reliability.py",
    "measure_tagged_aggregation.py",
    "measure_privacy_budget.py",
    "measure_correlated_fleets.py",
    "measure_difficulty_confound.py",
    "measure_secure_reliability.py",
    "measure_authority_anchors.py",
    "measure_audit_policy.py",
    "measure_blind_spot.py",
    "measure_channel_bias.py",
    "measure_selective_risk.py",
    "measure_error_shape.py",
    "measure_latent_blindspot.py",
    "measure_estimator_initialization.py",
)

#: The one warning family this command cannot expect to see, because it fires on
#: model-backed measurements the list above deliberately excludes.
VALIDITY_PREFIX = "validity."

#: Event families raised inside the library rather than by the measurement that provokes
#: them. A scoped tolerance for one of these is earned by the script driving that regime,
#: so the event name will not appear in the script's own source and reachability has to
#: stop looking for it there.
LIBRARY_RAISED = ("inference.",)

#: Warnings this project expects to see, each a real finding rather than a defect.
#: A warning absent from here is worth looking at; one of these going missing is worth
#: looking at harder, because it means a finding stopped reporting itself.
EXPECTED_WARNINGS = {
    "power.claim_unresolved",
    "tagging.aggregate_leak_invisible",
    "consensus.cliff",
    "budget.no_finite_guarantee",
    "difficulty.manufactured_by_reviewers",
    "difficulty.ability_inverted",
    # A selection policy beating the uniform floor is the finding, not a defect, so it
    # is announced at WARNING and expected here. If it stops firing, either a corpus
    # change made selection worthless or the comparison broke.
    "audit.policy_beats_uniform",
    # Finding 21's result: outside its stated condition the targeted policy stops
    # helping. Expected, because the experiment exists to produce it.
    "blindspot.policy_advantage_inverted",
    # Finding 19's limit: the unanimous fleet is not repaired by any budget the sweep
    # can afford. This did not fire while the draw was re-sampled per budget, because
    # one draw cleared the bar at 180 of 200 anchors on the twenty tasks left to score.
    # Across 21 nested draws none does, which is what the unanimity row was carried to
    # show. It going quiet again would mean a composition started repairing that the
    # finding says cannot.
    "authority.not_repaired",
    # Finding 28's bound, and the reason that run is not silent: at the noise level the
    # random-error control needs, the estimator stops converging. The cells it affects
    # are named in the artifact and no claim rests on them, so this is a flag travelling
    # with a number rather than a defect. It going quiet would mean the control was no
    # longer being measured at a rate that breaks a healthy fleet.
    "selective_risk.estimator_did_not_converge",
    # Finding 25's result: an initialisation escape exists at the crossing composition.
    # Expected because the sweep was built to find out whether one does, and it does.
    # If it stops firing, no start beats the majority vote anywhere -- a stronger result
    # than the one published, and one that should retire finding 25's caveat rather than
    # pass silently.
    "estimator_initialization.escape_found",
    # `inference.glad_did_not_converge` is deliberately NOT here. It was, for about an
    # hour, while this project believed non-convergence was a property of GLAD. It is
    # not: the implementation was missing the Gaussian priors Whitehill et al. specify
    # in their section 3.1, and with them every composition settles in under 40
    # iterations. Listing it as expected would now hide a real regression.
    # Everything below fires only on model-backed measurements, which this command does
    # not run. `core_missing` exempts the family by this prefix, so a new validity
    # warning needs no second edit -- and anything outside the family is checked.
    "validity.small_n",
    "validity.below_majority",
    "validity.degenerate_predictions",
    "validity.high_unparsed",
    "validity.over_escalation",
    "validity.saturated",
    "validity.skewed_classes",
    "validity.empty",
}


#: Warnings expected from ONE script and nowhere else, keyed by that script.
#:
#: Scoped rather than added to the set above, because the set above is global and a
#: global expectation is how a real regression gets filed as routine. The comment on
#: `inference.glad_did_not_converge` records that exact near-miss: non-convergence was
#: briefly believed to be a property of the estimator and was in fact a missing prior.
#: `measure_selective_risk.py` is the one measurement that deliberately runs a fleet
#: noisy enough to break the EM fit -- it is the control the abstention claims are
#: bounded by -- and the cells affected are named in its artifact. The same warning from
#: any other script is still unexpected, which is the point of the scoping.
SCRIPT_SCOPED_WARNINGS = {
    "measure_selective_risk.py": frozenset({"inference.federated_em_did_not_converge"}),
    #: Finding 29 runs the same high-noise cells for the same reason -- they are the
    #: regime whose diagnosis is the question -- and its own miss warning is expected
    #: while the dispersion index gets two near-tie cells wrong.
    "measure_error_shape.py": frozenset(
        {"inference.federated_em_did_not_converge", "error_shape.rule_choice_missed"}
    ),
    #: Finding 30's sweep refuses a slice draw whose carriage is lopsided and carries on
    #: with a smaller denominator, publishing the per-row draw count so the reader can see
    #: it. That path was tolerated in the script and not here, so the one refusal it is
    #: designed to absorb would have failed the gate under "unexpected warning" -- a
    #: deliberate tolerance and a guard disagreeing about the same event. It stays scoped:
    #: the same warning from anywhere else is still unexpected. It is also not in the
    #: global set above, because it fires only on a lopsided draw and demanding it fire
    #: would make a well-behaved corpus look broken.
    "measure_latent_blindspot.py": frozenset({"latent_blindspot.size_refused"}),
}


def verify_scoped_exemptions() -> list[str]:
    """Check that every scoped tolerance still names something real.

    `EXPECTED_WARNINGS` is checked in both directions: an unexpected one fails, and one
    that stops firing fails too, because a finding that quietly stops reporting itself
    looks exactly like a clean run. `SCRIPT_SCOPED_WARNINGS` had neither. It is the more
    dangerous list of the two, because a scoped entry widens what one script may say
    without anybody being told when the reason expires.

    Firing is the wrong thing to require of these -- they are regime-dependent, and
    `latent_blindspot.size_refused` fires only on a corpus that refuses a draw, which the
    committed one does not. So the check is reachability: the script must exist, must be
    one this command runs, and must still contain the event it is excused for. An entry
    naming an event no script emits is a tolerance with nothing behind it, sitting ready
    to absorb the next thing that happens to use that name.
    """
    problems: list[str] = []
    for script, events in SCRIPT_SCOPED_WARNINGS.items():
        path = ROOT / "scripts" / script
        if script not in SCRIPTS:
            problems.append(f"{script} is scoped but is not in SCRIPTS, so it never runs here")
            continue
        if not path.is_file():
            problems.append(f"{script} is scoped but no longer exists")
            continue
        source = path.read_text(encoding="utf-8")
        for event in sorted(events):
            # The emitting script, or the library it calls. `inference.*` is raised inside
            # `pharos.inference` and only travels through the measurement, so a scoped
            # entry for it is earned by the script provoking that regime rather than by
            # the string appearing in its file.
            reachable = event in source or event.startswith(LIBRARY_RAISED)
            if not reachable:
                problems.append(
                    f"{script} is excused for {event!r}, which it no longer emits. "
                    "Remove the entry, or the exemption is covering something else now."
                )
    return problems


def run(script: str, *, debug: bool) -> tuple[list[dict[str, object]], int]:
    """Run one script and return its structured log records and exit code."""
    env_level = "DEBUG" if debug else "INFO"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts" / script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PHAROS_LOG_LEVEL": env_level},
        check=False,
    )
    records: list[dict[str, object]] = []
    for line in (completed.stdout + completed.stderr).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "logger" in parsed:
            records.append(parsed)
    return records, completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="include routine DEBUG metrics")
    args = parser.parse_args()

    total: Counter[str] = Counter()
    unexpected: list[tuple[str, dict[str, object]]] = []
    seen_warnings: set[str] = set()
    failures: list[str] = []

    # Before running anything, because a stale tolerance is cheaper to report than to
    # diagnose and this needs no measurement to decide.
    stale_exemptions = verify_scoped_exemptions()

    #: Which scoped tolerances actually suppressed something this run. Ruff's `RUF100`
    #: is the standard here -- a suppression is valid only if the thing it excuses is
    #: really triggered -- and a tolerance that suppresses nothing is the shape a stale
    #: one has. It is reported rather than failed, because some of these are legitimately
    #: regime-dependent: `latent_blindspot.size_refused` fires only on a corpus that
    #: refuses a draw, and the committed one does not. Failing on unused would delete a
    #: correct entry; saying nothing lets an expired one sit forever. Naming it is the
    #: re-review that security tooling schedules on a timer, run every time instead.
    used_scope: dict[str, set[str]] = {}

    for script in SCRIPTS:
        records, code = run(script, debug=args.debug)
        if code != 0:
            failures.append(f"{script} exited {code}")
        levels: Counter[str] = Counter(str(r.get("level", "?")) for r in records)
        events: Counter[str] = Counter(
            str(r.get("metric") or r.get("event") or "?") for r in records
        )
        summary = ", ".join(f"{lvl.lower()}={n}" for lvl, n in sorted(levels.items()))
        print(f"\n{script}  ({len(records)} records: {summary or 'none'})")
        for name, count in events.most_common():
            print(f"    {name} x{count}")
        for record in records:
            if record.get("level") not in {"WARNING", "ERROR", "CRITICAL"}:
                continue
            name = str(record.get("event") or record.get("metric") or "?")
            seen_warnings.add(name)
            if name in SCRIPT_SCOPED_WARNINGS.get(script, frozenset()):
                used_scope.setdefault(script, set()).add(name)
            if name not in EXPECTED_WARNINGS and name not in SCRIPT_SCOPED_WARNINGS.get(
                script, frozenset()
            ):
                unexpected.append((script, record))
        total.update(levels)

    print("\n" + "=" * 68)
    print("totals: " + ", ".join(f"{k.lower()}={v}" for k, v in sorted(total.items())))

    if failures:
        print("\nSCRIPTS THAT FAILED:")
        for f in failures:
            print(f"  {f}")

    if stale_exemptions:
        print("\nSCOPED TOLERANCES THAT NO LONGER NAME ANYTHING:")
        for problem in stale_exemptions:
            print(f"  {problem}")

    dormant = sorted(
        (script, event)
        for script, events in SCRIPT_SCOPED_WARNINGS.items()
        for event in events
        if event not in used_scope.get(script, set())
    )
    if dormant:
        print("\nscoped tolerances that suppressed nothing this run -- re-read the reason:")
        for script, event in dormant:
            print(f"  {script}: {event}")

    if unexpected:
        print("\nUNEXPECTED WARNINGS -- these are the ones to look at:")
        for script, record in unexpected:
            name = str(record.get("event") or record.get("metric"))
            detail = {
                k: v
                for k, v in record.items()
                if k not in {"timestamp", "level", "logger", "message", "event", "metric"}
            }
            print(f"  {script}: {name} {detail}")
    else:
        print("\nno unexpected warnings")

    missing = EXPECTED_WARNINGS - seen_warnings
    # Only the ones these scripts can emit; the validity family fires on model-backed
    # measurements that this command deliberately does not run.
    #
    # Derived by subtraction rather than listed again. This used to be a second literal
    # set naming each warning a second time, and it drifted the first time it was
    # extended: `authority.not_repaired` was added to EXPECTED_WARNINGS and not here, so
    # the one finding whose limit is that it never repairs would have stopped announcing
    # that limit without failing anything. A second hand-maintained copy of a list is a
    # list that will disagree with the first; the exemption is a property of the
    # `validity.` family, so name the property.
    core_missing = {w for w in missing if not w.startswith(VALIDITY_PREFIX)}
    if core_missing:
        print(
            "\nEXPECTED WARNINGS THAT DID NOT FIRE -- a finding may have stopped reporting itself:"
        )
        for name in sorted(core_missing):
            print(f"  {name}")

    print("=" * 68)
    return 1 if (failures or unexpected or core_missing or stale_exemptions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
