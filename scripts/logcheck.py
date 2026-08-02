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
)

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
    # `inference.glad_did_not_converge` is deliberately NOT here. It was, for about an
    # hour, while this project believed non-convergence was a property of GLAD. It is
    # not: the implementation was missing the Gaussian priors Whitehill et al. specify
    # in their section 3.1, and with them every composition settles in under 40
    # iterations. Listing it as expected would now hide a real regression.
    "validity.small_n",
    "validity.below_majority",
    "validity.degenerate_predictions",
    "validity.high_unparsed",
    "validity.over_escalation",
    "validity.saturated",
    "validity.skewed_classes",
    "validity.empty",
}


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
            if name not in EXPECTED_WARNINGS:
                unexpected.append((script, record))
        total.update(levels)

    print("\n" + "=" * 68)
    print("totals: " + ", ".join(f"{k.lower()}={v}" for k, v in sorted(total.items())))

    if failures:
        print("\nSCRIPTS THAT FAILED:")
        for f in failures:
            print(f"  {f}")

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
    core_missing = missing & {
        "power.claim_unresolved",
        "tagging.aggregate_leak_invisible",
        "consensus.cliff",
        "budget.no_finite_guarantee",
        "difficulty.manufactured_by_reviewers",
        "difficulty.ability_inverted",
    }
    if core_missing:
        print(
            "\nEXPECTED WARNINGS THAT DID NOT FIRE -- a finding may have stopped reporting itself:"
        )
        for name in sorted(core_missing):
            print(f"  {name}")

    print("=" * 68)
    return 1 if (failures or unexpected or core_missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
