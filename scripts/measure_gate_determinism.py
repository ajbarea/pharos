#!/usr/bin/env python3
"""Surface baselines at full precision, so two machines can be compared exactly.

The README claims the gate produces bit-identical surface baselines on a WSL laptop and
on an RHEL 9 cluster node with a different CPU count, kernel, and libc, and says the
property is the claim while the values move whenever the generator does. A claim of
bit-identity cannot be checked against rounded output: 0.6378 on two machines is
consistent with a difference in the eleventh decimal, which is exactly the kind of
difference a different BLAS or a different core count produces.

So this emits `float.hex()`, which is exact and unambiguous, alongside the decimal a
human can read. Comparing two runs is then a diff rather than a judgement call.

The gate is deterministic and model-free: it generates a corpus from a seed, fits the
surface probes under a fixed cross-validation split, and compares against a permutation
null. Nothing here calls a model or touches the network.

    uv run python scripts/measure_gate_determinism.py --out results/gate_determinism.json

Two machines, then `--compare`:

    uv run python scripts/measure_gate_determinism.py --compare a.json b.json
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

from pharos.export import corpus_bytes, sha256
from pharos.gate import run_gate
from pharos.generate import GeneratorConfig, generate
from pharos.provenance import run_provenance
from pharos.telemetry import get_logger, log_execution_context, record

#: The seeds the paper's gate table is built from, so this checks the published values
#: rather than a convenient subset of them.
SEEDS = (1, 7, 11, 23, 101, 202, 303)
EVENTS = 400

LOG = get_logger()


def measure() -> dict[str, Any]:
    """One gate run per seed, recorded exactly."""
    log_execution_context()
    baselines: dict[str, Any] = {}
    for seed in SEEDS:
        corpus = generate(GeneratorConfig(seed=seed, n_events=EVENTS))
        #: What the gate actually scored, so `--compare` can ask whether two machines saw
        #: the same corpus instead of inferring it from a commit id. A commit is a proxy
        #: and a bad one in both directions: a docstring fix changes it without changing
        #: the corpus, and the two artifacts here differed by exactly such a commit.
        digest = sha256(corpus_bytes(corpus))
        result = run_gate(corpus)
        # Coerced rather than assumed: the gate types this as `int | float`, and an
        # AUC of exactly 1 or 0 would arrive as an int, which has no `.hex()`. A
        # degenerate corpus is precisely when this script should still produce output.
        baseline = float(result.surface_baseline)
        baselines[str(seed)] = {
            #: Exact. Two machines agreeing here agree in every bit of the mantissa.
            "hex": baseline.hex(),
            "decimal": baseline,
            "corpus_sha256": digest,
        }
        record("gate.surface_baseline", baseline, seed=seed)
    return {
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            #: Two of the three things the README names as differing between the
            #: machines. The third, CPU count, is in the execution context.
            "libc": " ".join(platform.libc_ver()),
        },
        "n_events": EVENTS,
        "baselines": baselines,
        "provenance": run_provenance(n_seeds=len(SEEDS), n_events=EVENTS),
    }


def compare(left: Path, right: Path) -> int:
    """Exit non-zero when any seed disagrees in any bit.

    Reports every disagreement rather than the first. A single differing seed and all
    seven differing are different diagnoses -- one suggests a boundary case in the data,
    the other a systematically different numerical environment -- and stopping at the
    first would hide which.
    """
    a = json.loads(left.read_text(encoding="utf-8"))
    b = json.loads(right.read_text(encoding="utf-8"))

    # Refuse when the two machines scored different corpora, because then a difference
    # in baselines says nothing about the machines. Asked of the corpus digest rather
    # than the commit id: the commit is a proxy that is wrong in both directions, and it
    # was wrong here -- the first two artifacts differed by a commit that touched no
    # generator code at all, and refusing on that would have made the check unrunnable
    # in exactly the workflow its own instructions describe.
    # Missing digests are checked first, and separately. "Cannot tell whether the corpora
    # match" and "the corpora do not match" are different findings with different
    # remedies, and testing them in the other order reports the second for the first: one
    # artifact carrying a digest against one that does not compares a string to None,
    # which is unequal, which reads as a corpus difference nobody can act on.
    if not all("corpus_sha256" in v for v in (*a["baselines"].values(), *b["baselines"].values())):
        print(
            "refusing to compare: an artifact predates the corpus digest, so whether the\n"
            "  two machines scored the same corpus cannot be established. Regenerate both.",
            file=sys.stderr,
        )
        return 2
    differing = [
        seed
        for seed in sorted(set(a["baselines"]) & set(b["baselines"]), key=int)
        if a["baselines"][seed]["corpus_sha256"] != b["baselines"][seed]["corpus_sha256"]
    ]
    if differing:
        print(
            f"refusing to compare: the corpora differ at seed(s) {', '.join(differing)}.\n"
            "  The generator defines the corpus, so this compares two different things\n"
            "  and any difference in baselines would say nothing about the machines.\n"
            f"  {left.name} is at {a['provenance']['git_commit'][:12]}, "
            f"{right.name} at {b['provenance']['git_commit'][:12]}.",
            file=sys.stderr,
        )
        return 2

    print(f"{left.name}: {a['machine']['platform']}  python {a['machine']['python']}")
    print(f"{right.name}: {b['machine']['platform']}  python {b['machine']['python']}")
    print()

    mismatches = []
    for seed in sorted(set(a["baselines"]) | set(b["baselines"]), key=int):
        x = a["baselines"].get(seed)
        y = b["baselines"].get(seed)
        if x is None or y is None:
            mismatches.append((seed, "missing", "missing"))
            continue
        same = x["hex"] == y["hex"]
        print(f"  seed {seed:>3}  {x['decimal']!r:22} {'==' if same else '!='} {y['decimal']!r}")
        if not same:
            mismatches.append((seed, x["hex"], y["hex"]))

    print()
    if mismatches:
        print(f"{len(mismatches)} of {len(a['baselines'])} seeds differ:")
        for seed, x, y in mismatches:
            print(f"  seed {seed}: {x}  vs  {y}")
        LOG.warning(
            "gate.cross_machine_mismatch",
            extra={"event": "gate.cross_machine_mismatch", "n_mismatched": len(mismatches)},
        )
        return 1
    print(f"bit-identical on all {len(a['baselines'])} seeds")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("LEFT", "RIGHT"),
        help="Compare two artifacts and exit non-zero if any seed differs",
    )
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)

    payload = measure()
    print(f"{payload['machine']['platform']}  python {payload['machine']['python']}")
    for seed, value in payload["baselines"].items():
        print(f"  seed {seed:>3}  {value['decimal']!r:22}  {value['hex']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
