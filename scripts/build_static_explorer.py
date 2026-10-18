#!/usr/bin/env python3
"""Freeze the explorer into files GitHub Pages can serve, with no backend.

Every view but one -- corpus, lattice, gate, analyst review, and the exits -- is a
deterministic function of a seed and a few small inputs, so none of them needs a
live server. The model view does, because it calls a model; in a frozen copy it
says so and shows how to run the explorer locally, rather than failing.

Why freeze rather than host. A university cluster cannot serve a public API: login
nodes forbid long-running processes and compute nodes are scheduler-allocated,
walltime-limited, and unroutable. Every other option is a service that has to stay
alive, funded, and reachable for as long as anyone might follow a link to it. A
static bundle has none of those failure modes, which for a resource paper is worth
more than live arbitrary-seed generation.

**Nothing here reimplements Pharos.** Every value in the bundle is produced by
calling the same endpoint functions the live app serves, so the page stays a client
rather than a second implementation. The page itself is copied unmodified; a small
shim already in it routes `api()` through the bundle when one is present.

The cost is that only the seeds listed here work in the browser. The page says so,
and points at the CLI for anything else.

    uv run python scripts/build_static_explorer.py --out docs/explorer
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from pharos.provenance import run_provenance
from pharos.web import STATIC, create_app

#: Seeds frozen into the bundle. The CI gate list, trimmed: every extra seed
#: multiplies the review section, which is the largest part of the payload.
SEEDS: tuple[int, ...] = (1, 7, 101)

#: Matches the page's own defaults, so the first click needs no waiting and no
#: explanation of why this seed works and that one does not.
CORPUS_EVENTS = 40
GATE_EVENTS = 120
GATE_NULL_TRIALS = 8


def build_bundle(app: Any) -> dict[str, Any]:
    """Call every model-free endpoint over the frozen input space.

    Routes are read off the app rather than imported by name, so an endpoint that
    is renamed or removed fails here loudly instead of silently dropping a tab.
    """
    routes = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        endpoint = getattr(route, "endpoint", None)
        if path and endpoint:
            routes[path] = endpoint

    missing = {
        "/api/vocabulary",
        "/api/corpus",
        "/api/gate",
        "/api/review/all",
        "/api/lattice",
        "/api/escapes",
    } - set(routes)
    if missing:
        raise RuntimeError(f"pharos.web no longer serves {sorted(missing)}; update this script")

    corpus = {str(seed): routes["/api/corpus"](seed=seed, events=CORPUS_EVENTS) for seed in SEEDS}
    gate = {
        str(seed): routes["/api/gate"](seed=seed, events=GATE_EVENTS, null_trials=GATE_NULL_TRIALS)
        for seed in SEEDS
    }
    # Both proposed verdicts for every task, keyed the way the page asks for them.
    review = {
        str(seed): {
            "true": routes["/api/review/all"](seed=seed, verdict=True)["tasks"],
            "false": routes["/api/review/all"](seed=seed, verdict=False)["tasks"],
        }
        for seed in SEEDS
    }

    return {
        "provenance": run_provenance(),
        # The argument, not just the mechanism. Frozen like everything else here: it
        # reads `results/` at build time, so a page served from Pages shows whatever
        # the artifacts said at the commit that built it, named in the provenance.
        "escapes": routes["/api/escapes"](),
        "seeds": list(SEEDS),
        "corpus_events": CORPUS_EVENTS,
        "gate_events": GATE_EVENTS,
        "gate_null_trials": GATE_NULL_TRIALS,
        "vocabulary": routes["/api/vocabulary"](),
        "corpus": corpus,
        "gate": gate,
        "review": review,
        "dominance": routes["/api/lattice"](),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/explorer"),
        help="directory to write index.html and bundle.json into",
    )
    args = parser.parse_args()

    app = create_app()
    bundle = build_bundle(app)

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STATIC / "index.html", args.out / "index.html")
    payload = json.dumps(bundle, separators=(",", ":"), sort_keys=True)
    (args.out / "bundle.json").write_text(payload, encoding="utf-8")

    kb = len(payload.encode("utf-8")) / 1024
    print(f"wrote {args.out / 'index.html'}")
    print(f"wrote {args.out / 'bundle.json'}  ({kb:,.0f} KiB, {len(SEEDS)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
