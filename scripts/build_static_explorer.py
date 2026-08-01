#!/usr/bin/env python3
"""Freeze the explorer into files GitHub Pages can serve, with no backend.

The explorer's four model-free tabs -- corpus, lattice, gate, analyst review --
are deterministic functions of a seed and a few small inputs. Nothing about them
needs a live server; they need a server today only because that is how the page
was first built. The fifth tab does need one, because it calls a model, and it is
disabled here rather than left to fail.

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
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

from pharos.labels import Capacity, Compartment, Sensitivity
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

#: Task indexes offered per seed in the review tab. One triage task per event, so
#: this is every task a 40-event corpus produces.
REVIEW_TASKS = 40


def _compartment_subsets() -> list[list[str]]:
    """Every compartment set, which is what makes the lattice a lattice.

    Enumerated rather than sampled: the interesting cells are the incomparable
    ones, and those are exactly the pairs a sampler would be most likely to miss.
    """
    members = list(Compartment)
    return [
        sorted(str(c) for c in combination)
        for size in range(len(members) + 1)
        for combination in itertools.combinations(members, size)
    ]


def _dominance_key(sensitivity: str, compartments: list[str], capacity: str | None = None) -> str:
    inner = ",".join(compartments)
    return f"{sensitivity}|{inner}" + (f"|{capacity}" if capacity else "")


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
        "/api/review",
        "/api/dominance",
    } - set(routes)
    if missing:
        raise RuntimeError(f"pharos.web no longer serves {sorted(missing)}; update this script")

    vocabulary = routes["/api/vocabulary"]()

    corpus = {str(seed): routes["/api/corpus"](seed=seed, events=CORPUS_EVENTS) for seed in SEEDS}
    gate = {
        str(seed): routes["/api/gate"](seed=seed, events=GATE_EVENTS, null_trials=GATE_NULL_TRIALS)
        for seed in SEEDS
    }

    review: dict[str, Any] = {}
    for seed in SEEDS:
        per_seed: dict[str, Any] = {}
        for index in range(REVIEW_TASKS):
            per_seed[str(index)] = {
                "true": routes["/api/review"](seed=seed, index=index, verdict=True),
                "false": routes["/api/review"](seed=seed, index=index, verdict=False),
            }
        review[str(seed)] = per_seed

    # Capacity is not part of the pair key, because the lattice does not consult it.
    # Dominance compares levels and compartments; the join's own capacity is the
    # item's, echoed rather than derived. Keying on all four capacities stored the
    # same 4,096 answers four times over and cost 8 MB to say nothing new.
    #
    # So two tables. `labels` holds every rendered label payload the page can
    # display, so no label string is ever formatted in JavaScript. `pairs` holds
    # what `pharos.labels` actually decided about each pair. `capacity_follows`
    # tells the shim which side the join's capacity comes from, so that rule stays
    # a fact emitted by Python rather than a second copy of it in the page.
    subsets = _compartment_subsets()
    labels: dict[str, Any] = {}
    for sensitivity in Sensitivity:
        for compartments in subsets:
            for capacity in Capacity:
                probe = routes["/api/dominance"](
                    {
                        "holder": {
                            "sensitivity": sensitivity.name,
                            "compartments": compartments,
                            "capacity": capacity.name,
                        },
                        "item": {
                            "sensitivity": sensitivity.name,
                            "compartments": compartments,
                            "capacity": capacity.name,
                        },
                    }
                )
                labels[_dominance_key(sensitivity.name, compartments, capacity.name)] = probe[
                    "holder"
                ]

    pairs: dict[str, Any] = {}
    for holder_sensitivity in Sensitivity:
        for holder_compartments in subsets:
            holder = {
                "sensitivity": holder_sensitivity.name,
                "compartments": holder_compartments,
                "capacity": Capacity.FREETEXT.name,
            }
            inner: dict[str, Any] = {}
            for item_sensitivity in Sensitivity:
                for item_compartments in subsets:
                    item = {
                        "sensitivity": item_sensitivity.name,
                        "compartments": item_compartments,
                        "capacity": Capacity.FREETEXT.name,
                    }
                    answer = routes["/api/dominance"]({"holder": holder, "item": item})
                    join = answer["join"]
                    inner[_dominance_key(item_sensitivity.name, item_compartments)] = {
                        "holder_may_read_item": answer["holder_may_read_item"],
                        "item_may_read_holder": answer["item_may_read_holder"],
                        "incomparable": answer["incomparable"],
                        "join": _dominance_key(join["sensitivity"], join["compartments"]),
                    }
            pairs[_dominance_key(holder_sensitivity.name, holder_compartments)] = inner

    dominance = {"capacity_follows": "item", "labels": labels, "pairs": pairs}

    return {
        "provenance": run_provenance(),
        "seeds": list(SEEDS),
        "corpus_events": CORPUS_EVENTS,
        "gate_events": GATE_EVENTS,
        "gate_null_trials": GATE_NULL_TRIALS,
        "review_tasks": REVIEW_TASKS,
        "vocabulary": vocabulary,
        "corpus": corpus,
        "gate": gate,
        "review": review,
        "dominance": dominance,
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
