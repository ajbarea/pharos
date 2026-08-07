#!/usr/bin/env python3
"""Whether the cliff is a property of the estimator or of where the estimator was started.

Findings 12 and 17 through 24 all rest on one behaviour: once enough of the fleet holds
the wrong standard, Dawid-Skene reports the wrong standard as the truth. `inference.py`
says in its own module docstring that EM started from the majority vote "is drawn toward
whatever the majority believes", and every published number was measured from exactly
that start.

That is a researcher degree of freedom sitting underneath the headline result, and it is
the one a reviewer reaches for first. The Dawid-Skene log-likelihood is non-convex;
majority-vote initialisation carries no global-optimality guarantee, and the standard
remedies are older than this project. Zhang, Chen, Zhou and Jordan (JMLR 17, 2016;
arXiv:1406.3824) initialise from a spectral method of moments and prove optimal
convergence for the two-stage estimator. The plain remedy, named in the same literature,
is many random restarts scored by likelihood. Neither has been tried here.

So the objection to answer is precise: *is the wrong answer the maximum-likelihood fit,
or merely the basin the conventional start falls into?* Those have opposite consequences.
A local optimum is escapable and the finding weakens to a statement about one
initialiser. A global optimum is not escapable by any initialiser, and the finding
strengthens into a statement about identifiability, which is where the related work
already points -- FedDS (Dong, Zhu, Shang and Xue, Information Sciences 745:123425, 2026)
needs every client's confusion matrix diagonally dominant, and a fleet whose majority
holds the wrong standard is where that fails.

The decisive comparison is the oracle start, and it is a diagnostic rather than a method:
seed EM at the ground truth and let it run. If it stays, the truth is a fixed point and
the two solutions can be compared on likelihood. If it walks away to the majority answer,
the truth is not even a local optimum and the question is settled without needing a
better initialiser to exist.

Needs no model and no network.

    uv run python scripts/measure_estimator_initialization.py --out results/estimator_initialization.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from measure_audit_policy import EVENTS
from measure_authority_anchors import REPAIRED, majority
from measure_secure_reliability import contributions_for, fleet_of

from pharos.analyst import Proposal
from pharos.disclosure import KEEP_COMPARTMENTS
from pharos.generate import GeneratorConfig, generate
from pharos.inference import agreement_with, dawid_skene, log_likelihood
from pharos.labels import declassify
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.telemetry import get_logger, progress, record

LOG = get_logger()

#: Fleet sizes. Nine is where every published governance number was measured; fifteen is
#: the smallest fleet finding 24 showed surviving a bare majority, so it carries a
#: composition that is broken at nine and healthy at fifteen and separates "the start
#: matters" from "the composition matters".
FLEETS = (9, 15)

#: Corpus draws. Not a robustness garnish -- the first run of this sweep was a single
#: draw and read, at the crossing, as a clean escape: the truth fit better and a restart
#: found it. Three more draws split two against two, and the effect is bimodal rather
#: than present. This repository has made the one-draw mistake twice before (finding 19's
#: anchor prices, finding 5's shot count), so the sweep reports a count over draws and
#: never a value from one.
DRAWS = (1, 7, 11, 23, 101, 202, 303, 404)

#: Random restarts per composition. Each is a fresh uniform draw over [0, 1] per task,
#: which is the remedy the crowdsourcing literature names for majority-vote EM not
#: reaching the global optimum. Enough that a basin holding a better answer would be
#: found: at 32 draws a basin covering a tenth of the space is missed with probability
#: 0.034, and one covering a twentieth with probability 0.19 -- so a *clean* sweep here
#: bounds how large an unfound basin could be rather than proving none exists.
RESTARTS = 32

#: Likelihoods within this of each other are the same optimum. Two EM runs stopping at
#: the same fixed point differ in the last places by iteration order alone, and calling
#: that "a better solution" would manufacture an escape out of floating-point noise.
LIKELIHOOD_TIE = 1e-6


def starts(
    tasks: list[str], truth: dict[str, bool], rng: random.Random
) -> dict[str, dict[str, float] | None]:
    """Every initialisation to compare, keyed by name. `None` is the published default.

    `truth` is supplied to exactly one of these and it is not a method -- an estimator
    that needs the answer to find the answer has no use. It is the diagnostic that makes
    the rest interpretable.
    """
    return {
        # The published start, and the only one any finding uses.
        "majority": None,
        # Everything undecided. The uninformative start: if the majority answer is
        # reachable from no information at all, it is not an artifact of being pointed
        # at it.
        "uninformative": dict.fromkeys(tasks, 0.5),
        # The answer. Not a method; the diagnostic that separates "EM prefers the wrong
        # answer" from "EM was aimed at it".
        "oracle": {t: (1.0 if truth[t] else 0.0) for t in tasks},
        # The inverse of the answer, as a control. A start this hostile that still lands
        # where the others land says the fixed point is reached from anywhere.
        "adversarial": {t: (0.0 if truth[t] else 1.0) for t in tasks},
        **{f"random-{i}": {t: rng.random() for t in tasks} for i in range(RESTARTS)},
    }


def fit_from(
    contributions: list[tuple[str, str, bool]],
    truth: dict[str, bool],
    initial: dict[str, float] | None,
) -> dict[str, Any]:
    """One EM run, scored on likelihood and on agreement with the world."""
    estimate = dawid_skene(contributions, initial_posterior=initial)
    return {
        "log_likelihood": round(log_likelihood(contributions, estimate), 6),
        "agreement": round(agreement_with(estimate.labels(), truth), 4),
        "iterations": estimate.iterations,
        "converged": estimate.converged,
    }


def composition_row(
    fleet: int, n_wrong: int, tasks: Any, proposals: Any, truth: dict[str, bool], seed: int
) -> dict[str, Any]:
    """Every start, on one fleet composition."""
    contributions = contributions_for(fleet_of(n_wrong, fleet), tasks, proposals, seed=seed)
    task_ids = sorted({t for t, _, _ in contributions})
    rng = random.Random(seed + n_wrong)  # noqa: S311

    runs = {
        name: fit_from(contributions, truth, initial)
        for name, initial in starts(task_ids, truth, rng).items()
    }

    published = runs["majority"]
    oracle = runs["oracle"]
    restarts = [runs[f"random-{i}"] for i in range(RESTARTS)]
    # Selected the way the literature selects: highest likelihood wins, and the selector
    # never sees an agreement score. Picking the best-agreeing restart instead would be
    # an oracle wearing a restart's clothes.
    best_restart = max(restarts, key=lambda r: r["log_likelihood"])

    better = [
        r
        for r in runs.values()
        if r["log_likelihood"] > published["log_likelihood"] + LIKELIHOOD_TIE
    ]

    return {
        "fleet": fleet,
        "n_wrong": n_wrong,
        "share": round(n_wrong / fleet, 4),
        "is_majority": n_wrong == majority(fleet),
        "published_recovers": published["agreement"] >= REPAIRED,
        "published_agreement": published["agreement"],
        "published_log_likelihood": published["log_likelihood"],
        "oracle_agreement": oracle["agreement"],
        "oracle_log_likelihood": oracle["log_likelihood"],
        "best_restart_agreement": best_restart["agreement"],
        #: How many of the restarts land on the truth, as opposed to whether any did.
        #: One in thirty-two and thirty in thirty-two are the same boolean and different
        #: deployment stories, and the second is the only one that is a method.
        "restarts_recovering": sum(1 for r in restarts if r["agreement"] >= REPAIRED),
        "restarts": RESTARTS,
        #: The diagnostic. A truth-seeded run that scores no higher than the
        #: majority-seeded one says the wrong answer fits the data at least as well, so
        #: the failure is identifiability and not initialisation.
        "oracle_beats_published": oracle["log_likelihood"]
        > published["log_likelihood"] + LIKELIHOOD_TIE,
        #: Whether the truth is a fixed point at all. If EM seeded at the answer walks
        #: away from it, no initialiser reaches it and the comparison above is moot.
        "oracle_start_holds": oracle["agreement"] >= REPAIRED,
        #: The literature's plain remedy, scored the way the literature scores it: many
        #: random starts, keep the highest likelihood, and take what it gives you.
        "best_restart_recovers": best_restart["agreement"] >= REPAIRED,
        #: The one that matters for the published claim: did *any* start both fit
        #: strictly better and recover the truth?
        "escape_exists": bool(better) and max(r["agreement"] for r in better) >= REPAIRED,
    }


def summarise(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """One composition across every draw, as counts rather than as a value.

    `cells` holds the same (fleet, n_wrong) from each corpus draw. Only the draws where
    the published start actually fails can carry an escape -- elsewhere there is nothing
    for a better initialiser to find -- so every rate below is over that subset and
    carries its own denominator.
    """
    broken = [c for c in cells if not c["published_recovers"]]
    gaps = sorted(
        round(c["oracle_log_likelihood"] - c["published_log_likelihood"], 4) for c in broken
    )
    return {
        "fleet": cells[0]["fleet"],
        "n_wrong": cells[0]["n_wrong"],
        "share": cells[0]["share"],
        "is_majority": cells[0]["is_majority"],
        "draws": len(cells),
        "draws_broken": len(broken),
        "draws_with_an_escape": sum(1 for c in broken if c["escape_exists"]),
        "draws_where_the_truth_fits_better": sum(1 for c in broken if c["oracle_beats_published"]),
        "draws_where_the_truth_is_a_fixed_point": sum(1 for c in broken if c["oracle_start_holds"]),
        "draws_where_restarts_recover": sum(1 for c in broken if c["best_restart_recovers"]),
        #: Out of `restarts` per draw. Zero everywhere would say the basin is never
        #: found; a small non-zero number says it is found unreliably, which is worse
        #: than either for anyone proposing restarts as the fix.
        "restart_recovery_rate": (
            round(sum(c["restarts_recovering"] for c in broken) / (len(broken) * RESTARTS), 4)
            if broken
            else None
        ),
        #: Signed, and the sign is the finding. Positive means the truth is the better
        #: fit and the published answer is a local optimum; negative means the published
        #: answer is the better fit and no likelihood-guided search will leave it.
        "likelihood_gap_range": [gaps[0], gaps[-1]] if gaps else None,
        "likelihood_gap_median": gaps[len(gaps) // 2] if gaps else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleets", type=int, nargs="+", default=list(FLEETS))
    parser.add_argument("--draws", type=int, nargs="+", default=list(DRAWS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for seed in args.draws:
        progress("estimator_initialization.draw", seed=seed)
        print(f">>> draw {seed}")
        tasks = build_triage_tasks(generate(GeneratorConfig(seed=seed, n_events=EVENTS)))
        proposals = {
            t.task_id: Proposal(
                t.task_id, not t.significant, declassify(t.label, KEEP_COMPARTMENTS)
            )
            for t in tasks
        }
        truth = {t.task_id: t.significant for t in tasks}
        for fleet in args.fleets:
            for n_wrong in range(1, fleet + 1):
                row = composition_row(fleet, n_wrong, tasks, proposals, truth, seed)
                row["draw"] = seed
                cells.setdefault((fleet, n_wrong), []).append(row)

    rows = [summarise(group) for group in cells.values()]
    # A composition nobody's draw broke has no escape to find and no denominator to
    # report one against. Scoring it would count "never needed rescuing" as "never
    # rescued", which is the censoring bug finding 20's sweep already hit once.
    priced = [row for row in rows if row["draws_broken"]]

    escapes = [row for row in priced if row["draws_with_an_escape"]]
    truth_fits_better = [row for row in priced if row["draws_where_the_truth_fits_better"]]
    truth_is_stable = [row for row in priced if row["draws_where_the_truth_is_a_fixed_point"]]
    restart_wins = [row for row in priced if row["draws_where_restarts_recover"]]

    for row in priced:
        record(
            "estimator_initialization.likelihood_gap",
            row["likelihood_gap_median"],
            fleet=row["fleet"],
            n_wrong=row["n_wrong"],
        )

    payload = {
        "fleets": args.fleets,
        "draws": args.draws,
        "restarts": RESTARTS,
        "likelihood_tie": LIKELIHOOD_TIE,
        "repaired_threshold": REPAIRED,
        "rows": rows,
        "priced_compositions": len(priced),
        "escape_compositions": [
            {
                "fleet": r["fleet"],
                "n_wrong": r["n_wrong"],
                "share": r["share"],
                "draws_with_an_escape": r["draws_with_an_escape"],
                "draws_broken": r["draws_broken"],
            }
            for r in escapes
        ],
        "invariants": {
            #: The headline, and it is false at exactly one composition. False does not
            #: fail the build -- it is the result, and it is why finding 25 exists.
            "no_initialisation_escapes_the_cliff": not escapes,
            #: Where the escape lives. At the crossing itself the published answer is a
            #: local optimum in some draws; past it the published answer is the better
            #: fit and there is nothing to escape to.
            "the_escape_is_confined_to_the_crossing": all(r["is_majority"] for r in escapes),
            #: The identifiability half, and the sharpest line in the artifact. Past the
            #: crossing the truth is still *reachable* -- seeded there, EM stays -- but it
            #: is the strictly worse fit, so selecting by likelihood rejects it. A better
            #: initialiser cannot help a search whose objective prefers the wrong answer.
            "likelihood_selection_would_pick_the_truth_wherever_it_is_reachable": len(
                truth_fits_better
            )
            == len(truth_is_stable),
            #: The literature's standard remedy, tried and reported either way.
            "random_restarts_never_recover_the_truth": not restart_wins,
        },
        "provenance": run_provenance(fleets=args.fleets, draws=args.draws, restarts=RESTARTS),
    }

    print()
    print("  compositions the published start breaks, counted over draws")
    header = f"    {'fleet':>6}{'wrong':>7}{'broken':>8}{'escapes':>9}{'restart%':>10}{'median dlogL':>14}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for row in priced:
        rate = row["restart_recovery_rate"]
        print(
            f"    {row['fleet']:>6}{row['n_wrong']:>7}"
            f"{row['draws_broken']:>4}/{row['draws']:<3}"
            f"{row['draws_with_an_escape']:>9}"
            f"{rate if rate is not None else 0:>10.3f}"
            f"{row['likelihood_gap_median']:>14.3f}"
        )

    print()
    for name, value in payload["invariants"].items():
        print(f"{name:<58} {value}")

    moved = {name: value for name, value in payload["invariants"].items() if value is False}
    if moved:
        LOG.warning(
            "estimator_initialization.escape_found",
            extra={
                "event": "estimator_initialization.escape_found",
                "moved": sorted(moved),
                "compositions": [(r["fleet"], r["n_wrong"]) for r in escapes],
            },
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
