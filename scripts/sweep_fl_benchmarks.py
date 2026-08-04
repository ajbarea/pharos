"""Aggregation error for six server-side FL rules under sign-flip poisoning.

Each condition draws an independent fleet per round, so the rounds are the sample and
the spread across them is reported alongside the mean. A single-run mean is not a score:
at 20 clients the draw moves the number by more than several of the gaps being compared.

This artifact carries no validity assessment and is not quoted in the manuscript. It
exists to size the problem, not to settle it. Two things it must not be asked to do:
rank rules whose intervals overlap, and support a claim about any rule at an attack
ratio where the honest majority assumption behind that rule no longer holds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from pharos.fl import (
    Bulyan,
    FedAvg,
    FedMedian,
    GeometricMedian,
    Krum,
    MultiKrum,
    PrivacyBudget,
    TrimmedMean,
    add_gaussian_dp_noise,
    apply_sign_flip,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def l2_distance(vec1: list[float], vec2: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(vec1, vec2, strict=True)))


def _percentile_interval(values: list[float], draws: int = 2000) -> tuple[float, float]:
    """Bootstrap percentile interval for the mean of `values`.

    Seeded independently of the sweep so that resampling noise cannot move with the
    experiment's own seed and be mistaken for an effect of it.
    """
    rng = random.Random(0)  # noqa: S311
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def generate_fleet_gradients(
    n_clients: int = 20,
    dim: int = 100,
    n_byzantine: int = 0,
    seed: int = 42,
) -> tuple[list[float], list[list[float]], list[bool]]:
    """Generates ground truth honest gradient and client updates with n_byzantine sign-flip attackers."""
    rng = random.Random(seed)  # noqa: S311

    true_gradient = [rng.gauss(0.0, 1.0) for _ in range(dim)]

    clients: list[list[float]] = []
    is_byzantine: list[bool] = []

    for i in range(n_clients):
        if i < n_byzantine:
            # Byzantine sign-flip attack
            poisoned = apply_sign_flip(true_gradient, severity=1.5)
            clients.append(poisoned)
            is_byzantine.append(True)

        else:
            # Honest client with mild Gaussian noise
            client_grad = [g + rng.gauss(0.0, 0.1) for g in true_gradient]
            clients.append(client_grad)
            is_byzantine.append(False)

    return true_gradient, clients, is_byzantine


def run_fl_benchmark_sweep(
    n_clients: int = 20,
    dim: int = 100,
    n_rounds: int = 50,
    seed: int = 42,
    out: Path | None = None,
) -> dict[str, Any]:
    print("Aggregation error under sign-flip poisoning")

    # f is the adversary count each rule is *told* to expect, held at 4 (20% of the
    # fleet) across every attack ratio. That is deliberate and it is the realistic case:
    # a defender does not know the true ratio. It also means the 30% and 50% columns
    # measure each rule under a misspecified f, which is a different question from its
    # behaviour when f is correct, and the two must not be conflated.
    strategies = {
        "FedAvg": FedAvg(),
        "FedMedian": FedMedian(),
        "TrimmedMean": TrimmedMean(k=4),
        "Krum": Krum(f=4),
        "MultiKrum": MultiKrum(f=4, m=12),
        "Bulyan": Bulyan(f=4),
        "GeometricMedian": GeometricMedian(),
    }

    attack_ratios = [0.0, 0.10, 0.30, 0.50]  # 0%, 10%, 30%, 50%
    dp_configs = [
        ("No DP", None),
        ("DP (ε=1.0, δ=1e-5)", PrivacyBudget(epsilon=1.0, delta=1e-5, clip_norm=1.0)),
    ]

    sweep_results: list[dict[str, Any]] = []

    for ratio in attack_ratios:
        n_byzantine = int(n_clients * ratio)
        print(
            f"\n--- Evaluating Attack Ratio: {int(ratio * 100)}% ({n_byzantine}/{n_clients} Byzantine Nodes) ---"
        )

        for dp_name, dp_budget in dp_configs:
            for strat_name, strat in strategies.items():
                errors: list[float] = []
                latencies: list[float] = []

                for r in range(n_rounds):
                    true_grad, clients, _ = generate_fleet_gradients(
                        n_clients=n_clients,
                        dim=dim,
                        n_byzantine=n_byzantine,
                        seed=seed + r,
                    )

                    # Apply DP noise if enabled
                    if dp_budget is not None:
                        clients = [
                            add_gaussian_dp_noise(
                                c,
                                dp_budget,
                                rng=random.Random(seed + r * 100 + i),  # noqa: S311
                            )
                            for i, c in enumerate(clients)
                        ]

                    t0 = time.perf_counter()
                    aggregated = strat.aggregate(clients)
                    dt_ms = (time.perf_counter() - t0) * 1000.0

                    err = l2_distance(aggregated, true_grad)
                    errors.append(err)
                    latencies.append(dt_ms)

                mean_err = sum(errors) / len(errors)
                mean_lat = sum(latencies) / len(latencies)
                lo, hi = _percentile_interval(errors)

                entry = {
                    "strategy": strat_name,
                    "attack_ratio": ratio,
                    "n_byzantine": n_byzantine,
                    "dp_mode": dp_name,
                    "n_rounds": len(errors),
                    "mean_l2_error": round(mean_err, 4),
                    "l2_error_ci95": [round(lo, 4), round(hi, 4)],
                    "mean_latency_ms": round(mean_lat, 4),
                }
                sweep_results.append(entry)
                print(
                    f"  [{strat_name:15s} | {dp_name:18s}] "
                    f"L2 {mean_err:.4f} [{lo:.4f}, {hi:.4f}] | {mean_lat:.4f} ms"
                )

    # Each rule against its own no-attack error, which is the only comparison that does
    # not smuggle in a ranking. Krum's error is high at every ratio including zero, so
    # against a fleet-wide constant it reads as damaged when it is merely coarse; against
    # itself it reads as unmoved, which is the true statement. A previous version of this
    # script assigned a PROTECTED/POISONED label from a hardcoded list of strategy names,
    # so the column reported which rules the author expected to win.
    baseline = {
        (r["strategy"], r["dp_mode"]): r["mean_l2_error"]
        for r in sweep_results
        if r["attack_ratio"] == 0.0
    }
    for r in sweep_results:
        base = baseline.get((r["strategy"], r["dp_mode"]))
        r["error_vs_own_no_attack"] = round(r["mean_l2_error"] / base, 3) if base else None

    from pharos.provenance import run_provenance

    output_payload = {
        "experiment": "Aggregation error under sign-flip poisoning",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": {
            "n_clients": n_clients,
            "gradient_dim": dim,
            "n_rounds": n_rounds,
            "f_supplied_to_rules": 4,
        },
        "records": sweep_results,
        "provenance": run_provenance(seed=seed),
    }

    # Only when asked. This wrote `results/fl_benchmarks.json` unconditionally, so
    # `tests/test_scripts.py` -- which calls this with 5 clients, 10 dimensions and 1
    # round to exercise the helpers -- replaced the committed 20-client, 50-round
    # artifact with a toy every time the suite ran. The artifact was found carrying
    # `n_clients: 5, n_rounds: 1` and errors an order of magnitude off, and nothing
    # failed, because a smaller sweep is still a well-formed sweep.
    #
    # Same defect and same fix as `measure_adversarial_robustness.py`, which was
    # deleted for it earlier today; this one was missed because its write is at the
    # bottom of a long function rather than in an obvious `main`.
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run_fl_benchmark_sweep(
        n_clients=args.clients,
        dim=args.dim,
        n_rounds=args.rounds,
        seed=args.seed,
        out=args.out,
    )
