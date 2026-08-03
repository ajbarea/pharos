"""Server-side FL aggregation rules, for the shared-adapter half of the router.

Each rule is implemented from its paper's definition of the *aggregation step* only.
That boundary matters: a method whose contribution is client-side (FedProx) reduces to
FedAvg here, and saying so is the point rather than an omission. Every class states
which paper it comes from and, where the two differ, what this implementation does not
do. An aggregation rule named after a paper it does not implement is worse than no
rule at all, because the name is what a reader checks against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FedAvg:
    """Federated Averaging (McMahan et al., AISTATS 2017)."""

    def aggregate(self, updates: list[list[float]]) -> list[float]:
        n = len(updates)
        if n == 0:
            return []
        dim = len(updates[0])
        return [sum(u[d] for u in updates) / n for d in range(dim)]


@dataclass(frozen=True)
class FedProx:
    """FedAvg server step (Li et al., MLSys 2020).

    FedProx's contribution is a proximal term in the *client's* local objective, so its
    server-side aggregation is identical to FedAvg by construction. `mu` is carried here
    to be handed to a local trainer; it does not and should not affect this step. Kept
    as a distinct class so a configuration can name the method it is running.
    """

    mu: float = 0.01

    def aggregate(self, updates: list[list[float]]) -> list[float]:
        n = len(updates)
        if n == 0:
            return []
        dim = len(updates[0])
        return [sum(u[d] for u in updates) / n for d in range(dim)]


@dataclass(frozen=True)
class FedMedian:
    """Coordinate-wise median (Yin et al., ICML 2018)."""

    def aggregate(self, updates: list[list[float]]) -> list[float]:
        n = len(updates)
        if n == 0:
            return []
        dim = len(updates[0])
        res = []
        for d in range(dim):
            vals = sorted(u[d] for u in updates)
            mid = n // 2
            med = vals[mid] if n % 2 != 0 else (vals[mid - 1] + vals[mid]) / 2.0
            res.append(med)
        return res


@dataclass(frozen=True)
class TrimmedMean:
    """Coordinate-wise trimmed mean (Yin et al., ICML 2018)."""

    k: int = 2

    def aggregate(self, updates: list[list[float]]) -> list[float]:
        n = len(updates)
        if n == 0:
            return []
        dim = len(updates[0])
        res = []
        for d in range(dim):
            vals = sorted(u[d] for u in updates)
            trimmed = vals[self.k : n - self.k] if n > 2 * self.k else vals
            res.append(sum(trimmed) / len(trimmed))
        return res


def _krum_scores(updates: list[list[float]], f: int) -> list[int]:
    """Update indices ordered by Krum score, best first.

    The score is the sum of the n-f-2 smallest squared distances to other updates, which
    is the quantity all three Krum-family rules below rank on. Shared so they cannot
    drift apart: the same ordering has to mean the same thing in each.
    """
    n = len(updates)
    dim = len(updates[0])
    scores = []
    for i in range(n):
        dists = sorted(
            sum((updates[i][d] - updates[j][d]) ** 2 for d in range(dim))
            for j in range(n)
            if i != j
        )
        scores.append((sum(dists[: max(1, n - f - 2)]), i))
    scores.sort()
    return [i for _, i in scores]


@dataclass(frozen=True)
class Krum:
    """Byzantine-robust single selection (Blanchard et al., NIPS 2017)."""

    f: int = 2

    def aggregate(self, updates: list[list[float]]) -> list[float]:
        if not updates:
            return []
        return updates[_krum_scores(updates, self.f)[0]]


@dataclass(frozen=True)
class MultiKrum:
    """Mean of the m best-scoring updates by Krum score (Blanchard et al., NIPS 2017).

    Multi-Krum is defined in the same paper as Krum, not in the Bulyan paper that builds
    on it; the citation here read El Mhamdi et al. 2018 and that was wrong.
    """

    f: int = 2
    m: int = 4

    def aggregate(self, updates: list[list[float]]) -> list[float]:
        if not updates:
            return []
        dim = len(updates[0])
        selected = _krum_scores(updates, self.f)[: self.m]
        return [sum(updates[i][d] for i in selected) / len(selected) for d in range(dim)]


@dataclass(frozen=True)
class Bulyan:
    """Selection by Krum, then a trimmed coordinate-wise mean (El Mhamdi et al., ICML 2018).

    Both stages are load-bearing. The first selects theta = n - 2f candidates; the second
    takes, per coordinate, the mean of the beta = theta - 2f values closest to that
    coordinate's median. Without the second stage this is Multi-Krum wearing Bulyan's
    name, which is what it was: the selection ran and its output was returned directly,
    so the rule the paper contributes was never executed.
    """

    f: int = 2

    def aggregate(self, updates: list[list[float]]) -> list[float]:
        n = len(updates)
        if n == 0:
            return []
        dim = len(updates[0])
        theta = max(1, n - 2 * self.f)
        selected = _krum_scores(updates, self.f)[:theta]
        pool = [updates[i] for i in selected]
        beta = max(1, theta - 2 * self.f)
        res = []
        for d in range(dim):
            vals = sorted(u[d] for u in pool)
            mid = len(vals) // 2
            median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
            closest = sorted(vals, key=lambda v: abs(v - median))[:beta]
            res.append(sum(closest) / len(closest))
        return res


@dataclass(frozen=True)
class GeometricMedian:
    """RFA Weiszfeld iteration (Pillutla et al., IEEE TSP 2022)."""

    max_iter: int = 20
    eps: float = 1e-5

    def aggregate(self, updates: list[list[float]]) -> list[float]:
        n = len(updates)
        if n == 0:
            return []
        dim = len(updates[0])
        y = [sum(u[d] for u in updates) / n for d in range(dim)]
        for _ in range(self.max_iter):
            weights = []
            for u in updates:
                dist = math.sqrt(sum((u[d] - y[d]) ** 2 for d in range(dim)))
                weights.append(1.0 / max(dist, self.eps))
            total_w = sum(weights)
            y = [sum(weights[i] * updates[i][d] for i in range(n)) / total_w for d in range(dim)]
        return y


# ArKrum (Yang and Imam, arXiv:2505.17226) was defined here as a subclass that called
# Multi-Krum with an explicit f and m. That is not ArKrum: the method's contribution is
# estimating the adversary count from a median filter so that f does not have to be
# supplied, which is the parameter it was being handed. Removed rather than corrected,
# because implementing it faithfully needs the full text and a validation run of its own,
# and a wrong implementation under a real citation is the failure mode with no floor.


Strategy = FedAvg | FedProx | FedMedian | TrimmedMean | Krum | MultiKrum | Bulyan | GeometricMedian
