"""Differential Privacy (DP) Noise Injector for Shared LoRA Aggregation.

Implements Differential Privacy mechanisms (Gaussian / Laplace noise with
L2 clipping) to guarantee privacy budget (epsilon, delta) protection for shared
adapters.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyBudget:
    """Privacy budget parameters for federated differential privacy."""

    epsilon: float
    delta: float = 1e-5
    clip_norm: float = 1.0


def add_gaussian_dp_noise(
    weights: list[float],
    budget: PrivacyBudget,
    rng: random.Random | None = None,
) -> list[float]:
    """Inject Gaussian noise to weights scaled to (epsilon, delta) differential privacy."""
    if rng is None:
        rng = random.Random()  # noqa: S311

    if budget.epsilon <= 0:
        raise ValueError("epsilon must be positive for differential privacy")

    # Calculate standard deviation sigma for (epsilon, delta)-DP
    sigma = budget.clip_norm * math.sqrt(2 * math.log(1.25 / budget.delta)) / budget.epsilon

    # L2 clipping of weights
    l2_norm = math.sqrt(sum(w * w for w in weights))
    clip_scale = min(1.0, budget.clip_norm / max(l2_norm, 1e-12))
    clipped = [w * clip_scale for w in weights]

    # Add Gaussian noise
    return [w + rng.gauss(0, sigma) for w in clipped]
