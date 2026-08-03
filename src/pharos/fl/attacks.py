"""Byzantine attack primitives, for exercising the aggregation rules.

Round-level attack simulations: sign-flipping, IPM (Inner Product Manipulation),
Gaussian noise injection, and Sybil client injection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AttackConfig:
    """Configuration for simulated Byzantine round attack."""

    attack_type: str
    byzantine_fraction: float = 0.2
    severity: float = 1.0


def apply_sign_flip(weights: list[float], severity: float = 1.0) -> list[float]:
    """Sign-flip attack: invert weight signs to pull gradients in opposite direction."""
    return [-w * severity for w in weights]


def apply_gaussian_noise(
    weights: list[float], scale: float = 0.5, rng: random.Random | None = None
) -> list[float]:
    """Add zero-mean Gaussian noise to corrupt client updates."""
    if rng is None:
        rng = random.Random()  # noqa: S311
    return [w + rng.gauss(0, scale) for w in weights]
