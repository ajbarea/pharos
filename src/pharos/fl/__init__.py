"""Pharos Federated Learning & Byzantine Robustness Module."""

from pharos.fl.attacks import AttackConfig, apply_gaussian_noise, apply_sign_flip
from pharos.fl.dp import PrivacyBudget, add_gaussian_dp_noise
from pharos.fl.strategy import (
    Bulyan,
    FedAvg,
    FedMedian,
    FedProx,
    GeometricMedian,
    Krum,
    MultiKrum,
    Strategy,
    TrimmedMean,
)

__all__ = [
    "AttackConfig",
    "Bulyan",
    "FedAvg",
    "FedMedian",
    "FedProx",
    "GeometricMedian",
    "Krum",
    "MultiKrum",
    "PrivacyBudget",
    "Strategy",
    "TrimmedMean",
    "add_gaussian_dp_noise",
    "apply_gaussian_noise",
    "apply_sign_flip",
]
