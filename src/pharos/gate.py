"""The shortcut gate: can plant membership be predicted without reading anything?

Planted ground truth invites a model to learn the artifact of insertion instead
of the property under test. For a federated fleet the consequence is worse than
one inflated score, because a shared adapter would propagate the shortcut to
every deployment: convergence would look healthy while the fleet had learned an
insertion tell.

So the corpus has to earn its use. A probe gets only non-semantic features, no
content words at all, and has to fail. If it succeeds, the generator leaked
something and the corpus version is not usable.

Two decisions make this a real gate rather than a formality.

The split holds out whole centers rather than sampling at random. A random split
lets a surface feature present in both halves go unpunished, which is the
documented way benchmarks hide their own shortcuts; holding out a center forces
any tell to transfer across sources to count.

The verdict takes the strongest of several probes, not an average. A gate should
assume the most capable attacker available, so a linear model and a
gradient-boosted tree both run and the worse news wins.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pharos.generate import Report
from pharos.world import ReportType, Voice

#: The features the probe is allowed. Shape only. Adding anything derived from
#: content words would defeat the purpose of the gate.
SURFACE_FEATURES: tuple[str, ...] = (
    "char_len",
    "word_count",
    "sentence_count",
    "mean_sentence_len",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "has_timestamp",
    "report_type_id",
    "voice_id",
)

_PUNCTUATION = set(".,;:()/-")

#: The band that counts as chance. Pass/fail, not advisory.
DEFAULT_BAND: tuple[float, float] = (0.45, 0.55)


def surface_features(report: Report) -> dict[str, float]:
    """Non-semantic shape features for one report.

    Every value is a count, a ratio, or a category index. None of them can be
    computed from what the report says, only from how it is shaped.
    """
    text = report.text
    words = text.split()
    sentences = [s for s in text.split(".") if s.strip()]
    length = max(len(text), 1)
    return {
        "char_len": float(len(text)),
        "word_count": float(len(words)),
        "sentence_count": float(len(sentences)),
        "mean_sentence_len": float(len(words) / max(len(sentences), 1)),
        "digit_ratio": sum(c.isdigit() for c in text) / length,
        "upper_ratio": sum(c.isupper() for c in text) / length,
        "punct_ratio": sum(c in _PUNCTUATION for c in text) / length,
        "has_timestamp": float(any(w.endswith("Z") and w[:-1].isdigit() for w in words)),
        "report_type_id": float(sorted(ReportType).index(report.report_type)),
        "voice_id": float(sorted(Voice).index(report.voice)),
    }


@dataclass(frozen=True, slots=True)
class GateResult:
    """The verdict, and enough context to reproduce and argue with it."""

    auc: float
    band: tuple[float, float]
    n_train: int
    n_test: int
    held_out_centers: tuple[str, ...]
    per_probe_auc: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        low, high = self.band
        return low <= self.auc <= high

    def as_dict(self) -> dict[str, object]:
        return {
            "auc": round(self.auc, 4),
            "band": list(self.band),
            "passed": self.passed,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "held_out_centers": list(self.held_out_centers),
            "per_probe_auc": {k: round(v, 4) for k, v in self.per_probe_auc.items()},
        }


def _matrix(reports: list[Report]) -> tuple[np.ndarray, np.ndarray]:
    rows = [surface_features(r) for r in reports]
    features = np.array([[row[name] for name in SURFACE_FEATURES] for row in rows], dtype=float)
    labels = np.array([float(r.is_plant) for r in reports], dtype=float)
    return features, labels


def _held_out_centers(reports: list[Report], holdout: int) -> tuple[str, ...]:
    """The last `holdout` center ids in sorted order, so the split is deterministic."""
    center_ids = sorted({r.center.center_id for r in reports})
    if len(center_ids) <= holdout:
        raise ValueError(
            f"cannot hold out {holdout} of {len(center_ids)} centers; need at least one to train on"
        )
    return tuple(center_ids[-holdout:])


def run_gate(
    reports: list[Report],
    *,
    band: tuple[float, float] = DEFAULT_BAND,
    holdout_centers: int = 1,
) -> GateResult:
    """Try to predict plant membership from shape alone. Chance means the corpus is clean.

    Raises `ValueError` when the split leaves a side without both classes, since
    an AUC over one class is undefined and silently returning 0.5 would be a
    gate that always passes.
    """
    held_out = _held_out_centers(reports, holdout_centers)
    train = [r for r in reports if r.center.center_id not in held_out]
    test = [r for r in reports if r.center.center_id in held_out]

    if len({r.is_plant for r in train}) < 2 or len({r.is_plant for r in test}) < 2:
        raise ValueError("both splits need both classes for AUC to be defined")

    x_train, y_train = _matrix(train)
    x_test, y_test = _matrix(test)

    scaler = StandardScaler().fit(x_train)
    probes = {
        "logistic": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            scaler.transform(x_train),
            scaler.transform(x_test),
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(random_state=0, max_iter=200),
            x_train,
            x_test,
        ),
    }

    per_probe: dict[str, float] = {}
    for name, (model, fit_x, score_x) in probes.items():
        model.fit(fit_x, y_train)
        scores = model.predict_proba(score_x)[:, 1]
        per_probe[name] = float(roc_auc_score(y_test, scores))

    # The strongest attacker sets the verdict. An AUC below chance is just as
    # informative as one above it, so distance from 0.5 is what ranks probes.
    worst = max(per_probe.items(), key=lambda kv: abs(kv[1] - 0.5))
    return GateResult(
        auc=worst[1],
        band=band,
        n_train=len(train),
        n_test=len(test),
        held_out_centers=held_out,
        per_probe_auc=per_probe,
    )
