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
    per_fold_auc: dict[str, tuple[float, ...]] = field(default_factory=dict)
    fold_spread: dict[str, float] = field(default_factory=dict)

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
            "per_fold_auc": {k: [round(x, 4) for x in v] for k, v in self.per_fold_auc.items()},
            "fold_spread": {k: round(v, 4) for k, v in self.fold_spread.items()},
            "n_folds": len(self.held_out_centers),
        }


def _matrix(reports: list[Report]) -> tuple[np.ndarray, np.ndarray]:
    rows = [surface_features(r) for r in reports]
    features = np.array([[row[name] for name in SURFACE_FEATURES] for row in rows], dtype=float)
    labels = np.array([float(r.is_plant) for r in reports], dtype=float)
    return features, labels


def _center_ids(reports: list[Report]) -> tuple[str, ...]:
    center_ids = tuple(sorted({r.center.center_id for r in reports}))
    if len(center_ids) < 2:
        raise ValueError("cannot hold out a center when the corpus has fewer than two")
    return center_ids


def _fold_auc(train: list[Report], test: list[Report]) -> dict[str, float]:
    """AUC per probe for one fold, or an empty dict when the fold is undefined."""
    if len({r.is_plant for r in train}) < 2 or len({r.is_plant for r in test}) < 2:
        return {}
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
    out: dict[str, float] = {}
    for name, (model, fit_x, score_x) in probes.items():
        model.fit(fit_x, y_train)
        out[name] = float(roc_auc_score(y_test, model.predict_proba(score_x)[:, 1]))
    return out


def run_gate(
    reports: list[Report],
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> GateResult:
    """Try to predict plant membership from shape alone. Chance means the corpus is clean.

    Uses leave-one-center-out cross-validation rather than a single held-out
    center. That is a correctness requirement, not a refinement. With four
    centers a single fold tests on roughly a quarter of the corpus, and the
    sampling error on an AUC over that many rows is around four points, so a
    band of five points either side of chance sits inside the gate's own noise.
    An early single-fold version of this gate duly passed two seeds and failed
    three at values it could not distinguish from chance.

    Averaging over every fold uses the whole corpus for testing, roughly halving
    the spread, and the per-fold values are reported so a wide spread is visible
    rather than hidden inside a mean.

    Raises `ValueError` when no fold is well defined, since silently returning
    chance would be a gate that always passes.
    """
    center_ids = _center_ids(reports)
    per_fold: dict[str, list[float]] = {}
    fold_sizes: list[int] = []
    for held_out in center_ids:
        train = [r for r in reports if r.center.center_id != held_out]
        test = [r for r in reports if r.center.center_id == held_out]
        fold = _fold_auc(train, test)
        if not fold:
            continue
        fold_sizes.append(len(test))
        for probe, auc in fold.items():
            per_fold.setdefault(probe, []).append(auc)

    if not per_fold:
        raise ValueError("no usable fold: every split lacked both classes")

    mean_auc = {probe: float(np.mean(values)) for probe, values in per_fold.items()}
    spread = {probe: float(np.max(values) - np.min(values)) for probe, values in per_fold.items()}

    # The strongest attacker sets the verdict, ranked by distance from chance.
    # An AUC below chance is as informative as one above it.
    worst_probe = max(mean_auc, key=lambda probe: abs(mean_auc[probe] - 0.5))
    return GateResult(
        auc=mean_auc[worst_probe],
        band=band,
        n_train=len(reports) - (fold_sizes[0] if fold_sizes else 0),
        n_test=sum(fold_sizes),
        held_out_centers=center_ids,
        per_probe_auc=mean_auc,
        per_fold_auc={probe: tuple(values) for probe, values in per_fold.items()},
        fold_spread=spread,
    )
