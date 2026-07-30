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

**Why this is a calibration instrument and not a purity test.** An early version
demanded an AUC at chance and could never have been satisfied. Ground truth here
is defined by the presence of particular content, so plants necessarily carry the
significant facts more often than background does, and any surface statistic of
those facts therefore carries some information. Measured on this corpus: every
report holds exactly two fact sentences of fourteen words and nine digits each,
and plants still run 49.29 words against 49.63, because the fact *mix* differs by
construction. The only way to reach a true chance AUC would be a vocabulary whose
every rendering is a surface twin of every other on character count, punctuation,
and capitalisation as well.

So the useful questions are two, and the gate answers both.

Is the leak real? Compare the observed statistic against a permutation null,
where labels are shuffled so no relationship survives. On this corpus the null
sits at 0.4986 with a standard deviation of 0.0216, which both confirms the gate
is unbiased and gives the band an empirical basis rather than an assumed one.

How large is it? The observed AUC is the **surface baseline**: the score a model
can reach while reading nothing. Any downstream triage number has to be reported
against it, because a triage F1 is meaningless without knowing what shape alone
already achieves.
"""

import random
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pharos.generate import Report
from pharos.telemetry import record, span
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
    null_mean: float | None = None
    null_sd: float | None = None
    null_p95: float | None = None
    null_trials: int = 0

    @property
    def passed(self) -> bool:
        """Whether the observed statistic is inside the nominal band.

        Retained as the strict ideal. For content-defined ground truth it is
        usually unreachable, so `leak_is_significant` is the operational check and
        `surface_baseline` is the number downstream scores are reported against.
        """
        low, high = self.band
        return low <= self.auc <= high

    @property
    def surface_baseline(self) -> float:
        """The AUC a model reaches reading nothing. Report triage scores against this."""
        return self.auc

    @property
    def leak_is_significant(self) -> bool | None:
        """Whether the leak exceeds what label shuffling alone produces.

        `None` when no permutation null was computed, so a caller can never mistake
        an unmeasured null for a clean one.
        """
        if self.null_p95 is None:
            return None
        return self.auc > self.null_p95

    @property
    def null_z(self) -> float | None:
        """Standard deviations between the observed statistic and the null mean."""
        if self.null_mean is None or not self.null_sd:
            return None
        return (self.auc - self.null_mean) / self.null_sd

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
            "surface_baseline": round(self.surface_baseline, 4),
            "null_mean": None if self.null_mean is None else round(self.null_mean, 4),
            "null_sd": None if self.null_sd is None else round(self.null_sd, 4),
            "null_p95": None if self.null_p95 is None else round(self.null_p95, 4),
            "null_trials": self.null_trials,
            "null_z": None if self.null_z is None else round(self.null_z, 2),
            "leak_is_significant": self.leak_is_significant,
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


class _Sweep(NamedTuple):
    """One cross-validated sweep: the verdict statistic and everything behind it."""

    verdict: float
    mean_auc: dict[str, float]
    per_fold: dict[str, list[float]]
    fold_sizes: list[int]


def _verdict_auc(reports: list[Report], center_ids: tuple[str, ...]) -> _Sweep:
    """Cross-validated AUC per probe, plus the verdict statistic and fold sizes."""
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
    worst = max(mean_auc, key=lambda probe: abs(mean_auc[probe] - 0.5))
    return _Sweep(mean_auc[worst], mean_auc, per_fold, fold_sizes)


def permutation_null(
    reports: list[Report], *, trials: int = 20, seed: int = 0
) -> tuple[float, float, float]:
    """Mean, standard deviation, and 95th percentile of the gate under shuffled labels.

    Shuffling `is_plant` destroys any real relationship, so what the gate reports
    on permuted labels is its own false-positive distribution. This is what gives
    the band an empirical basis, and it is also how the gate proves it is not
    itself the source of an apparent leak.
    """
    center_ids = _center_ids(reports)
    rng = random.Random(seed)
    labels = [r.is_plant for r in reports]
    stats: list[float] = []
    for _ in range(trials):
        rng.shuffle(labels)
        shuffled = [replace(r, is_plant=lab) for r, lab in zip(reports, labels, strict=True)]
        try:
            sweep = _verdict_auc(shuffled, center_ids)
        except ValueError:
            continue
        stats.append(sweep.verdict)
    if not stats:
        raise ValueError("permutation null produced no usable trial")
    ordered = sorted(stats)
    sd = float(np.std(stats, ddof=1)) if len(stats) > 1 else 0.0
    return float(np.mean(stats)), sd, ordered[max(int(0.95 * len(ordered)) - 1, 0)]


def run_gate(
    reports: list[Report],
    *,
    band: tuple[float, float] = DEFAULT_BAND,
    null_trials: int = 0,
    null_seed: int = 0,
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
    with span("gate.sweep", n_reports=len(reports), n_folds=len(center_ids)):
        sweep = _verdict_auc(reports, center_ids)
    mean_auc = sweep.mean_auc
    per_fold = sweep.per_fold
    fold_sizes = sweep.fold_sizes
    spread = {probe: float(np.max(v) - np.min(v)) for probe, v in per_fold.items()}

    null_mean = null_sd = null_p95 = None
    if null_trials > 0:
        null_mean, null_sd, null_p95 = permutation_null(reports, trials=null_trials, seed=null_seed)

    # Emit the measurements structurally as well as returning them, so a run is
    # analysable from its own log stream without re-deriving anything.
    for probe, value in mean_auc.items():
        record("gate.probe_auc", value, probe=probe, n_reports=len(reports))
    record("gate.surface_baseline", sweep.verdict, n_reports=len(reports), n_folds=len(center_ids))
    if null_mean is not None:
        record("gate.null_mean", null_mean, trials=null_trials)
        record("gate.null_z", (sweep.verdict - null_mean) / (null_sd or 1.0), trials=null_trials)

    return GateResult(
        auc=sweep.verdict,
        band=band,
        n_train=len(reports) - (fold_sizes[0] if fold_sizes else 0),
        n_test=sum(fold_sizes),
        held_out_centers=center_ids,
        per_probe_auc=mean_auc,
        per_fold_auc={probe: tuple(v) for probe, v in per_fold.items()},
        fold_spread=spread,
        null_mean=null_mean,
        null_sd=null_sd,
        null_p95=null_p95,
        null_trials=null_trials,
    )
