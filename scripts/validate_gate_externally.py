#!/usr/bin/env python3
"""Does the calibration-instrument finding hold outside our own generator?

The gate section makes a general claim: *content-defined ground truth cannot have a
chance-level surface baseline*. An instance belongs to the positive class because it
carries certain content, so positives carry that content more often, so the content
distribution differs by class, so any surface statistic correlated with it leaks.

That argument is about content-defined labels in general, but it has only ever been
measured on the corpus we wrote. That is thin support for a claim that broad, and it
is exactly the kind of claim that is comfortable to believe and expensive to be wrong
about. This runs the same probe against public datasets nobody here constructed.

**The sharpest case is an adversarially filtered dataset.** SWAG and HellaSwag were
built by iteratively discarding examples a discriminator could solve, specifically to
remove stylistic artifacts. If our claim is right, a content-defined corpus cannot
reach chance on surface features. If adversarial filtering worked, it should. Those
cannot both be true in full generality, and finding out which gives way is a result
either direction:

- Surface signal survives filtering -> the claim generalises, and filtering does not
  reach the floor it aims at.
- Filtering reaches chance -> our claim is too strong as stated, and the honest
  version is narrower: content-defined ground truth carries surface signal *unless
  the construction explicitly removes it*, which is achievable and which we declined
  to do.

**Two differences from the Pharos gate, both stated rather than hidden.** External
corpora have no watch centres, so this uses stratified k-fold rather than
leave-one-group-out; the permutation null is computed under the identical procedure,
so the baseline-versus-null comparison stays valid even though the absolute AUCs are
not directly comparable to ours. And two of the ten Pharos features are categorical
identifiers with no external analogue, so the probe here gets eight rather than ten.
Fewer features is the conservative direction: it can only understate a leak.

    uv run --with datasets scripts/validate_gate_externally.py
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pharos.provenance import run_provenance
from pharos.telemetry import log_execution_context, progress

#: The eight Pharos surface features computable from text alone. `report_type_id`
#: and `voice_id` are corpus-specific identifiers and have no external counterpart.
TEXT_FEATURES: tuple[str, ...] = (
    "char_len",
    "word_count",
    "sentence_count",
    "mean_sentence_len",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "has_timestamp",
)

_PUNCTUATION = set(".,;:()/-")


def text_surface_features(text: str) -> list[float]:
    """The same shape statistics `pharos.gate.surface_features` computes, minus the
    two corpus-specific identifiers. No content word is ever read."""
    words = text.split()
    sentences = [s for s in text.split(".") if s.strip()]
    length = max(len(text), 1)
    return [
        float(len(text)),
        float(len(words)),
        float(len(sentences)),
        float(len(words) / max(len(sentences), 1)),
        sum(c.isdigit() for c in text) / length,
        sum(c.isupper() for c in text) / length,
        sum(c in _PUNCTUATION for c in text) / length,
        float(any(w.endswith("Z") and w[:-1].isdigit() for w in words)),
    ]


@dataclass(frozen=True, slots=True)
class Corpus:
    """A public dataset reduced to what the probe needs."""

    name: str
    texts: list[str]
    labels: list[int]
    note: str

    @property
    def prevalence(self) -> float:
        return sum(self.labels) / max(len(self.labels), 1)


def _probe_auc(x_train, y_train, x_test, y_test) -> dict[str, float]:
    """One fold, both probes. Identical estimators to `pharos.gate`."""
    scaler = StandardScaler().fit(x_train)
    out: dict[str, float] = {}
    for name, model in (
        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("gradient_boosting", HistGradientBoostingClassifier(random_state=0, max_iter=200)),
    ):
        if len(set(y_test)) < 2 or len(set(y_train)) < 2:
            continue
        fitted = model.fit(scaler.transform(x_train), y_train)
        scores = fitted.predict_proba(scaler.transform(x_test))[:, 1]
        out[name] = float(roc_auc_score(y_test, scores))
    return out


def surface_baseline(x, y, *, folds: int = 4, seed: int = 0) -> tuple[float, dict[str, float]]:
    """Cross-validated AUC per probe; the probe furthest from chance sets the verdict.

    The worst-news-wins rule is Pharos's, kept deliberately: a gate should assume the
    most capable attacker available rather than average one away.
    """
    per_probe: dict[str, list[float]] = {}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(x, y):
        fold = _probe_auc(x[train_idx], y[train_idx], x[test_idx], y[test_idx])
        for probe, auc in fold.items():
            per_probe.setdefault(probe, []).append(auc)
    if not per_probe:
        raise ValueError("no usable fold")
    mean = {p: float(np.mean(v)) for p, v in per_probe.items()}
    worst = max(mean, key=lambda p: abs(mean[p] - 0.5))
    return mean[worst], mean


def permutation_null(x, y, *, trials: int, folds: int, seed: int) -> tuple[float, float, float]:
    """What the same procedure produces on shuffled labels: its own false-positive
    distribution. This is what makes an absolute AUC interpretable across datasets
    whose splits and class balances differ."""
    rng = np.random.default_rng(seed)
    stats = []
    shuffled = np.array(y)
    for trial in range(trials):
        rng.shuffle(shuffled)
        try:
            verdict, _ = surface_baseline(x, shuffled, folds=folds, seed=seed)
        except ValueError:
            continue
        stats.append(verdict)
        if (trial + 1) % max(1, trials // 4) == 0:
            progress("external.null_progress", trial=trial + 1, trials=trials)
    if not stats:
        raise ValueError("permutation null produced no usable trial")
    ordered = sorted(stats)
    sd = float(np.std(stats, ddof=1)) if len(stats) > 1 else 0.0
    return float(np.mean(stats)), sd, ordered[max(int(0.95 * len(ordered)) - 1, 0)]


# ---------------------------------------------------------------------------
# Loaders. Each returns a binary task whose positive class is defined by CONTENT,
# which is the property under test. Where a dataset is multi-class, one class is
# taken as positive rather than collapsing it arbitrarily.
# ---------------------------------------------------------------------------
def load_corpora(limit: int, seed: int) -> list[Corpus]:
    from datasets import load_dataset

    out: list[Corpus] = []

    def take_rows(rows, name, note):
        """`rows` yields (text, label) pairs already binarised."""
        texts, labels = [], []
        for text, label in rows:
            if not text or not isinstance(text, str):
                continue
            texts.append(text)
            labels.append(int(label))
            if len(texts) >= limit:
                break
        if len(set(labels)) < 2:
            print(f"  skip {name}: single class after sampling", file=sys.stderr)
            return
        out.append(Corpus(name=name, texts=texts, labels=labels, note=note))
        print(f"  {name}: {len(texts)} rows, prevalence {sum(labels) / len(labels):.3f}")

    def hellaswag_rows():
        """One row per candidate ENDING, labelled 1 for the true continuation.

        The first version probed the shared context against the answer index, which is
        incoherent: the context is identical whichever ending is correct, so the label
        carried no relationship to the text and the probe scored chance for a reason
        that had nothing to do with adversarial filtering. What filtering actually
        targets is whether a candidate ending betrays itself by its shape, so that is
        what gets probed.
        """
        ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
        for row in ds:
            endings = row.get("endings") or []
            try:
                correct = int(row["label"])
            except (KeyError, TypeError, ValueError):
                continue
            for index, ending in enumerate(endings):
                yield ending, int(index == correct)

    def interleaved(dataset, text_key, label_fn):
        """Alternate classes while streaming.

        Several test splits are class-ordered, so a streaming head yields one class
        only. IMDb was dropped for exactly that reason on the first run.
        """
        buckets: dict[int, list[str]] = {0: [], 1: []}
        for row in dataset:
            text = row.get(text_key)
            if not text or not isinstance(text, str):
                continue
            buckets[int(label_fn(row))].append(text)
            while buckets[0] and buckets[1]:
                yield buckets[0].pop(), 0
                yield buckets[1].pop(), 1

    specs = [
        (
            "hellaswag_endings",
            hellaswag_rows,
            "adversarially filtered; one row per candidate ending, positive = true continuation",
        ),
        (
            "ag_news",
            lambda: interleaved(
                load_dataset("fancyzhx/ag_news", split="test", streaming=True),
                "text",
                lambda r: int(r["label"] == 0),
            ),
            "topic classification, World vs rest; no adversarial construction",
        ),
        (
            "imdb",
            lambda: interleaved(
                load_dataset("stanfordnlp/imdb", split="test", streaming=True),
                "text",
                lambda r: int(r["label"]),
            ),
            "sentiment; long free text, no length control",
        ),
    ]
    for name, rows, note in specs:
        try:
            take_rows(rows(), name, note)
        except Exception as exc:  # a dataset moving or going gated is not our bug
            print(f"  skip {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=4000, help="rows per dataset")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--null-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    log_execution_context()
    print("loading public corpora")
    corpora = load_corpora(args.limit, args.seed)
    if not corpora:
        print("no corpora loaded", file=sys.stderr)
        return 1

    results = []
    for corpus in corpora:
        x = np.array([text_surface_features(t) for t in corpus.texts])
        y = np.array(corpus.labels)
        baseline, per_probe = surface_baseline(x, y, folds=args.folds, seed=args.seed)
        null_mean, null_sd, null_p95 = permutation_null(
            x, y, trials=args.null_trials, folds=args.folds, seed=args.seed
        )
        z = (baseline - null_mean) / (null_sd or 1.0)
        row = {
            "dataset": corpus.name,
            "note": corpus.note,
            "n": len(corpus.texts),
            "prevalence": round(corpus.prevalence, 4),
            "surface_baseline": round(baseline, 4),
            "per_probe_auc": {k: round(v, 4) for k, v in per_probe.items()},
            "null_mean": round(null_mean, 4),
            "null_sd": round(null_sd, 4),
            "null_p95": round(null_p95, 4),
            "z": round(z, 2),
            "exceeds_null": bool(baseline > null_p95),
        }
        results.append(row)
        print(
            f"\n{corpus.name}: baseline {baseline:.4f} vs null "
            f"{null_mean:.4f} +/- {null_sd:.4f}  z={z:+.2f}  "
            f"{'ABOVE null' if row['exceeds_null'] else 'at chance'}"
        )

    print("\n" + "=" * 74)
    # str(...) because `results` rows are dict[str, object]; without it the join
    # below is an overload the checker cannot match and a latent TypeError if a
    # dataset name ever arrives as something other than a string.
    above = [str(r["dataset"]) for r in results if r["exceeds_null"]]
    at_chance = [str(r["dataset"]) for r in results if not r["exceeds_null"]]
    print(f"surface signal above the null : {', '.join(above) or 'none'}")
    print(f"indistinguishable from chance : {', '.join(at_chance) or 'none'}")
    print(
        "\nPharos measures 0.63-0.67 against nulls near 0.50. These are NOT directly\n"
        "comparable in absolute terms: different splits, class balances, and text\n"
        "lengths. What IS comparable is the sign of baseline minus null, which is the\n"
        "claim under test."
    )
    print("=" * 74)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(seed=args.seed),
                    "method": {
                        "features": list(TEXT_FEATURES),
                        "folds": args.folds,
                        "null_trials": args.null_trials,
                        "split": "stratified k-fold (external corpora have no centre analogue)",
                        "note": (
                            "Eight features rather than Pharos's ten; the two omitted are "
                            "corpus-specific identifiers. Fewer features can only understate "
                            "a leak."
                        ),
                    },
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
