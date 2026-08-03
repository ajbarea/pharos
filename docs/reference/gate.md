# The Shortcut Gate

The Pharos shortcut gate (`pharos.gate`) prevents models from learning superficial artifacts of insertion rather than genuine task structure.

!!! danger "Why Shortcut Detection Matters"
    In a federated fleet, a model that learns insertion tells or formatting shortcuts will propagate those spurious features to all shared adapters, producing false metric convergence across deployments.

---

## Probe Surface Features

The gate probe inspects **10 non-semantic surface features** (zero content words):

```python
from pharos.gate import surface_features

# Feature vector extracted per report:
# [char_len, word_count, sentence_count, mean_sentence_len, digit_ratio,
#  upper_ratio, punct_ratio, has_timestamp, report_type_id, voice_id]
surface_features(report)
```

---

## Probe Architecture & Evaluation

1. **Leave-One-Center-Out Cross-Validation**: Prevents data leakage between split folds while preserving sample size stability.
2. **Worst-News-Wins Ranking**: Both a linear classifier and a gradient-boosted tree evaluate the features; the probe closest to predicting the label sets the final score.

---

## Historical Leakage Refinements

The synthetic corpus generator underwent four rounds of adversarial refinement to eliminate surface shortcuts:

| Round | Surface AUC | Identified Shortcut | Generator Fix |
| :---: | :---: | :--- | :--- |
| **1** | 0.737 | `digit_ratio` anomaly in significant facts | Enforced 1 timestamp + 2 integer numerals per rendering |
| **2** | 0.581 | Planted event reports were slightly longer | Equalized and balanced word counts across fact templates |
| **3** | 0.572 | Structural asymmetry: plants were the only triple facts | Introduced background decoy triples to match class structure |
| **4** | ~0.550 | Slot width variances | Standardized all non-time slots to 2-digit integer templates |

!!! tip "Structural Normalization Rule"
    Tuning a leaked feature in prose simply shifts the leak. Removing the underlying structural asymmetry eliminates the whole artifact class.

---

## Surface Baseline Calibration

Content-defined ground truth cannot achieve a theoretical 0.500 chance-level surface baseline because plants inherently carry specific fact combinations.

```console
$ uv run python -m pharos.cli gate --seed 7 --events 400
surface baseline  0.6545  (ceiling 0.72)
permutation null  0.4979 +/- 0.0300  p95 0.5437  over 20 trials
leak vs null      z = +5.22  significant: True
VERDICT           USABLE
```

### Usability Criteria (`Manifest.usable`)

1. ✅ **Label Variation**: Labels must vary across the generated corpus.
2. ✅ **Permutation Null Verification**: Computed explicitly via label shuffling ($p_{\text{null}} \approx 0.50$).
3. ✅ **Ceiling Cap**: Surface baseline AUC must remain below `MAX_SURFACE_BASELINE` (**0.72**).

---

## External Corpus Benchmark Validation

`scripts/validate_gate_externally.py`

Running the exact same shortcut probe against established public NLP corpora:

| Public Corpus | Dataset Type | $n$ Tasks | Surface Baseline | Permutation Null | $z$-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`ag_news`** | Unfiltered News | 3,800 | 0.6642 | $0.5006 \pm 0.020$ | **+8.09** |
| **`imdb`** | Unfiltered Reviews | 12,000 | 0.5648 | $0.4996 \pm 0.010$ | **+7.01** |
| **`hellaswag`** | **Adversarially Filtered** | 12,000 | 0.5358 | $0.5005 \pm 0.010$ | **+3.65** |

!!! note "Adversarial Filtering Limits"
    Even heavily adversarially-filtered datasets like HellaSwag retain statistically significant surface baselines ($z = +3.65$).

