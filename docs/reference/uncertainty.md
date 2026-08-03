# Uncertainty Quantification & Estimands

`pharos.uncertainty` provides cluster-bootstrap confidence intervals and variance decomposition across evaluation tasks.

---

## Task Sample Variance vs. Decode Instability

| Uncertainty Dimension | Cause | Reduction Strategy |
| :--- | :--- | :--- |
| **Between-Task Variance** | Task difficulty variation across sampled events | Increase task sample size ($n$) |
| **Within-Task Variance** | Prompt-caching or decode transitions | Managed by single-pass execution at temperature 0 |

!!! info "Variance Decomposition"
    `variance_split` calculates the proportion of uncertainty originating from task selection versus decoding variance, informing compute budget allocation.

---

## Single-Run vs. Consensus Estimands

In edge triage deployments, a single node executes one pass without voting. Evaluating single-pass behavior requires matching estimands:

```python
from pharos.uncertainty import summarize

m = summarize(trials, label="8-shot")
m.single_run.point  # Expected performance of a single-pass edge agent
m.consensus  # Majority vote performance over repeated passes
m.consensus_gain  # Accuracy differential (can be negative if errors are systematic)
```

!!! danger "Single-Pass vs. Voting Misalignment"
    Reporting majority-vote consensus scores for a single-pass edge deployment overstates operational performance. If model errors are systematic, consensus voting locks in errors.

---

## Cluster-Bootstrap Resampling

Resampling individual evaluation calls treats repeated runs on the same task as independent events, artificially shrinking confidence intervals by $\sqrt{k}$.

`cluster_bootstrap` resamples **entire task clusters** alongside all associated runs to preserve empirical error bounds.

---

## Statistical Separation & Comparisons

`resolves(a, b)` returns `False` if either condition's 95% confidence interval overlaps the other's point estimate.

```python
from pharos.uncertainty import resolves

# Returns False if two measurement conditions cannot be statistically separated
resolves(condition_a, condition_b)
```

