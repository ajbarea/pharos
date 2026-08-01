# Reporting a noisy measurement

`pharos.uncertainty`

This page describes what a model-dependent score has to carry.

!!! warning "Built for a claim that was then retracted"
    This module was written because
    [finding 9](../findings.md#9-a-measurement-that-repeats-one-prompt-measures-the-wrong-thing)
    reported 10% run-to-run instability. That turned out to be a cache warm-up
    artifact, and single-pass Pharos measurements reproduce exactly.

    The module is kept because its **between-task** term was always the larger one
    and was never reported: 30 tasks is a small sample however stable the decode.
    Its **within-task** term now measures a cold-to-warm cache transition rather
    than noise, and should be read that way -- near-zero is the expected value, and
    a large one means the probe repeated a prompt rather than that the model is
    unreliable.

## Two error sources, and only one of them shrinks

| Source | What it is | Reduced by |
| --- | --- | --- |
| Between-task | Some events are harder; a different sample would score differently | more tasks |
| Within-task | The same task answered differently on repeat calls | *not noise here -- see above* |

A binomial interval over `n` tasks accounts for the first and silently assumes the
second is zero. `variance_split` reports the share, which is the actionable number:
it says whether a measurement is limited by sample size or by instability, and
therefore which axis of the experiment budget is worth spending.

## The estimand has to match the deployment

The reflex on discovering run-to-run noise is to average several runs per task and
put an interval on the average. That estimates the performance of a system that
**votes**, and a Pharos fleet does not vote: one edge agent answers once and the
answer is acted on.

So `single_run` is the headline, weighting every `(task, run)` pair equally, and
`consensus` sits beside it as what voting would buy:

```python
from pharos.uncertainty import summarize

m = summarize(trials, label="8-shot")
m.single_run.point  # what one pass achieves
m.consensus  # what a majority vote over the same passes achieves
m.consensus_gain  # the difference, which can be negative
```

Quoting the consensus number for a single-shot deployment overstates it. The gap
between them is a design input, not a rounding detail -- and it can be **negative**,
because voting locks in an error on tasks the model usually gets wrong.

## Cluster bootstrap, not the naive one

Runs of the same task are correlated. Resampling individual `(task, run)` pairs
treats five calls on one task as five independent observations and shrinks the
interval by roughly `sqrt(k)` for no reason at all. `cluster_bootstrap` resamples
whole tasks with all their runs attached.

The test suite asserts the direction rather than trusting it: an interval computed
by flattening runs into separate clusters must come out **narrower** than the
clustered one, so a regression that undoes the correction fails the build.

## Reading an ordering off a table

`resolves(a, b)` is false when either interval covers the other's point estimate.
That is the question a reader actually has -- *does the ordering in this table
survive the noise* -- and overlapping intervals answer it. It is deliberately
conservative and deliberately not a p-value.

The measurement scripts print the pairs that are **not** separated, which is the
honest direction: a reader assumes a table's ordering is real unless told otherwise.

## What this does not fix

The intervals here are percentile intervals over the two sources Pharos can
resample. Published analysis of LLM measurement error finds that standard intervals
under-cover because they omit variation from **prompt phrasing, judge model choice,
and model version**, and that the under-coverage gets *worse* with more data, not
better.

None of that is addressed here. An interval from this module is a floor on the
uncertainty, not a guarantee, and a result that depends on the difference between
two conditions whose intervals nearly separate should be treated as unresolved.

## Where the numbers come from

```bash
uv run python scripts/measure_rule_learnability.py --repeats 5   # finding 5
uv run python scripts/measure_decode_stability.py                # finding 9
```

`--repeats 1` is still the default for a quick look, and it prints no interval at
all rather than a degenerate one, because a point with an invented range is worse
than a point that admits it is alone.
