# The shortcut gate

Planted ground truth invites a model to learn the artifact of insertion rather
than the property under test. In a federated fleet that is worse than one inflated
score: a shared adapter would propagate the shortcut to every deployment, so
convergence would look healthy while the fleet had learned an insertion tell.

So a corpus has to earn its use. A probe gets only non-semantic features, ten of
them, no content words at all.

## What the probe sees

```python
from pharos.gate import surface_features

surface_features(report)
# char_len, word_count, sentence_count, mean_sentence_len, digit_ratio,
# upper_ratio, punct_ratio, has_timestamp, report_type_id, voice_id
```

Shape only. If any of these predicts plant membership well, the corpus is
predictable without being read.

## Two decisions that make this a real gate

**Leave-one-center-out cross-validation, not a random split.** A random split lets
a surface feature present in both halves go unpunished. And a *single* held-out
center tests on a quarter of the corpus, where AUC sampling error is wider than
the pass band itself, so a single-fold gate reports its own noise as leakage. An
earlier version did exactly that: it passed two seeds and failed three at values
it could not distinguish from chance.

**The strongest probe sets the verdict, never an average.** A linear model and a
gradient-boosted tree both run, ranked by distance from chance, and the worse news
wins. A gate should assume the most capable attacker available.

## What the gate caught

Each fix is a permanent constraint on the vocabulary, not a one-off patch. The
generator was rejected four times.

| Round | AUC | Cause | Fix |
| --- | --- | --- | --- |
| 1 | 0.737 | Significant facts had numeric slots, most fillers had none. `digit_ratio` alone hit 0.652 | One timestamp and two integer numerals in every rendering; no decimals |
| 2 | 0.581 | Rendering length: pattern facts averaged 13.33 words against 12.64, and plants carry all three | Balanced word counts |
| 3 | 0.572 | Hand-balancing oscillated. The leak was structural: plants were the only events with a deterministic triple, so **any** per-fact property became a class signal | Background draws a decoy triple, so both classes are one fixed triple plus two fillers |
| 4 | ~0.55 | Slot widths varied, so which slots a fact used was a digit signature | Every non-time slot is a two-digit integer, so every rendering carries exactly nine digits |

Round 3 is the one worth internalizing: **tuning a leaked property just moves the
leak.** Removing the asymmetry that produced it removes the whole class.

## A calibration instrument, not a purity test

The gate originally demanded an AUC at chance. That requirement could never have
been met, and working out why produced the most useful result in the project.

**Content-defined ground truth cannot have a chance-level surface baseline.** A
plant is a plant *because* it carries the significant facts, so plants carry those
facts more often than background does, so the fact mix differs by class, so any
surface statistic of those facts carries some information. Measured after four
rounds of normalization: every report holds exactly two fact sentences of fourteen
words and nine digits each, and plants still average 49.29 words against 49.63.
The residual is the mix, and the mix is the definition.

Reaching a true chance baseline would need a vocabulary whose every rendering is a
surface twin of every other on character count, punctuation, and capitalisation as
well. That is achievable and it is not obviously worth it.

So the gate answers two questions instead of pronouncing on purity.

**Is the leak real?** Compare against a **permutation null**, where labels are
shuffled so no relationship survives. Measured: a null mean near 0.50 with a
standard deviation around 0.02 to 0.03. That confirms the gate is unbiased and
gives the band an empirical basis rather than an assumed one.

**How large is it?** The observed AUC is the **surface baseline**: what a model
scores while reading nothing. Every downstream triage number is reported against
it, because an F1 is meaningless without knowing what shape alone already
achieves.

## When a corpus is usable

`Manifest.usable` requires three things, and deliberately does **not** require a
chance baseline:

1. The labels vary. A constant label cannot evaluate a disclosure boundary.
2. A permutation null was actually computed. An unmeasured baseline is not a small
   baseline, and `leak_is_significant` returns `None` rather than `False` when no
   null exists, so a caller cannot mistake one for the other.
3. The baseline sits under `MAX_SURFACE_BASELINE` (0.72). Above that, shape
   explains too much of the task for a triage score to mean anything.

Current state at 400 events: seed 1 gives 0.6547 against a null of 0.4854 +/-
0.0317, seed 7 gives 0.6588 against 0.5047 +/- 0.0361, seed 101 gives 0.6675
against 0.4912 +/- 0.0287. All significant, all under the ceiling, all usable.

## The claim holds on corpora we did not build

`scripts/validate_gate_externally.py`

Everything above was measured on a corpus we wrote, which is thin evidence for a
claim about content-defined labels *in general*. So the same probe -- same
estimators, same worst-news-wins rule, same permutation null -- was run against
public datasets built by others.

| Corpus | Construction | n | Baseline | Null | z |
| --- | --- | --- | --- | --- | --- |
| `ag_news` | no filtering step | 3,800 | 0.6642 | 0.5006 | **+8.09** |
| `imdb` | no filtering step | 12,000 | 0.5648 | 0.4996 | **+7.01** |
| `hellaswag_endings` | **adversarially filtered** | 12,000 | 0.5358 | 0.5005 | **+3.65** |

**Every corpus exceeds its own null**, so the phenomenon is not an artifact of our
generator. The unexpected part is the ordering: leakage falls monotonically with how
much construction effort went into removing it.

The HellaSwag row carries the weight. Adversarial filtering is the standard remedy
and was applied there specifically to remove the signal this probe looks for. It
works -- the leak is about a third the size of the unfiltered case -- and it still
does not reach chance. An iterative discriminator can drop the examples it can solve,
but not the fact that positives and negatives are drawn from different content
distributions.

!!! warning "What this does not license"
    Absolute AUCs are **not** comparable across these corpora or to Pharos: splits,
    class balances, and text lengths all differ. Only the sign and significance of
    baseline-minus-null are. And three datasets are not a survey.

!!! note "A construction error worth avoiding"
    The first version of this probe scored HellaSwag at chance and it was wrong. It
    compared the *shared context* against the answer index, but the context is
    identical whichever ending is correct, so the label had no relationship to the
    text being measured. It now emits one row per candidate ending, labelled for the
    true continuation, which is the property filtering actually targets. The wrong
    version would have supported a confident and false conclusion.

## Answerability and non-leakage pull against each other

The ceiling was raised from 0.65 to 0.72 once generation began guaranteeing
coverage. Requiring that every fact of an event actually be rendered ties report
composition to the event's fact set, which lifted the baseline from about 0.55 to
0.63-0.67 on a corpus that is now *correct*.

This is a real tension rather than a bug. The alternative was to keep a tighter
ceiling and reject corpora for being answerable, which is the wrong trade. The
surface baseline is a published property of a correct corpus, not a defect to
drive to chance.

## What the gate cannot catch

A surface probe tests whether **shape** predicts the label. It cannot test whether
the label is **derivable from the content**. Those are different properties and
the second needs its own checks.

This is not hypothetical. Only 34% of significant events once rendered all three
of their defining facts, so two thirds of the positive class was unanswerable from
its own prompt, and a measurement built on it produced a conclusion that had to be
retracted. The gate passed the whole time. Semantic integrity checks now live in
the test suite: every fact of an event is rendered by some channel that can carry
it, no report asserts a fact outside its event, and the per-report fact count
stays constant so length carries no class signal.
