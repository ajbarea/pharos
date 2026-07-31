# Findings

What has been measured, and the script that reproduces each number. The
**argument** these support belongs to the manuscript; this page is the index.

All model-dependent numbers below used `qwen2.5:7b-instruct` on an 8 GB RTX 3060
Ti. Each committed artifact under `results/` carries the version, commit, model,
and seed that produced it.

!!! note "Regenerating"
    `make results` reruns all four scripts into `results/`. It needs Ollama
    serving the model named in each script, and it is not fast: the label
    fidelity measurement alone is 72 sequential model calls.

## 1. Leave-one-out attribution cannot produce a correct governed label

`scripts/measure_label_fidelity.py`

Eight-source summarization turns, exact leave-one-out, 24 turns: **61.8% source
recall at 97.6% precision, and a wrong governed label on 8 of 24 turns (33%).** Six
of the eight leak; two creep.

!!! warning "Corrected 2026-07-30"
    First measured at n=8 on the corpus *before* the coverage fix in finding 3.
    Three claims did not survive remeasurement. A wrong label on **half** of turns
    is **33%** at n=24. **Always under-restrictive** is refuted: creep occurs in 2
    of 24. And the cited move to an **incomparable** label
    (`RESTRICTED[LIAISON,PARTNER,SENSOR]` to `PROTECTED[LEGAL]`) did not recur and
    should not be quoted. Source recall did reproduce, 0.618 against 0.62.

The cause is corroboration. Leave-one-out asks which single source is
load-bearing, and a fact reported through several channels has none: drop any one
copy and the fact survives in the others, so no source is blamed and none of their
labels enters the join. Corroboration is not an edge case in this domain, it is
what channels are for.

Read the precision alongside the recall. At 0.976 the ablation almost never blames
a source that did not contribute, it simply misses most of the ones that did. That
asymmetry is why the failure is mostly leak, and why a quarter of turns receive a
label that under-protects their sources.

Leave-one-out is also the ceiling that cheaper estimators approximate, so nothing
faster repairs it. At 67% exactly correct it is not a **usable** labelling
mechanism, though the original "rules out the whole family" was stated more
strongly than 24 turns can support.

The replacement costs nothing: given what the output asserts, join the labels of
every source that *could* have asserted it. One detection pass, no ablation sweep,
and conservative by construction, so the error direction is creep rather than leak.

## 2. The design is bimodal on one policy ruling

`scripts/measure_federation_eligibility.py`

Three aggregator ceilings, four capacities, 40 turns. Turns average **2.15
compartments of 4**, and most already sit high on the level ladder, because a
summary over eight sources joins nearly everything.

| Declassification policy | FREETEXT | SPAN | SCALAR | ENUM |
| --- | --- | --- | --- | --- |
| keep compartments (fail-closed default) | 0-38% | 0-38% | 0-38% | 0-38% |
| drop compartments for low capacity | 0-38% | 0-38% | **100%** | **100%** |

!!! warning "Corrected 2026-07-30"
    Also first measured at n=8 on the pre-coverage-fix corpus: the mean was
    reported as 2.88 (now 2.15) and the keep-compartments row as 0-12% (now
    0-38%). The **shape** is unchanged and is the part that matters, and it now
    rests on 40 turns rather than 8.

So "may a low-capacity verdict shed the compartments of its sources?" is not a
detail. Answer no and the fleet is a set of unconnected local learners. Answer yes
and verdict-shaped outputs federate completely while prose never does, which
reproduces the design's split table from measurement rather than assertion.

See [the label lattice](reference/label-lattice.md).

## 3. A corpus bug, a retracted finding, and a real benchmark target

`scripts/measure_triage_lift.py`

This one went wrong before it went right, and the sequence is the useful part.

The first measurement said the specialist could not do its own task: triage
accuracy 0.35 against a majority floor of 0.725. The conclusion drawn was that a
7B model cannot evaluate a three-way conjunction over facts split across reports.

**That conclusion was wrong, and the cause was a corpus bug.** Only **34% of
significant events actually rendered all three of their defining facts**, because
channels were chosen before coverage was checked and any shortfall was padded from
the whole vocabulary. Two thirds of the positive class was unanswerable from its
own prompt. The model was being scored on evidence it was never shown, and its
escalate-on-anything behaviour was a reasonable response to partial evidence.

The [shortcut gate](reference/gate.md) could never have caught this: a surface
probe tests whether shape predicts the label, not whether the label is derivable
from the content. Semantic integrity needs its own checks, and they are now in the
test suite.

On the corrected corpus:

| Setup | Accuracy | Majority | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Rule stated, plain prompt | 0.433 | 0.633 | 0.393 | 1.000 | 0.564 |
| **Rule stated, checklist prompt** | **1.000** | 0.633 | 1.000 | 1.000 | **1.000** |
| Rule withheld, plain prompt | 0.367 | 0.633 | 0.367 | 1.000 | 0.537 |
| Rule withheld, brief reasoning | 0.667 | 0.633 | 0.524 | 1.000 | 0.688 |

The model does the conjunction perfectly **when the rule is given and the prompt
structures the check**. Which exposed a design error: stating the rule leaves
nothing for the fleet to learn, and learning the analyst's rule from
accept/revise/reject is the entire premise of personalization.

Withholding the rule gives the benchmark its proper shape. A ceiling of F1 1.000,
known reachable, against a base of 0.537 to 0.688 that over-escalates at recall
1.000 and precision 0.37 to 0.52. **That gap is the target.**

## 3b. Over-escalation is universal, and scale does not fix it

`scripts/sweep_models.sh`, `scripts/compare_models.py`

Finding 3 was measured on one model, so its central observation could have been a
fact about `qwen2.5:7b-instruct` rather than about the task. Five models, three
families, two size classes, 40 tasks each, rule withheld:

| Model | Family | Acc | Majority | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| **qwen2.5-3b** | Qwen | **0.650** | 0.625 | 0.517 | 1.000 | **0.682** |
| llama3.1-8b | Llama | 0.525 | 0.625 | 0.441 | 1.000 | 0.612 |
| llama3.2-3b | Llama | 0.475 | 0.625 | 0.417 | 1.000 | 0.588 |
| qwen2.5-7b | Qwen | 0.450 | 0.625 | 0.405 | 1.000 | 0.577 |
| mistral-7b | Mistral | 0.425 | 0.625 | 0.395 | 1.000 | 0.566 |

Two things hold across every model, and they are the claims:

**Recall is 1.000 everywhere.** Not approximately, exactly. Every model escalates
every significant event, and also escalates most routine ones, with precision
between 0.395 and 0.517. The over-escalation reported in finding 3 is not an
artifact of one model, it is what the task does to all of them when the rule is
withheld.

**Scale does not help.** The best score belongs to the *smallest* model tested, and
8B does not beat 3B. Whatever is missing is not capacity.

Both are consistent with finding 5's conclusion, and strengthen it. A model that
reaches F1 1.000 when handed the rule is not short of capability, and now we know
it is not short of parameters either. It is short of the rule, so rule
*acquisition* is the whole question.

!!! warning "What this does not claim"
    40 tasks per model. Differences of roughly 0.1 sit inside the noise at this
    size, so **the ordering between models is not claimed** and should not be
    quoted as a ranking. What is claimed is the part that is unanimous: recall
    1.000 for all five, and no model clearly clearing the majority floor.

    All five ran at 100% GPU residency, checked per model rather than assumed.

## 4. Answerability and surface non-leakage pull against each other

Fixing coverage raised the surface baseline from about 0.55 to 0.63-0.67, and
widening channel sets did not bring it back down. Once every fact of an event must
appear across a fixed number of reports, report composition is tightly determined
by the event's fact set.

A real tension rather than a bug. It argues for treating the surface baseline as a
published property of a *correct* corpus rather than a defect to drive to chance.

## 5. In-context learning does not close the gap

`scripts/measure_rule_learnability.py`

The design's premise is that a fleet learns analytic craft from an analyst's
accept, revise, and reject decisions. Withholding the rule and supplying labelled
examples instead is the cheap form of that question.

| Condition | Shots | F1 |
| --- | --- | --- |
| Zero-shot floor | 0 | 0.720 |
| Verdict-only examples | 2 / 4 / 8 | 0.615 / 0.640 / 0.636 |
| Examples with the officer's stated reason | 2 / 4 / 8 | 0.522 / 0.706 / 0.556 |
| **Rule stated, checklist prompt (ceiling)** | n/a | **1.000** |

**Examples close none of the gap**, and richer examples do not rescue it.

Three caveats, because this result is easy to over-read:

- Twenty evaluation tasks per condition. Differences of about 0.1 sit inside the
  noise, so the ordering *between* conditions is not claimed, only that none
  reaches the ceiling.
- Eight shots is roughly 3,600 words before the target case, so long-context
  dilution is an unseparated confound.
- **In-context learning is not gradient learning.** Eight examples in a prompt and
  a LoRA trained on thousands of tuples are different mechanisms, and the first
  failing does not establish that the second will.

The useful conclusion is about sequencing. The cheap proxy for the design's
central premise came back negative-to-inconclusive, so the premise cannot be
validated cheaply and **the adapter experiment is no longer optional: it is the
thing that decides.** What the ceiling establishes is where the bottleneck sits. A
model that reaches F1 1.000 when told the rule is not short of capability, it is
short of the rule, so rule *acquisition* is the whole question.
