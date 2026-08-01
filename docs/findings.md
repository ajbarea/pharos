# Findings

What has been measured, and the script that reproduces each number. The
**argument** these support belongs to the manuscript; this page is the index.

Findings 1 to 3 and 5 were measured with `qwen2.5:7b-instruct` on an 8 GB RTX 3060
Ti. Finding 3b sweeps six models from 3B to 14B, and finding 6 fine-tunes
`Qwen2.5-3B-Instruct`; both need a cluster A100 for their largest runs.

Every artifact in `results/` carries the version, commit, platform, model, and seed
behind it, so any number here traces back to the run and the machine that produced
it. That matters because the model-dependent numbers are not bit-reproducible across
machines, while the gate is.

!!! note "Regenerating"
    `make results` reruns the four Ollama-backed measurements into `results/`. It
    needs Ollama serving the model each script names, and it is slow: the label
    fidelity pass alone is 216 sequential model calls. Findings 3b and 6 are cluster
    jobs and are not part of this target.

## Status: provisional

**The testbed is still being built, and these numbers should be read as provisional
rather than settled.** That is a deliberate position, not modesty, and the history
supports it: of the first three findings, two did not reproduce when remeasured at
larger n on a corrected corpus, and the third had to be retracted outright after a
generator bug turned out to have made two thirds of the positive class unanswerable.

What that means for a reader:

- **Mechanisms are the durable part.** That leave-one-out under-attributes because
  corroborated facts have no single load-bearing source, or that eligibility is
  bimodal on the compartment-shedding ruling, survived remeasurement. The
  *severities* attached to them did not.
- **Every number carries its own verdict.** `pharos.validity` inspects each
  measurement for the conditions that make a score misleading -- small n, a class
  floor the score does not clear, degenerate predictions, unparsed answers -- and
  marks the artifact `quotable: false` when it trips one. Most of the sweep results
  are currently not quotable, and say so.
- **Numbers below are traceable, not authoritative.** Each artifact in `results/`
  records the version, commit, platform, model, and seed behind it, so a figure can
  be checked rather than trusted.

The gate's calibration finding is the one result here with support from outside this
generator, having been reproduced on three public corpora. Treat the rest as the
current state of an instrument still under construction.

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
fact about `qwen2.5:7b-instruct` rather than about the task. Six models, three
families, 3B to 14B, 40 rule-withheld tasks each:

| Model | Family | Size | Acc | Majority | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-3b | Qwen | 3B | 0.625 | 0.625 | 0.500 | 1.000 | **0.667** |
| qwen2.5-14b | Qwen | 14B | 0.525 | 0.625 | 0.441 | 1.000 | 0.612 |
| llama3.1-8b | Llama | 8B | 0.500 | 0.625 | 0.429 | 1.000 | 0.600 |
| llama3.2-3b | Llama | 3B | 0.475 | 0.625 | 0.417 | 1.000 | 0.588 |
| qwen2.5-7b | Qwen | 7.6B | 0.450 | 0.625 | 0.405 | 1.000 | 0.577 |
| mistral-7b | Mistral | 7B | 0.425 | 0.625 | 0.395 | 1.000 | 0.566 |

Three things hold across every model, and they are the claims:

**Recall is 1.000 everywhere.** Not approximately, exactly, including at 14B. Every
model escalates every significant event and also escalates most routine ones, at
precision between 0.395 and 0.500. The over-escalation reported in finding 3 is a
property of the task under a withheld rule, not an idiosyncrasy of one model.

**No model beats the majority-class floor.** The best accuracy exactly ties it at
0.625. A reader comparing any F1 here to 0.5 would badly overstate what is
happening.

**Scale does not help, and this now controls for family.** Within Qwen alone, 3B
scores 0.667 and 14B scores 0.612: a 4.7x parameter increase with no improvement.
The 14B run required a cluster A100 because it exceeds 8 GB of VRAM, which is the
whole reason it is here.

All three are consistent with finding 5 and strengthen it. A model that reaches
F1 1.000 when handed the rule is short of neither capability nor parameters. It is
short of the rule, so rule *acquisition* is the whole question.

!!! warning "What this does not claim"
    40 tasks per model. Differences of roughly 0.1 sit inside the noise, so **the
    ordering between models is not claimed** and should not be quoted as a ranking.
    What is claimed is what is unanimous: recall 1.000 for all six, and no model
    clearing the majority floor.

    All six ran at 100% GPU residency, checked per model rather than assumed.

### These numbers are not bit-reproducible, and the gate is

Worth stating precisely, because the two halves of this repository behave
differently and the difference is by design.

Re-running the five smaller models on cluster hardware changed **2 of 200
judgements**: `qwen2.5-3b` and `llama3.1-8b` each flipped exactly one task of forty;
the other three were identical. Temperature is 0 and the seed is fixed, so this is
runtime and quantization numerics moving a borderline verdict, not sampling.

The [gate](reference/gate.md), by contrast, reproduces **bit-identically** across the
same two machines. That asymmetry is the reason model calls are confined to
`pharos.attribute`: the acceptance decision that licenses a corpus cannot drift with
a backend, while the scores measured *on* that corpus carry roughly a
one-percent floor on cross-platform agreement. Quote model-dependent numbers with
the platform named.

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

| Condition | Shots | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Zero-shot floor | 0 | 0.389 | 0.875 | 0.538 |
| Labelled examples | 2 | 0.269 | 1.000 | 0.424 |
| Labelled examples | 4 | 0.292 | 0.875 | 0.438 |
| Labelled examples | 8 | 0.400 | 1.000 | 0.571 |
| **Rule stated, checklist prompt (ceiling)** | n/a | **1.000** | **1.000** | **1.000** |

**Examples close almost none of the gap.** The best few-shot condition closes 7% of
the distance from the zero-shot floor to the ceiling, and two of the three land
*below* the floor they were meant to lift. No condition clears its own
majority-class accuracy either, which sits between 0.724 and 0.759.

!!! warning "Corrected 2026-08-01"
    This table previously carried a second series -- examples annotated with the
    officer's stated reason -- and a different set of F1 values. No committed
    script produces that condition and no committed artifact holds those numbers,
    so both are withdrawn. The values above are `results/learnability.json`, which
    is what `scripts/measure_rule_learnability.py` writes and what the manuscript's
    table is generated from. The conclusion is unchanged; only the evidence for it
    is now checkable.

Three caveats, because this result is easy to over-read:

- Thirty evaluation tasks per condition. Differences of about 0.1 sit inside the
  noise, so the ordering *between* conditions is not claimed, only that none
  reaches the ceiling.
- Eight shots is roughly 1,300 words of examples before the target case, so
  long-context dilution is an unseparated confound.
- **In-context learning is not gradient learning.** Eight examples in a prompt and
  a LoRA trained on 1,140 tuples are different mechanisms, and the first failing
  does not establish that the second will.

The useful conclusion is about sequencing. The cheap proxy for the design's
central premise came back negative-to-inconclusive, so the premise cannot be
validated cheaply and **the adapter experiment is no longer optional: it is the
thing that decides.** What the ceiling establishes is where the bottleneck sits. A
model that reaches F1 1.000 when told the rule is not short of capability, it is
short of the rule, so rule *acquisition* is the whole question.

## 6. Gradient learning does close the gap, on clean labels

`scripts/train_adapter.py`, `results/adapter_learnability.json`

Finding 5 ended by saying the adapter experiment was the thing that decides. It has
now run: a LoRA fine-tune of Qwen2.5-3B-Instruct on an A100-40GB, rank 16, three
epochs over 1,140 training tuples, evaluated on 60 held-out tasks.

| | Accuracy | Majority | Precision | Recall | F1 | Unparsed |
| --- | --- | --- | --- | --- | --- | --- |
| Base | 0.300 | 0.717 | 0.319 | 0.882 | 0.469 | 8 / 60 |
| **Adapter** | **1.000** | 0.667 | 1.000 | 1.000 | **1.000** | 0 / 60 |
| Ceiling (rule stated, checklist) | 1.000 | | 1.000 | 1.000 | 1.000 | |

**The rule is gradient-learnable where in-context examples reached none of it.**
That is the substantive result, and it is the cleanest contrast in this document:
same withheld rule, same task, same ceiling, and the mechanism is the only thing
that changed.

Four things that keep this honest:

- **The split is clean, and this was verified rather than assumed.** One task per
  event, so an index split is an event split: zero event overlap, zero exact-text
  overlap between evaluation and training, and zero duplicate texts corpus-wide.
  A perfect score is the shape of a leak, so the leak was the first thing checked.
- **The score is saturated, and saturation is not the same as certainty.** Zero
  errors in 60 bounds the true error rate only at 5% by the rule of three, and a
  ceiling cannot rank anything above it. `pharos.validity` now flags this
  automatically; it did not before, and the gap was found by this run passing every
  existing check while plainly deserving a caveat.
- **The base row is worse than it looks and should not be quoted alone.** It sits
  below its own majority floor and left 8 of 60 answers unparsable, both of which
  the validity checks flagged at runtime. It is a fair starting point for a delta,
  not a measurement of the model.
- **Clean labels are not analyst decisions.** The training signal here is the
  generator's ground truth. The design's actual premise is learning from accept,
  revise, and reject -- indirect, noisier, and far less abundant. This result
  removes the *capability* objection and leaves the *supervision* question
  untouched, which is the next experiment rather than a conclusion of this one.
