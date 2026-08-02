# Findings

What has been measured, and the script that reproduces each number. The
**argument** these support belongs to the manuscript; this page is the index.

Findings 1 to 3 and 5 were measured with `qwen2.5:7b-instruct` on an 8 GB RTX 3060
Ti, finding 5 at 600 evaluation tasks. Finding 3b sweeps six models from 3B to 14B,
and findings 6 and 10 fine-tune `Qwen2.5-3B-Instruct`; both need a cluster A100, and
finding 10's 600-task evaluation is a 2h39m job on one. Findings 7
and 8 call no model at all: they regenerate the corpus from its seed and review
verdicts already committed to `results/`, so both reproduce exactly and run in CI.
Finding 9 is a measurement-design result and retracts an earlier version of itself.
Finding 11 calls no model either -- it is a property of the corpus's label structure
and of who can read what -- so it also reproduces exactly and runs in CI.

Every artifact in `results/` carries the version, commit, platform, model, and seed
behind it, so any number here traces back to the run and the machine that produced
it. That matters because the model-dependent numbers are not bit-reproducible across
machines, while the gate is.

!!! note "Regenerating"
    `make results` reruns the four Ollama-backed measurements into `results/`. It
    needs Ollama serving the model each script names, and it is slow: the label
    fidelity pass alone is 216 sequential model calls. Findings 3b and 6 are cluster
    jobs and are not part of this target. `make review` regenerates finding 7 and
    `make linkage` finding 11; neither needs a model.

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

## What these sizes can resolve

`scripts/measure_power.py`, `results/power.json`

Model-dependent findings here run anywhere from 24 to 600 tasks, and that spread is
the point rather than an inconsistency: the effects claimed differ by an order of
magnitude, so the size each one needs does too. This prices every claim instead of
arguing about a single convention, and the two claims it flagged as underpowered have
since been rerun at the size it named. The interval machinery is the same one the
measurements use, and the class balance is read from the corpus rather than assumed.

| Evaluation size | 95% half-width | Smallest resolvable difference |
| --- | --- | --- |
| 24 | 0.208 | 0.417 |
| 30 | 0.183 | 0.367 |
| 40 | 0.150 | 0.300 |
| 60 | 0.117 | 0.233 |
| 120 | 0.079 | 0.158 |
| 240 | 0.065 | 0.129 |
| 600 | 0.035 | 0.070 |
| 2000 | 0.020 | 0.041 |

The second column is the number that matters: two conditions separate only when the
gap between them exceeds roughly one half-width on each side, which is exactly how
`uncertainty.resolves` decides.

**Every headline claim against the size it was actually run at:**

A claim measured against a *fixed* reference is cheaper to resolve than one comparing
two measured conditions, and the table prices them differently. The majority floor,
the stated-rule ceiling, and an adversary's guessing prior are all computed exactly
from a generated corpus, so they carry no sampling noise of their own and the gap has
to clear one half-width rather than two. Charging them double was an error in an
earlier version of this section, and it understated two claims.

| Finding | n | Gap it rests on | vs | Verdict | Claim |
| --- | --- | --- | --- | --- | --- |
| 3b | 40 | 0.000 | constant | unresolved (needs n>2000) | qwen2.5-3b (0.625) clears the majority floor (0.625) |
| 3b | 40 | 0.200 | constant | **resolved** | mistral-7b (0.425) is below the majority floor (0.625) |
| 5 | 600 | 0.009 | condition | unresolved (needs n>2000) | 8 shots (0.514) beats 0 shots (0.523) -- REFUTED at n=600 |
| 5 | 600 | 0.055 | condition | unresolved (needs n≥2000) | 2 shots (0.468) is worse than 0 shots (0.523) -- direction only |
| 5 | 600 | 0.486 | constant | **resolved** | 8 shots (0.514) is below the stated-rule ceiling (1.000) |
| 5 | 600 | 0.179 | constant | **resolved** | 8 shots (0.514) is below the majority floor (0.693) |
| 6 | 60 | 0.531 | condition | **resolved** | adapter (1.000) beats the base model (0.469) |
| 10 | 600 | 0.560 | condition | **resolved** | any-one adapter matches teacher (1.000) not world (0.440) |
| 10 | 600 | 0.118 | condition | **resolved** | inattentive adapter (0.893) beats its own teacher (0.775) |
| 11 | 200 | 0.100 | constant | **resolved** | linkage recovery (0.205) beats the guessing prior (0.105) |
| 11 | 50 | 0.820 | condition | **resolved** | RESTRICTED analysts (0.820) are recovered where OPEN (0.000) are not |

Eight of eleven are resolved, and the pattern is the useful part. **The claims this
project makes strongly rest on large gaps and are comfortably resolved**; the ones
that are not are mostly ones it already declines to make. Two deserve naming:

!!! warning "Corrected 2026-08-01"
    This table previously charged every claim two half-widths, including those
    measured against an exactly-known reference, and reported "mistral-7b is below
    the majority floor" as unresolved at n=40. That contradicted
    [finding 3b's own interval table](#3b-over-escalation-is-universal-and-scale-does-not-fix-it),
    where mistral's `[0.275, 0.600]` excludes the 0.625 floor outright. The
    inconsistency was in this section, not in finding 3b, and the corrected
    accounting reconciles the two. No measurement was rerun and no score changed.

- **"qwen2.5-3b clears the majority floor" has a gap of exactly zero.** It ties the
  floor at 0.625. No sample size resolves a tie, so that is not a claim awaiting
  more data -- it is permanently unresolvable as posed, and
  [finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) states it
  as unresolved rather than as a negative.
- **"8 shots beats 0 shots" was bought, and the claim died.** This table used to say
  it needed n≈600, an order of magnitude more than the 30 it was measured at. We ran
  600. The lift is not there:
  [finding 5](#5-in-context-learning-does-not-close-the-gap) now reports 8 shots at
  0.514 against a zero-shot 0.523, and the row above records the refutation rather
  than deleting it. The companion claim, that 2 shots is *worse* than 0, is the
  closest thing to a resolved difference here and still is not one.

That second point is why this section is not a limitation. A static benchmark with 30
items is stuck with 30; a generator is limited only by compute, and 600 tasks is
roughly two hours of local inference. The table is therefore a purchase order rather
than a disclaimer, and the two purchases it has prompted so far both ended in a
retraction. That is the instrument working: an unresolved claim is not a quiet claim,
it is a claim with a price tag on it.

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

**No model beats the majority-class floor** -- but at n=40 that is partly a
statement about the sample. The best accuracy exactly ties it at 0.625, and a reader
comparing any F1 here to 0.5 would badly overstate what is happening. What the
intervals add is where the claim stops:

| Model | Accuracy | 95% interval | Clears 0.625? |
| --- | --- | --- | --- |
| qwen2.5-3b | 0.625 | [0.475, 0.775] | **unresolved** |
| qwen2.5-14b | 0.525 | [0.375, 0.675] | **unresolved** |
| llama3.1-8b | 0.500 | [0.325, 0.675] | **unresolved** |
| llama3.2-3b | 0.475 | [0.325, 0.625] | no |
| qwen2.5-7b | 0.450 | [0.300, 0.625] | no |
| mistral-7b | 0.425 | [0.275, 0.600] | no |

Cluster bootstrap over tasks, computed from the per-task rows in the committed
artifacts by `scripts/compare_models.py`. No model was re-called.

Three intervals reach above the floor. For those the honest statement is *this
sample does not show them clearing it*, which is weaker than *they do not clear it*.
The unanimous claims -- recall exactly 1.000, precision between 0.395 and 0.500 --
are unaffected, because they are not close calls.

!!! warning "Corrected 2026-08-01"
    This paragraph read "no model beats the majority-class floor" without
    qualification. That is right about the point estimates and overstates what 40
    tasks establish. The intervals were computed post hoc from artifacts that
    already existed; nothing was remeasured, and no number in the table above
    changed.

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

`scripts/measure_rule_learnability.py`, `results/learnability.json`

The design's premise is that a fleet learns analytic craft from an analyst's
accept, revise, and reject decisions. Withholding the rule and supplying labelled
examples instead is the cheap form of that question.

**600 evaluation tasks per condition**, one pass each, `qwen2.5:7b-instruct`.

| Condition | Shots | Precision | Recall | F1 | Accuracy | 95% interval | Unparsed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot floor | 0 | 0.382 | 0.918 | 0.540 | 0.523 | [0.484, 0.565] | 2 |
| Labelled examples | 2 | 0.365 | 0.972 | 0.530 | **0.468** | [0.427, 0.509] | 17 |
| Labelled examples | 4 | 0.391 | 0.978 | 0.559 | 0.515 | [0.475, 0.557] | 21 |
| Labelled examples | 8 | 0.385 | 0.978 | 0.553 | 0.514 | [0.473, 0.556] | 1 |
| **Rule stated, checklist prompt (ceiling)** | n/a | **1.000** | **1.000** | **1.000** | **1.000** | | |

**Examples close 4% of the gap to the ceiling**, and no condition comes near its own
majority-class accuracy, which sits at 0.693. The floor is not close: the best
condition is 0.18 below it.

**Two shots looks *worse* than none, and the data does not quite resolve it.** The
gap runs the opposite way to what a reader expects, 0.468 against 0.523, and it is
the closest any pair here comes to separating. It does not separate. The gap is
0.0551 against a difference half-width of 0.0576, so the honest verdict is a
direction, not a difference.

That distinction is worth spelling out because two criteria disagree on this pair and
only one of them is right. Neither interval covers the other's point, which is what
`uncertainty.resolves` tests, and on that basis an earlier draft of this section
claimed the comparison as resolved. But "do two conditions differ" is a question about
an interval on their *difference*, and that interval is wider than either input.
`uncertainty.resolves_difference` applies the correct rule and returns false here.
The weaker test was mislabelled "conservative" in its own docstring; it is the
permissive one, and it has been corrected.

What the other columns show is a coherent mechanism for the direction, whether or not
the size is resolved. Recall climbs from 0.918 to 0.972 while precision falls from
0.382 to 0.365, so the examples are not teaching the rule, they are teaching the model
to escalate more. That is the same failure
[finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) found in
every model tested, and supplying examples does not repair it.

No pair separates under the difference test. Four, eight, and zero shots are
indistinguishable from each other at this size.

!!! warning "Retracted 2026-08-01: the eight-shot lift"
    An earlier version of this table, measured at **30** tasks, reported 8 shots at
    accuracy 0.571 against a zero-shot floor of 0.493 and described the best
    condition as closing 19% of the gap to the ceiling. The
    [power analysis](#what-these-sizes-can-resolve) priced that 0.078 gap as
    unresolvable at 30 tasks and put the size needed at roughly 600.

    Remeasured at 600: **8 shots scores 0.514 against a zero-shot 0.523**. The lift
    is not smaller than reported, it is absent, and the point estimate is slightly
    negative. The claim is withdrawn.

    One further statement from that version survives and is worth keeping: no pair of
    shot counts is separated. That was true at 30 tasks and remains true at 600 under
    the difference test, which is a stronger result than it sounds, because it now
    rests on intervals a fifth as wide.

    This is the second finding here overturned by buying more evaluation rather than
    by any change of method, and both were flagged in advance by the power table.
    That is the table working as intended.

Three caveats, because this result is still easy to over-read:

- **Unparsed answers are not evenly distributed.** 21 of 600 at four shots against 1
  at eight, and an unparsed answer is carried separately rather than scored as wrong.
  The variation is unexplained and is a reason to treat the 4-versus-8 ordering as
  noise even beyond what the intervals say.
- Eight shots is roughly 1,300 words of examples before the target case, so
  long-context dilution is an unseparated confound.
- **In-context learning is not gradient learning.** Eight examples in a prompt and
  a LoRA trained on 1,140 tuples are different mechanisms, and the first failing
  does not establish that the second will --
  [finding 6](#6-gradient-learning-does-close-the-gap-on-clean-labels) shows it does
  not.

The useful conclusion is about sequencing, and it is now stronger than when the
lift looked real. The cheap proxy for the design's central premise came back
**negative**, not merely inconclusive, so the premise cannot be validated cheaply and
the adapter experiment is the thing that decides. What the ceiling establishes is
where the bottleneck sits: a model that reaches F1 1.000 when told the rule is not
short of capability, it is short of the rule, so rule *acquisition* is the whole
question.

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

## 7. Review is abundant; what it costs is correctness

`scripts/measure_analyst_review.py`, `results/analyst_review.json`

Finding 6 removed the capability objection and left the supervision question
untouched: the rule is learnable *from clean labels*, and the design's premise is
learning from an analyst's accept, revise, and reject decisions instead. This
measures what that substitution costs, at the boundary rather than only on the
verdict.

Eight reviewers, each a parameter grid entry in `pharos.analyst` differing from the
by-the-book reviewer on exactly one axis, review the 40 triage proposals each of the
six sweep models actually produced -- 1,920 decisions in all. The proposals are
real; the reviewers are not, and the caveat at the bottom of this section is the
load-bearing one.

| Reviewer | Differs by | Targets | Correct | Escalated | Recovered | Addressed |
| --- | --- | --- | --- | --- | --- | --- |
| by-the-book | (control) | 1.00 | 1.000 | 0.525 | 0.00 | **1.00** |
| two-of-three | threshold 2 | 1.00 | **0.575** | 0.525 | 0.00 | 1.00 |
| any-one | threshold 1 | 1.00 | **0.425** | 0.525 | 0.00 | 1.00 |
| inattentive | slips 15% | 1.00 | 0.875 | 0.525 | 0.00 | 1.00 |
| terse | revises 25% | 0.83-0.88 | 1.000 | 0.525 | 0.00 | 1.00 |
| unexplained | names no grounds | 1.00 | 1.000 | 0.525 | 0.00 | 1.00 |
| releaser | sheds compartments | 1.00 | 1.000 | 0.00 | **1.00** | 1.00 |
| no-escalation | will not escalate | 1.00 | 1.000 | 0.00 | 0.00 | **0.00** |

*Targets*: share of decisions handing the learner a verdict to train on. *Correct*:
share of those targets matching the world. *Escalated*: share of decisions passed to
an authority rather than settled. *Recovered*: share of the 21 unreleasable proposals
whose correction clears the ceiling unaided. *Addressed*: share either recovered or
escalated. The by-the-book row is a control, not a result: its threshold is the
world's own rule and it never slips, so 1.000 is what it is defined to produce.

**"Far less abundant" was wrong, and this project wrote it.** The README and finding
6 both described analyst decisions as scarcer than labels. For a binary verdict they
are not: an acceptance is a target, a revision is a target, an escalation carries one
too, and only a bare rejection costs anything. A reviewer who revises just a quarter
of the time still supplies a usable target on 0.83 to 0.88 of decisions. Scarcity is
the wrong worry.

**The targets are the reviewer's opinion, not the model's.** Target accuracy is
identical across all six models -- not close, identical, because a corrected verdict
is the reviewer's own call and an accepted one is a call the reviewer agreed with.
Review does not import what the model knew; it exports the reviewer's standard. That
is the mechanism behind the next row, and it is why the number is a property of the
reviewer alone.

**A reviewer with the wrong threshold teaches a rule that loses to guessing.** At
threshold 2 the target stream matches the world on 0.575, at threshold 1 on 0.425.
The majority floor on this evaluation set is 0.625. Both are below it, so a learner
trained on those corrections would do worse than always answering ROUTINE -- and it
would do so while every acceptance and revision looked like ordinary supervision.
This is not a hypothetical reviewer. Threshold 1 is over-escalation, and finding 3b
measured over-escalation in **every** model tested, recall 1.000 at precision 0.395
to 0.500. A fleet learning from reviewers who fail on the axis its models already
fail on does not get corrected; it gets confirmed.

**Unnamed grounds cost nothing in volume and everything in attribution.** The
`unexplained` reviewer accepts and revises exactly as often as by-the-book, and
supplies exactly as many correct targets, so no yield statistic distinguishes them.
What it withholds is which of the two assertions was wrong. A proposal is a verdict
*and* the label it would be released under, so an objection that does not say which
one failed asks the learner to fix an error it cannot locate. That axis exists
because the output is governed; a preference over ungoverned text has nothing
corresponding to it.

**The disclosure boundary moves, but only by an authority, and the toll is half the
stream.** This paragraph replaces one that was wrong, and the correction is the most
useful thing in this finding.

Of the 40 proposals, 21 carry a compartment the aggregator is not cleared for. No
keep-compartments reviewer's own correction clears the ceiling -- recovery is
**0.00**, exactly as first measured, and that part reproduces. The first version
concluded from it that *review cannot move the disclosure boundary*. That conclusion
was an artifact of the instrument: `shared_eligible` returned a boolean, so a
reviewer had two doors, and the option a real analyst reaches for first was not
representable.

Given a third disposition, all 21 are **authorizable**: the block is a compartment
shortfall, which finding 2 had already identified as a policy act rather than an
engineering problem, and a policy act is what an authority exists to perform. Every
escalating reviewer addresses **1.00** of the blocked set. The `no-escalation` row
differs from `by-the-book` on that one parameter and addresses **0.00**, so the whole
of the original claim sits in that parameter.

What replaces it is a cost rather than an impossibility. **21 of 40 decisions become
escalations** -- 52.5% of the review stream lands on whoever rules on compartments,
for every reviewer applying the fail-closed default. The boundary is movable; what a
fleet has to budget for is the authority's queue. The reviewer who sheds compartments
escalates nothing and recovers everything, which is the same bimodality finding 2
reported, now with its price visible on both sides.

Ensemble agreement over the four actions is Fleiss' kappa 0.435 to 0.450 across the
six models. That number is a property of the grid this project
chose and should not be read as an estimate of anything.

!!! warning "Corrected 2026-08-01: the release claim"
    First published as *review cannot unblock the disclosure boundary*, on a run
    where `shared_eligible` returned a boolean and a reviewer therefore had two
    doors. Recovery of 0.00 reproduces exactly; the conclusion drawn from it does
    not. With a third disposition every blocked proposal is addressed, and the real
    result is the 52.5% escalation load that replaces the impossibility. The
    `no-escalation` row reproduces the original run and is kept for that purpose.

!!! warning "The reviewers are parameters, not people"
    A simulated analyst here is a decision procedure with named parameters, chosen so
    the feedback-generating process has ground truth by construction. It is
    deliberately not a persona-prompted language model: that technique explains only
    a small share of real annotator variance and compresses disagreement rather than
    reproducing it, which would put an unquantified error in the position of the
    thing being measured. The consequence is that every row above bounds a
    *mechanism* -- at this threshold, this slip rate, this feedback bandwidth, this
    much survives -- and none of them estimates what a human analyst would do. The
    corpus metadata carries the same cap under `rai:dataLimitations`.

## 8. Being right and sloppy beats being wrong and careful

`scripts/measure_review_sweep.py`, `results/review_sweep.json`

Finding 7 reported target accuracy for eight named reviewers. Each was one point in
a space, and a point cannot say where the edge is. This sweeps the two axes that
move it, over 40 tasks and five review seeds per cell.

The two axes differ in kind, and separating them is the whole point. A reviewer's
**standard** is how many of the three defining facts they require; getting it wrong
is a *systematic* error, reproducible and patterned. Their **carefulness** is how
often they fail to apply their own standard; getting that wrong is *random* error.

Target accuracy against the world, `*` marking cells that clear the 0.625 majority
floor:

| Standard | slip 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 | 0.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| needs 1 of 3 | 0.425 | 0.420 | 0.445 | 0.430 | 0.425 | 0.475 | 0.480 | 0.525 |
| needs 2 of 3 | 0.575 | 0.575 | 0.580 | 0.540 | 0.550 | 0.505 | 0.515 | 0.465 |
| **needs 3 of 3** | **1.000**\* | 0.965\* | 0.920\* | 0.830\* | 0.815\* | 0.685\* | 0.520 | 0.540 |

**No wrong standard clears the floor at any carefulness.** Every cell in the top two
rows sits below 0.625, including the ones where the reviewer never slips. A learner
handed those targets would do better ignoring them and always answering ROUTINE.
This is asserted over the whole grid in the test suite rather than spot-checked,
because it is the claim the finding rests on.

**A correct standard tolerates a great deal of carelessness.** It clears the floor
through a 30% slip rate (0.685) and only fails at 40%. So the exchange rate is
lopsided: **one step of standard error costs what a 40% slip rate costs**, and two
steps cost more than any slip rate measured up to 50%.

That is the quantitative form of a qualitative claim in the learning-from-noisy-
labels literature -- that systematic annotator bias damages a model more than random
noise, because the errors are class-dependent and get learned rather than averaged
away. Here the ratio is roughly eight to one against the systematic error.

**Noise partially rescues a wrong standard, which is the counterintuitive part.** The
needs-1-of-3 reviewer *improves* from 0.425 to 0.525 as slip rises to 50%, and the
correct reviewer falls to 0.540 at the same point. Both converge toward chance,
because a reviewer flipping coins is no longer applying any rule. Confidently wrong
is worse than random, and adding noise to a badly biased reviewer moves them toward
the middle rather than further away.

**What this means for a fleet.** Recruiting effort should go to establishing that
reviewers hold the right standard, not to keeping them attentive. The usual
annotation-quality apparatus -- attention checks, redundancy, inter-rater agreement
-- measures carefulness, and every reviewer in the top two rows would pass all of it
while agreeing perfectly with themselves. Agreement is not correctness, and here it
is not even correlated with it.

!!! warning "What this does not measure"
    Target accuracy is a property of the review stream, not of a model trained on
    it. Whether a learner reproduces the reviewer's standard, or is pulled toward
    the world's rule by its own prior, is the open experiment: an adapter trained
    on each reviewer's targets and evaluated against **both** the world and the
    teacher. That is a GPU job and it has not been run. The reviewers here remain
    parameterised decision procedures, so every row bounds a mechanism and
    estimates no population.


## 9. A measurement that repeats one prompt measures the wrong thing

`scripts/measure_decode_stability.py`, `results/decode_stability.json`

!!! danger "Retracted and replaced 2026-08-01, same day it was published"
    This finding first claimed that **a single-pass score is not reproducible when
    the model reasons**, on the strength of 3 of 30 tasks disagreeing across
    identical calls. That claim is **false** and the measurement behind it was
    designed wrong. It is left here in corrected form rather than deleted, because
    the mistake is more useful than the original finding was.

The original probe called *one prompt* several times in a row and counted
disagreements. A real measurement calls each task *once*, in sequence. Those are
different experiments, and only the second one is the reproducibility question.

| Design | Rule stated, 8 tokens | Rule withheld, 320 tokens |
| --- | --- | --- |
| Repeat one prompt 3x | 0 / 30 | **4 / 30 (13.3%)** |
| Repeat the whole sweep | **0 / 30** | **0 / 30** |

**Single-pass measurements reproduce exactly.** Two complete sweeps, each task
called once in the same order, agree on every task in both decode regimes, and
produce identical accuracy.

**The 13.3% is a warm-up transition, not noise.** Following it produced the
mechanism: in every varied case the *first* call against a prompt differs and every
call after it is byte-identical. Five repeats of a summarization prompt gave
`[A, B, B, B, B]` on 3 of 3 varied prompts. So repeating one prompt measures the
cold-to-warm transition of a prefix cache and reports it as irreproducibility. A
real measurement never repeats a prompt, so every call is cold, and cold calls agree
with each other.

**What this cost, and what survives.** The retracted version was cited in a second
place before it was checked: finding 5's remeasurement warning attributed its
caution to a 10% instability that does not exist. What survives the correction is
worth keeping:

- **The between-task interval is real and was missing.** Thirty tasks is a small
  sample whatever the decode does, and finding 5's conditions genuinely do not
  separate ([see its table](#5-in-context-learning-does-not-close-the-gap)). That
  conclusion rests on sampling uncertainty across tasks, which the cluster bootstrap
  measures correctly and which no amount of reproducibility removes.
- **Finding 3b's correction stands entirely.** It was computed from single-pass
  artifacts by bootstrapping over tasks, and single-pass artifacts are now known to
  be exactly reproducible.
- **The cross-platform figure in the manuscript stands**, and its original reading
  was right. Two of 200 judgements changing across machines is a platform effect,
  measured under a decode that reproduces perfectly on one machine.

**What generalises.** Not a claim about temperature or decode length, but about
measurement design: *a probe that repeats one input is not measuring what a sweep
over many inputs will do*, and if the backend caches anything, the two differ
systematically rather than randomly. That is worth checking before reporting either.

!!! note "Where this leaves the uncertainty layer"
    `pharos.uncertainty` was built for the retracted claim and is kept, because the
    thing it actually measures -- sampling uncertainty over tasks -- was missing
    before and is still missing everywhere else. Its within-task component is now
    known to capture a cache transition rather than noise, and
    [its reference page](reference/uncertainty.md) says so.

## 10. A fleet learns its analyst's standard, not the world's

`scripts/train_adapter.py --reviewer`, `cluster/review-adapter.sbatch`,
`results/review_adapter-*.json`

Finding 6 trained an adapter on the generator's ground truth and reached F1 1.000,
which settled capability. Finding 8 then showed a reviewer's target stream can be
systematically wrong and that no amount of carefulness repairs it. Neither said what
a *model* does with such targets. This does.

Four LoRA adapters on Qwen2.5-3B-Instruct, A100-40GB, 1,140 training tuples each,
identical except for who labelled them. Every adapter is scored twice on the same 60
held-out tasks and the same decode: once against the world, once against **its own
teacher's answers**.

| Teacher | Standard | Targets matching world | Adapter vs **world** | Adapter vs **teacher** | Inherited error |
| --- | --- | --- | --- | --- | --- |
| by-the-book | 3 of 3 | 1.000 | 0.995 | 0.995 | -0.005 |
| inattentive | 3 of 3, slips 15% | 0.855 | **0.893** | 0.775 | **+0.038** |
| two-of-three | 2 of 3 | 0.728 | 0.732 | **0.983** | +0.004 |
| any-one | 1 of 3 | 0.439 | 0.440 | **1.000** | +0.001 |

Accuracy on **600** held-out tasks whose events are disjoint from training. The base
model scores accuracy 0.317 with 74 of 600 answers unparsable, so every row is a large
gain over not training at all. The last column is the adapter's agreement with the
world minus its teacher's, and it is the cleanest statement of the finding.

**A wrong standard is inherited almost exactly.** The two systematically-wrong
teachers hand over their error rate to within **0.004 and 0.001**: a teacher whose
targets agree with the world on 0.728 produces an adapter agreeing on 0.732, and one
at 0.439 produces 0.440. The adapter did not partially absorb the reviewer's rule or
average it against its prior; it learned "two of three" and "one of three" and now
agrees with the world exactly as much as that rule does. This is the concrete form of
what the noisy-annotation literature calls absorbing an unreliable annotator's errors
into the parameters, and at 600 tasks the correspondence is close enough to read as an
identity rather than a tendency.

**Random error is filtered; systematic error is not.** The contrast is the finding,
and the careless teacher is the only row that breaks the inheritance pattern. It slips
15% of the time, its targets match the world on 0.855, and its adapter matches the
world on 0.893 while matching the teacher on only 0.775. It tracks the underlying rule
more closely than the teacher that taught it. Given a consistent wrong rule instead,
the adapter simply became that rule.

!!! success "Bought and confirmed 2026-08-01"
    This was the one claim here the [power analysis](#what-these-sizes-can-resolve)
    priced as unresolvable at the size it was run. At 60 tasks the gap between the
    careless adapter and its teacher was 0.083 against a minimum detectable 0.233, and
    this section said so and declined to claim a margin.

    Remeasured at 600 the gap **widened to 0.118**, against a difference half-width of
    0.042, so it now resolves. Both figures score one decode against two references
    over the same evaluation set, which makes the real test paired and therefore
    tighter than the independent one quoted. The direction reported at 60 tasks was
    right, and the margin is now measured rather than declined.

    Worth stating alongside [finding 5](#5-in-context-learning-does-not-close-the-gap),
    where the same purchase destroyed the claim. The power table is not a retraction
    machine; it prices claims, and the prices come back both ways.

So the two error types finding 8 separated at the level of the target stream stay
separated after training. Carelessness is partly repaired by learning, by nearly four
points. **Wrongness is preserved by it** to within half a point, which is the more
unsettling half of the result: there is no dilution, no regression toward the model's
prior, and nothing in the training signal that notices the rule is wrong.

**What this means for the design.** The premise is that a fleet learns analytic
tradecraft from analyst decisions. It does -- exactly, faithfully, and without any
check on whether the tradecraft is correct. Personalization works as specified, and
that *is* the risk: an over-escalating analyst produces an over-escalating agent, and
over-escalation is what every model in
[finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) already
does. The mitigation is not more data. It is annotator-reliability weighting, which
needs annotator identity retained through training -- something Pharos has by
construction and most annotation pipelines discard.

!!! note "What this does not settle"
    Four teachers, one model, one corpus, one seed, single-pass evaluation. Every
    contrast in the table is now resolved at 600 tasks, including the careless
    teacher's, so the caveat that remains is about generality rather than about size.
    The teachers are parameterised decision procedures, so this bounds a mechanism and
    estimates nothing about human analysts. One model and one corpus is the real
    limit: that a 3B Qwen inherits a wrong rule this exactly says nothing yet about
    whether a larger model or a different task shape would.

## 11. The gate clears every item, and the stream still names the analyst

`scripts/measure_fleet_linkage.py`, `src/pharos/fleet.py`, `results/fleet_linkage.json`

Every gate above decides one item at a time. `shared_eligible` asks whether *this*
entry may train an adapter released at a ceiling; `disclosure.decide` asks whether
*this* derived label may leave and on whose authority. Both are correct at what they
do, and neither is asked the question the design's motivating claim actually turns
on: whether a fleet can learn across analysts **without leaking what any analyst
works on**.

A fleet does not disclose an item. It discloses a stream, and a stream carries
something no item in it carries -- the set of tasks a contributor was able to weigh
in on at all. That set is a function of the contributor's clearance, because an
analyst can only decide what an analyst can read.

The attack reads **no content**: not the report text, not the verdict, not the
released label. It sees only which task identifiers appear under a pseudonym, and
matches that against what each of the 16 candidate compartment sets would have been
able to reach. That is the sharpest available form of the claim, because there is no
text in it to redact and no capacity to lower. "Reads no content" is enforced rather
than asserted: a test flips every verdict in the stream and requires the attack's
output to be unchanged, so a future edit that lets content reach the scorer fails the
build.

200 tasks, a fleet of 200 analysts drawn from the full clearance space, every item in
the stream already approved by the per-item gates.

| Ruling | Contributions | Compartment set recovered | Mean anonymity set |
| --- | --- | --- | --- |
| shed compartments | 7,053 | **0.205** [0.150, 0.265] | 10.0 |
| keep compartments | 1,000 | 0.000 [0.000, 0.000] | 16.0 |

Guessing the fleet's most common beat scores 0.105, and the interval excludes it, so
the attack is doing real work. But the aggregate is the least interesting number here.

**Identifiability is not spread evenly across the fleet. It concentrates on exactly
the analysts with the most to protect.**

| Contributor's own clearance | n | Recovered | Mean anonymity set |
| --- | --- | --- | --- |
| OPEN | 66 | 0.000 [0.000, 0.000] | 16.0 |
| INTERNAL | 48 | 0.000 [0.000, 0.000] | 12.0 |
| PROTECTED | 36 | 0.000 [0.000, 0.000] | 7.0 |
| RESTRICTED | 50 | **0.820** [0.700, 0.920] | 2.3 |

The mechanism is structural rather than statistical. Of the 64 candidate clearances,
only **17 produce distinct reachable task sets**, and 12 of those 17 belong to
exactly one compartment set while a single set is shared by all 16. A cleared analyst
can reach the rare, heavily compartmented tasks, and those tasks are diagnostic; an
uncleared one reaches only the handful of uncompartmented tasks everyone reaches, and
is hidden in the whole candidate space. The pattern holds across five fleet draws
(RESTRICTED 0.720 to 0.820, every other level exactly 0.000), so it is not an artifact
of one sample.

**Nor of one corpus.** Across five corpus seeds the reachable-set *sizes* differ
substantially, yet the collapse structure and the recovery rate are identical to three
decimals. That is not a coincidence and not a bug: which compartment sets are mutually
distinguishable is a property of the label lattice and the channel-to-compartment
mapping, not of which events happened to be generated. It becomes so once the corpus
is large enough to populate every compartment cell, which the saturation grid puts at
**150 events**:

| Events | Structures across 5 corpus seeds | Recovery |
| --- | --- | --- |
| 20 | 5 distinct | 0.000 to 0.205 |
| 40 | 5 distinct | 0.075 to 0.205 |
| 80 | 2 distinct | 0.170 to 0.205 |
| **150** | **1** | **0.205** |
| 200 | 1 | 0.205 |
| 400 | 1 | 0.205 |

Two consequences. The headline runs at 200 events, above saturation, so it measures
the lattice rather than a draw. And every small corpus **understates** the leak rather
than overstating it, which is the safe direction but means a short run cannot be
quoted as reassurance. It also settles which uncertainty belongs on the number: the
interval is over the fleet, because the fleet is the only thing being sampled.

A tie is scored as a failure to identify, never as a hit. That is why three of the
four rows are exactly zero: an adversary left choosing among seven candidates has not
named anyone, and crediting the tie-break for a coin flip would report a convention as
a result.

**The controls, each priced in the training volume it destroys.**

| Control | Recovered | Anonymity set | Volume kept | Analysts silenced |
| --- | --- | --- | --- | --- |
| none | 0.205 | 10.0 | 1.000 | 0 |
| k-anonymity, k=10 | 0.205 | 10.0 | 0.992 | 0 |
| k-anonymity, k=25 | 0.130 | 10.0 | 0.845 | 0 |
| k-anonymity, k=50 | 0.000 | 13.3 | 0.265 | 0 |
| suppress rarest 25% | 0.130 | 10.1 | 0.925 | 0 |
| **suppress rarest 50%** | **0.000** | 12.5 | **0.764** | 0 |
| subsample p=0.5 | 0.185 | 10.1 | 0.511 | 5 |
| subsample p=0.05 | 0.025 | 8.6 | 0.048 | 108 |
| **pool contributors** | **0.000** | **200.0** | **1.000** | 0 |

Three things fall out of that table.

**Textbook k-anonymity at a small k is a no-op**, because in a fleet of 200 even a
rare clearance cell has several holders: k=10 changes nothing and costs almost
nothing. It only bites at k=50, where it takes 73% of the training data with it.
Choosing the threshold by rank instead of as an absolute count reaches the same
protection for under a third of that cost (23.6% of the volume against 73.5%), and is
the version worth deploying.

**Subsampling is dominated on both axes.** At p=0.05 it still leaks while destroying
95% of the volume and silencing 108 of 200 analysts. It removes contributions
uniformly when the identifying signal is concentrated in the rare ones. (This was our
first guess at the mechanism and it was wrong -- we assumed the signal lived in the
*pattern of absences* and would survive suppression of rare tasks. Suppressing the
rarest half drives recovery to zero, so it does not.)

**Pooling contributors is total and free.** Recovery goes to zero, the anonymity set
becomes the whole fleet, and not one training example is lost. The reason it costs
nothing is structural to the design rather than lucky: personalization is the part
that stays local, so the shared side never needed to know who sent what. The cost
that remains is the secure-aggregation protocol itself, which is real engineering and
is not modelled here.

**The tension this creates with finding 10.** Finding 10 concluded that the
mitigation for learning a wrong standard is annotator-reliability weighting, which
*requires annotator identity retained through training*. This finding says annotator
identity is exactly what makes the leak attributable to a person. The two mitigations
are in direct conflict, and the conflict is not resolvable by tuning: one wants a
per-contributor weight and the other wants no per-contributor anything. Naming it is
worth more than picking a side, and the design owes an answer -- most likely a
reliability estimate computed under the same secure aggregation that hides the
identity, which neither finding measures.

**What this means for the design.** A disclosure gate that certifies items is not a
privacy mechanism for a fleet, and no amount of per-item correctness makes it one.
The ruling that makes federation viable at all
([finding 2](#2-the-design-is-bimodal-on-one-policy-ruling)) is the same ruling that
opens this channel: shedding compartments takes contributions from 1,000 to 7,053 and
recovery from 0.000 to 0.205. Privacy and utility here are one knob, not two, and it
is the knob finding 2 already identified as the design's hinge.

!!! note "What this does not settle"
    One corpus, one adversary, and a clearance-recovery attack rather than a
    membership or reconstruction attack. The adversary is given the corpus's label
    structure, which is honest for a published benchmark and generous for a deployed
    system. Analysts are parameterised clearances, not people, and a real watch floor
    would show correlations between beat and workload that this does not model --
    every one of which would make the attack easier, not harder. Nothing here measures
    a trained model: the leak is in the contribution stream, upstream of any gradient.

    **The load-bearing assumption is that the aggregator can tell which task a
    contribution concerns.** That holds when the federation unit is a labeled example,
    which is this design's premise and the reason its disclosure gate operates on
    content at all. It does *not* hold when only gradients are shared, and the
    established federated-inference attacks target that gradient channel instead. So
    this result does not transfer to gradient-only federation, and the gradient-channel
    attacks do not transfer here. Naming the channel is the whole of the claim.
