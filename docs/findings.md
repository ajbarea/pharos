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
and of who can read what -- so it also reproduces exactly and runs in CI. Findings 18
to 21 are the same kind: they are properties of an estimator and a protocol over the
committed corpus and call nothing. The masked **aggregate** is bit-identical by
construction, being integer; the EM around it is float and runs through `libm`, so it
reproduces to float tolerance rather than bit-for-bit. This project has already
retracted an unqualified "bit-identical" twice, and this is the scope that survives.

Every artifact in `results/` carries the version, commit, platform, model, and seed
behind it, so any number here traces back to the run and the machine that produced
it. That matters because the model-dependent numbers are not bit-reproducible across
machines, while the gate is.

!!! note "Regenerating"
    `make results` reruns the four Ollama-backed measurements into `results/`. It
    needs Ollama serving the model each script names, and it is slow: the label
    fidelity pass alone is 360 sequential model calls. Findings 3b and 6 are cluster
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

## The findings, by what they are about

Grouped for navigation only. The numbering is chronological, so a group's members are
scattered through it, and nothing in the grouping is a claim about how they relate.

| | |
| --- | --- |
| **Attribution and policy** | [1. Leave-one-out cannot produce a governed label](#1-leave-one-out-attribution-cannot-produce-a-correct-governed-label) · [2. Bimodal on one policy ruling](#2-the-design-is-bimodal-on-one-policy-ruling) |
| **Triage baselines and scale** | [3. A corpus bug and a retracted finding](#3-a-corpus-bug-a-retracted-finding-and-a-real-benchmark-target) · [3b. Over-escalation is universal](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) · [4. Answerability against non-leakage](#4-answerability-and-surface-non-leakage-pull-against-each-other) |
| **Learning the rule** | [5. In-context learning does not close the gap](#5-in-context-learning-does-not-close-the-gap) · [6. Gradient learning does, on clean labels](#6-gradient-learning-does-close-the-gap-on-clean-labels) · [10. A fleet learns its analyst's standard](#10-a-fleet-learns-its-analysts-standard-not-the-worlds) |
| **What review costs** | [7. Review is abundant; correctness is not](#7-review-is-abundant-what-it-costs-is-correctness) · [8. Right and sloppy beats wrong and careful](#8-being-right-and-sloppy-beats-being-wrong-and-careful) |
| **Measurement design** | [9. Repeating one prompt measures the wrong thing](#9-a-measurement-that-repeats-one-prompt-measures-the-wrong-thing) · [17. Item difficulty does not separate the two](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst) · [27. Four guards could be inverted with the suite still green](#27-four-guards-could-be-inverted-with-the-suite-still-green-and-coverage-called-them-all-covered) |
| **Disclosure and identity** | [11. The stream still names the analyst](#11-the-gate-clears-every-item-and-the-stream-still-names-the-analyst) · [12. Reliability needs identity where it matters](#12-reliability-cannot-be-estimated-without-identity-where-it-matters) · [13. A tag can replace identity](#13-a-reliability-tag-can-replace-identity-and-the-leak-metric-cannot-tell-you-when) · [16. The cliff assumes independent fleets](#16-the-cliff-is-safe-only-because-the-fleets-were-drawn-independently) |
| **Cost of running it** | [14. What the agent costs on its hardware](#14-what-the-agent-costs-on-the-hardware-it-is-meant-to-run-on) · [15. The budget is spent on the wrong variable](#15-the-standard-privacy-mechanism-spends-the-budget-on-the-wrong-variable) · [19. What an authority of record costs](#19-an-authority-of-record-repairs-the-cliff-and-its-price-explodes) |
| **Estimating under aggregation** | [18. The cliff survives the protocol](#18-the-estimate-moves-under-secure-aggregation-and-the-cliff-does-not-move-with-it) · [19. An authority repairs it, at a price](#19-an-authority-of-record-repairs-the-cliff-and-its-price-explodes) · [20. Audit where the fleet splits](#20-audit-where-the-fleet-splits-and-the-prediction-that-said-otherwise) · [21. Where that policy stops working](#21-the-corpus-the-audit-policy-cannot-handle-built-on-purpose) · [22. The trace it leaves anyway](#22-the-trace-a-blind-spot-leaves-after-it-stops-leaving-disagreement) · [23. Provenance finds what is findable](#23-once-the-detector-names-the-channel-provenance-finds-the-corrupted-items-that-are-findable-at-all) |
| **What a deployment does about it** | [28. Detection converts into coverage, not correction](#28-the-open-problem-answered-detection-converts-into-coverage-not-into-correction) · [29. The shape of the error, read from the aggregate](#29-the-shape-of-the-error-is-visible-from-the-aggregate-and-it-is-cheap-to-read-wrong) · [30. The blind spot with no name](#30-the-blind-spot-with-no-name-is-detected-the-same-and-located-by-the-same-signal-read-one-sided) |
| **Whether a result survives a parameter nobody chose** | [24. The crossing is a distribution over corpora](#24-the-crossing-is-a-distribution-over-corpora-not-a-share-and-a-majority-was-nine-analysts-talking) · [25. Not where we pointed the estimator](#25-the-cliff-is-not-where-we-pointed-the-estimator-except-at-the-crossing-itself) · [26. One of the headlines was the corpus](#26-findings-20-to-23-were-measured-on-one-corpus-and-one-of-their-headlines-was-that-corpus) |

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

<!-- BEGIN GENERATED: power-claims -->
| Finding | n | Gap it rests on | vs | Verdict | Claim |
| --- | --- | --- | --- | --- | --- |
| 3b | 40 | 0.125 | constant | unresolved (needs n≥120) | qwen2.5-3b (0.775) clears the majority floor (0.650) |
| 3b | 40 | 0.154 | constant | **resolved** | mistral-7b (0.487) is below the majority floor (0.641) |
| 5 | 2400 | 0.003 | condition | unresolved (needs n>2000) | 8 shots (0.485) beats 0 shots (0.481) -- REFUTED at n=2400 |
| 5 | 2400 | 0.049 | condition | **resolved** | 2 shots (0.433) is worse than 0 shots (0.481) -- direction only |
| 5 | 2400 | 0.515 | constant | **resolved** | 8 shots (0.485) is below the stated-rule ceiling (1.000) |
| 5 | 2400 | 0.200 | constant | **resolved** | 8 shots (0.485) is below the majority floor (0.685) |
| 6 | 60 | 0.463 | condition | **resolved** | adapter (1.000) beats the base model (0.537) |
| 10 | 600 | 0.533 | condition | **resolved** | any-one adapter matches teacher (1.000) not world (0.467) |
| 10 | 600 | 0.059 | condition | unresolved (needs n≥1000) | inattentive adapter (0.923) beats its own teacher (0.864) |
| 11 | 200 | 0.100 | constant | **resolved** | linkage recovery (0.205) beats the guessing prior (0.105) |
| 11 | 50 | 0.820 | condition | **resolved** | RESTRICTED analysts (0.820) are recovered where OPEN (0.000) are not |

**8 of 11** resolve at the size they were run.
<!-- END GENERATED: power-claims -->

The pattern is the useful part. **The claims this
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

!!! warning "Corrected 2026-08-05: this table was reading a hand-typed copy of itself"
    Every claim above used to be written out by hand --- the effect size and the
    quoted scores together --- in a table inside `scripts/measure_power.py`. It was
    prose wearing a data structure's clothes, and it fed a *generated* block, so the
    page rendered it faithfully and it went stale silently when the corpus was
    re-measured on 2026-08-04.

    The visible symptom was the bullet that used to sit here, reporting
    `qwen2.5-3b (0.625) clears the majority floor (0.625)` and reasoning at length
    about a dead heat that no sample size could resolve. The 0.625 belonged to
    **qwen2.5-14b**, a different model in the same table. The artifact says
    qwen2.5-3b scores **0.775** against a floor of **0.650** --- not a tie, a gap of
    0.125, unresolved at n=40 and resolvable at n=120. Everything the old bullet
    concluded about ties being permanently unresolvable was sound reasoning applied
    to two numbers that were never measured together.

    The claims are now read from the artifacts, so only the *shape* of each claim is
    written down. Three of the eleven remain unresolved at the size they were run,
    and the table above says which and what each would need.

- **"8 shots beats 0 shots" was bought, and the claim died.** This table used to say
  it needed n≈600, an order of magnitude more than the 30 it was measured at. We ran
  it. The lift is not there:
  [finding 5](#5-in-context-learning-does-not-close-the-gap) reports 8 shots at
  0.485 against a zero-shot 0.481, a gap of 0.004 that would need more than 2000
  tasks to separate, and the row above records the refutation rather than deleting
  it. The companion claim, that 2 shots is *worse* than 0, is the closest thing to a
  resolved difference here and still is not one.

That bullet is why this section is not a limitation. A static benchmark with 30
items is stuck with 30; a generator is limited only by compute, and 600 tasks is
roughly two hours of local inference. The table is therefore a purchase order rather
than a disclaimer, and the two purchases it has prompted so far both ended in a
retraction. That is the instrument working: an unresolved claim is not a quiet claim,
it is a claim with a price tag on it.

## Measurement health

`pharos.validity` inspects every measurement for the conditions that make a score
misleading and marks the artifact unquotable when it trips one. That flag used to be
computed, warned about on the console, and then read by nothing, which made it a
console message rather than a property of the record.

It is published here instead of policed. A guard forbidding prose from quoting a
flagged number would be wrong: quoting one as evidence that something *failed* is
exactly what the flag asserts, and is what
[finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) correctly
does with six models that do not clear their own majority floor. What the flag forbids
is quoting such a number as evidence of capability.

This table is generated from `results/` and CI fails when it drifts from them.

<!-- BEGIN GENERATED: measurement-health -->
| Artifact | n | Quotable | Why not |
| --- | --- | --- | --- |
| `adapter_learnability` | 60 | **no** | **base**: 9/60 answers were unparsable (15%); every other number describes only the remainder; accuracy 0.392 does not beat the majority floor 0.647: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `analyst_review` | 40 | yes | - |
| `audit_policy` | 2 | **no** | n=2 is below 30; treat differences as provisional |
| `authority_anchors` | 11 | **no** | n=11 is below 30; treat differences as provisional |
| `blind_spot` | 105 | yes | - |
| `channel_bias` | 200 | yes | - |
| `consensus_reliability` | 200 | yes | - |
| `correlated_fleets` | 60 | yes | - |
| `decode_stability` | 30 | yes | - |
| `difficulty_confound` | 200 | yes | - |
| `edge_cost` | 199 | yes | - |
| `error_shape` | 69 | yes | - |
| `fleet_linkage` | 200 | yes | - |
| `label_fidelity` | 40 | yes | - |
| `latent_blindspot` | 200 | yes | - |
| `learnability` | 600 | **no** | **rows[0]**: accuracy 0.482 does not beat the majority floor 0.685: this is not evidence of capability |
| `learnability_replication` | 600 | **no** | **rows[0]**: accuracy 0.507 does not beat the majority floor 0.685: this is not evidence of capability |
| `privacy_budget` | 200 | yes | - |
| `review_adapter-any-one-xseed101` | 600 | **no** | **adapter**: accuracy 0.465 does not beat the majority floor 0.680: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-any-one` | 600 | **no** | **adapter**: accuracy 0.467 does not beat the majority floor 0.690: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-by-the-book-xseed101` | 600 | **no** | **base**: 75/600 answers were unparsable (12%); every other number describes only the remainder; accuracy 0.347 does not beat the majority floor 0.674: this is not evidence of capability |
| `review_adapter-by-the-book` | 600 | **no** | **base**: 82/600 answers were unparsable (14%); every other number describes only the remainder; accuracy 0.371 does not beat the majority floor 0.680: this is not evidence of capability |
| `review_adapter-inattentive-xseed101` | 600 | yes | - |
| `review_adapter-inattentive` | 600 | yes | - |
| `review_adapter-t1s0.05` | 600 | **no** | **adapter**: accuracy 0.465 does not beat the majority floor 0.690: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-t1s0.1` | 600 | **no** | **adapter**: accuracy 0.467 does not beat the majority floor 0.690: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-t1s0.15` | 600 | **no** | **adapter**: accuracy 0.388 does not beat the majority floor 0.690: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-t1s0.2` | 600 | **no** | **adapter**: accuracy 0.435 does not beat the majority floor 0.690: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-t1s0.3` | 600 | **no** | **adapter**: accuracy 0.310 does not beat the majority floor 0.690: this is not evidence of capability; every prediction was positive: the model is not discriminating, and precision/recall describe the class balance rather than the model; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-t1s0.4` | 600 | **no** | **adapter**: accuracy 0.390 does not beat the majority floor 0.690: this is not evidence of capability |
| `review_adapter-t1s0.5` | 600 | **no** | **adapter**: accuracy 0.310 does not beat the majority floor 0.690: this is not evidence of capability; every prediction was positive: the model is not discriminating, and precision/recall describe the class balance rather than the model; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-t1s0` | 600 | **no** | **base**: 82/600 answers were unparsable (14%); every other number describes only the remainder; accuracy 0.371 does not beat the majority floor 0.680: this is not evidence of capability |
| `review_adapter-t2s0.05` | 600 | yes | - |
| `review_adapter-t2s0.1` | 600 | yes | - |
| `review_adapter-t2s0.15` | 600 | yes | - |
| `review_adapter-t2s0.2` | 600 | yes | - |
| `review_adapter-t2s0.3` | 600 | **no** | **adapter**: accuracy 0.680 does not beat the majority floor 0.690: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-t2s0.4` | 600 | **no** | **adapter**: accuracy 0.600 does not beat the majority floor 0.690: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `review_adapter-t2s0.5` | 600 | **no** | **adapter**: accuracy 0.328 does not beat the majority floor 0.690: this is not evidence of capability |
| `review_adapter-t2s0` | 600 | yes | - |
| `review_adapter-t3s0.05` | 600 | yes | - |
| `review_adapter-t3s0.1` | 600 | yes | - |
| `review_adapter-t3s0.15` | 600 | yes | - |
| `review_adapter-t3s0.2` | 600 | yes | - |
| `review_adapter-t3s0.3` | 600 | yes | - |
| `review_adapter-t3s0.4` | 600 | **no** | **adapter_vs_teacher**: accuracy 0.548 does not beat the majority floor 0.560: this is not evidence of capability |
| `review_adapter-t3s0.5` | 600 | **no** | **adapter**: accuracy 0.360 does not beat the majority floor 0.690: this is not evidence of capability |
| `review_adapter-t3s0` | 600 | yes | - |
| `review_adapter-two-of-three-xseed101` | 600 | yes | - |
| `review_adapter-two-of-three` | 600 | yes | - |
| `review_sweep` | 40 | yes | - |
| `secure_reliability` | 200 | yes | - |
| `selective_risk` | 140 | yes | - |
| `tagged_aggregation` | 200 | yes | - |
| `triage_lift-llama3.1-8b` | 40 | **no** | accuracy 0.600 does not beat the majority floor 0.650: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `triage_lift-llama3.2-3b` | 40 | yes | - |
| `triage_lift-mistral-7b` | 40 | **no** | accuracy 0.487 does not beat the majority floor 0.641: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `triage_lift-qwen2.5-14b` | 40 | **no** | accuracy 0.625 does not beat the majority floor 0.650: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `triage_lift-qwen2.5-3b` | 40 | yes | - |
| `triage_lift-qwen2.5-7b` | 40 | **no** | accuracy 0.450 does not beat the majority floor 0.650: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `triage_lift` | 40 | **no** | accuracy 0.450 does not beat the majority floor 0.650: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |

**27 of 61** assessed artifacts are flagged. A flagged number may still be quoted as evidence that something *failed*, which is what the flag asserts; it may not be quoted as evidence of capability.

**Carrying no validity assessment, which is a gap rather than a pass:** `corpus_sensitivity`, `estimator_initialization`, `governance_sensitivity`.

Exempt, because there is no sampling question to answer:

- `adapter_replication` -- compares assessed adapter artifacts against their own replicates; the question is whether two runs agree, which no sampling flag answers
- `external_gate_validation` -- carries its own permutation-null statistics per corpus
- `federation_eligibility` -- deterministic over the label lattice; nothing is sampled
- `fl_benchmarks` -- sizes the problem rather than settling it, is quoted nowhere in the manuscript, and reports a bootstrap interval per condition instead of a flag
- `fleet_sensitivity` -- a sweep over a nuisance parameter; reports invariants, samples nothing
- `gate_determinism` -- reports the gate's surface baseline at full precision on one machine; the result is the comparison against another machine, not the number
- `gate_determinism-cluster` -- the second machine of that comparison
- `guard_mutations` -- four deterministic edits, each either noticed by the suite or not; there is no population the four are drawn from and no n a flag could be computed against
- `power` -- prices hypothetical evaluation sizes; simulates outcomes rather than measuring any
- `teacher_fleet` -- aggregates assessed adapter artifacts; adds no measurement of its own
- `triage_lift` -- superseded by the per-model triage_lift-* artifacts, which are assessed
<!-- END GENERATED: measurement-health -->

## 1. Leave-one-out attribution cannot produce a correct governed label

`scripts/measure_label_fidelity.py`

<!-- BEGIN GENERATED: label-fidelity -->
| Turns | Source recall | Source precision | Exact label | Leak | Creep |
| --- | --- | --- | --- | --- | --- |
| 40 | **0.705** | **0.942** | 33 | 4 | 3 |

Eight-source summarization turns under exact leave-one-out, 360 model calls. A wrong governed label on **7 of 40 turns (18%)**: 4 leak and 3 creep. Recall is what the ablation misses; precision is what it wrongly blames.
<!-- END GENERATED: label-fidelity -->

!!! warning "Corrected 2026-07-30"
    First measured at n=8 on the corpus *before* the coverage fix in finding 3.
    Three claims did not survive remeasurement. A wrong label on **half** of turns
    is **33%** at n=24. **Always under-restrictive** is refuted: creep occurs in 2
    of 24. And the cited move to an **incomparable** label
    (`RESTRICTED[LIAISON,PARTNER,SENSOR]` to `PROTECTED[LEGAL]`) did not recur and
    should not be quoted. Source recall did reproduce, 0.618 against 0.62.

!!! warning "Corrected again 2026-08-06, and the headline is generated now"
    The corpus was re-measured at n=40 on 2026-08-04 and the headline above was not
    updated, so this page carried the n=24 figures --- 61.8% recall, 97.6% precision,
    8 of 24 wrong --- while the paragraphs below it quoted the n=40 ones. Both were on
    the page at once, three screens apart.

    A sentence that has now been wrong twice for the same reason should stop being a
    sentence, so the headline is a generated block and cannot drift from the artifact
    again. The direction of the correction is worth noting: the mechanism looks
    *better* at n=40 than at n=24 (18% wrong rather than 33%), and the finding stands
    either way, because it rests on leave-one-out being unable to produce a correct
    label at all rather than on how often it fails.

The cause is corroboration. Leave-one-out asks which single source is
load-bearing, and a fact reported through several channels has none: drop any one
copy and the fact survives in the others, so no source is blamed and none of their
labels enters the join. Corroboration is not an edge case in this domain, it is
what channels are for.

Read the precision alongside the recall. At 0.942 the ablation rarely blames a source
that did not contribute, it simply misses many of the ones that did. That asymmetry is
why the failure is mostly leak, and why a tenth of turns (4 of 40) receive a label that
under-protects their sources.

Leave-one-out is also the ceiling that cheaper estimators approximate, so nothing
faster repairs it. At 67% exactly correct it is not a **usable** labelling
mechanism, though the original "rules out the whole family" was stated more
strongly than 24 turns can support.

!!! note "Re-measured 2026-08-03 on the corrected corpus, and once at the wrong n"
    Precision moved from 0.976 to **0.942** and recall from 0.618 to **0.705**; the
    outcome split is **33 exact, 4 leak, 3 creep of 40**. The shape is unchanged and is
    the durable part: still mostly leak, still creep in a small minority.

    This was first re-run at n=24, because `make results` said `--tasks 24` while the
    committed artifact was n=40 -- the documented way to regenerate the artifact could
    not reproduce it, which is now fixed and asserted in the test suite. That run
    reported a fifth leaking and **one incomparable label**, and on the strength of it
    this note briefly said the 2026-07-30 retraction of the incomparable claim was
    reversing. It is not. At the correct sample size no incomparable label appears, the
    retraction stands as written, and the apparent reversal was an artifact of a sample
    size nobody chose. Recorded rather than quietly deleted, because it is the second
    time in two days that a wrong `--tasks` produced a publishable-looking claim.

The replacement costs nothing: given what the output asserts, join the labels of
every source that *could* have asserted it. One detection pass, no ablation sweep,
and conservative by construction, so the error direction is creep rather than leak.

## 2. The design is bimodal on one policy ruling

`scripts/measure_federation_eligibility.py`

Three aggregator ceilings, four capacities, 40 turns. Turns average **just under two
compartments of 4**, and most already sit high on the level ladder, because a
summary over eight sources joins nearly everything.

| Declassification policy | FREETEXT | SPAN | SCALAR | ENUM |
| --- | --- | --- | --- | --- |
| keep compartments (fail-closed default) | 0-52% | 0-52% | 0-52% | 0-52% |
| drop compartments for low capacity | 0-52% | 0-52% | **100%** | **100%** |

!!! warning "The shape is the finding; the cell values are not bit-reproducible"
    This has now been measured four times. The mean compartments went 2.88 at n=8 on
    the pre-coverage-fix corpus, then 2.15, then 1.98, then **1.875**; the
    keep-compartments row went 0-12%, then 0-38%, then 0-50%, then **0-52%**. The
    **shape is identical in all four**: the bimodality sits on the compartment-shedding
    ruling and nowhere else.

    The last two differ by decode variation alone, not by code or corpus. This
    measurement calls a model -- `label_by_provenance` classifies each turn's content
    -- so it is subject to exactly what
    [finding 9](#9-a-measurement-that-repeats-one-prompt-measures-the-wrong-thing)
    describes, and two runs at the same `--tasks 40` on the same commit landed 0.1
    compartments and two percentage points apart. Quote the mechanism and the
    bimodality. Do not quote a cell to three figures, and do not read a two-point
    move between runs as a change in anything.

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

| Model | Family | Size | Acc | 95% interval | Majority | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-3b | Qwen | 3B | **0.775** | [0.650, 0.900] | 0.650 | 0.609 | 1.000 | **0.757** |
| llama3.2-3b | Llama | 3B | **0.675** | [0.525, 0.825] | 0.650 | 0.518 | 1.000 | 0.683 |
| qwen2.5-14b | Qwen | 14B | 0.625 | [0.475, 0.775] | 0.650 | 0.483 | 1.000 | 0.651 |
| llama3.1-8b | Llama | 8B | 0.600 | [0.450, 0.750] | 0.650 | 0.467 | 1.000 | 0.636 |
| mistral-7b | Mistral | 7B | 0.487 | [0.325, 0.650] | 0.641 | 0.412 | 1.000 | 0.583 |
| qwen2.5-7b | Qwen | 7.6B | 0.450 | [0.300, 0.600] | 0.650 | 0.389 | 1.000 | 0.560 |

!!! danger "Re-measured 2026-08-03 on the corrected corpus, and one claim did not survive"
    Every row above was re-run after the generator defect in
    [5c](#5c-the-generator-defect-behind-5b-and-the-fix) was fixed. The previous
    version of this table reported that **no model cleared the majority-class floor**,
    and listed that among the three unanimous properties. **Two models now clear it**,
    so that claim is withdrawn as stated. The two that did not survive contact with a
    corrected corpus are the two that were closest to their floor, which is what the
    power table had already said about them.

**Recall is 1.000 everywhere.** Not approximately, exactly, including at 14B. Every
model escalates every significant event and also escalates most routine ones, at
precision between 0.389 and 0.609. The over-escalation reported in finding 3 is a
property of the task under a withheld rule, not an idiosyncrasy of one model. This is
the one claim that has now held across two different corpora.

**Two of six clear the majority-class floor, and three are unresolved.** `qwen2.5-3b`
at 0.775 and `llama3.2-3b` at 0.675 sit above their floor of 0.650, and `qwen2.5-3b`'s
interval clears it outright. For `qwen2.5-14b`, `llama3.1-8b` and `mistral-7b` the
point estimate is below the floor and the interval reaches it, so the supportable
statement is *this sample does not show them clearing it* rather than that they do not.
Only `qwen2.5-7b` is below it with room to spare. A reader comparing any F1 here to 0.5
would still badly overstate what is happening, which is why both floors are reported.

**Scale does not help, and this controls for family.** Within Qwen alone, 3B scores
0.775 and 14B scores 0.625: a 4.7x parameter increase that makes things *worse*, and by
more than it did on the previous corpus. The 14B run needs more than 8 GB of VRAM, which
is the whole reason it is in the sweep.

That is still consistent with finding 5 and still strengthens it. A model that reaches
F1 1.000 when handed the rule is short of neither capability nor parameters. It is
short of the rule, so rule *acquisition* is the whole question.

!!! warning "What this does not claim"
    40 tasks per model, so this is not a ranking. Six of the fifteen pairwise
    comparisons separate at 95% against roughly 0.8 expected by chance, which is more
    than multiplicity explains and still supports only a partial order: nine pairs do
    not separate at all. The table is reported so the unanimity is visible, not to rank
    systems, and the one number to quote from it is recall.

    All six ran at 100% GPU residency for the 3B-8B models; the 14B sat at 64%, checked
    per model rather than assumed.

### These numbers are not bit-reproducible, and the gate is

Worth stating precisely, because the two halves of this repository behave
differently and the difference is by design.

Re-running the five smaller models on cluster hardware changed **2 of 200
judgements**: `qwen2.5-3b` and `llama3.1-8b` each flipped exactly one task of forty;
the other three were identical. Temperature is 0 and the seed is fixed, so this is
runtime and quantization numerics moving a borderline verdict, not sampling.

The **corpus** reproduces bit-identically across the same two machines -- the SHA-256
of the serialized corpus agrees at every seed. The gate's **score** does not, which we
report because `scripts/measure_gate_determinism.py` measured it rather than assuming
it: two of seven seeds disagree, by 1.2e-05 and 6.5e-06, with numpy, scikit-learn,
scipy and the BLAS all at identical versions. The BLAS selects kernels by processor --
AVX-512 on the cluster's Xeon, AVX2 on the laptop's Ryzen -- which changes the
reduction order inside the probe fit.

What matters is that no *verdict* moves: the largest disagreement is about four orders
of magnitude below the gap between any measured baseline (0.638--0.660) and the 0.720
acceptance ceiling. That asymmetry is still the reason model calls are confined to
`pharos.attribute`, but the accurate form of it is that the acceptance **decision**
does not drift with the platform, not that the number is identical. The scores measured
*on* the corpus carry roughly a one-percent floor on cross-platform agreement. Quote
model-dependent numbers with the platform named.

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
| Zero-shot floor | 0 | 0.368 | 0.899 | 0.522 | 0.482 | [0.441, 0.523] | 4 |
| Labelled examples | 2 | 0.365 | 0.978 | 0.532 | **0.433** | [0.392, 0.476] | 41 |
| Labelled examples | 4 | 0.357 | 0.984 | 0.524 | 0.436 | [0.396, 0.476] | 10 |
| Labelled examples | 8 | 0.380 | 1.000 | 0.550 | 0.485 | [0.444, 0.527] | 4 |
| **Rule stated, checklist prompt (ceiling)** | n/a | **1.000** | **1.000** | **1.000** | **1.000** | | |

**Examples close none of the gap to the ceiling**, and no condition comes near its own
majority-class accuracy, which sits at 0.685. The floor is not close: the best
condition is 0.20 below it, and on the corrected corpus **every** condition now sits
below the majority floor rather than merely short of the ceiling. Every row is marked
unquotable by `pharos.validity` for exactly that reason, which is the correct reading:
none of these numbers is evidence of capability.

**Eight shots against zero is +0.003, which is the finding.** Their intervals overlap
almost completely, [0.444, 0.527] against [0.441, 0.523]. This is the third
independent measurement of that comparison and the second on which the lift is
absent; the one run that showed a lift was measured on the pre-fix corpus and is
discussed in [5b](#5b-the-intervals-above-understate-the-uncertainty-measured-by-replication).

**Two shots looks *worse* than none, and we decline the comparison anyway.** The gap
runs the opposite way to what a reader expects, 0.433 against 0.482, and it is the
closest any pair here comes to separating, at 0.049 against overlapping intervals.

That comparison has now returned three different verdicts across three runs, and the
reason to decline it is measured rather than argued. One run separated it by a margin
of 0.0013 -- the pair cleared its own criterion by roughly a thousandth. Another put
the gap at 0.0551 against a half-width of 0.0576 and the criterion returned *false*.
This one does not separate at all. A test whose verdict flips on a quantity far
smaller than the measurement's own run-to-run variation is not testing anything, and
"do two examples hurt" has changed sign twice on data differing only by which
afternoon it was collected and which corpus generator produced it.

We therefore report the direction and decline the difference, which is what the earlier
run's arithmetic said and what this run's replication says for a better reason.

That distinction is worth spelling out because two criteria disagree on this pair and
only one of them is right. Neither interval covers the other's point, which is what
`uncertainty.resolves` tests, and on that basis an earlier draft of this section
claimed the comparison as resolved. But "do two conditions differ" is a question about
an interval on their *difference*, and that interval is wider than either input.
`uncertainty.resolves_difference` applies the correct rule and returns false here.
The weaker test was mislabelled "conservative" in its own docstring; it is the
permissive one, and it has been corrected.

What the other columns show is a coherent mechanism for the direction, whether or not
the size is resolved. Recall climbs from 0.899 at zero shots to **exactly 1.000** at
eight, while precision barely moves --- 0.368 to 0.380 --- so the examples are not
teaching the rule. They are teaching the model to escalate more: false positives rise
from 290 to 307 while false negatives fall from 19 to 0. That is the same
over-escalation
[finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) found in every
model tested, and supplying examples does not repair it.

!!! warning "Corrected 2026-08-06: read the confusion matrix, not the recall"
    This paragraph previously said the model "escalates every single task" at eight
    shots, and called that "a classifier that answers SIGNIFICANT to everything". The
    artifact says otherwise: at eight shots the matrix is 188 true positives, 307 false
    positives and **101 true negatives**. A hundred and one tasks were answered ROUTINE.

    Recall of exactly 1.000 means every *positive* was escalated, not that every task
    was. Reading a saturated recall as universal escalation is the specific misreading
    the validity module exists to flag, and
    [the health table](#measurement-health) records it correctly for this row: the
    eight-shot entry carries the "recall 1.000 while false positives exceed true
    positives" concern and *not* the "every prediction was positive" concern that a
    genuinely degenerate run would carry. The prose was contradicted by a generated
    block two screens above it.

    The same sentence also said precision "falls from 0.368 to 0.380", which is a rise
    described as a fall. The direction of the argument survives either way, because it
    never rested on precision moving: it rests on where the errors go.

No other pair separates under the difference test. Four, eight, and zero shots are
indistinguishable from each other at this size.

### 5b. The intervals above understate the uncertainty, measured by replication

This measurement has now been run twice, end to end, and the two runs disagree by more
than the intervals allow. That is a fact about every model-backed number in this
document, not just this one, so it is recorded here where the evidence is.

!!! danger "Withdrawn 2026-08-02, hours after publication. The comparison is confounded."
    This section originally asserted that the two runs were "the same measurement in
    every respect that could matter ... the same prompts". That could not be verified,
    and is probably false.

    `generate()` draws every event first and *then* renders, from a single RNG stream.
    Events are therefore stable across corpus sizes -- same ids, same facts, same
    governed labels, which is why the eval set checked out -- but rendering begins at a
    stream position that depends on `n_events`, so **the same task is worded differently
    in corpora of different sizes**. The eval task ids are identical for every
    `n_events` at or above 632; the prompt hash is different for every one of them.

    Run B used `--events 800`. Run A predates the change that made artifacts record
    `n_events` at all, so its corpus size is unknown and its prompts cannot be
    reconstructed. The observed differences are therefore confounded between decode
    variation and a difference in the prose the model was shown, and nothing here
    separates them.

    The provenance gap that makes this unresolvable is the one closed earlier the same
    day, by recording `n_events` in the artifact. Finding this is what the fix was for;
    it simply arrived one run too late to rescue these two.

    **What stands:** the finding itself, which never depended on this section. Every
    value in both runs sits between 0.45 and 0.55 against a ceiling of 1.000 and a floor
    of 0.693. **What is withdrawn:** the claim that a 95% interval failed to cover an
    identical re-run, and the claim that the shot-count ordering reverses between
    identical runs. Both may well be true; neither is established by these two runs.

    A controlled replication -- two runs at the same recorded `--events`, on the same
    commit -- **has now been run, and is reported below.** The table immediately
    following is retained as the record of what was seen in runs A and B, not as
    evidence for what caused it.

The two runs used the same 600 evaluation tasks, verified identical by task id and
ground truth, the same model, and temperature 0.0 with seed 7 on both. They may not have
used the same prompts, for the reason given above.

| Shots | Run A, 2026-08-01 | Run B, 2026-08-02 | Difference | Run A's 95% interval |
| --- | --- | --- | --- | --- |
| 0 | 0.5234 | 0.5126 | -0.011 | [0.484, 0.565] |
| 2 | 0.4683 | 0.4534 | -0.015 | [0.427, 0.509] |
| **4** | **0.5147** | **0.4523** | **-0.062** | **[0.475, 0.557]** |
| 8 | 0.5142 | 0.5492 | +0.035 | [0.473, 0.556] |

**Two things were observed. Neither is attributed, for the reason above.**

*A 95% interval did not cover the other run's point estimate.* At four shots, Run A
reported [0.475, 0.557] and Run B's point is 0.4523, below the bottom of it. Read as
decode variation this would say the interval understates uncertainty; read as a corpus
difference it says only that two different renderings score differently, which is
unremarkable. The runs cannot distinguish these.

*The ordering of the conditions differs.* Run A puts eight shots at 0.5142, below its
zero-shot 0.5234; Run B puts eight shots at 0.5492, above its zero-shot 0.5126. Same
ambiguity: a genuine reversal between identical runs would be the stronger form of the
power argument, and a reversal between differently-worded corpora would be a fact about
two corpora.

### 5c. The generator defect behind 5b, and the fix

`src/pharos/generate.py`, `tests/test_generate.py`

The confound named in 5b was a defect in the generator, not a fact about decoding. It
is recorded here because the manuscript's self-audit section cites this page as the
long-form record of it.

`generate()` drew **every** event before rendering **any** of them, from one
`random.Random(seed)`. Event drawing consumes a variable number of values, so rendering
began at a stream position set by `n_events`. Measured against the pre-fix generator at
seed 7:

| Compared corpora | Report ids | Fact assignment | Governed label | Rendered text |
| --- | --- | --- | --- | --- |
| 40 vs 80 events | 100% | 21% | 52% | **0%** |
| 200 vs 600 events | 100% | 16% | 52% | **0%** |
| 200 vs 800 events | 100% | 14% | 52% | **0%** |
| 632 vs 800 events | 100% | 15% | 50% | **0%** |

**No report was worded identically across two corpus sizes, at any pair.** The checks a
reader would think to run -- do the task ids line up, does the ground truth agree --
pass at 100%, which is why this survived as long as it did. Any two model-backed
measurements taken at different `--events` were comparing different corpora.

The fix derives a stream per event from `sha256(f"{seed}:{index}:{purpose}")`, so a
smaller corpus is an exact prefix of a larger one, field for field including text.
`tests/test_generate.py::test_a_smaller_corpus_is_an_exact_prefix_of_a_larger_one`
asserts every field; it fails on the first report against the pre-fix generator.

!!! warning "The corpus changed, so this page currently spans two of them"
    The gate was re-run first and still passes, at a surface baseline of 0.6545 against
    a ceiling of 0.72 (it was 0.6559 before), and the class-conditional channel mix is
    unchanged to within a tenth of a point, so the corpus remains usable and remains
    the same instrument. Individual findings' numbers moved anyway, and two conclusions
    moved with them ([3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it)
    and [17](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst)).

    **Re-derived on the corrected corpus (2026-08-03):** the model-free measurements,
    the six-model sweep, label fidelity, federation eligibility, decode stability, and
    rule learnability.

    **Still on the pre-fix corpus:** `adapter_learnability` and the eight
    `review_adapter-*` artifacts, which are LoRA fine-tuning runs and need a cluster
    A100 rather than the local 8 GB card. `edge_cost` was on that list and is not any
    more; it re-ran on 2026-08-04 once the machine was quiet.
    **Findings 6 and 10 rest on the remainder**, so until they are re-run, any comparison
    between one of them and a finding above it is a comparison across two corpora --
    which is the exact confound
    [5b](#5b-the-intervals-above-understate-the-uncertainty-measured-by-replication)
    was withdrawn for. Do not draw one.

    `external_gate_validation` is unaffected: it runs the gate against three public
    corpora and never calls this generator.

!!! danger "How this was found, which is the part worth keeping"
    Not by inspection. The property was asserted in a test whose three load-bearing
    assertions -- `fact_ids`, `label`, `text` -- had been deleted, leaving three that
    pass whatever the generator does. The suite was green at 508 passing and 94.45%
    branch coverage with the defect fully present, and the manuscript already described
    the fix as shipped. A test named for a property it no longer checks is worse than no
    test, because it answers the question a reviewer would otherwise ask.

### 5d. The controlled replication, and what it settles

`results/learnability.json`, `results/learnability_replication.json`

Runs A and B could not separate decode variation from a corpus difference. Runs C and D
can, because the generator defect that confounded A and B is
[fixed](#5c-the-generator-defect-behind-5b-and-the-fix) and both were run at
`--events 800`, seed 7, 600 evaluation tasks, temperature 0, on commits whose
corpus-defining code and prompt path are **byte-identical** (`git diff` over
`generate.py`, `labels.py`, `world.py`, `tasks.py`, `scenarios/` and
`measure_rule_learnability.py` is empty between them). The only difference between C
and D is when they ran.

| Shots | Run C | Run D | Difference | Unparsed C -> D |
| --- | --- | --- | --- | --- |
| 0 | 0.4815 | **0.5067** | **+0.0252** | 4 -> 0 |
| 2 | 0.4329 | **0.4549** | **+0.0220** | 41 -> 35 |
| 4 | 0.4356 | 0.4356 | 0.0000 | 10 -> 10 |
| 8 | 0.4849 | 0.4849 | 0.0000 | 4 -> 4 |

**Reproduction is per-condition, not global.** Four and eight shots return *bit-identical
confusion matrices* -- 183/330/74/3 and 188/307/101/0, every cell -- while zero and two
shots move by more than two points. Finding 9's claim that a single-pass measurement
reproduces exactly is therefore true of half these conditions and false of the other
half, in one run of one script.

**The two that moved are exactly the two whose unparsed count changed**, and the
verdicts moved with them: at zero shots, 14 items crossed from false positive to true
negative on top of the 4 that became parseable. Four conditions is not enough to call
that a mechanism, and it is enough to say the variation is not uniform decode noise --
it is concentrated in the conditions where the model's output format was itself
unstable.

!!! danger "The headline comparison changes sign between two controlled runs"
    Eight shots against zero is **+0.0034 in run C and -0.0218 in run D**. The
    comparison finding 5 withdraws does not merely fail to reach significance; its
    *direction* is not stable across two runs that differ in nothing but the hour they
    were executed.

    The largest per-condition move is 0.0252, which is **seven times** the 0.0034 gap
    that a reader might have called a lift. This is the evidence 5b reached for and did
    not have: the withdrawal of the eight-shot lift is not caution, it is required. A
    difference smaller than the measurement's own controlled run-to-run variation is not
    a small effect, it is not an effect.

**What this does *not* say.** Two runs bound nothing tightly, and this is two runs. It
establishes that variation at this magnitude occurs, not its distribution. The six-point
precaution stated below was set before this measurement and survives it comfortably; the
useful sharpening is that the variation is condition-dependent, so a per-condition
interval computed within a single run does not capture it.

**What survives untouched.** The finding is that in-context learning does not close the
gap, and every value in both runs sits between 0.45 and 0.55 against a stated-rule
ceiling of 1.000 and a majority floor of 0.693. A ten-point band is irrelevant to a
claim about a fifty-point shortfall, whichever source it comes from. This section was
never load-bearing for the finding; it was an attempt to strengthen a caveat, and it
overreached.

**The precaution stands even though the evidence for it does not.** Whichever source
produced these differences, both are live in any model-backed comparison here, so the
rule is unchanged: a gap under roughly six points between two model-backed conditions
is unsupported without replication at a *recorded* corpus size. Two claims sit in that
range and both were already declined on other grounds:
[finding 5](#5-in-context-learning-does-not-close-the-gap)'s shot-count ordering, and
the six-model ordering in
[finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it), which the
text refuses to present as a ranking. The claims this project does make -- recall
exactly 1.000 in every model, a fleet inheriting its teacher's error rate to within
0.002, a consensus cliff from 1.000 to 0.660 -- are all far outside that band.

!!! note "What would settle it, and what is queued"
    Two runs bound nothing even when they are controlled, and these were not. Two
    instruments settle it between them.

    *For decode variation:* the per-task cross-run disagreement rate, measured by
    [finding 9](#9-a-measurement-that-repeats-one-prompt-measures-the-wrong-thing),
    whose sample was raised from 30 to 300 because 0 of 30 admits a rate as high as
    9.5% -- more than enough to produce everything above on its own.

    *For the corpus:* a repeat of this measurement at `--events 800`, matching Run B
    exactly, with the size now recorded in the artifact. If the numbers reproduce, the
    differences above were rendering. If they do not, they were decode.

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

- **Unparsed answers are not evenly distributed.** 10 of 600 at four shots against 4
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
| by-the-book | (control) | 1.00 | 1.000 | 0.65 | 0.00 | **1.00** |
| two-of-three | threshold 2 | 1.00 | **0.750** | 0.65 | 0.00 | 1.00 |
| any-one | threshold 1 | 1.00 | **0.475** | 0.65 | 0.00 | 1.00 |
| inattentive | slips 15% | 1.00 | 0.875 | 0.65 | 0.00 | 1.00 |
| terse | revises 25% | 0.95 | 1.000 | 0.65 | 0.00 | 1.00 |
| unexplained | names no grounds | 1.00 | 1.000 | 0.65 | 0.00 | 1.00 |
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

**A reviewer with the wrong threshold teaches a degraded rule, and the worst of them
teaches one that loses to guessing.** At threshold 2 the target stream matches the world
on 0.750, at threshold 1 on 0.475. The majority floor on this evaluation set is 0.650.
Threshold 1 is below it, so a learner trained on those corrections would do worse than
always answering ROUTINE -- and it would do so while every acceptance and revision looked
like ordinary supervision. Threshold 2 sits above the floor and still costs a quarter of
its targets, which is supervision that is worth having and quietly wrong.
This is not a hypothetical reviewer. Threshold 1 is over-escalation, and finding 3b
measured over-escalation in **every** model tested, recall 1.000 at precision 0.389
to 0.609. A fleet learning from reviewers who fail on the axis its models already
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

Target accuracy against the world, `*` marking cells that clear the 0.650 majority
floor:

| Standard | slip 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 | 0.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| needs 1 of 3 | 0.475 | 0.470 | 0.465 | 0.490 | 0.455 | 0.445 | 0.480 | 0.465 |
| needs 2 of 3 | 0.750\* | 0.710\* | 0.685\* | 0.685\* | 0.635 | 0.640 | 0.600 | 0.490 |
| **needs 3 of 3** | **1.000**\* | 0.965\* | 0.920\* | 0.830\* | 0.815\* | 0.685\* | 0.520 | 0.540 |

!!! danger "Re-measured 2026-08-03, and the headline claim of this finding is withdrawn"
    This said **no wrong standard clears the floor at any carefulness**, and said it was
    asserted over the whole grid in the test suite rather than spot-checked. The grid
    assertion is what caught its failure: on the corrected corpus a two-of-three reviewer
    clears the floor at four of eight slip rates, topping out at 0.750 against a floor of
    0.650. A learner handed *those* targets would not do better ignoring them.

    The test was narrowed to the most-wrong standard, which still never clears, and a
    second test now asserts the exchange rate directly, since that is what this finding
    is actually about and it does not depend on where the floor happens to sit.

**The most wrong standard never clears the floor.** Every cell in the needs-1-of-3 row
sits below 0.650, including where the reviewer never slips at all.

**A correct standard tolerates a great deal of carelessness.** It clears the floor
through a 30% slip rate (0.685) and only fails at 40%. The exchange rate is still
lopsided, and slightly less so than previously measured: **one step of standard error
costs what a 30% slip rate costs** (a two-of-three reviewer's 0.750 is what the correct
standard scores at 30% slip), and two steps cost more than any slip rate measured up to
50%, since the needs-1-of-3 row never reaches even the correct standard's worst cell.

That is the quantitative form of a qualitative claim in the learning-from-noisy-
labels literature -- that systematic annotator bias damages a model more than random
noise, because the errors are class-dependent and get learned rather than averaged
away. The direction reproduces on both corpora. The magnitude is what the exchange
rate above states and nothing more.

!!! warning "Two smaller claims here did not survive the corrected corpus"
    **A stated ratio of "roughly eight to one against the systematic error"** is
    withdrawn. It was never derived in the text from a cell in the grid, and no
    arithmetic over the current grid produces it. What the grid supports is the
    exchange rate: one step of standard error is worth about 30 points of slip, two
    steps more than 50.

    **"Noise partially rescues a wrong standard"** is refuted. It rested on the
    needs-1-of-3 row climbing from 0.425 to 0.525 as slip rose to 50%. That row is now
    flat -- 0.475 at no slip, 0.465 at 50%, varying by 0.045 across the whole range with
    no trend -- which is what a reviewer whose standard already ignores the rule should
    look like when noise is added to it. The correct reviewer still falls to 0.540 at
    50%, so the two rows still converge; they converge because the *correct* one
    degrades, not because the wrong one improves. The claim was reading a trend into
    eight cells that spanned a tenth of a point.

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

<!-- BEGIN GENERATED: teacher-inheritance -->
| Teacher | Standard | Targets matching world | Adapter vs **world** | Adapter vs **teacher** | Inherited error |
| --- | --- | --- | --- | --- | --- |
| by-the-book | 3 of 3 | 1.0000 | 0.9967 | 0.9967 | -0.0033 |
| inattentive | 3 of 3, slips 15% | 0.8553 | 0.8967 | 0.7817 | +0.0414 |
| two-of-three | 2 of 3 | 0.7009 | 0.6933 | 0.9750 | -0.0076 |
| any-one | 1 of 3 | 0.4465 | 0.4667 | 1.0000 | +0.0202 |

Accuracy on 600 held-out tasks, 1140 training tuples per adapter. The last column is the adapter's agreement with the world minus its teacher's: near zero means the adapter reproduced its teacher's error rate rather than diluting it.
<!-- END GENERATED: teacher-inheritance -->

Accuracy on **600** held-out tasks whose events are disjoint from training. The base
model scores accuracy 0.320 with 82 of 600 answers unparsable, so every row is a large
gain over not training at all. The last column is the adapter's agreement with the
world minus its teacher's, and it is the cleanest statement of the finding.

**A wrong standard is inherited almost exactly.** The two systematically-wrong
teachers hand over their error rate to within **0.008 and 0.020**: a teacher whose
targets agree with the world on 0.701 produces an adapter agreeing on 0.693, and one
at 0.447 produces 0.467. The adapter did not partially absorb the reviewer's rule or
average it against its prior; it learned "two of three" and "one of three" and now
agrees with the world exactly as much as that rule does. This is the concrete form of
what the noisy-annotation literature calls absorbing an unreliable annotator's errors
into the parameters, and at 600 tasks the correspondence is close enough to read as an
identity rather than a tendency.

**Random error is filtered; systematic error is not.** The contrast is the finding,
and the careless teacher is the only row that breaks the inheritance pattern. It slips
15% of the time, its targets match the world on 0.855, and its adapter matches the
world on 0.902 while matching the teacher on only 0.790. It tracks the underlying rule
more closely than the teacher that taught it. Given a consistent wrong rule instead,
the adapter simply became that rule.

!!! success "Bought and confirmed 2026-08-01"
    This was the one claim here the [power analysis](#what-these-sizes-can-resolve)
    priced as unresolvable at the size it was run. At 60 tasks the gap between the
    careless adapter and its teacher was 0.083 against a minimum detectable 0.233, and
    this section said so and declined to claim a margin.

    Remeasured at 600 the gap **widened to 0.112**, against a difference half-width of
    0.042, so it now resolves. Both figures score one decode against two references
    over the same evaluation set, which makes the real test paired and therefore
    tighter than the independent one quoted. The direction reported at 60 tasks was
    right, and the margin is now measured rather than declined.

    Worth stating alongside [finding 5](#5-in-context-learning-does-not-close-the-gap),
    where the same purchase destroyed the claim. The power table is not a retraction
    machine; it prices claims, and the prices come back both ways.

So the two error types finding 8 separated at the level of the target stream stay
separated after training. Carelessness is partly repaired by learning, by nearly four
points. **Wrongness is preserved by it** to within two points, which is the more
unsettling half of the result: there is no dilution, no regression toward the model's
prior, and nothing in the training signal that notices the rule is wrong.

!!! warning "Corrected 2026-08-06: the inheritance was tighter on the page than in the data"
    Both tables above were hand-typed and every row had drifted. One drift carried the
    finding's headline: the page said the two systematically-wrong teachers hand over
    their error rate *"to within 0.002 and 0.001"*. The artifacts say **0.008 and
    0.020** --- an order of magnitude larger, and the claim that inheritance is an
    *identity* rests on exactly that gap being small.

    The conclusion survives, because two points of slack across a 600-task evaluation
    is still inheritance rather than dilution, and the direction is unchanged. But
    "to within 0.002" was a stronger sentence than anything measured supported, and it
    was the sentence a reader would quote. Both tables are generated now.

**What this means for the design.** The premise is that a fleet learns analytic
tradecraft from analyst decisions. It does -- exactly, faithfully, and without any
check on whether the tradecraft is correct. Personalization works as specified, and
that *is* the risk: an over-escalating analyst produces an over-escalating agent, and
over-escalation is what every model in
[finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) already
does. The mitigation is not more data. It is annotator-reliability weighting, which
needs annotator identity retained through training -- something Pharos has by
construction and most annotation pipelines discard.

**It survives a corpus the adapter has never seen any part of.** The result above
evaluates on held-out *events* from the same corpus. A stronger test evaluates on a
different corpus instantiation entirely, sharing no events, no vessels and no
renderings, which is what contamination-resistance guidance asks of a benchmark.
Repeating the sweep with the evaluation drawn from seed 101 (2h54m, same A100):

<!-- BEGIN GENERATED: teacher-inheritance-xseed -->
| Teacher | Standard | Targets matching world | Adapter vs **world** | Adapter vs **teacher** | Inherited error |
| --- | --- | --- | --- | --- | --- |
| by-the-book | 3 of 3 | 1.0000 | 0.9917 | 0.9917 | -0.0083 |
| inattentive | 3 of 3, slips 15% | 0.8575 | 0.9483 | 0.8233 | +0.0908 |
| two-of-three | 2 of 3 | 0.7046 | 0.7033 | 0.9983 | -0.0013 |
| any-one | 1 of 3 | 0.4534 | 0.4650 | 1.0000 | +0.0116 |

Accuracy on 600 held-out tasks, 1740 training tuples per adapter. The last column is the adapter's agreement with the world minus its teacher's: near zero means the adapter reproduced its teacher's error rate rather than diluting it.
<!-- END GENERATED: teacher-inheritance-xseed -->

Both conclusions hold on a corpus with zero overlap. The systematically wrong teachers
are still tracked closely, now within about two points rather than half a point, and
the careless teacher is still the only one improved upon: 0.943 against its teacher's
0.823, a gap of 0.125 against a difference half-width of 0.038, resolved on its own
terms.

!!! warning "Do not read the two tables as a controlled comparison"
    The cross-corpus run trained on **1,740** tuples where the same-corpus run trained
    on **1,140**. That is not an oversight in either run but a consequence of the
    design: when evaluation comes from a different corpus, nothing has to be held out
    of the training one. So the differences *between* the tables, the filtering effect
    growing from +0.041 to +0.091 and the inheritance loosening from two points to
    two, are confounded with 53% more training data and cannot be attributed to
    cross-corpus evaluation. What each table supports on its own is the same pair of
    conclusions, which is the claim being made.

### At fleet scale: 24 teachers, both axes crossed

`scripts/measure_teacher_fleet.py`, `results/teacher_fleet.json`

Four teachers is four case studies, and two of the three claims above are ones four
points cannot separate: the named teachers vary standard *and* carefulness together, so
"systematic" and "random" are never independently manipulated. The grid crosses three
escalation thresholds with eight slip rates and trains all 24, one A100 job per point.
It also runs on the **corrected** generator, so unlike the tables above it is directly
comparable with every other finding on this page.

<!-- BEGIN GENERATED: teacher-fleet -->
| Teacher | Targets vs world | Adapter vs **world** | Adapter vs **teacher** | Ceiling `1-s` | Inherited |
| --- | --- | --- | --- | --- | --- |
| `t1s0` † | 0.447 | 0.467 | 1.000 | 1.000 | +0.020 |
| `t1s0.05` † | 0.457 | 0.465 | 0.923 | 0.950 | +0.008 |
| `t1s0.1` † | 0.454 | 0.467 | 0.918 | 0.900 | +0.012 |
| `t1s0.15` † | 0.468 | 0.388 | 0.793 | 0.850 | -0.079 |
| `t1s0.2` † | 0.446 | 0.435 | 0.773 | 0.800 | -0.011 |
| `t1s0.3` † | 0.475 | 0.310 | 0.640 | 0.700 | -0.165 |
| `t1s0.4` † | 0.482 | 0.390 | 0.562 | 0.600 | -0.092 |
| `t1s0.5` † | 0.484 | 0.310 | 0.473 | 0.500 | -0.174 |
| `t2s0` | 0.701 | 0.702 | 0.990 | 1.000 | +0.001 |
| `t2s0.05` | 0.696 | 0.702 | 0.940 | 0.950 | +0.006 |
| `t2s0.1` | 0.651 | 0.693 | 0.867 | 0.900 | +0.042 |
| `t2s0.15` | 0.639 | 0.692 | 0.805 | 0.850 | +0.053 |
| `t2s0.2` | 0.627 | 0.715 | 0.775 | 0.800 | +0.088 |
| `t2s0.3` † | 0.599 | 0.680 | 0.640 | 0.700 | +0.081 |
| `t2s0.4` † | 0.551 | 0.600 | 0.538 | 0.600 | +0.049 |
| `t2s0.5` † | 0.500 | 0.328 | 0.503 | 0.500 | -0.172 |
| `t3s0` | 1.000 | 0.997 | 0.997 | 1.000 | -0.003 |
| `t3s0.05` | 0.947 | 0.982 | 0.945 | 0.950 | +0.034 |
| `t3s0.1` | 0.910 | 0.960 | 0.853 | 0.900 | +0.050 |
| `t3s0.15` | 0.864 | 0.923 | 0.787 | 0.850 | +0.059 |
| `t3s0.2` | 0.804 | 0.920 | 0.733 | 0.800 | +0.116 |
| `t3s0.3` | 0.723 | 0.835 | 0.622 | 0.700 | +0.112 |
| `t3s0.4` | 0.595 | 0.722 | 0.548 | 0.600 | +0.127 |
| `t3s0.5` † | 0.492 | 0.360 | 0.503 | 0.500 | -0.132 |

† marks the 12 adapters the validity check refuses, for accuracy beneath the majority floor or for recall bought with more false positives than true ones.

| Threshold | n | Median inherited | Beat their teacher | Quotable |
| --- | --- | --- | --- | --- |
| 1 | 8 | -0.0449 | 2 | 0 |
| 2 | 8 | +0.0458 | 5 | 5 |
| 3 | 8 | +0.0549 | 6 | 7 |
<!-- END GENERATED: teacher-fleet -->

**The two conclusions survive, and one number does not.** Teachers who never slip hand
their disagreement over at a median of **+0.0008** -- the inheritance claim, now on
three independent points rather than two. Teachers who do slip are improved upon at a
median of **+0.034**, with **12 of 21** adapters ending closer to the world than the
teacher that taught them. What does not survive is "close enough to read as an
identity": across 24 teachers, adapter-teacher agreement runs from 1.000 down to 0.473
with a median of 0.781. Four teachers could not have distinguished a tendency from an
identity, and the grid does.

**The apparent decay is a ceiling, not a failure to inherit.** A teacher slipping at
rate *s* disagrees with its own rule that often, so an adapter that had learned the
rule *perfectly* would still match those labels only `1-s` of the time. Measured
against that bound the residual has median **-0.030** and never exceeds 0.078 in
magnitude. Equivalently, in **20 of 24** cases the adapter agrees with its teacher more
closely than the teacher agrees with *itself* on a second pass, by a median of 0.041.
The adapter learns the teacher's rule with the teacher's noise averaged out -- what the
weak-to-strong literature calls convergence to the *posterior mean* teacher rather than
to any individual one ([Xu et al. 2025](https://arxiv.org/abs/2505.24313)).

**Whether the fleet helps depends on which error the teacher makes.** Conditioning on the threshold instead of the slip rate splits the grid cleanly -- see the second table above.

Training denoises a teacher who is right and careless. It cannot rescue a teacher who
is careful and wrong, and where the standard *itself* is the error, more of it hurts.
A fleet-wide average would report a net gain and hide this entirely.

!!! danger "Agreement with the analyst is not an acceptance criterion"
    `t1s0` reproduces its teacher at **1.000** -- perfect fidelity, the best score in
    the table -- while agreeing with the world on **0.467**, below the 0.690 obtainable
    by answering "escalate" to everything. The validity check refuses it on both
    counts. An operator holding only the fidelity column would have shipped a model
    indistinguishable from a stub.

    It is not an isolated row: **three** adapters reproduce their teacher above 0.9
    and are refused, and half the grid is refused on the same grounds.

    This is the governance consequence of the whole section. Fidelity to the teacher
    and usefulness to the mission are separate measurements, and only one of them can
    be an acceptance gate.

!!! note "What this does not settle"
    One model, single-pass evaluation. Size and contamination are settled: 24 teachers
    crossing both axes, 600-task contrasts, and both conclusions surviving a corpus
    with no shared events. The teachers are parameterised decision procedures, so this
    bounds a mechanism and estimates nothing about human analysts. One model is the
    real limit: that a 3B Qwen inherits a wrong rule this exactly says nothing yet
    about whether a larger model or a different task shape would.

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
| shed compartments | 6,606 | **0.205** [0.150, 0.265] | 9.98 |
| keep compartments | 400 | 0.000 [0.000, 0.000] | 16.00 |

Guessing the fleet's most common beat scores 0.105, and the interval excludes it, so
the attack is doing real work. But the aggregate is the least interesting number here.

!!! warning "Every number in this finding is a lower bound"
    These are what *our* attack recovers, not what is recoverable. An attack result
    bounds leakage from below and never from above: a better adversary, a richer
    auxiliary corpus, or a scheme that exploits content as well as participation would
    recover at least this much and possibly more. The honest reading of 0.205 is "at
    least 0.205", and of a control that drives recovery to 0.000 is "this attack is
    defeated", which is a weaker statement than "the channel is closed".

    That distinction is not hypothetical here.
    [Finding 13](#13-a-reliability-tag-can-replace-identity-and-the-leak-metric-cannot-tell-you-when)
    exhibits a scheme this attack scores at 0.000 while clearance leaks in aggregate at
    0.820, so the repository already contains a case where a zero from this metric meant
    nothing. Read every 0.000 below as "not by this route".

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

<!-- BEGIN GENERATED: linkage-controls -->
| Mitigation | Recovered | Mean anonymity set | Volume kept | Analysts silenced |
| --- | --- | --- | --- | --- |
| no mitigation | 0.205 | 9.98 | 100.0% | 0 |
| k-anonymity, k=10 | 0.205 | 9.98 | 98.7% | 0 |
| k-anonymity, k=25 | 0.110 | 10.27 | 83.1% | 0 |
| k-anonymity, k=50 | **0.000** | 13.32 | 20.2% | 0 |
| k-anonymity, k=100 | **0.000** | 16.00 | 6.1% | 0 |
| suppress rarest 25% | 0.110 | 10.27 | 92.0% | 0 |
| suppress rarest 50% | **0.000** | 11.56 | 74.5% | 0 |
| suppress rarest 75% | **0.000** | 12.46 | 42.4% | 0 |
| subsample p=0.5 | 0.170 | 9.28 | 51.1% | 23 |
| subsample p=0.2 | 0.100 | 8.40 | 19.3% | 64 |
| subsample p=0.05 | 0.040 | 6.37 | 4.7% | 116 |
| pool every analyst | **0.000** | 200.00 | 100.0% | 0 |

Bold marks a mitigation that drives recovery to zero. The guessing prior is 0.105, so a row at or below it has closed *this* attack; read the volume and silenced columns for what that cost. Contributions under the two rulings are 6,606 when compartments are shed and 400 when they are kept.
<!-- END GENERATED: linkage-controls -->

!!! warning "Corrected 2026-08-06: every cell of this table had drifted"
    The control ladder was hand-typed and had gone stale against the artifact in almost
    every column. Contributions were published as 7,053 and 1,000 where the measurement
    says 6,606 and 400; k=25 and the rarity rows were wrong in recovery, anonymity and
    volume; subsampling at p=0.5 silenced 23 analysts rather than 5, and at p=0.05 it
    silenced 116 rather than 108.

    Nothing in the argument moved, and that is the uncomfortable part. Every conclusion
    below still follows: small-k anonymity is a no-op, only the settings that destroy
    most of the volume close the attack, and pooling closes it at no volume cost while
    destroying what the fleet is for. A table can be wrong in every cell and still
    support its own conclusions, which is precisely why the conclusions cannot be the
    thing that checks it. The ladder is generated now.

Three things fall out of that table, and all of them are statements about this attack
rather than about the channel.

**Textbook k-anonymity at a small k is a no-op**, because in a fleet of 200 even a
rare clearance cell has several holders: k=10 changes nothing and costs almost
nothing. It only bites at k=50, where it takes 79.8% of the training data with it.
Choosing the threshold by rank instead of as an absolute count reaches the same
protection for a third of that cost (25.5% of the volume against 79.8%), and is
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
opens this channel: shedding compartments takes contributions from 400 to 6,606 and
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

## 12. Reliability cannot be estimated without identity where it matters

`scripts/measure_consensus_reliability.py`, `results/consensus_reliability.json`

[Finding 10](#10-a-fleet-learns-its-analysts-standard-not-the-worlds) and
[finding 11](#11-the-gate-clears-every-item-and-the-stream-still-names-the-analyst)
point in opposite directions, and the conflict is not rhetorical. A learner acquires
its teacher's standard exactly, so the mitigation is annotator-reliability weighting,
which needs contributor identity retained through training. But contributor identity
is what makes a fleet's contributions attributable to a person, and pooling
contributors is the only control that removes that leak at no cost in volume. One
wants identity; the other wants none.

The obvious reconciliation is to estimate reliability from **agreement among
contributions on the same task**, which needs no identity at all. Several analysts
reach the same task, so a contribution that disagrees with its task's consensus could
be discounted without anyone knowing who sent it. This measures whether that works.

Nine analysts, 200 tasks, sweeping how many hold a wrong standard. Three aggregation
strategies: pool everything; drop contributors whose own agreement with the world is
low (**needs identity**, and is an oracle rather than a method, since it is handed the
answer it would have to estimate); or take each task's majority verdict (**needs no
identity**).

| Wrong standard | Unweighted | Oracle (identity + truth) | Consensus | **Dawid-Skene** |
| --- | --- | --- | --- | --- |
| 0 of 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 of 9 | 0.887 | 1.000 | 1.000 | 1.000 |
| 4 of 9 | 0.849 | 1.000 | 1.000 | 1.000 |
| **5 of 9** | 0.811 | **1.000** | **0.660** | **0.660** |
| 7 of 9 | 0.735 | 1.000 | 0.660 | 0.660 |
| 9 of 9 | 0.660 | none | 0.660 | 0.660 |

**It is a cliff, not a curve, and it sits exactly at the majority crossing.** Up to 4
of 9, consensus is worth precisely as much as knowing who everyone is. At 5 of 9 it
collapses to 0.660, which is not a degraded score but *the wrong standard's own
agreement with the world* -- the bottom row is the same number, because a fleet that is
entirely wrong and a fleet that is merely mostly wrong return the same rule. Past the
crossing, consensus returns the wrong rule intact and with full confidence.

The [fleet sweep](#16-the-cliff-is-safe-only-because-the-fleets-were-drawn-independently)
shows both properties are structural: across fleets of 5 to 51 the cliff lands at the
majority threshold every time and the value it lands on is 0.660 every time.

**The canonical estimator falls off the same cliff, to the same value.** Dawid and
Skene (1979) infer per-contributor error rates and true labels jointly by EM, using no
ground truth, and surveys of truth inference report it beating majority voting. Here it
matches majority voting exactly at every composition here, collapse included --- at
this fleet size, which the sweep above qualifies: from fifteen contributors it gains one
step of margin before falling. The mechanism
is pinned in `tests/test_inference.py`: EM started from the majority vote is drawn
toward whatever the majority believes, so once most contributors hold the wrong
standard it concludes the correct minority *are* the unreliable ones. It rates them
below the wrong majority, inverts the labels, and emits no signal that anything went
wrong.

That row matters more than the majority-vote one. An earlier version of this finding
compared only against a plain vote, which made "reliability cannot be estimated" a claim
about the weakest available estimator. It is now a claim about the strongest one in
standard use.

The 9-of-9 row reports the oracle as **none** rather than 0.000. With every
contributor below the floor it drops all of them, and an empty stream is an absence of
evidence rather than a perfectly wrong answer. Scoring it 0.000 would say the
identity-based control failed, where what it did was correctly refuse everything.

**Why this is a negative result and not a reassurance.** The identity-free method
works exactly in the regime where you do not need it, and fails exactly in the regime
where you do. Two findings here say that regime is the realistic one:
[finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) found
over-escalation in **every** model tested, and
[finding 8](#8-being-right-and-sloppy-beats-being-wrong-and-careful) found that
agreement is not correctness and that a careful, self-consistent, mutually-agreeing
set of reviewers can be uniformly wrong. A fleet of model-assisted analysts is
precisely the population in which a wrong standard can hold the majority. There,
consensus does not merely fail to help; it certifies the error.

So the conflict between findings 10 and 11 stands. It is not resolvable by pooling,
and this measurement closes the cheapest escape from it rather than opening one.

!!! note "What this does not settle"
    One wrong standard (needing 2 of 3 rather than 3 of 3), one fleet size, one
    corpus. The oracle is a bound and not a candidate method: a real reliability
    estimator has to infer what this one is told. Dawid-Skene is canonical but not the only
    estimator; LFC and BCC lead on nominal data in Zheng et al.'s survey and neither is
    tested here. What no agreement-based method escapes is the crossing itself, since
    past it the wrong standard is what the agreement is *about*, and that argument
    covers the untested ones. What would genuinely resolve the conflict is a reliability
    estimate computed *under* secure aggregation rather than one recovered from
    pooled outputs, and that is not measured here.

## 13. A reliability tag can replace identity, and the leak metric cannot tell you when

`scripts/measure_tagged_aggregation.py`, `results/tagged_aggregation.json`

Three findings box each other in. A learner acquires its teacher's standard exactly,
so reliability weighting needs contributor identity
([finding 10](#10-a-fleet-learns-its-analysts-standard-not-the-worlds)). Identity is
what makes contributions attributable to a person, and pooling is the only free
control ([finding 11](#11-the-gate-clears-every-item-and-the-stream-still-names-the-analyst)).
Consensus cannot recover reliability from pooled outputs
([finding 12](#12-reliability-cannot-be-estimated-without-identity-where-it-matters)).
Identity is also what an audit trail needs, and auditability is not optional here: the
application this testbed serves is specified in terms of **auditable** training
signals, and provenance documentation for high-risk systems became an EU AI Act
obligation in August 2026.

The move left is a tag coarser than a person. Label each contribution with the
contributor's *reliability tier* rather than their identity: the aggregator can weight,
and nobody is named.

| Tagging scheme | Identifies an individual | Infers clearance | Groups |
| --- | --- | --- | --- |
| per-person (finding 11 baseline) | 0.205 | 1.000 | 200 |
| pooled, no tag | 0.000 | 0.330 | 1 |
| **tier, independent of clearance** | **0.000** | **0.330** | 3 |
| **tier, correlated with clearance** | **0.000** | **0.820** | 3 |

Naming the fleet's most common clearance level scores 0.330, so that is the floor a
tag has to beat to be leaking anything.

**The two tier rows are indistinguishable on the metric finding 11 uses and opposite
on the one that matters.** Both name nobody. The independent tag infers clearance at
0.330, exactly the prior, and discloses nothing. The correlated tag infers it at
**0.820**, because its three groups turn out to be *tier0 = 100% OPEN*, *tier1 = 100%
INTERNAL*, and *tier2 = the two highest levels*. It is a clearance label wearing a
reliability name, and every contribution carries it.

**This is the lower-bound problem made concrete.** Finding 11's recovery figures bound
leakage from below, and here is a scheme where the bound is vacuous: 0.000 from that
attack alongside 0.820 clearance inference from a different question asked of the same
stream. Any control validated only against one attack inherits this.

**It is also a limitation of finding 11's instrument, and worth stating plainly.**
That measurement scores an analyst as recovered only when the inference is unique
*and* attributable to them, which is the right question for "was someone named" and
the wrong one for "was a group characterised." Any tag coarser than a person scores
0.000 there by construction. Reading that zero as a privacy property would have let
this repository's own instrument certify a scheme that discloses clearance on every
record.

**What this means for the design.** The tag threads all three needles, but only under
an assumption the mechanism does not enforce and cannot check for itself: that
reliability tier is independent of clearance. On a real watch floor that assumption is
doubtful, since seniority plausibly drives both how much someone is read into and how
reliable their judgement is. So the deployable version of this is not "tag by tier",
it is "tag by tier **and measure the association between tier and clearance before
shipping**, and treat a non-trivial association as a disclosure." The measurement is
cheap; the assumption is the expensive part.

!!! note "What this does not settle"
    Three tiers, one fleet, one correlation structure, and clearance *level* rather
    than compartment set as the attribute inferred. The correlated case here is close
    to worst-case by construction, since tier is defined directly from level; a real
    association would be weaker and the leak proportionally smaller, though not zero.
    The independent case is best-case for the same reason. What the pair establishes
    is that the two ends differ enormously while looking identical to a per-analyst
    metric, not where a particular deployment would land between them.

## 14. What the agent costs on the hardware it is meant to run on

`scripts/measure_edge_cost.py`, `results/edge_cost.json`

Every finding above asks whether the triage agent is *correct*. None asks whether it
fits. The application is edge triage on nodes "operating on the scale of a laptop",
and a correct agent that does not fit is not a result.

The gap turned out to be half-closed already and never reported. Findings 1 to 3 and 5
were all measured on an 8 GB consumer GPU, which is laptop-class hardware, so those
accuracy numbers were edge numbers the whole time. What was missing was a cost attached
to them.

Three costs, because they bind at different moments.

| Cost | Measured | What it governs |
| --- | --- | --- |
| **Sync**, per node per round | **57.1 MiB** (bf16) | Bandwidth to personalize |
| **Footprint**, resident | 4,466 MiB (7B) / 1,841 MiB (3B) | Whether the node can answer at all |
| **Cold start** | **3.7 s** | Waking a node that slept |
| **Warm decision** | **0.321 s** median, 0.337 s p95 (n=199) | Throughput once resident |

**A round costs the adapter, not the model.** The adapter is 29,933,568 of
3,115,872,256 parameters, 0.961%, so a personalization round puts **57.1 MiB** on the
wire at bf16 and the base never moves. Doubling that to 114 MiB is the cost of saving
a checkpoint in fp32 without noticing, which is worth stating because it is invisible
until someone measures the file.

**One resident node sustains about 11,200 triage decisions per hour** at 0.321 s
median. Waking a node costs 3.7 s, or what **11 warm decisions** cost, which is the
number that matters for a duty cycle: a node that sleeps between watches pays it every
time, and one held resident pays it once.

!!! warning "The cold start is forced, and it has to be"
    An earlier run of this measurement reported a 0.6 s cold start and labelled it
    "the weights reaching VRAM". They had never left: the server keeps a model resident
    for minutes after a request, so the figure silently depended on whether anything
    had touched the model recently. The script now evicts the model before timing, so
    the label is true by construction rather than by luck. Run twice back to back it
    reports 4.2 s and 3.6 s where the unforced version reported 0.6 s.

    This is [finding 9](#9-a-measurement-that-repeats-one-prompt-measures-the-wrong-thing)
    wearing different clothes. There a repeated prompt measured a warm cache and was
    reported as irreproducibility; here a repeated *run* measured a warm model and was
    nearly reported as a cold start. Both come from the same place: a first call is not
    like the calls after it, and a measurement that does not say which kind it took is
    not saying what it measured.

!!! note "What this does not settle"
    One machine, one server, one quantization, `num_predict` of 8 because a triage
    answer is one word. The spread is tight (0.297 s to 0.337 s) but tightness on one
    machine is not generality.

    This was measured at nineteen warm calls until 2026-08-04 and marked unquotable for
    it. That was the smaller problem. At nineteen the reported p95 was
    `warm[min(n-1, int(0.95*n))]`, which is `warm[18]` -- **the largest value in the
    sample**, published under a percentile's name, and the maximum of nineteen draws is
    the noisiest statistic they admit. At 199 the same index is the tenth largest, which
    is an estimate. The corpus had to grow too: `--tasks 200` was silently capped at 59
    because only 60 events were generated, so the requested sample was really the corpus
    size. Nothing
    here measures energy or thermal throttling, which current on-device guidance treats
    as the binding constraint on sustained mobile inference and which an 8 GB desktop
    card will not reproduce. The sync figure is arithmetic on parameter counts rather
    than a file on a wire, so it omits protocol overhead and any compression.

## 15. The standard privacy mechanism spends the budget on the wrong variable

`scripts/measure_privacy_budget.py`, `src/pharos/budget.py`, `results/privacy_budget.json`

Finding 11's control ladder was k-anonymity, rarity suppression, subsampling, and
pooling. Every one of those we invented. **Differential privacy, the field's default
control and the mechanism this work's motivating abstract promises, was simply absent**,
and stayed absent across three findings, because a self-generated list carries no signal
that it is incomplete.

Closing the omission turned out to matter more than adding a row.

**Value noise cannot help, at any epsilon.** The attack reads the map from pseudonym to
task-identifier set. It never inspects a verdict, a released label, or a word of text.
Any mechanism acting on contribution *values* is therefore the identity function as far
as the attack is concerned, which is a proposition rather than a measurement. Measured
anyway, because a claim that a standard defence is useless should be demonstrated:

| Verdict flip rate | 0.0 | 0.1 | 0.25 | **0.5** |
| --- | --- | --- | --- | --- |
| Recovery | 0.205 | 0.205 | 0.205 | **0.205** |

At a flip rate of 0.5 the verdict carries no information whatsoever and recovery has
not moved. This is the shape of the mistake the finding exists to prevent: noise added
to what a system *reports* while the leak is in what it *touches*.

**Participation noise works, and the composed budget is what it costs.** The variable
that leaks is whether a contribution about task T exists under pseudonym P at all, so
the mechanism is randomized response over that indicator: keep an eligible contribution
with probability `keep`, and fabricate one on an unreachable task with probability
`fabricate`. The second half is what buys deniability and is exactly what subsampling
lacks.

<!-- BEGIN GENERATED: privacy-budget -->
| Keep | Fabricate | Recovered | Label noise | epsilon per indicator | epsilon composed |
| --- | --- | --- | --- | --- | --- |
| 0.9 | 0.1 | 0.210 | 0.3603 | 2.20 | 435.1 |
| 0.8 | 0.2 | 0.155 | 0.5602 | 1.39 | 274.5 |
| 0.7 | 0.3 | 0.095 | 0.6823 | 0.85 | 167.8 |
| 0.6 | 0.4 | 0.070 | 0.7716 | 0.41 | 67.5 |

Participation noise against a baseline recovery of 0.205 over 200 analysts. `epsilon composed` is the better of basic and advanced composition across 198 indicators at delta = 1e-05. Read the label-noise column beside it: the setting that suppresses the attack corrupts most of the labels, and an epsilon in the hundreds is a budget in name rather than a guarantee.
<!-- END GENERATED: privacy-budget -->

Recovery here is again what *this* attack achieves, so the column bounds leakage from
below; the epsilon columns, by contrast, bound it from above for any adversary, which
is exactly why a mechanism with a stated budget is worth more than a control validated
against one attack. That contrast is the reason this finding is worth having even
though its numbers are worse than the ladder's.

**The two epsilon columns are the finding.** Randomized response bounds the likelihood
ratio for *one* indicator, and the attack observes all of them: two clearances in this
fleet are separated by as many as **198 tasks**, so the guarantee against an adversary
telling one from the other is the budget composed over that set. At the strongest
setting tested, a per-indicator ε of **0.41** looks excellent and the composed ε is
**66.7**, which is not a privacy guarantee in any usual sense. It costs **75.6% label
noise** to get there.

Quoting the per-indicator figure alone would describe a mechanism far stronger than the
one deployed. That is not a hypothetical failure; it is the number a deployment would
naturally report.

Two smaller results in the table. **Subsampling has infinite epsilon**, which is why it
sat in finding 11's ladder as a cost with no protection beside it: dropping
contributions makes a stream sparser without making any surviving contribution
deniable. And recovery at `keep=0.9, fabricate=0.1` is *higher* than the baseline,
0.210 against 0.205, which at 200 analysts is one person and is read as noise rather
than as noise-helping-the-attack.

!!! note "What this does not settle"
    One mechanism family, one fleet, and epsilon computed for an adversary
    distinguishing two clearances rather than for the full recovery task the attack
    actually performs, which would need a different accounting. The composed bound uses
    the widest separation in the fleet, so it is the worst case rather than the typical
    one. A mechanism designed for set-valued participation, rather than randomized
    response applied per indicator, might compose far better and is not tested here.
    Nothing about the arithmetic says DP is the wrong tool; it says this application of
    it does not deliver a usable guarantee at any utility worth having.

## 16. The cliff is safe only because the fleets were drawn independently

`scripts/measure_correlated_fleets.py`, `results/correlated_fleets.json`

[Finding 12](#12-reliability-cannot-be-estimated-without-identity-where-it-matters)
swept the number of wrong analysts as a free parameter and found a cliff at the
majority crossing. That answers "what happens at 5 of 9" and leaves the question a
deployment actually faces untouched: **how often does a fleet end up there?**

Every fleet measured in this repository has been drawn i.i.d. Nothing enforces that,
and it is the most favourable assumption available. Analysts share a training pipeline,
inherit a house style, and are corrected by the same supervisors, so a wrong standard
propagates through a cohort rather than appearing independently in each person.

Three structures, all with the *same* expected error rate, differing only in how it is
distributed. P(wrong majority) is exact, from a binomial over schools rather than over
people. Expected agreement composes that exact probability with agreements measured on
real fleets through the real pipeline.

| Population error rate | Independent | Three schools | One culture | Understatement |
| --- | --- | --- | --- | --- |
| **0.1** | **0.001** | 0.028 | **0.100** | **112×** |
| 0.2 | 0.020 | 0.104 | 0.200 | 10.2× |
| 0.3 | 0.099 | 0.216 | 0.300 | 3.0× |
| 0.4 | 0.267 | 0.352 | 0.400 | 1.5× |
| 0.5 | 0.500 | 0.500 | 0.500 | 1.0× |

**The understatement is largest exactly where a deployment would cite it for
reassurance.** At a 10% population error rate, independence says one fleet in a
thousand crosses the majority; a single shared training culture says one in ten. That
is the regime where a designer would say "a wrong majority essentially cannot happen",
and it is the regime where the assumption is off by two orders of magnitude. By the
time the rates converge, at 0.5, the reassurance was worthless anyway.

Expected agreement follows: 1.000 against 0.964 at a 10% error rate, 0.993 against
0.927 at 20%. Dawid-Skene tracks consensus exactly in every cell, which independently
reproduces finding 12's central result on a different fleet distribution.

!!! success "The 112x is a floor, and it grows with the fleet"
    Fleet size is swept here for the same reason it is swept in
    [finding 17](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst):
    nine was not chosen on principle. P(wrong majority), independent against one
    culture:

    | Fleet | rate 0.1: independent (exact) | one culture | understatement | rate 0.3: independent | one culture | understatement |
    | --- | --- | --- | --- | --- | --- | --- |
    | 5 | 8.56e-03 | 0.100 | 12x | 0.163 | 0.300 | 2x |
    | **9** (committed) | 8.91e-04 | 0.100 | **112x** | 0.099 | 0.300 | 3x |
    | 15 | 3.36e-05 | 0.100 | **2,974x** | 0.050 | 0.300 | 6x |
    | 25 | 1.62e-07 | 0.100 | **616,966x** | 0.018 | 0.300 | 17x |
    | 51 | 1.98e-13 | 0.100 | **5.0e11x** | 0.001 | 0.300 | **214x** |

    **The understatement grows monotonically with fleet size, at every rate.** The
    mechanism is not subtle: independent draws concentrate as the fleet grows, so
    P(wrong majority) falls toward zero, while a shared culture is a single coin
    whatever the headcount.

    !!! warning "Corrected 2026-08-05: these were published as "unbounded", and they are not"
        The three widest fleets carried `unbounded` here, on the stated grounds that
        the independent probability "rounds to zero at three decimals and the ratio
        stops being finite". It does not stop being finite --- the *display* rounded to
        zero and the artifact divided by that rounded value, so the ratio was recorded
        as null. The same rounding put 111.1 in the artifact where the exact binomial
        gives 112.2, which is the value the prose had already been corrected to.

        `measure_fleet_sensitivity.py` now computes the ratio from the exact binomial
        and records `exact_independent` beside the rounded rate. The corrected numbers
        are larger than the claim they replace, and they make the argument stronger
        rather than weaker: an understatement of 616,966x at 25 analysts is a sharper
        statement than "unbounded", which reads as a limit artifact and invites exactly
        the dismissal it deserved.

    So the reported 112x is not the headline number, it is the **smallest** the
    understatement gets in any fleet a deployment would plausibly field. This is the
    one place in this document where sweeping a parameter made a finding stronger
    rather than narrower, and the direction is the uncomfortable one: the i.i.d.
    assumption gets more dangerous as the fleet scales, which is the opposite of how a
    designer would expect a statistical assumption to behave.

!!! warning "The first version of this measurement was noise, and it is worth saying why"
    P(wrong majority) was initially *estimated* from 40 drawn fleets. Its standard
    error near 0.1 is about 0.05, and the result duly showed clustered cells **below**
    independent ones at two rates, an ordering the mathematics forbids. The quantity has
    a closed form; sampling it invented noise and then reported the noise as a reversal.

    The rewrite computes it exactly and samples only the conditional agreements, which
    is cheaper as well as correct. The lesson generalises past this script: **before
    estimating anything, check whether it can be derived.** A simulated quantity and a
    computed one look identical in a results table.

!!! note "What this does not settle"
    Equal-sized schools, a single wrong standard, and correlation modelled as
    all-or-nothing per school. Real cohorts overlap, partially agree, and drift, and
    every one of those sits between the two extremes bracketed here rather than outside
    them. The conditional agreements rest on 60 draws per cell, which is thin for the
    conditionals though it does not touch the exact probabilities. What the pair of
    columns establishes is that the i.i.d. assumption is load-bearing and unstated, not
    where a particular organisation would land between them.

## 17. Adding item difficulty does not separate a hard case from a wrong analyst

`scripts/measure_difficulty_confound.py`, `src/pharos/inference.py`, `results/difficulty_confound.json`

[Finding 12](#12-reliability-cannot-be-estimated-without-identity-where-it-matters)
showed agreement-based estimators failing once a wrong standard holds the majority, and
the natural objection is that the estimator was too simple. Dawid-Skene blames the
annotator for every disagreement. The obvious missing term is the **item**: some cases
sit near the boundary and everyone struggles with them. Whitehill et al. (NIPS 2009)
add exactly that, estimating labeler ability, item difficulty and the true label
jointly. That model is now implemented here as `pharos.inference.glad`.

This corpus can test it unusually well, because it carries a real difficulty structure
built for an unrelated reason. The significant class is a conjunction of three facts,
and three of the ten background patterns carry **two** of those three. Those routine
items sit one fact from the boundary:

Items whose signature overlap puts them one fact from the boundary are counted in the
generated table below, beside the difficulty each fleet estimates for them. Overlap 2 is
the near-boundary class: routine under the correct rule, and exactly the class a
two-of-three reviewer gets wrong.

The 52 near-boundary items are also **exactly** the items a reviewer holding a
two-of-three standard gets wrong. "This item is ambiguous" and "this reviewer applies
the wrong rule" predict identical data.

**The control decides it.** Estimated difficulty by true overlap, under fleets that
differ only in composition:

<!-- BEGIN GENERATED: difficulty-confound -->
| Fleet | ovl=0 | ovl=1 | **ovl=2** | ovl=3 | Spread | Converged |
| --- | --- | --- | --- | --- | --- | --- |
| correct fleet (control) | 1.085 | 1.085 | 1.085 | 1.085 | 1.000 | yes, 1 iter |
| correct, 15% random slip | 1.578 | **2.120** | 1.954 | 1.683 | 1.343 | yes, 10 iter |
| 3 of 9 wrong standard | 0.911 | 0.911 | **2.318** | 0.911 | 2.543 | yes, 6 iter |
| 5 of 9 wrong standard | 0.885 | 0.885 | **3.421** | 0.885 | 3.867 | yes, 25 iter |

| Signature facts present | ovl=0 | ovl=1 | **ovl=2** | ovl=3 |
| --- | --- | --- | --- | --- |
| Items | 40 | 39 | 52 | 69 |

Estimated item difficulty by true signature overlap, under fleets differing only in composition. **Bold** in each row marks that row's own peak, read from the artifact rather than asserted in prose.
<!-- END GENERATED: difficulty-confound -->

**A correct fleet finds no difficulty structure at all.** Spread 1.0, flat across the
corpus, because the correct rule resolves a two-of-three item unambiguously: two is not
three, so the item is routine and there is nothing to be uncertain about. Every bit of
the structure in the rows below it is manufactured by the reviewers.

So the estimator does not fail by being unable to find the wrong analysts. It fails by
**relabelling a wrong standard as a property of the data**, and the two diagnoses call
for opposite actions:

- *"These items are hard"* → clarify the guidance, accept lower accuracy on them, and
  leave the reviewers alone.
- *"A third of the fleet holds the wrong rule"* → retrain those reviewers, and expect
  full accuracy afterwards.

A governance process reading GLAD's output takes the first, which is the wrong one.

**And the ability estimate names the wrong people.** Difficulty is not the only parameter
GLAD reports. It also scores each reviewer, and that score is the one a supervisor would
actually query, because it answers *whom do I retrain*:

| Fleet | wrong-standard reviewers | correct reviewers | Verdict |
| --- | --- | --- | --- |
| 3 of 9 wrong standard | 1.62 | 4.00 | correct: the wrong rule scores 2.5x lower |
| 5 of 9 wrong standard | **3.89** | **2.00** | inverted: the wrong rule scores 1.95x *higher* |

Below the majority the estimate is right, and a supervisor acting on it retrains the
three reviewers who hold the wrong rule. Above it the ranking flips, and the same
supervisor acting on the same field retrains the four who are correct while certifying
the wrong standard as the expert one. The ability score tracks the majority rather than
the truth, which is finding 12's cliff again in the one parameter that was supposed to
survive it: item difficulty was added to explain away disagreement the fleet could not
resolve, and it relocates the error rather than isolating it.

This is why the failure is worse than a low agreement number. At 5 of 9, agreement of
0.660 at least announces that something is wrong. The ability column does not: it is
confident, well separated, and backwards. The inversion holds at every corpus size
swept below, which is more than can be said for the CC-Rasch row.

**Random error is not exempt, and has a different signature.** The 15%-slip row inflates
difficulty with no wrong standard anywhere, but the shape is different in the way that
matters: it is elevated across *every* band, 1.58 to 2.12, rather than concentrated on
one. The systematic rows do the opposite --- three of the four bands sit at or below the
control value and a single band carries the whole effect.

That is the discriminating shape, and it is not "which band peaks". This section
previously said the slip row peaks at overlap 0, *furthest* from the boundary, and drew
the contrast from that. It peaks at overlap 1, and the claim was never measured: it was
written beside a run and never re-read against the artifact. The contrast survives the
correction because it never depended on which band happened to be highest --- random
error raises the floor everywhere, systematic error raises one band and leaves the rest
flat. Any annotator error inflates apparent difficulty; only systematic error inflates it
*at the boundary* while leaving the others at the control value.

It is a shape rather than a magnitude: the slip row's spread of 1.343 is smaller than
3-of-9's 2.543, but a single summary number cannot tell you which band carries it, and
the peak alone cannot either.

!!! warning "This section published different numbers, and the reason is worth recording"
    Every magnitude above changed on 2026-08-02. The first version of this measurement
    reported spreads of 18.9 and 34.1, an ability inversion of 31x, and a random-slip
    row that did not converge at all -- with a whole subsection explaining that GLAD is
    "an unregularised maximum-likelihood fit whose parameters are unbounded" and that
    its magnitude therefore could not be quoted.

    That explanation was about our implementation, not about GLAD. Whitehill et al.
    specify priors in their section 3.1, and this had omitted them: *"In our
    implementation we used Gaussian priors (mu = 1, sigma = 1) for alpha. For beta, we
    need a prior that does not generate negative values. To do so we re-parameterized
    beta = e^beta' and imposed a Gaussian prior (mu = 1, sigma = 1) on beta'."* Without
    those terms the residual never reaches zero as the sigmoid saturates, the
    parameters climb without bound, and EM never settles. With them, every composition
    converges in under 40 iterations and the divergence disappears entirely.

    **The finding survives and the caveat does not.** The control is still exactly flat,
    the structure still appears only when reviewers hold a wrong standard, it is still
    localised on precisely the band a two-of-three reviewer errs on, and the ability
    estimate still inverts at the majority. What changed is that the effect is smaller
    than first reported -- an inversion of 1.95x rather than 31x -- and that nothing
    needs withdrawing for want of convergence, because everything now converges.

    The lesson is narrower than "check your implementation". It is that a *negative*
    result about someone else's method is the one to re-derive from their paper before
    publishing, because a missing regulariser looks exactly like the method failing,
    and the failure is more interesting than the bug so it gets less scrutiny. Zheng et
    al.'s report that GLAD converges slowly made the wrong answer more plausible, not
    less.

!!! note "Independent corroboration, from a benchmark that was not looking for this"
    Zheng et al. report that "the methods that model task difficulty (GLAD) or
    latent topics (Multi) in tasks do not perform significantly better in quality;
    moreover, they often take more time to converge" -- measured across real datasets,
    against many alternatives, with no wrong-standard construction anywhere in sight.
    That is the *what*. This finding supplies a *why* for at least one common case: where
    the hard items and the reviewer's blind spot coincide, the difficulty term has an
    error to absorb, and absorbing it is not the same as modelling it.

    Zheng, Li, Li, Shan and Cheng, *Truth Inference in Crowdsourcing: Is the Problem
    Solved?*, PVLDB 10(5):541-552, 2017.

**The strongest objection is that GLAD is too coarse, and it does not hold.** A
two-of-three reviewer is not globally unreliable. They are exactly right on the
significant class and wrong only on routine items at the boundary, and GLAD gives each
reviewer a single ability number that cannot express that. This is not our observation:
Singer et al. make it about GLAD directly, that a single ability per annotator
"prevents them from distinguishing majority-class competence from minority-class
competence", and their **CC-Rasch** model conditions both ability and difficulty on the
class to fix it. It is five days old at time of writing and is the best available
answer to the objection, so it is implemented in `pharos.inference.cc_rasch` and run on
the same fleets:

| Fleet | Dawid-Skene | GLAD | CC-Rasch | Routine-class gap |
| --- | --- | --- | --- | --- |
| correct (control) | 1.000 | 1.000 | 1.000 | - |
| correct, 15% random slip | 0.990 | 0.990 | 0.979 | - |
| 3 of 9 wrong standard | 1.000 | 1.000 | 1.000 | **+4.12** |
| **5 of 9 wrong standard** | **0.660** | **0.660** | **1.000** | **+3.92** |

Read on its own, that bottom row says class-conditioning *answers* the objection:
CC-Rasch recovers the truth exactly where Dawid-Skene and GLAD both collapse to 0.660.
**It does not, and the reason it does not is the more useful result.**

!!! danger "CC-Rasch is bimodal here, and the committed corpus size sits on one mode"
    The corpus size is a nuisance parameter: it should move the estimate a little and
    move no conclusion at all. Sweeping it moves this conclusion all the way.

    | `--events` | Dawid-Skene | CC-Rasch | Routine-class gap |
    | --- | --- | --- | --- |
    | **200** (committed) | 0.660 | **1.000** | +3.92 |
    | 300 | 0.669 | 0.669 | - |
    | 400 | 0.656 | 0.656 | +0.002 |
    | 500 | 0.637 | 0.637 | - |
    | 600 | 0.635 | 0.635 | +0.027 |
    | 700 | 0.633 | 0.633 | did not converge |
    | 800 | 0.637 | **1.000** | +4.32 |
    | 900 | 0.626 | **1.000** | +3.97 |

    CC-Rasch does not degrade smoothly between these. It lands on one of two answers:
    the truth exactly, or the wrong-standard majority exactly, with nothing in between.
    That is the signature of label switching -- two mirror-image solutions fitting the
    data equally well, with the draw deciding which one EM reaches. The identifiability
    machinery in `pharos.inference` was added to prevent precisely this and is evidently
    not sufficient to.

    **So no CC-Rasch number here is quotable in either direction.** Not the 1.000, which
    would say the objection is answered, and not the 0.660s, which would say it fails
    identically. Five of eight sizes support the second reading and three the first,
    which is a statement about the estimator's identifiability rather than about
    class-conditional modelling. Fixing it is open work.

!!! danger "Fleet size decides which estimators agree, and 9 is the size where they all look alike"
    Fleet size is the other researcher degree of freedom here, and nothing chose 9 on
    principle. Sweeping it, at a bare majority holding the wrong standard:

    | Fleet | Dawid-Skene | GLAD | CC-Rasch | GLAD ability inverted |
    | --- | --- | --- | --- | --- |
    | 5 | 0.660 | 0.660 | 0.660 | yes |
    | **9** (committed) | **0.660** | **0.660** | 1.000 | yes |
    | 15 | **1.000** | 0.660 | 1.000 | yes |
    | 25 | **1.000** | 0.660 | 1.000 | yes |
    | 51 | **1.000** | 0.660 | 1.000 | yes |

    **Dawid-Skene recovers the truth at a bare majority once the fleet reaches 15**, and
    CC-Rasch already does at 9. **GLAD never does, at any size.** The three estimators
    only *all* fail together at a fleet of five, which is the reading the original claim
    rested on; by nine, two of three; from fifteen, only GLAD. The artifact records that
    as an invariant (`fleets_where_all_three_estimators_agree`) rather than leaving it
    to be read off the table, because an earlier draft of this paragraph said all three
    sat at 0.660 at nine while the table beside it showed CC-Rasch at 1.000.

    Two things this does *not* undo. GLAD's failure is size-independent, and its ability
    inversion holds at **every** size tested -- so the claim this finding is actually
    about, that adding an item-difficulty term converts a wrong standard into a property
    of the data, is the part that survives the multiverse. What it does undo is the
    sentence that all three agree to three decimals: true at 9, false from 15.

    The reason nobody had checked is that `FLEET = 9` was a module constant in this
    script and in `measure_correlated_fleets.py`, so the parameter that decides the
    result could not be varied without editing source. Both now take `--fleet`, and the
    default reproduces every committed number exactly.

    Reported as a multiverse rather than a single specification, after Linde et al.
    ([arXiv:2605.19745](https://arxiv.org/abs/2605.19745), 2026), whose point is that
    sweeping the defensible choices mostly exposes *computational failures that
    otherwise go unreported* -- which is what happened twice on this page.

What survives the sweep is the part that does not depend on CC-Rasch. Dawid-Skene and
GLAD agree with each other and drop to roughly 0.63-0.67 at the crossing at *every*
size, and GLAD's per-reviewer ability inverts at every size. The confound is established
against those two estimators. Whether a correctly identified class-conditional model
escapes it is now an open question, and this repository cannot currently answer it.

!!! note "What this does not settle"
    Two estimators of this family, one wrong standard, one corpus whose difficulty
    structure is discrete and known. The finding is not that item-difficulty modelling
    is worthless; it is that in a setting where the hard items and the reviewer's blind
    spot coincide by construction, it converts one into the other, and that coincidence
    is the normal case rather than a contrived one whenever a rule has a boundary.

    Both estimators are implemented here from their papers rather than adapted from
    reference code, and both needed a correction found by checking that EM's
    observed-data log-likelihood rises monotonically. GLAD was missing its priors.
    CC-Rasch's centring step shifted ability and difficulty in *opposite* directions,
    which is not the gauge transformation the constraint calls for -- the model depends
    on their difference, so both must move the same way. Before that fix its likelihood
    fell between iterations and it stalled on one composition. The monotonicity check
    is cheap, it is the definitive test that an EM implementation is doing EM, and it
    caught what reading the code twice did not.

## 18. The estimate moves under secure aggregation, and the cliff does not move with it

`scripts/measure_secure_reliability.py`, `results/secure_reliability.json`

This repository has carried one open problem since
[finding 13](#13-a-reliability-tag-can-replace-identity-and-the-leak-metric-cannot-tell-you-when).
[Finding 10](#10-a-fleet-learns-its-analysts-standard-not-the-worlds) needs contributor
identity to weight a fleet by reliability;
[finding 11](#11-the-gate-clears-every-item-and-the-stream-still-names-the-analyst)
shows identity is the leak; findings 12 and 13 close both ways of recovering the first
from the second *after* aggregation. The direction never tried was the other one:
compute the estimate **under** aggregation, so the per-analyst stream the attack reads
is never produced.

It works, and it is exact rather than approximate, because Dawid-Skene splits along
the seam a secure sum offers. Its M step is per contributor over that contributor's own
reports, so it runs locally and is never transmitted. Its E step needs a product over
contributors per task, which is a sum in logs, and a sum is what secure aggregation
reveals and all that it reveals. `pharos.secagg` supplies masked aggregation over a
64-bit ring and returns a `ServerView` with no per-client field on it; `pharos.inference`
supplies `federated_dawid_skene` on top.

**The port does not change the answer.** Across all ten fleet compositions, the largest
posterior disagreement between the federated and centralized estimators is
**3.8e-14**, with **zero** label disagreements and matching iteration counts.

| Wrong of 9 | Centralized | Federated | Posterior gap |
| --- | --- | --- | --- |
| 0-4 | 1.0000 | 1.0000 | ≤ 7.9e-15 |
| **5** | **0.6598** | **0.6598** | 3.8e-14 |
| 6-9 | 0.6598 | 0.6598 | ≤ 2.2e-15 |

**And the cliff does not move.** It sits at 5 of 9 under both, which was the prediction
stated before the run. That localises the problem, and the localisation is the finding:
the cliff is not a leak and not an artifact of pooling, it is **non-identifiability**.
Dawid-Skene's parameters are identified only up to a relabelling of the latent class,
and the tie-break the literature relies on is diagonal dominance -- FedDS
(Dong, Zhu, Shang and Xue, *Information Sciences* 745:123425, 2026) assumes exactly
that in its Eq. (16). A fleet whose majority holds the wrong standard is the fleet
where that assumption is false.

!!! note "FedDS is the nearest relative, and it sits on the wrong side of the leak"
    FedDS brings Dawid-Skene to federated learning to weight clients by estimated
    reliability without a labelled public dataset at the server, which is the same
    problem this finding addresses. It runs EM **at the server** over each client's
    prediction vector on an unlabelled public set, so the server observes precisely the
    per-client stream finding 11 attacks, and the paper does not discuss secure
    aggregation. The contribution here is not the estimator, which is theirs and Dawid
    and Skene's; it is that the estimator can be computed with the server holding only
    sums, at no cost in accuracy, and that doing so leaves the identifiability failure
    exactly where it was.

### What the aggregate discloses anyway

Closing the naming channel opens a counting one, and it is worth stating precisely
rather than claiming the protocol leaks nothing. Majority-vote initialization is a
per-task mean; a mean needs its denominator; that denominator is the number of analysts
who could read the task. Nobody is named. But a task's readership is decided by the join
of its sources, so **reading one headcount is reading, exactly and with no inference, how
many analysts are cleared for that join.**

Over a 200-analyst fleet, every distinct source-join in the stream yields an **exactly
correct** headcount, with no estimation error at all:

<!-- BEGIN GENERATED: secure-readership -->
| Sensitivity \| compartments | Read off the aggregate | True | Adversary's prior |
| --- | --- | --- | --- |
| 0 \| (none) | **200** | 200 | 200.00 |
| 1 \| SENSOR | **67** | 67 | 75.00 |
| 2 \| SENSOR | **43** | 43 | 50.00 |
| 2 \| LEGAL | **34** | 34 | 50.00 |
| 3 \| LIAISON | **32** | 32 | 25.00 |
| 3 \| PARTNER | **25** | 25 | 25.00 |
| 2 \| LEGAL,SENSOR | **19** | 19 | 25.00 |
| 3 \| LIAISON,SENSOR | **15** | 15 | 12.50 |
| 3 \| LEGAL,PARTNER | **10** | 10 | 12.50 |
| 3 \| PARTNER,SENSOR | **10** | 10 | 12.50 |
| 3 \| LEGAL,LIAISON | **9** | 9 | 12.50 |
| 3 \| LEGAL,PARTNER,SENSOR | **6** | 6 | 6.25 |
| 3 \| LIAISON,PARTNER,SENSOR | **6** | 6 | 6.25 |
| 3 \| LEGAL,LIAISON,SENSOR | **5** | 5 | 6.25 |

14 of 14 joins yield an exactly correct headcount. Bold marks the exact ones; a row that stopped being exact would lose its bold here rather than in a sentence nobody reran.
<!-- END GENERATED: secure-readership -->

The prior column is what makes this a disclosure rather than a restatement of the fleet
size: clearances are drawn uniformly from the candidate space, so an adversary who knew
only the draw would expect 12.50 analysts on LEGAL+LIAISON and learns the answer is
exactly 9. The channel is **task-side rather than person-side** -- it discloses the
fleet's clearance census, not who holds what -- and on a watch floor "how many people
are read into LIAISON" is not obviously the safer of the two.

!!! warning "Stated as structure, not as a score"
    Finding 11's attack is not reported here at 0.000. Its input is the set of task
    identifiers appearing under each pseudonym, and under secure aggregation there are
    no pseudonyms, so the attack has no input rather than a poor one. Scoring an
    undefined observation as a defeated attack would credit the protocol with a
    measurement nobody made. What is measured is what replaces it.

    The masking is real and its cancellation is checked. Key agreement, threshold secret
    sharing for dropout recovery, and authentication are **not** implemented: they decide
    who may run the protocol and what happens when a client vanishes mid-round, neither
    of which changes what a server learns from a completed round, which is the quantity
    under study.

## 19. An authority of record repairs the cliff, and its price explodes

`scripts/measure_authority_anchors.py`, `results/authority_anchors.json`

!!! warning "Corrected 2026-08-06: the four-wrong row was repairing nothing"
    This finding scored a repair as `agreement >= 0.95`, full stop. Its two siblings ---
    the audit-policy sweep of [finding 20](#20-audit-where-the-fleet-splits-and-the-prediction-that-said-otherwise)
    and the blind-spot sweep of [finding 21](#21-the-corpus-the-audit-policy-cannot-handle-built-on-purpose)
    --- both additionally require that some *unanchored* label actually changed, because
    anchoring a task the estimator gets wrong removes it from the denominator and lifts
    the score without correcting anything. Finding 21's threshold table was withdrawn
    over exactly that. This script measures the same quantity on the same corpus and the
    guard was never mirrored to it.

    Mirroring it moves one published number. The **four-wrong row** reported a price of
    **0 anchors**, which read as "a minority wrong standard repairs itself for free". It
    now reads **not reached**, which is the truthful statement: at four of nine wrong the
    estimator makes no errors to begin with, so there is nothing for an authority to
    correct and no budget at which a correction occurs. A control row was being published
    as a result.

    Every other price is unchanged --- 12 at a bare majority, 80 at six of nine, 150 at
    seven, never at nine --- because in those compositions the anchors were doing real
    work all along. The correction narrows what the finding claims rather than moving it.
    (Those three numbers moved later, when finding 26 swept the corpus rather than the
    anchor draw. The retraction below carries the ranges; only the "never at nine"
    survived as stated.)

Finding 18 localised the cliff to non-identifiability, and
[finding 17](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst)
already showed no better estimator escapes it from the data alone. What breaks a
relabelling degeneracy is an **exogenous** label: a task whose disposition is asserted
by an authority rather than inferred from the fleet. That is the *authority of record*
the build order has owed since step 3, and this prices it.

**The scoring rule is the methodology.** An anchored task's label was handed over, so
counting it would measure how many answers the authority supplied rather than what they
bought. Every number below is computed **only over unanchored tasks**.

<!-- BEGIN GENERATED: authority-grid -->
| Wrong of 9 | 0 | 1 | 2 | 3 | 5 | 8 | 12 | 20 | 30 | 50 | 80 | 100 | 120 | 150 | 180 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | 0.660 | 0.660 | 0.660 | 0.660 | 0.656 | 0.653 | 0.645 | 0.629 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | 0.660 | 0.660 | 0.660 | 0.660 | 0.656 | 0.653 | 0.645 | 0.629 | 0.647 | 0.653 | 0.633 | **1.000** | 1.000 | 1.000 | 1.000 |
| 7 | 0.660 | 0.660 | 0.660 | 0.660 | 0.656 | 0.653 | 0.645 | 0.629 | 0.647 | 0.653 | 0.633 | 0.652 | 0.692 | **1.000** | 1.000 |
| 9 | 0.660 | 0.660 | 0.660 | 0.660 | 0.656 | 0.653 | 0.645 | 0.629 | 0.647 | 0.653 | 0.633 | 0.652 | 0.692 | 0.667 | 0.636 |

Agreement over unanchored tasks only, at anchor seed 909. Bold marks each composition's first budget reaching 0.95; that crossing moves with the draw, which is what the table below reports.
<!-- END GENERATED: authority-grid -->

**The price is not linear and it is not a curve.** It is a second threshold, and it
moves far faster than the fleet's error does:

<!-- BEGIN GENERATED: authority-price -->
| Wrong of 9 | Audited items needed (median) | Share of the round | Range over 21 draws | Draws that reached it |
| --- | --- | --- | --- | --- |
| 4 | not reached within 180 | — | — | 0 of 21 |
| 5 | 12 | 6.0% | 2–30 | 21 of 21 |
| 6 | 80 | 40.0% | 50–100 | 21 of 21 |
| 7 | 150 | 75.0% | 120–180 | 21 of 21 |
| 9 | not reached within 180 | — | — | 0 of 21 |

Threshold for 'repaired' is agreement ≥ 0.95 on unanchored tasks, over a corpus of 200. The median is taken over 21 anchor draws with draws that never repair ordered last, so a composition most draws fail to repair reports no number rather than the sweep's upper edge.
<!-- END GENERATED: authority-price -->

At a bare majority an authority ruling on a **median of twelve items in two hundred**
restores the estimate on the **93** that remain scorable. One analyst further and the
same repair costs **two fifths of the round**; two further, three quarters.

!!! danger "Retraction: these prices are one corpus, and it is a cheap one"
    Finding 26 swept the corpus these anchors are drawn from. The twelve above is the
    second-cheapest of eight draws: the bare-majority price spans a **median of 5 to 80**
    audited items, six of nine **80 to 120**, and seven of nine **120 to 180** — and at
    seven of nine one draw prices nothing at all, repairing in only 5 of its 21 anchor
    draws. The 21-draw median quoted here is a robustness check over *anchors*, which
    reads like one over the experiment and is not: the corpus was the dimension nobody
    varied.

    What survives untouched is the negative, which is the half this finding is quoted on:
    **no budget on the ladder repairs unanimity, in any draw.** Quote the range for a
    price and the invariant for the bound.

The
mechanism is visible in the M step: an anchored task constrains every contributor's
confusion matrix, but the unanchored majority still outvotes it, so the anchors have to
reach a share that dominates the estimate rather than merely inform it.

!!! danger "A partial budget is briefly worse than none"
    The curve is not monotone. At 6 of 9 wrong, 80 anchors scores **0.633** against
    **0.660** with no anchors at all. A programme that funds an audit at a fraction of
    what the crossing requires does not buy a fraction of the benefit; it buys slightly
    less than nothing until it clears the threshold. This is the practical warning in
    the finding, and it is the reason to report the threshold rather than a rate.

!!! warning "Retracted: this finding published a single draw, and the price moved"
    Earlier versions of this section reported one number per composition, read off one
    anchor draw: 5 items at a bare majority, 100 at six wrong, 150 at seven, and **180
    at unanimity**. Those were not reproducible. Making the draw *nested* across
    budgets --- so that a larger budget audits a superset rather than a fresh sample,
    which is the property the targeted policies in
    [finding 20](#20-audit-where-the-fleet-splits-and-the-prediction-that-said-otherwise)
    have by construction --- changed which items each seed picks and moved the
    bare-majority price from 5 to 30, with no change to the method being priced.

    A quantity that mobile under a re-draw is not a price, so the sweep now runs 21
    draws and the table reports the median with its range. The bare-majority price
    ranges **2 to 30 items across draws**, a factor of fifteen, and reporting any single
    one of those as *the* answer overstates the precision by that much. The spread is
    the finding the point estimate was hiding.

    The unanimity row's 180 does not survive at all: across all 21 draws, **no budget up
    to 180 of 200 repairs the unanimous fleet**. The earlier 1.000 came from one draw
    clearing the bar on the twenty tasks left unanchored, which `validity.small_n`
    flags. The corrected result agrees with what the design predicted, and the
    prediction is now stated as measured rather than as an aside.

!!! note "What the unanimity row is and is not"
    At 9 of 9 the fleet is unanimous, so there is no disagreement to estimate from and
    the estimator has nothing to work with. No affordable budget repairs it, and that is
    the point of carrying the row: it shows where the mechanism stops being a mechanism.
    An authority that has to rule on nine items in ten has not been assisted by a fleet
    even when the remaining tenth comes out right, so the row is a control rather than a
    result.

    Anchors are drawn uniformly and without regard to difficulty. An authority that
    audited the *hardest* items would score better and would be assuming the question:
    knowing which items are hard is knowing where the fleet is wrong, which is what the
    estimate was meant to establish. Uniform is the honest floor; a targeted policy can
    only beat it. The draw is nested across budgets so that a column of the sweep moves
    the budget alone: with a fresh sample per budget, two adjacent cells differ in both
    how many items were audited and which, and a threshold read off that comparison
    cannot say which of the two it is measuring.

## 20. Audit where the fleet splits, and the prediction that said otherwise

`scripts/measure_audit_policy.py`, `results/audit_policy.json`

[Finding 19](#19-an-authority-of-record-repairs-the-cliff-and-its-price-explodes) drew
its anchors uniformly and said so as a limitation: an authority auditing the *hardest*
items would be assuming the question, because knowing which items are hard is knowing
where the fleet is wrong. Uniform is an honest floor and a poor proposal. This prices
what a selection policy buys on top of it.

**A policy may read only what the aggregator can see** under
[finding 18](#18-the-estimate-moves-under-secure-aggregation-and-the-cliff-does-not-move-with-it)'s
protocol: per-task vote sums, per-task contributor counts, and the estimator's own
posterior. It may not read a per-analyst stream, because there is none, and it may not
read ground truth, because then it is an oracle. The oracle is measured anyway, apart
from the rest, so that a policy scoring near it is known to be near the ceiling rather
than merely better than the floor.

<!-- BEGIN GENERATED: audit-policy -->
| Wrong of 9 | `uniform` | `margin` | `posterior` | `consensus` | `oracle` † |
| --- | --- | --- | --- | --- | --- |
| 5 | 8 (2–20) | **2** | 2 | 80 | 2 |
| 6 | 45 (30–60) | **20** | 20 | 80 | 20 |
| 7 | 80 (60–80) | **30** | 30 | 95 | 30 |

Items an authority must rule on to repair the estimate, out of **97 auditable** tasks (200 in the corpus). Lower is better; bold is the winning deployable policy. † `oracle` reads ground truth and is a bound rather than a method.

`uniform` is a draw, not a rule: its cell is the median over 21 draws with the full range in brackets. The other policies select from the aggregate and have no draw to vary.
<!-- END GENERATED: audit-policy -->

**Uncertainty sampling wins, and it ties the oracle exactly.** Auditing the tasks the
fleet splits on repairs six-of-nine at **20** items against uniform's median 45, and
seven-of-nine at **30** against 80. Scoring against the estimator's posterior instead
of the raw votes gives the identical answer at every cell.

!!! note "How much of that margin is the policy, and how much is the draw"
    `margin` is deterministic given the aggregate; `uniform` is a sample, so the
    comparison is only as good as the baseline's spread. Over 21 draws it is wide, and
    the margin does not survive everywhere:

    - At **five** wrong, uniform's *best* draw needs **2** items --- exactly what
      `margin` needs. The bare-majority advantage is a median effect, not a guarantee,
      and a single lucky audit would have shown no advantage at all.
    - At **six** and **seven** wrong it holds against every draw: uniform's best is 30
      and 60 against `margin`'s 20 and 30, so no draw of the baseline reaches the
      targeted policy.

    The claim that survives is therefore the one about the compositions where the price
    had exploded, which is also where it matters. Reporting the baseline as a point
    estimate would have made all three look alike.

!!! danger "The prediction this script was written to test was wrong"
    It predicted that uncertainty sampling would **lose**, and lose to the policy that
    inverts it. The reasoning: the failure this whole line of work is about is a wrong
    standard held *confidently* by a majority, so the corrupted items ought to be the
    ones the fleet agrees on, and a budget spent where the fleet disagrees ought to be
    spent where the fleet is already right.

    That conflated two different things. A wrong majority means the votes on a
    corrupted item break the wrong way; it does not mean the item is unanimous. The
    mistaken reviewers here differ from the correct ones by one escalation threshold,
    so they diverge only on **boundary** items --- and a boundary item is exactly an
    item the fleet splits on. Disagreement is not orthogonal to the failure. It is the
    failure's signature.

    **`consensus` is much worse --- it needs 80 items where `margin` needs 20.**

    !!! danger "Retracted 2026-08-05: it is not *harmful*, only useless"
        This paragraph used to claim that a badly chosen audit "corrupts the estimate
        it was meant to repair", and read the fall from 0.660 to 0.108 as evidence of
        it. An independent review checked label by label: `consensus` changes **zero**
        unanchored labels at budgets 2 through 60, and at 80 and 95 every change it
        makes is a *correction*. The fall is exactly $(64-b)/(97-b)$ --- pure
        denominator, the same scoring artifact retracted in finding 21.

        The stated mechanism (that anchoring agreed items makes the estimator trust the
        wrong majority harder) did not happen; the estimate is untouched. A badly
        chosen audit wastes the budget and does nothing else, which is a duller claim
        and the one the data supports.

!!! warning "Why it ties the oracle, and what that costs the claim"
    `margin`'s selection is a **subset of the items the fleet gets wrong** at every
    budget that fits inside that set: the estimator gets **33** of the auditable 97
    wrong at zero anchors, and up to a budget of 33 every item `margin` picks is one of
    them --- 20 of 20 at a budget of 20, 30 of 30 at 30. Above 33 neither `margin` nor
    the oracle can keep the property, because there are only 33 wrong items to pick;
    both saturate there. That is why it matches the oracle exactly rather than
    approaching it, and it is a property of a corpus whose difficulty structure is
    discrete and known, where the hard items and the reviewer's blind spot coincide by
    construction.
    [Finding 17](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst)
    already names that coincidence and reports what it costs a different estimator.

    So the claim that travels is conditional: **when a wrong standard manifests as
    boundary disagreement, audit where the fleet splits.** The claim that does *not*
    travel is that disagreement sampling is optimal in general. Where a wrong standard
    is genuinely unanimous, the signal this policy reads does not exist, and nothing
    here says what to do instead.

### Half of finding 19's budget was unspendable

Only **97 of 200** tasks are ever observed by the aggregator at these compositions. The
rest were rejected by every reviewer, so no contributor's confusion matrix touches them
and an anchor placed there constrains nothing. Finding 19 drew uniformly from all 200,
which means roughly half its budget went to tasks that could not repay it.

Restricting the same uniform draw to the auditable pool is by itself a large
improvement --- six-of-nine falls from 100 items to 45 --- before any policy is applied.
Read finding 19's "50% of the round" as 50% of the corpus and **47.4%** of what was
actually auditable -- about half either way.

### A fallible authority

Finding 19 gave the authority perfect ground truth. Chew and Williams
([arXiv:2607.15455](https://arxiv.org/abs/2607.15455), July 2026) build their method on
the opposite assumption, that audit labels are themselves noisy and that auditor
disagreement reflects genuine ambiguity rather than only random error; they sample the
audit set by probability rather than selecting it. Sweeping the authority's own error
rate under the winning policy:

| Wrong of 9 | 0% | 5% | 10% | 20% |
| --- | --- | --- | --- | --- |
| 5 | 2 | 2 | 2 | 2 |
| 6 | 20 | 20 | 20 | 30 |
| 7 | 30 | 30 | 30 | 30 |

**The threshold is robust to an imperfect authority.** An auditor wrong one time in ten
buys the same repair as a perfect one at every composition; only at one error in five
does six-of-nine slip from 20 items to 30. That is the reassuring direction, and it is
worth stating precisely why: the anchors are exogenous, so what they contribute is
information the fleet does not already contain, and a fifth of that being wrong still
leaves four fifths pulling against a majority that was uniformly wrong.

## 21. The corpus the audit policy cannot handle, built on purpose

`scripts/measure_blind_spot.py`, `results/blind_spot.json`

[Finding 20](#20-audit-where-the-fleet-splits-and-the-prediction-that-said-otherwise)
carried a scope condition: its policy ties the oracle *because* every wrong standard
measured to that point was a shifted escalation threshold, which keys on how much
evidence a task shows. Two reviewers differing on that axis disagree exactly on the
boundary items, so "the fleet is split" and "the fleet is wrong" were the same set by
construction. That is a property of the generator, not of the world. The honest way to
find out what the policy is worth is to remove it.

**A wrong standard of a different shape.** A reviewer who discounts a *channel* rather
than misjudging a quantity: they read every report and decline to credit the ones
arriving through one compartment. Nothing about them is noisy or inconsistent. PARTNER
is the channel, and the script now *refuses to run* on a channel too entangled with
difficulty --- mean evidence 1.88 on tasks carrying PARTNER against 1.72 on tasks that
do not, where SENSOR is 2.00 against 0.48 and is rejected outright.

!!! warning "Corrected 2026-08-05: the mechanism is anti-correlation, not orthogonality"
    This finding originally claimed the blind spot picks its slice "by provenance
    instead of by difficulty". That is wrong, and structurally so. Blinding only ever
    *removes* evidence and the rule needs 3 of 3, so a verdict can flip only on a task
    whose visible evidence was exactly 3 --- meaning the affected slice sits at
    $3.00$ mean evidence **for every compartment**, PARTNER and SENSOR alike. It is a
    difficulty stratum, and the easiest one.

    The old guard compared that affected mean to the corpus mean with a slack of 1.5
    against a gap that is always 1.25, so it could not fail, and passed for the very
    channel the text singled out as unusable. Prose calling it "asserted rather than
    trusted" described code that did not exist.

    What actually makes the experiment work is **anti-correlation**: the corrupted
    slice sits at the *opposite* difficulty extreme from the boundary items a threshold
    error hits, which is exactly why the fleet is otherwise unanimous there. That is
    both true and the stronger argument. The guard now tests the statistic that
    discriminates --- evidence on tasks carrying the channel against tasks that do not
    --- and rejects SENSOR, LIAISON and LEGAL while passing PARTNER.

A fleet-wide PARTNER blind spot changes **20 verdicts of 200, every one on a task
showing all three defining facts.** The corrupted slice is the *unambiguous* end of the
corpus --- precisely where a threshold-shifted reviewer is always right, and where a
fleet is otherwise unanimous and correct.

!!! danger "Retracted 2026-08-05: the repair table below measured nothing being repaired"
    An independent review reproduced this artifact and checked it label by label.
    Agreement here is scored over *unanchored* tasks, which means anchoring a task the
    estimator gets **wrong** removes an error from the denominator and raises the score
    **without correcting anything**. Every "budget to repair" cell in the first published
    version of this finding was that artifact: at 7-of-9 blind with `margin` at 12
    items, the reported $0.9574$ is $180/188$ with **8 of the 20 corrupted labels still
    wrong**. Not one unanchored corrupted label is corrected here by any policy at any
    share.

    `repaired` now requires a label to have actually changed (`corrected > 0`), and
    under that definition **nothing in this finding repairs at all** --- the threshold
    table is dashes end to end, which is the honest result. The claim that the oracle
    "still repairs the estimate at 12 items, so the tasks remain fixable" is
    **withdrawn**.

    The finding is unaffected, because it never rested on that table. The hit-rate
    collapse below is the result, and it is measured directly.

### The advantage does not degrade. It disappears at unanimity.

<!-- BEGIN GENERATED: blind-spot -->
| Blind of 9 | `uniform` | `margin` | `posterior` | `consensus` | `channel` | `oracle` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | -- | -- | -- | -- | -- | -- |
| 3 | -- | -- | -- | -- | -- | -- |
| 5 | -- | -- | -- | -- | -- | -- |
| 7 | -- | -- | -- | -- | -- | -- |
| 8 | -- | -- | -- | -- | -- | -- |
| 9 | -- | -- | -- | -- | -- | -- |

Audited items needed to repair; `--` is not reached within 95. Below: the share of a 20-item audit landing on a genuinely corrupted task.

| Blind of 9 | `uniform` | `margin` | `posterior` | `consensus` | `channel` | `oracle` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.10 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 8 | 0.10 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 9 | 0.10 | 0.15 | 0.20 | 0.15 | 1.00 | 1.00 |
<!-- END GENERATED: blind-spot -->

While a sighted minority remains, at seven and eight of nine, the corrupted tasks still
draw dissent and `margin` still finds them: **every one of its twenty picks lands on a
corrupted task**, and it repairs where uniform never does inside the sweep. At nine of
nine the signal does not weaken, it ceases to exist --- there is nobody left to
disagree --- and `margin`'s hit rate falls to **0.15**, next to uniform's 0.10. Chance.

**The oracle is unaffected in the only respect that is measured here: it still selects
corrupted tasks at 1.00 while every deployable policy falls to chance.** That is what
makes this a *policy* failure rather than a hard case --- the corrupted items remain
perfectly identifiable to something that can see the answer, and become invisible to
everything that cannot. What is gone is the observable, not the tasks.

Whether those tasks are *fixable* by anchoring is not something this experiment shows,
and an earlier version of this paragraph claimed it was. See the retraction above.

!!! note "The prediction was half right, and the half it got wrong matters"
    The script predicted the advantage would be lost **and then invert** --- that
    uniform would start beating the targeted policy. The advantage is lost, abruptly and
    exactly at unanimity as predicted. It does not invert: at nine of nine both policies
    fail, and `margin` sits a hair above uniform rather than below it.

    "Both failed" and "the targeted policy did worse" are different claims, and the
    first version of this script reported them with one boolean that called the former
    an inversion. The artifact now records a three-valued verdict. That is the same
    two-valued-API failure this repository has already had decide a finding once.

!!! danger "What this costs finding 20, stated plainly"
    Finding 20's advice --- audit where the fleet splits --- is **safe only inside its
    stated condition**, and this is what falling outside it looks like: not graceful
    degradation, but a policy that scores at chance while an oracle on the same data
    scores perfectly. A deployment cannot tell from the inside which regime it is in,
    because the observable that would distinguish them is the one that has vanished.

    The transferable version is therefore a *pair*: audit where the fleet splits, and
    separately maintain some source of exogenous ground truth that does not depend on
    the fleet disagreeing with itself. A shared blind spot in a fleet trained by one
    pipeline is not an exotic hypothetical --- it is
    [finding 16](#16-the-cliff-is-safe-only-because-the-fleets-were-drawn-independently)'s
    house-style correlation taken to its limit.

!!! warning "A confound found while building this, and worth naming"
    The first run measured nothing at all, and reported success while doing it. Blinding
    a compartment selects tasks *carrying* that compartment, and carrying a compartment
    is what makes a task unreleasable --- so under the fail-closed default every affected
    task was escalated on disclosure grounds, contributed no verdict, and the blind spot
    reached the aggregator on exactly zero tasks. The blind spot was perfectly correlated
    with the release gate.

    Running the fleet under
    [finding 2](#2-the-design-is-bimodal-on-one-policy-ruling)'s compartment-shedding
    ruling puts all 200 tasks in the stream and all 20 corrupted verdicts with them.
    That two independent mechanisms keyed on the same field is exactly the kind of thing
    a generated benchmark hides until something is measured against it.

## 22. The trace a blind spot leaves after it stops leaving disagreement

`scripts/measure_channel_bias.py`, `results/channel_bias.json`

[Finding 21](#21-the-corpus-the-audit-policy-cannot-handle-built-on-purpose) ended
badly and left a specific question. A fleet unanimously sharing a channel blind spot is
corrupted on a slice of the corpus, every deployable audit policy falls to chance on it,
and an oracle on the same data still finds every affected item --- so the tasks are
identifiable in principle and invisible in practice. **A deployment cannot tell which
regime it is in.**

That conclusion had a hidden premise. Every signal used to that point was built from
*disagreement*, and unanimity is the absence of disagreement. But disagreement is not
the only observable. The fleet's verdict **rate** is one too, and it can be conditioned
on public structure rather than on who said what.

**The statistic is a conditional independence test.** For an unbiased fleet, whether a
task happens to carry a particular channel should say nothing about the verdict once you
already know how much evidence the task shows:

$$V \perp C \mid E$$

with $V$ the fleet's significant-rate, $C$ whether the task carries the channel, and $E$
the count of defining facts visible. A shared blind spot breaks exactly that: at a fixed
evidence level, tasks whose evidence arrives through the discounted channel are called
routine more often. **Conditioning on $E$ is what makes it a test rather than a
correlation** --- channels are not spread evenly across difficulty here, and an
unconditioned comparison would report an effect for any channel, which is the confound
finding 21 had to retract a claim over.

Significance is a within-stratum permutation null, matching the gate's idiom: channel
labels are shuffled *inside* each evidence level, preserving the difficulty distribution
and destroying only the association under test.

<!-- BEGIN GENERATED: channel-bias -->
**Verdict noise 0.00**

| Blind of 9 | `SENSOR` | `LIAISON` | `LEGAL` | `PARTNER` ‡ |
| --- | --- | --- | --- | --- |
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 1.0000 | 1.0000 | 0.0545 | **0.0002** (-0.028) |
| 2 | 1.0000 | 1.0000 | 0.0545 | **0.0002** (-0.056) |
| 3 | 1.0000 | 1.0000 | 0.0545 | **0.0002** (-0.083) |
| 4 | 1.0000 | 1.0000 | 0.0545 | **0.0002** (-0.111) |
| 5 | 1.0000 | 1.0000 | 0.0545 | **0.0002** (-0.139) |
| 7 | 1.0000 | 1.0000 | 0.0545 | **0.0002** (-0.194) |
| 9 | 1.0000 | 1.0000 | 0.0545 | **0.0002** (-0.250) |
| threshold control | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Verdict noise 0.05**

| Blind of 9 | `SENSOR` | `LIAISON` | `LEGAL` | `PARTNER` ‡ |
| --- | --- | --- | --- | --- |
| 0 | 0.4070 | 0.8110 | 0.7980 | 0.0643 |
| 1 | 0.7020 | 0.9540 | 0.5980 | 0.0012 |
| 2 | 0.8110 | 0.9810 | 0.3150 | **0.0002** (-0.077) |
| 3 | 0.8670 | 0.9190 | 0.2120 | **0.0002** (-0.105) |
| 4 | 0.9330 | 0.8780 | 0.2020 | **0.0002** (-0.129) |
| 5 | 0.9380 | 0.9500 | 0.0459 | **0.0002** (-0.145) |
| 7 | 0.8590 | 0.9500 | 0.0057 | **0.0002** (-0.181) |
| 9 | 0.8840 | 0.9330 | 0.0119 | **0.0002** (-0.242) |
| threshold control | 0.9070 | 0.4860 | 0.4600 | 0.1040 |

**Verdict noise 0.15**

| Blind of 9 | `SENSOR` | `LIAISON` | `LEGAL` | `PARTNER` ‡ |
| --- | --- | --- | --- | --- |
| 0 | 0.1120 | 0.7570 | 0.6450 | 0.3720 |
| 1 | 0.2610 | 0.5120 | 0.6640 | 0.1860 |
| 2 | 0.4500 | 0.4520 | 0.5690 | 0.0217 |
| 3 | 0.7050 | 0.3790 | 0.5370 | 0.0017 |
| 4 | 0.8260 | 0.2800 | 0.6250 | **0.0002** (-0.102) |
| 5 | 0.8610 | 0.5320 | 0.1900 | **0.0002** (-0.137) |
| 7 | 0.8390 | 0.7090 | 0.1810 | **0.0002** (-0.166) |
| 9 | 0.9080 | 0.6340 | 0.1360 | **0.0002** (-0.192) |
| threshold control | 0.8590 | 0.4110 | 0.6920 | 0.2810 |

One-sided permutation p-values against a within-stratum null over 4200 permutations, computed as (b+1)/(m+1) so the floor is 2.4e-04 and never zero. Bold is a detection at p ≤ 0.001, with the stratified gap beside it. ‡ is the channel the fleet actually discounts.

Read the **gap** for extent and the p-value for detection. The gap is linear in the blind share; the p-value saturates at its floor once an effect is comfortably significant and cannot distinguish shares above that point. p = 1.0000 on a noiseless fleet is not a missing value: with every analyst deterministic, each task's rate is fixed by its evidence stratum, every permutation returns the observed gap, and b = m exactly.
<!-- END GENERATED: channel-bias -->

**It survives unanimity.** At nine of nine blind --- precisely where finding 21's
policies score at chance --- the discounted channel sits at the smallest p-value 4200
permutations can produce, at every noise level measured, while no other channel and no
healthy fleet is detected anywhere. The statistic reads the *level* of the fleet's
verdict rather than its spread, so the disappearance of dissent costs it nothing. That
is the finding, and it holds.

!!! warning "Corrected 2026-08-06: this was a z-score, and should have been a p-value"
    A permutation test exists so the null's shape need not be assumed. This one then
    standardized against that null's mean and standard deviation and compared the
    result to three sigma, which puts the normality assumption back in the last step.
    It was wrong in three ways at once:

    - **It was undefined exactly where the controls live.** A noiseless fleet gives
      every permutation the same gap, so the null's standard deviation is zero and z
      is a division by it. The code turned that into `0.0`, and both negative controls
      "passed" from that division rather than from evidence. The finding described one
      of them as the load-bearing control that could have voided the result. It could
      not have.
    - **It invited a fix in kind.** The first repair drew the null 21 times and reported
      the median z, spending 4200 permutations to estimate a quantity that one pooled
      null of 4200 estimates better.
    - **It claimed unearned precision.** A z of 8 asserts a tail probability far beyond
      what any finite number of shuffles can evidence.

    The measurement now reports a one-sided permutation p-value,
    $p = (b + 1)/(m + 1)$, following Phipson and Smyth (2010): the permuted draws *are*
    an exact discrete null distribution rather than an estimate of a tail probability,
    so the observed value counts as one of its own draws. The naive $b/m$ understates
    by about $1/m$ and can report zero, which no finite number of permutations
    supports.

    Everything the old special-casing existed for dissolves. A noiseless control has
    every draw at least as extreme, so $b = m$ and $p = 1.0$ --- the correct answer, by
    construction, with no `n/a` and no refusal logic. The invariance across shares is
    gone too, and was never the right thing to report.

!!! info "Detection and extent are different numbers, and now say so"
    A p-value saturates. Once an effect is comfortably significant it sits on the floor
    $1/(m+1)$ --- here $2.4 \times 10^{-4}$ --- and stays there however much larger the
    effect grows, so equal p-values across shares say nothing about the shares. The
    **stratified gap** is the extent measure and always was: it is linear in the blind
    share, running $-0.028$ at one blind analyst to $-0.250$ at nine, in the ratio
    $1:9$.

    So the table reports the gap beside the p-value wherever a channel is detected.
    Detection reads the p-value, extent reads the gap, and neither is asked to do the
    other's job. The claim previously made here --- that a statistic which cannot vary
    with the share cannot report the share --- was true of the z-score and is not a
    property of the measurement.

**Where the floor actually sits.** At $\alpha = 0.001$, matching the one-sided tail of
the three-sigma convention the gate uses elsewhere:

| Verdict noise | Detected from | Not detected at |
| --- | --- | --- |
| 0.00 (idealized) | 1 of 9 | --- |
| 0.05 | 2 of 9 | 1 of 9 ($p = 0.0012$) |
| 0.15 (`inattentive`) | 4 of 9 | 3 of 9 ($p = 0.0017$) |

The noiseless row is the one this finding was first reported on, and it is a degenerate
fleet rather than a clean one. On a fleet that makes mistakes at the rate this project
already treats as realistic, the detector needs **roughly a third to a half** of the
fleet --- still well short of the unanimity that defeats every audit policy in
[finding 20](#20-audit-where-the-fleet-splits-and-the-prediction-that-said-otherwise), which
is the comparison the finding exists to make.

Note how narrowly the two borderline rows miss: $0.0012$ and $0.0017$ against a
threshold of $0.001$. A less strict $\alpha$ would move the floor down a row in both
cases. The threshold is held at the gate's own convention rather than chosen to make
the finding read better, which is the only defensible way to pick one.

**What this does not claim.** It detects a blind spot aligned with a *known, public*
partition of the corpus --- here the compartment a report arrived through. A shared error
that follows no observable partition would leave no trace in this statistic either, and
nothing here says how to find one. What has changed is narrower and still worth having:
the specific regime finding 21 left as undetectable is detectable, cheaply, from data the
aggregator already holds.

## 23. Once the detector names the channel, provenance finds the corrupted items that are findable at all

`scripts/measure_blind_spot.py`, `scripts/measure_audit_policy.py`,
`results/blind_spot.json`

[Finding 21](#21-the-corpus-the-audit-policy-cannot-handle-built-on-purpose) ends with
every deployable audit policy at chance on a unanimously blind fleet, because all of
them read *disagreement* and unanimity is its absence.
[Finding 22](#22-the-trace-a-blind-spot-leaves-after-it-stops-leaving-disagreement)
supplies the input that was missing: it says **which channel** is being discounted,
from the per-task sums the aggregator already holds. This wires the second into the
first.

**The policy.** Select tasks that carry the named channel, deepest evidence first. It
reads two things, both public corpus structure and both already read by finding 22's
detector: which tasks carry the channel, and how many defining facts each shows. It
never touches ground truth or the per-analyst stream, so it stays inside finding 20's
deployability rule.

Evidence depth is not decoration. Carrying the channel is necessary and not sufficient
--- the first version of this policy selected on provenance alone and landed **0.50**
of its audit on corrupted tasks, against uniform's 0.10. A blind analyst only flips a
verdict where the discounted channel was doing the work, which in this corpus is the
high-evidence end: the affected slice sits at 3.00 defining facts against a corpus mean
of 1.75.

**At the unanimity where every other deployable policy is at chance, it ties the
oracle.** Share of a 20-item audit landing on a corrupted task, at nine of nine blind:

| `uniform` | `margin` | `posterior` | `consensus` | `channel` | `oracle` |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 0.15 | 0.20 | 0.15 | **1.00** | 1.00 |

!!! warning "Finding every one of them is not repairing any of them"
    At a 20-item budget the channel policy drives `remaining_errors` to **zero**: every
    corrupted label in the corpus is correct afterwards, and agreement reaches 1.000.
    And `corrected` stays at **zero**, because not one *unanchored* label changed. The
    authority overrode the twenty items it ruled on, and the estimator learned nothing
    from any of them.

    **The oracle does exactly the same thing**, and that is the load-bearing half. An
    obstacle that defeats a policy handed the ground truth is not a selection problem,
    so no better selection rule closes it. What this finding delivers is the *selection*
    half of "what to do once the detector fires": you can now find the corrupted items
    in the regime where finding 21 said you could not. What remains open is whether
    anything can be learned from them, and the evidence here is that in this regime
    nothing can.

    Quoted without that paragraph, this reads as a solution. It is half of one.

!!! danger "Corrected 2026-08-07: the 1.00 below is this corpus, not this policy"
    Every number in this finding is measured on one corpus draw.
    [Finding 26](#26-findings-20-to-23-were-measured-on-one-corpus-and-one-of-their-headlines-was-that-corpus)
    sweeps eight. The channel policy recovers **0.75 to 1.00** across the seven draws that
    can host the experiment, and reaches 1.00 on the committed corpus alone. On five of
    the seven **the oracle itself does not find everything**, so no policy could have.

    The bound-tying claim nearly survives -- six draws of seven -- and at one draw the
    policy scores 0.90 where the oracle finds all of them, so it loses to the bound there
    rather than tying it.

    What survives unchanged is the comparison: every policy reading disagreement sits at
    or below 0.25 in every draw, so provenance's advantage is robust even where its
    absolute score is not. **Quote the range and the contrast, never the 1.00.**

**Scope condition, carried with the number every time.** The policy ties the oracle
*because* the corrupted slice in this corpus is exactly "carries the channel, and shows
deep evidence" by construction --- the same shape of condition
[finding 20](#20-audit-where-the-fleet-splits-and-the-prediction-that-said-otherwise) has to
carry. The claim that travels is conditional: **when a wrong standard follows an
observable partition of the corpus, audit that partition.** A shared error that follows
no observable partition leaves no trace for finding 22 to detect and no handle for this
policy to grip.

## 24. The crossing is a distribution over corpora, not a share, and "a majority" was nine analysts talking

Findings 19 through 23 all describe the same failure in the same words: *a majority
holds the wrong standard*. Every one of them was measured at nine analysts, where the
estimator recovers the truth at 4 and collapses at 5. The majority and the crossing are
the same cell at that fleet, so the phrase was never wrong. It was never tested either.

Sweeping the fleet and scanning **every** composition rather than the five the ladder
visits separates them.

<!-- BEGIN GENERATED: governance-sensitivity -->
| Fleet | Bare majority | Breaking share (median) | Breaking share (range) | Crossing at the majority | Survives a bare majority |
| --- | --- | --- | --- | --- | --- |
| 5 | 3 | 0.600 | 0.600 – 0.600 | 8/8 | 0/8 |
| 9 | 5 | 0.556 | 0.556 – 0.556 | 8/8 | 0/8 |
| 15 | 8 | 0.533 | 0.533 – 0.600 | 6/8 | 2/8 |
| 25 | 13 | 0.560 | 0.520 – 0.600 | 4/8 | 4/8 |

Every composition from 1 to the fleet size, one EM fit each, over 8 corpus draws, with recovery at agreement ≥ 0.95. The last two columns are counts of draws, not verdicts, and that is the finding: the crossing is a distribution spanning **0.520 – 0.600** rather than a constant, and it becomes *less* predictable as the fleet grows. At five and nine analysts every draw breaks at the same composition; at fifteen and twenty-five the same fleet survives a bare majority on some corpora and not others.

Agreement past the crossing takes 8 distinct values across draws (0.5773 – 0.7882). Within a single draw it is constant at every composition above the crossing, so the failure still has no gradient — a fleet is identified or it is not — but the level it falls to is a property of the corpus.
<!-- END GENERATED: governance-sensitivity -->

**The crossing is a distribution, not a constant, and it gets less predictable as the
fleet grows.** At five and nine analysts every one of eight corpus draws breaks at the
same composition. At fifteen and twenty-five they do not agree with each other: the same
fleet size survives a bare majority on some corpora and fails on others, 2 draws in 8 at
fifteen and 4 in 8 at twenty-five. Across the whole sweep the breaking share spans 0.520
to 0.600.

So the phrase every earlier finding used -- *a majority holds the wrong standard* --
names the mechanism exactly at nine analysts and is a coin flip at twenty-five. That is
the part that survives. What does **not** survive is the first version of this finding.

!!! danger "Retraction, 2026-08-07: this finding's own headline was one draw"
    Published a day earlier, finding 24 claimed the crossing sat at a fixed share
    bracketed by `0.533 < s ≤ 0.556`, that fleets of fifteen and twenty-five therefore
    *survive* a bare majority, and that agreement past the crossing is a constant 0.6598.
    All three came from one corpus per fleet, and the committed seed is favourable at
    both larger sizes.

    Over eight draws: the breaking share spans **0.520 – 0.600** and the bracket does not
    hold; a bare majority survives at fifteen in **2 draws of 8** rather than as a
    property; and post-crossing agreement takes **eight distinct values** from 0.5773 to
    0.7882 rather than one.

    This is the third time a single-draw quantity has been published here as a constant,
    after finding 19's anchor prices and finding 5's shot count. The scan now sweeps the
    draw and the artifact reports a median with its range. **A crossing quoted without a
    denominator is not a measurement.**

!!! danger "What this changes about findings 19 and 20"
    The prices those findings publish -- a median of 12 audited items to repair a bare
    majority, 20 at six of nine, 30 at seven -- are prices for **compositions** measured
    at nine analysts, and they stand. What does not stand is reading them as a price for
    *a bare majority* in general, in either direction: a bare majority is not reliably
    broken above nine analysts, and it is not reliably safe there either.

    Quote the composition and the fleet, never the phrase.

**This half was already on disk, and nobody read it back into the prose.**
[Finding 17's fleet sweep](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst)
recorded "Dawid-Skene recovers the truth at a bare majority once the fleet reaches 15"
when it ran, and `fleet_sensitivity.json` has carried `dawid_skene_survives_crossing:
true` for fleets 15, 25 and 51 ever since. Findings 19 through 23 were written after
that and went on calling the failure "a majority holds the wrong standard" anyway.

So the new part here is not the fact. It is that the crossing was scanned at every
composition rather than at the crossing alone, over eight draws rather than one; that its
location is a **distribution** whose spread grows with the fleet; and the observation that
the phrase all five later findings share is load-bearing in none of them. The rest is a
claim this repository had already measured and failed to propagate, which is the same
prose-outlives-its-evidence failure this page has now corrected five times.

**The depth is flat within a draw and not across draws.** Past the crossing, agreement is
the same value at every composition above it -- checked in all 32 fleet-and-draw cells --
so the failure still has no gradient: a fleet is identified or it is not. But *which*
level it falls to is a property of the corpus, ranging 0.5773 to 0.7882. The earlier claim
that the depth is a constant 0.6598 was the same one-draw error as the location.

**Reliability weighting buys margin, and less reliably than the first version claimed.**
Plain consensus flips at the majority by definition, at every fleet size. Dawid-Skene
does not, at nine. Above nine it sometimes does and sometimes does not, which is a weaker
statement than the one this finding first made. Finding 12 measured this at nine and
called it "one extra contributor of margin, not an escape", which remains the right
reading of the fleet it had.

!!! note "It is still not an escape"
    The margin is bounded by the same share. A house style that reaches 60% of a fleet
    defeats the estimator at every size measured, and [finding 16](#16-the-cliff-is-safe-only-because-the-fleets-were-drawn-independently)
    is the reason to expect a house style to reach exactly that far. A larger fleet raises
    the bar; it does not remove it.

**The depth does not move where the location does.** Agreement past the crossing is
0.6598 at every fleet and every composition above it. There is no gradual degradation
between the two levels and no partial failure to find: a fleet is identified or it is
not. That is what a relabelling of the latent class looks like, as opposed to a loss of
signal, and it is the sharpest evidence in the repository for the non-identifiability
reading that findings 18 and 19 argue from.

**Everything else in findings 19-23 survives the sweep.** Six invariants were checked at
four fleet sizes: selection beats a uniform draw wherever a repair is needed, and ties
the oracle bound; disagreement collapses to chance at unanimity; provenance recovers
every corrupted item there; no policy repairs an unanchored label, including the oracle;
and the blinded channel is detected at every noise level while every control stays
silent. All six still hold. What moved is confined to this finding's own claims about
where the crossing sits and how deep it goes.

!!! warning "Two defects the sweep found before it measured anything"
    All four scripts accepted `--fleet` and none of them scaled its *compositions* with
    it, so the flag produced a mislabelled artifact at any fleet but nine. At `--fleet 5`
    the audit policy measured 5-of-5 -- unanimity -- under the label of a bare majority,
    printed the two rows it had skipped as `none` beside it, where `none` elsewhere means
    *swept and never repaired*, and then named a "best deployable policy" chosen from a
    field in which every entry was unmeasured. The blind spot labelled its unanimity row
    "9 of 5". Compositions are now positions in the fleet
    (`measure_authority_anchors.ladder`), asserted to reproduce every committed constant
    exactly at nine.

    And a repair threshold of `None` meant two opposite things -- *no budget repaired it*
    and *there was nothing to repair*. The first run of this sweep read the second as the
    first and reported finding 20 as holding while the crossing row had quietly stopped
    being a failure at all. The two are separated in the artifact now.

A third defect belonged to the sweep itself and is worth recording because it is the
failure mode this whole page is about. A smoke run at 300 permutations reported finding
22 as moved. Nothing had moved: a permutation p-value floors at 1/(m+1), so 300
permutations put the entire attainable range above alpha and **no cell could be detected
at any effect size**. The sweep had manufactured a failure to replicate out of its own
argument. It now refuses to run below a permutation count that can clear alpha.

## 25. The cliff is not where we pointed the estimator, except at the crossing itself

Every finding from 12 onward rests on Dawid-Skene reporting the wrong standard as the
truth once enough of the fleet holds it. And `inference.py` has said in its own docstring,
since the day it was written, that EM started from the majority vote "is drawn toward
whatever the majority believes."

That is a researcher degree of freedom sitting directly underneath the headline result,
and it is the first thing a reviewer reaches for. The Dawid-Skene log-likelihood is
non-convex; majority-vote initialisation carries no global-optimality guarantee. Zhang,
Chen, Zhou and Jordan ([JMLR 17, 2016](https://jmlr.org/papers/volume17/14-511/14-511.pdf))
initialise from a spectral method of moments and prove optimal convergence for the
two-stage estimator precisely because the conventional start does not have that property;
the plainer remedy named in the same literature is many random restarts scored by
likelihood. Neither had ever been tried here.

So the question is sharp, and it has opposite answers: **is the wrong answer the
maximum-likelihood fit, or merely the basin the conventional start falls into?** A local
optimum is escapable, and findings 12 and 17 through 24 would shrink to statements about
one initialiser. A global optimum is not escapable by any initialiser at all.

<!-- BEGIN GENERATED: estimator-initialization -->
| Fleet | Wrong | Draws broken | Draws with an escape | Restarts recovering | Median log-likelihood gap |
| --- | --- | --- | --- | --- | --- |
| 9 | 5 *(crossing)* | 8/8 | 2 | 0.109 | -18.7 |
| 9 | 6 | 8/8 | 0 | 0.008 | -109.5 |
| 9 | 7 | 8/8 | 0 | 0.000 | +0.0 |
| 9 | 8 | 8/8 | 0 | 0.000 | -0.0 |
| 9 | 9 | 8/8 | 0 | 0.000 | +0.0 |
| 15 | 8 *(crossing)* | 6/8 | 1 | 0.193 | -32.0 |
| 15 | 9 | 8/8 | 0 | 0.082 | -85.9 |
| 15 | 10 | 8/8 | 0 | 0.027 | -177.3 |
| 15 | 11 | 8/8 | 0 | 0.004 | -268.2 |
| 15 | 12 | 8/8 | 0 | 0.000 | +0.0 |
| 15 | 13 | 8/8 | 0 | 0.000 | +0.0 |
| 15 | 14 | 8/8 | 0 | 0.000 | +0.0 |
| 15 | 15 | 8/8 | 0 | 0.000 | +0.0 |

Every composition the published start gets wrong, over 8 corpus draws and 32 random restarts each, plus an uninformative start, an adversarial one, and the ground truth. An *escape* is a start that both fits strictly better and recovers the truth.

The gap is signed and the sign is the finding: it is the truth's log-likelihood minus the published answer's, so **negative means the wrong answer is the better fit**. It is negative everywhere past the crossing and grows with the share, which is why no likelihood-guided initialisation helps there. Exactly zero means the truth is not a fixed point at all — seeded there, EM leaves.
<!-- END GENERATED: estimator-initialization -->

**The escape exists, and it is confined to the crossing composition.** At the cell where
the cliff begins, a better start sometimes recovers the truth: 2 draws in 8 at nine
analysts, 1 in 6 at fifteen. Roughly a tenth to a fifth of random restarts find it when
it is there. So at exactly one composition per fleet, the published number reports a
local optimum rather than the estimator's preference, and that cell now carries the
caveat.

**Past the crossing, no start escapes, and the artifact says why.** The gap between the
truth's likelihood and the published answer's runs to tens and then hundreds of nats
against the truth. At six of nine it is about 110 nats; at eleven of fifteen, 268. The
truth is still *reachable* there -- seeded at it, EM stays -- but it is the strictly worse
fit, so **selecting by likelihood rejects it**. That is the sharpest form of the result:
a better initialiser cannot rescue a search whose objective prefers the wrong answer.

!!! note "Why the oracle start is a diagnostic and not a method"
    One of the starts swept is the ground truth itself. An estimator that needs the
    answer to find the answer is useless, and it is not proposed as anything else. It is
    the only way to separate "EM prefers the wrong answer" from "EM was aimed at it",
    because it is the one start guaranteed to be inside the truth's basin if a basin
    exists. Where EM seeded at the truth walks away from it -- seven of nine and above --
    the truth is not a fixed point of the likelihood at all, and the question of finding
    it with a better initialiser does not arise.

**Random restarts are the wrong fix even where an escape exists.** Reported as a rate
rather than a boolean, because one restart in thirty-two and thirty in thirty-two are the
same yes and different methods. Where the escape exists, restarts find it around 11-19% of
the time; where it does not, they land on the inverted labelling instead -- agreement near
0.35, which is label switching, the failure the identifiability machinery in the
class-conditional model was added to prevent and
[finding 17's note](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst)
already records there.

!!! warning "What this does and does not license"
    It licenses keeping findings 12 and 17 through 24 as stated, with one added scope
    condition: **at the crossing composition itself, the reported failure is
    initialisation-dependent on some corpora.** Everywhere past it the failure is a
    property of the likelihood.

    It does not license the claim that no initialiser anywhere could do better. Thirty-two
    restarts bound how large an unfound basin could be -- a basin covering a tenth of the
    space is missed with probability 0.034 -- rather than proving none exists. A spectral
    initialisation of the kind Zhang et al. prove optimal is not implemented here, and is
    the obvious next thing to build.

This is the objection that would have been asked first and answered worst. It cost one
argument on an existing function and one script; it should have been run before finding
12 was published, not thirteen findings later.

## 26. Findings 20 to 23 were measured on one corpus, and one of their headlines was that corpus

[Finding 24](#24-the-crossing-is-a-distribution-over-corpora-not-a-share-and-a-majority-was-nine-analysts-talking)
established that the committed corpus seed is favourable: at fleets of fifteen and
twenty-five it survives a bare majority where most draws do not. That was measured on the
crossing scan alone. The four scripts behind findings 19 through 23 each hard-coded
`SEED = 7` and **did not accept a seed at all**, so the corpus was not a sweepable
dimension of this project's headline governance results.

It is now. Three defects surfaced before a single policy was scored, all of the same kind
as the `--fleet` defect finding 24 found, and all invisible from the committed artifact.

!!! danger "The script had never run on a corpus that was not seed 7"
    `measure_audit_policy.py`'s budget ladder ended at a hard-coded 95 while the
    *auditable pool* it draws from is a property of the draw. Over eight draws the pool
    ranges **83 to 99**, so the ladder exceeded it and the script exited non-zero on
    **four of eight** draws. The ladder is derived from the pool now and the artifact
    publishes both the requested and the used form.

    That 97 was not only a constant in code. It is quoted in `las-2027/ALIGNMENT.md` as a
    corollary a proposal should carry: *"only 97 of 200 tasks are auditable at all, so
    finding 19's share of the round overstates the real price."* The direction of that
    argument survives; the number does not.

<!-- BEGIN GENERATED: corpus-sensitivity -->
| Claim | Finding | Holds in | Of draws |
| --- | --- | --- | --- |
| `margin` ties the oracle bound at 5 of 9 | 20 | **8** | 8 |
| `margin` ties the oracle bound at 6 of 9 | 20 | **8** | 8 |
| `margin` ties the oracle bound at 7 of 9 | 20 | **6** | 8 |
| `margin` beats a uniform draw at 5 of 9 | 20 | **7** | 8 |
| `margin` beats a uniform draw at 6 of 9 | 20 | **8** | 8 |
| `margin` beats a uniform draw at 7 of 9 | 20 | **6** | 8 |
| Disagreement policies sit at chance at unanimity | 21 | **7** | 7 |
| Provenance ties the oracle bound | 23 | **6** | 7 |
| Provenance finds *every* corrupted item | 23 | **1** | 7 |
| No policy repairs an unanchored label | 23 | **7** | 7 |
| Nothing repairs unanimity, at any budget | 19 | **8** | 8 |
| Confidence abstention fails at unanimity | 28 | **7** | 7 |
| Its textbook inversion fails too | 28 | **7** | 7 |
| Provenance abstention beats every untargeted draw | 28 | **7** | 7 |
| Provenance abstention ties the bound | 28 | **5** | 7 |
| Confidence abstention works on *random* error | 28 | **5** | 7 |
| The shape index is calibrated on a healthy fleet | 29 | **7** | 7 |
| It rises with the shared share | 29 | **7** | 7 |
| It picks the winning rule in every cell | 29 | **3** | 7 |

Finding 29's index picks the rule that wins in **56 of 64** decidable cells across 7 draws. What a wrong call costs is **0.011 to 0.050** of published error rate, and the committed corpus is at the cheap end: quoting its 0.011 as the price of a wrong call would be reporting the draw again.
| Blinded channel detected, controls silent | 22 | **8** | 8 |

Fleet of 9, 8 corpus draws, every denominator stated. Finding 21's experiment needs a blind channel orthogonal to item difficulty and refuses to run where they are entangled, so it is constructible on **7 of 8** draws; a draw that cannot host the negative control says nothing about the finding and is excluded rather than counted against it.

The auditable pool an audit budget is a fraction of ranges **83 to 99**, not the 97 the script documented. Provenance recovers **0.75 to 1.00** of corrupted items against every disagreement-reading policy's 0.25 or less, so the *advantage* is robust even where the 1.00 is not.

Finding 19's anchor prices are a median over 21 *anchor* draws inside one corpus, and the corpus moves them. Across 8 draws, the median number of audited items an authority of record must rule on spans a bare majority **5 to 80**, six of nine **80 to 120**, seven of nine **120 to 180** (priced in 7 of 8 draws). The committed corpus ranks 2 of 8 at a bare majority, 2 of 8 at six, 2 of 7 at seven when the draws are ordered cheapest first, so it is not a worst case and not a typical one. The published single numbers were one draw's, not the price.

At seven of nine, one draw (seed 101) repairs in only **5 of 21** anchor draws, so it has no median at all: on that corpus an authority usually buys nothing within the ladder. That draw is reported here rather than dropped from the range.

What does not move is the negative. **No budget on the ladder repairs unanimity, in any draw** -- which is the claim finding 19 is quoted on outside this repository, and the one the open problem is shaped by.
<!-- END GENERATED: corpus-sensitivity -->

**The bound holds below the crossing, and not above it.** `margin` ties the oracle bound
in **8 of 8** draws at five and six wrong, and in **6 of 8** at seven. A policy tying the
bound means no better selection rule exists on this signal, so the tie is the claim worth
having — and it is now bounded by where the fleet is hardest to repair at all, rather than
stated everywhere.

!!! danger "Retraction: this sweep's own first headline counted a tie that never happened"
    The version of this finding published on 2026-08-07 reported the tie holding in
    **every draw at every composition** and called it the most robust result in the
    governance set. That was a scoring defect in the sweep, not a result. A cell where
    *neither* `margin` nor the oracle repaired at any budget — both thresholds `None` —
    compared equal and was counted as a tie. Seed 202 at seven wrong is such a cell, and
    that single cell was the whole of the invariant.

    Two consequences, both in the table above. The tie at seven wrong is 6 of 8, not
    7 of 7. And the denominator itself was wrong: `nothing_repaired` was derived from the
    policies failing rather than from the corpus, so seed 101 at seven wrong — a fleet
    that *was* broken and that nothing in the ladder repaired — was dropped from the
    denominator instead of counted. It is read off the budget-zero oracle row now, which
    is where `measure_governance_sensitivity` already read it.

    Found by an independent review of the pull request that introduced it, before it was
    merged. It is the same shape as the defect the finding is about: a comparison that
    was never made, reported as one that held.

**Finding 21's row was vacuous, and now is not.** `disagreement_policies_at_chance` read
its policy list from a `deployable` key the blind-spot artifact has never had, so the
comparison ran over an empty list and `all()` returned true without scoring a single
policy. The row read 7 of 7 and measured nothing. Scored against the four deployable
policies by name, it still reads **7 of 7** — the claim was true, and it had not been
tested. Every policy's rate is recorded per draw now, so the prose comparison can be
checked against the artifact rather than taken on trust.

What moves next is finding 23's number.

!!! danger "Retraction: 'provenance finds every corrupted item' is one draw in seven"
    [Finding 23](#23-once-the-detector-names-the-channel-provenance-finds-the-corrupted-items-that-are-findable-at-all)
    reports a hit rate of **1.00** against uniform's 0.10 and margin's 0.15, tying the
    oracle exactly. Across seven draws that can host the experiment, the channel policy
    recovers **0.75, 0.80, 0.80, 0.90, 0.90, 0.75** and — on the committed corpus alone —
    **1.00**.

    The 1.00 is not a property of the policy. On five of those seven draws **the oracle
    itself does not find everything**, so there was nothing for any policy to reach. And
    at one draw the channel policy scores 0.90 where the oracle finds all of them, so it
    **loses to the bound** rather than tying it: the tie holds in six of seven, not seven
    of seven.

    Two claims were being reported as one. *Ties the best any selection rule could do* is
    the result and nearly survives. *Finds every corrupted item* was the corpus.

**The advantage is robust, and it is the part to quote.** Provenance recovers 0.75 to 1.00
of corrupted items in every draw, while every policy that reads disagreement sits at or
below 0.25 in every draw. So the comparison finding 23 rests on holds everywhere it was
tested; only the absolute value moved. Quote the range and the contrast, never the 1.00.

**The negative half is the most robust thing here.** *No policy repairs an unanchored
label* holds in **7 of 7** draws, for the channel policy and the oracle alike. The open
question finding 23 leaves — whether an audited fleet can be made to generalise from what
the authority ruled on — is unaffected by any of this, which is worth stating because it
is the twelve-month deliverable.

!!! note "Finding 21's experiment cannot be built on every corpus, and that is not a failure"
    The blind-spot corpus needs a channel orthogonal to item difficulty, and
    `measure_blind_spot.py` refuses to run when the two are entangled — at seed 23 the
    `PARTNER` compartment carries mean evidence 2.12 against 1.61 without it. That refusal
    is the design working: a negative control that is not orthogonal is not a negative
    control.

    So the experiment is constructible on **7 of 8** draws, and every rate above carries
    that denominator. A draw that cannot host the control says nothing about the finding,
    which is a different thing from a draw that contradicts it, and collapsing the two
    would be the censoring error [finding 20's sweep](#20-audit-where-the-fleet-splits-and-the-prediction-that-said-otherwise)
    already made once.

**One more thing that moves, and it is small.** The `best_deployable` policy the artifact
names is `margin` in seven draws and `uniform` in one. A sweep that reports a single
winner from a single corpus is reporting the corpus.

This is the fourth single-draw quantity corrected here, after finding 5's shot count,
finding 19's anchor prices, and finding 24's crossing. The class is guarded rather than
re-noticed now: every one of these scripts takes `--seed`, and the test suite fails if a
new governance script does not.

**The last cell is filled, and it was the most expensive one.** Naming
`measure_authority_anchors` as the unswept dimension took an afternoon to close: the
script costs 26 seconds a draw, so the exclusion that called it too slow to sweep had
been guessing. Finding 19's prices move more than any other quantity here — the
bare-majority price spans a median of **5 to 80** audited items where 12 was published,
and the committed corpus is the second-cheapest draw of eight at every composition that
prices at all. The affordability claim was the optimistic end of a range.

The negative survives, and it is the half that carries the argument: **no budget on the
ladder repairs unanimity, in any of the eight draws**. So an authority of record is
costlier than published and still buys nothing where the fleet agrees, which is the shape
the open problem already had.

**What is pinned is now declared.** The multiverse is itself a researcher degree of
freedom, and a sweep that picks its dimensions after seeing which are kind to the result
is not a robustness check. The artifact carries a `multiverse` block naming what varies
(the corpus seed, eight draws) and what is held fixed with the reason for each: the fleet
size, which belongs to finding 24's sweep and would confound this one; the permutation
count, because a p-value floors at 1/(m+1); and the 21 anchor draws, which are the inner
multiverse this one wraps.

## 27. Four guards could be inverted with the suite still green, and coverage called them all covered

Every finding above rests on code that decides whether a number may be quoted. The
question this project asks of everything else had never been asked of that code: if the
condition were wrong, would anything say so?

Coverage cannot answer it. `validity.py` and `provenance.py` -- the two modules the
manuscript names as instruments, *a validity flag that travels with the number* and the
provenance stamp on every artifact -- were already at 100% line and branch coverage.
Coverage records that a line ran, and a guard runs identically whether or not anything
checks what it decided.

Mutation testing answers it directly: change the condition, and see whether the suite
objects. Scoped to those two modules, 438 mutants ran, 187 died and 251 survived. Most
survivors are string literals and log-message text, which are noise here. **Four are
guards, and all four survived the entire suite as it stood**, verified by applying each
to the real source and running the whole of `tests/` rather than the fast subset:

| mutation | what it breaks |
| --- | --- |
| `total = scored + unparsed` → `scored - unparsed` | the sample size every validity flag is computed against |
| `if total < SMALL_N` → `<=` | whether a measurement at exactly n=30 carries the caveat |
| `if not path` → `if path` | `executable` becomes empty in every artifact |
| `if ".venv" in path` → `not in` | the executable path stops being the stable relative form |

The first is the one worth reading twice. The self-audit already records a case where a
sample size was wired to the number of decode regimes rather than the number of tasks
compared, and its lesson was that a wrong `n` produces a plausible number rather than an
error. The arithmetic behind `n` can still be inverted here without a single test
failing. The instrument that caught that defect had the defect's own shape inside it.

Seven tests now pin these, and each was verified by re-applying its mutant and watching
the suite fail. `measure_guard_mutations.py` keeps the four as a standing check rather
than a one-off: it refuses a dirty tree or a red baseline, applies each mutation to the
real source, runs the whole suite, restores in a `finally` so an interrupt cannot leave a
mutated guard behind, and writes `results/guard_mutations.json`. The artifact records the
verdicts **now**, which is 4 of 4 killed; the table above is what they were before the
tests existed. A future change that stops killing one of them is therefore a visible
regression rather than a rediscovery. `[tool.mutmut]` records the wider scope for anyone
who wants the full 438: `uvx mutmut run`.

!!! warning "The first version of this measurement was wrong, in the way this file is about"
    The harness decided kill-versus-survive by grepping pytest's last two lines for
    `N failed`. Under `-q` the final line is `FAILED tests/...`, so the count line was
    never in the window and **every real failure read as a pass**. It reported four
    false survivals before anything was written down.

    It was caught by disbelief rather than by a check: tests asserting `report.n == 40`
    cannot pass against code that computes 20. The harness now reads pytest's exit
    status, and the numbers above are from the corrected run against a baseline that
    exits 0.

**What is not claimed.** 187/438 is not a mutation score for this project. It covers two
modules out of thirty, and the survivor count is inflated by string mutants nobody should
chase. `gate.py` belongs in scope and is excluded for cost alone -- its tests take 90
seconds, against half a second for the two modules here -- which is a budget, not a
verdict on how well the gate is tested.

## 28. The open problem, answered: detection converts into coverage, not into correction

`scripts/measure_selective_risk.py`, `results/selective_risk.json`

Findings 19 to 21 close off correction at unanimity from three directions -- no budget on
the anchor ladder repairs a label, no disagreement-reading policy beats chance, and
[finding 26](#26-findings-20-to-23-were-measured-on-one-corpus-and-one-of-their-headlines-was-that-corpus)
shows both hold across corpus draws.
[Finding 22](#22-the-trace-a-blind-spot-leaves-after-it-stops-leaving-disagreement) then
detects the regime anyway, and
[finding 23](#23-once-the-detector-names-the-channel-provenance-finds-the-corrupted-items-that-are-findable-at-all)
turns that detection into a selection rule that finds the corrupted items. What was left
open is the sentence between them: **the fleet knows which regime it is in, the items are
selectable, and auditing them still corrects nothing.** So what should a deployment
actually do?

**Auditing is not the only action.** A deployment that cannot correct a label can decline
to publish it. That is selective prediction, and the 2026 form of the rule states it as
agreement: predict only when the label is *forced*, that is when every consistent
hypothesis agrees, and abstain otherwise (Khosravani, [arXiv:2605.02611](https://arxiv.org/abs/2605.02611)).
Read across to a fleet of analysts, that rule says publish where the fleet is unanimous.
This corpus was built to say what happens to a rule like that when unanimity **is** the
failure.

**What is measured.** No authority, no anchors, no re-estimation. The estimator runs once
per fleet and each policy chooses labels to withhold. Two numbers per cell and neither is
readable alone: *selective risk*, the errors among labels still published, and *coverage*,
how many are published at all. Withholding here is deletion by design rather than by
accident -- nothing claims a repair, and a policy that withholds correct labels raises its
own risk, which is what makes the column honest without an oracle.

<!-- BEGIN GENERATED: selective-risk -->
**Slip rate 0.0** --- errors among published labels, 20 of the pool withheld

| Blind of 9 | base | `uniform` | `margin` | `posterior` | `consensus` | `channel` | `oracle` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 7 | 0.100 | 0.100 | 0.000 | 0.000 | 0.111 | 0.000 | 0.000 |
| 8 | 0.100 | 0.100 | 0.000 | 0.000 | 0.111 | 0.000 | 0.000 |
| 9 | 0.100 | 0.100 | 0.094 | 0.089 | 0.094 | 0.000 | 0.000 |

Share of those 20 withheld labels that were actually wrong (chance is the base rate in the column above):

| Blind of 9 | `uniform` | `margin` | `posterior` | `consensus` | `channel` | `oracle` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.10 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 8 | 0.10 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 9 | 0.10 | 0.15 | 0.20 | 0.15 | 1.00 | 1.00 |

**Slip rate 0.15** --- errors among published labels, 20 of the pool withheld

| Blind of 9 | base | `uniform` | `margin` | `posterior` | `consensus` | `channel` | `oracle` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.020 | 0.022 | 0.006 | 0.006 | 0.022 | 0.017 | 0.000 |
| 5 | 0.075 | 0.078 | 0.022 | 0.017 | 0.083 | 0.011 | 0.000 |
| 7 | 0.100 | 0.100 | 0.067 | 0.078 | 0.111 | 0.000 | 0.000 |
| 8 | 0.100 | 0.100 | 0.094 | 0.106 | 0.111 | 0.000 | 0.000 |
| 9 | 0.105 | 0.106 | 0.111 | 0.106 | 0.117 | 0.006 | 0.006 |

Share of those 20 withheld labels that were actually wrong (chance is the base rate in the column above):

| Blind of 9 | `uniform` | `margin` | `posterior` | `consensus` | `channel` | `oracle` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.15 | 0.15 | 0.00 | 0.05 | 0.20 |
| 5 | 0.05 | 0.55 | 0.60 | 0.00 | 0.65 | 0.75 |
| 7 | 0.10 | 0.40 | 0.30 | 0.00 | 1.00 | 1.00 |
| 8 | 0.10 | 0.15 | 0.05 | 0.00 | 1.00 | 1.00 |
| 9 | 0.10 | 0.05 | 0.10 | 0.00 | 1.00 | 1.00 |

At unanimity the textbook rule is at chance and provenance is exact. Withholding by the channel the detector named halves the wrong labels for 6% of coverage at slip 0.0 and 6% of coverage at slip 0.15, and clears them entirely for 10% of coverage at slip 0.0 and not reached within the sweep at slip 0.15. That is the price of the whole trade: **detection converts into coverage, not into correction**. Nothing here repairs a label, and the coverage column is what says so.

The same rule on a fleet with **no** blind spot withholds 10% of labels and catches 0 wrong ones, because there are none to catch. That is what a false detection costs, and it is a number rather than a reassurance.

The `channel` column is a proposal only where finding 22's detector fires, which is not every row: at 3 of 9 blind it is scored on a channel no deployment would have been told about. Every claim above is quantified over cells where the detector fired, and the artifact lists any that are not.

The control runs the other way. At slip 0.4 a fleet with no blind spot already carries estimator errors, and there the textbook rule works while provenance no longer beats an untargeted draw --- so the failure above is a property of the error's *shape*, not of abstention, and the remedy is bounded by the same distinction.
<!-- END GENERATED: selective-risk -->

**The crossing is sharp, and it is at unanimity.** With one dissenter left in nine, the
textbook rule is *exact*: `margin` and `posterior` withhold nothing but corrupted labels
and tie the oracle bound. At nine of nine the same rule lands 3 and 4 of a 20-label
withhold on a wrong label, against chance's 2. Nothing about the corpus changed; the
signal the rule reads stopped existing. `consensus`, the textbook inversion, does no
better -- at unanimity almost everything is unanimous, so ranking by agreement ranks
nothing.

**What the answer costs.** On the committed corpus, withholding by the channel the
detector named clears every label the blind spot corrupted at 10% of coverage, and it
needs no authority, no second opinion and no re-estimation. Across the eight corpus
draws finding 26 sweeps, seven can host the experiment: the published error rate falls
below every untargeted draw in **7 of 7**, and the withhold matches what withholding the
wrong labels themselves would buy in **5 of 7** -- the same two-in-eight shortfall
finding 26 measured for the audit form of this policy, and for the same reason. The qualifier is exact rather than decorative: at a 15%
slip rate one error survives the withhold, and it is an independent slip rather than the
shared standard -- the rule addresses the component it was aimed at and no other. That is the whole trade and it is worse than an audit would have been if an
audit worked: a deployment that detects a shared blind spot buys silence on a slice of its
corpus, not a corrected label anywhere. The exchange rate is set by how prevalent the
channel is, not by how many labels are wrong -- which is why a *false* detection is not
free either: on a fleet with no blind spot at all, the same rule withholds 10% of labels
and catches nothing.

**The bound, stated as prominently as the result.** All of the above holds where the
shared blind spot is the *whole* of the estimator's error -- the regime the artifact
identifies itself, by checking that a fleet with no blind spot has no errors at that slip
rate. Push independent noise up until a healthy fleet breaks (slip 0.40 here, where
aggregation stops absorbing it) and the picture inverts: the textbook rule works again,
because random error does show up as disagreement, and provenance stops beating an
untargeted draw, because the shared component is now a minority of what is wrong. Neither
rule is right in general. The distinction that travels is between error that is *shared*
and error that is *independent*, and a deployment cannot pick a rule without knowing which
it has -- which is what finding 22's detector is for.

**Across draws, in one place.** Confidence-based abstention fails at unanimity in 7 of 7
draws, its textbook inversion fails in 7 of 7, provenance works in 7 of 7 and ties the
bound in 5 of 7, a false detection costs 10% of coverage in every draw, and the
random-error control is measurable and positive in 5 of 7 -- on the other two the
estimator is degenerate enough at that noise level that no rule beats an untargeted
draw. `results/corpus_sensitivity.json` carries each denominator.

!!! note "A comparison this measurement got wrong first, in the way this project keeps getting things wrong"
    The first version compared each policy against the **median** of 21 uniform draws and
    reported that confidence-based abstention worked at unanimity, on a gap of 0.006 --
    one task, well inside the spread of the baseline it was beating. Withholding 20 random
    labels of two hundred removes an error now and then, so the median is not a floor. The
    comparison is now against the *best* of the 21 draws, and the claim reversed.

## 29. The shape of the error is visible from the aggregate, and it is cheap to read wrong

`scripts/measure_error_shape.py`, `results/error_shape.json`

[Finding 28](#28-the-open-problem-answered-detection-converts-into-coverage-not-into-correction)
ends on a fork it cannot resolve from inside. Withholding by the named channel is right
where the error is shared; withholding by confidence is right where it is independent;
each is wrong in the other regime. That measurement decided which regime it was in by
checking a **healthy control fleet's** error count -- which an experimenter has and a
deployment does not.

**The statistic.** Condition on the evidence stratum, as finding 22's detector does.
Within a stratum every task shows the same evidence, so a fleet applying one rule votes
the same way on all of them and the per-task vote sum varies only as its analysts slip.
Independent slips are exactly binomial. A *shared* error is not: it splits the stratum
into the tasks the shared standard corrupts and the tasks it does not, and a mixture of
two rates carries more variance than a binomial at their mean. So the **index of
dispersion** -- observed variance over binomial variance, pooled across strata -- sits at
1 under independent error and rises above it when part of the error is shared.

It reads per-task vote sums, per-task contributor counts, and the public evidence count.
That is **strictly less than finding 22's detector needs**: no channel to name, no
per-analyst stream, no ground truth, no control fleet. Overdispersion diagnostics are
standard wherever counts are modelled; what we did not find prior work for is this
application, and that is a statement about our search rather than about the literature.

<!-- BEGIN GENERATED: error-shape -->
**Index of dispersion** --- observed variance of the per-task vote sums over the binomial variance at the same rate, within evidence stratum. 1 is independent error; above 1 is a shared component. `--` is a fleet with no variance anywhere, which cannot be diagnosed rather than being diagnosed as clean.

| Blind of 9 | slip 0.0 | slip 0.05 | slip 0.15 | slip 0.25 | slip 0.4 |
| --- | --- | --- | --- | --- | --- |
| 0 | -- | 1.10 | 0.96 | 1.08 | 1.08 |
| 1 | 0.73 | 1.06 | 0.95 | 0.89 | 1.01 |
| 3 | 2.36 | 1.62 | 1.11 | 0.95 | 0.93 |
| 5 | 4.23 | 2.61 | 1.44 | 1.05 | 0.82 |
| 7 | 6.42 | 3.67 | 1.83 | 1.11 | 0.95 |
| 9 | 9.00 | 5.32 | 2.57 | 1.49 | 0.96 |

**Which rule wins, and which the index picks.** `C` is withholding by the named channel, `F` is withholding by confidence, `b` is both beating an untargeted draw, `-` is neither, `?` is a fleet the index cannot diagnose.

| Blind of 9 | slip 0.0 | slip 0.05 | slip 0.15 | slip 0.25 | slip 0.4 |
| --- | --- | --- | --- | --- | --- |
| 0 | ?/? | ?/F | ?/F | F/F | F/F |
| 1 | ?/F | ?/F | -/F | F/F | -/F |
| 3 | ?/C | ?/C | F/F | b/F | C/F ⚠ |
| 5 | ?/C | b/C | b/C | b/F | -/F |
| 7 | b/C | b/C | b/C | b/F | C/F ⚠ |
| 9 | C/C | C/C | C/C | C/C | -/F |

The index picks the rule that wins in **8 of 10** cells where a rule wins at all. The rate is the less useful half of that sentence: what a wrong call costs is **0.011** of published error rate at worst, because the cells it misses are ones where the winning rule beats an untargeted draw by about one task.

| Missed cell | Wins | Index says | Cost | Channel | Confidence | Best uniform |
| --- | --- | --- | --- | --- | --- | --- |
| 3 blind, slip 0.4 | channel | confidence | 0.0111 | 0.3056 | 0.3167 | 0.3111 |
| 7 blind, slip 0.4 | channel | confidence | 0.0111 | 0.3556 | 0.3667 | 0.3667 |
<!-- END GENERATED: error-shape -->

**What holds.** The index is calibrated on a fleet with no shared blind spot -- it sits
at 1 at every slip rate that produces any dispersion at all -- and it rises monotonically
with the share of the fleet carrying the blind spot, reaching 9.00 on a noiseless fleet
where every analyst is blind. It falls back toward 1 as independent noise rises, which is
the mechanism working as described rather than failing: noise fills in the gap between
the two groups a shared standard creates.

**What does not, and the prediction it refutes.** The docstring predicted that
thresholding the index would pick the winning rule in most cells *and that its failures
would land in the ambiguous cells where neither rule dominates*. The first half holds and
the second is wrong. Both misses are cells where a rule genuinely wins and the index
confidently names the other one -- a wrong recommendation rather than an abstention,
which is the worse failure mode.

**Why the rate is the less useful half.** Both missed cells sit at the highest slip rate,
where the estimator is already carrying 65 and 76 wrong labels of two hundred, and where
the winning rule beats the best untargeted draw by about one task. Following the index
there costs **0.011** of published error rate on this corpus. The accuracy rate says 8 of
10; the cost says the two misses are nearly free, and the second number is the one a
deployment acts on.

**And the cost is a draw, which the sweep caught before this page quoted it as a
property.** Across the seven corpus draws that host the test the index picks correctly in
**56 of 64** decidable cells, and the worst wrong call costs **0.011 to 0.050** -- the
committed corpus sitting at the cheap end of that range. An earlier version of this
paragraph, and of the manuscript passage it feeds, said "a hundredth of the published
error rate" without a denominator. That is the single-draw shape this project has
retracted five times, caught here by the sweep rather than by a reader.

**The honest form of the answer.** The shape of the error *is* estimable from what the
aggregator already holds, and the estimate is worth acting on because its errors are
confined to the regime where the choice barely matters -- at a price that ranges to five
hundredths of the published error rate across draws rather than the one hundredth this
corpus shows. What it cannot do is diagnose a
fleet with no disagreement anywhere: with every analyst deterministic and identical there
is no variance to compare against, and the artifact reports that cell as **undiagnosable**
rather than as clean. That distinction is the same one this page keeps arriving at from
different directions -- a silent instrument and an instrument reporting nothing wrong are
different claims.

!!! warning "The first version of this test could not fire"
    The null is parametric -- simulate each stratum's counts as binomial at its own rate
    -- and a simulated p-value floors at 1/(m+1). The script documented 400 draws as
    putting that floor "comfortably below" an alpha of 0.001. It does not: 1/401 is
    0.0025, which is **larger**, so every cell would have read as not overdispersed no
    matter how extreme, including the fleet whose index is 9.00. The arithmetic is now
    asserted at run time and refuses, in the same form
    `measure_corpus_sensitivity.py` already used on finding 22's permutation count.

## 30. The blind spot with no name is detected the same, and located by the same signal read one-sided

`scripts/measure_latent_blindspot.py`, `results/latent_blindspot.json`

This repository has carried one residue since
[finding 23](#23-once-the-detector-names-the-channel-provenance-finds-the-corrupted-items-that-are-findable-at-all),
and its README states it in these words: finding 22 detects a blind spot aligned with a
**known, public partition** of the corpus, and "a shared error that follows no observable
partition would leave no trace in it either, and nothing here says how to find one."

That sentence makes two claims and they come apart. The channel scan does lose everything,
because a scan enumerates a small public family and an unnameable slice leaves nothing to
enumerate. The **index of dispersion** never read a partition in the first place --- it
reads per-task vote sums within an evidence stratum --- and the question is whether it
notices the difference. It does not.

**The construction, matched to finding 21's in every respect but one.**
`AnalystPolicy.distrusted_reports` is the same wrong standard as `blind_compartment` with
its handle removed: the blind analysts decline to credit a set of individual reports
rather than a channel. The set is drawn so it corrupts what PARTNER corrupts on this
corpus --- 20 verdicts of 200, all on tasks showing all three defining facts, so a lost
report drops the reviewer below a conjunction of three --- and drawn uniformly within that
stratum, which is what makes membership unpredictable from any column the corpus records.
A balance precondition refuses a draw whose compartment carriage sits above the 99th
percentile of uniform draws from the same pool, because a slice that *did* load onto a
channel would be found by the scan and this finding would report the scan working as the
scan failing.

<!-- BEGIN GENERATED: latent-blindspot -->
**What each detector says, by what the blind spot is keyed on.** `scan` is the channel detector of finding 22, which enumerates the corpus's compartments. `index` is the dispersion statistic of finding 29, which enumerates nothing. The latent columns are the median and range over 5 slice draws; `--` is a fleet with no variance to read.

| Blind of 9 | slip | scan, channel-keyed | scan, latent | index, channel-keyed | index, latent (range) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0 | PARTNER | silent | 0.73 | 0.73 (0.73-0.73) |
| 3 | 0.0 | PARTNER | silent | 2.36 | 2.36 (2.36-2.36) |
| 5 | 0.0 | PARTNER | silent | 4.23 | 4.23 (4.23-4.23) |
| 7 | 0.0 | PARTNER | silent | 6.42 | 6.42 (6.42-6.42) |
| 9 | 0.0 | PARTNER | silent | 9.00 | 9.00 (9.00-9.00) |
| 0 | 0.15 | silent | silent | 0.96 | 0.96 (0.96-0.96) |
| 1 | 0.15 | silent | silent | 0.95 | 0.93 (0.89-0.96) |
| 3 | 0.15 | silent | silent | 1.11 | 1.12 (1.03-1.18) |
| 5 | 0.15 | PARTNER | silent | 1.44 | 1.46 (1.36-1.53) |
| 7 | 0.15 | PARTNER | silent | 1.83 | 1.84 (1.70-1.92) |
| 9 | 0.15 | PARTNER | silent | 2.57 | 2.53 (2.50-2.57) |

**Localization at unanimity**, where every disagreement-reading rule is already at chance. Share of a 20-item withhold that landed on a label the estimator got wrong, over every slice draw and slip rate. `oracle` reads ground truth and is a bound rather than a method.

| Rule | median | range |
| --- | --- | --- |
| `uniform` | 0.10 | 0.05-0.15 |
| `margin` | 0.15 | 0.05-0.25 |
| `posterior` | 0.15 | 0.10-0.25 |
| `consensus` | 0.10 | 0.00-0.25 |
| `deviation` | 1.00 | 0.85-1.00 |
| `shortfall` | 1.00 | 1.00-1.00 |
| `oracle` | 1.00 | 1.00-1.00 |

**As the corrupted slice grows past the majority of its stratum.** The pool of tasks a discounted report can flip is 69 on this corpus, so the crossing sits between the two middle rows.

| Slice | slip | errors | `uniform` | `deviation` | `shortfall` |
| --- | --- | --- | --- | --- | --- |
| 10 of 69 | 0.0 | 10 | 0.05 | 0.50 | 0.50 |
| 10 of 69 | 0.15 | 11 | 0.05 | 0.55 | 0.50 |
| 20 of 69 | 0.0 | 20 | 0.10 | 1.00 | 1.00 |
| 20 of 69 | 0.15 | 21 | 0.15 | 0.95 | 1.00 |
| 30 of 69 | 0.0 | 30 | 0.15 | 1.00 | 1.00 |
| 30 of 69 | 0.15 | 31 | 0.15 | 0.65 | 1.00 |
| 40 of 69 | 0.0 | 40 | 0.20 | 0.00 | 1.00 |
| 40 of 69 | 0.15 | 40 | 0.20 | 0.30 | 1.00 |
| 50 of 69 | 0.0 | 50 | 0.25 | 0.05 | 1.00 |
| 50 of 69 | 0.15 | 50 | 0.25 | 0.20 | 1.00 |
| 60 of 69 | 0.0 | 60 | 0.30 | 0.55 | 1.00 |
| 60 of 69 | 0.15 | 60 | 0.30 | 0.25 | 0.50 |
<!-- END GENERATED: latent-blindspot -->

**Detection survives the loss of the partition, exactly.** At every share of a
deterministic fleet the two constructions give the *same* index --- 0.73, 2.36, 4.23,
6.42, 9.00 --- and the reason is worth stating because it is also the limit of the
statistic: within a stratum the index depends on how many tasks the shared standard
corrupts and not on which ones, so two slices of the same size in the same stratum are the
same number to it. Once analysts slip independently the two fleets are different draws and
the indices separate by hundredths; the channel-keyed value then sits **inside the range
five latent slice draws produce, in every cell**. The statistic cannot tell the two
constructions apart. The scan, on the same fleets, names PARTNER on one and is silent on
the other at every share and both slip rates.

**Localization survives too, and that was predicted to fail.** The docstring's third
prediction was that ranking tasks by their within-stratum residual --- the per-task
summands of the index --- would do no better than an untargeted draw at unanimity, on
finding 21's reasoning that a unanimously blind fleet leaves nothing to distinguish a
corrupted task from a merely unusual one. It is wrong. At unanimity `deviation` lands a
median **1.00** of a 20-item withhold on a wrong label against the **best** of 21
untargeted draws at 0.25, tying the oracle, with no channel named and no detection required
first. The three disagreement-reading rules are all *below* that untargeted floor on the
same fleets --- `margin` and `posterior` at 0.15, `consensus` at 0.10 --- which is finding
28's result arriving again from a fleet with no channel in it.

The reconciliation is the sentence above: the index is invariant to *which* tasks a
standard corrupts, and its summands are not. A sum over residuals discards the information
a rank over the same residuals keeps, and finding 29 read only the sum.

**What the two-sided form costs, measured rather than asserted.** The fourth prediction
was that if localization worked at all it would work only while the corrupted slice was a
minority of its stratum, and the sweep was run to find the crossing. It is there. Once the
corrupted slice sets its own stratum's rate the *clean* tasks become the outliers, and
`deviation` falls to or below an untargeted draw in **5 of the 12 swept cells, none of them
below 40 of the 69 eligible tasks** --- the pool's majority is 35, and the first inversion
is on the other side of it. The worst cell is 40 of 69 on a noiseless fleet, where it lands
**0.00** of a 20-item withhold on a wrong label against an untargeted 0.35. That is the
failure mode that matters, because a rule that is merely uninformative wastes a budget and
a rule that is anti-informative spends it on correct labels. Degradation starts earlier
than inversion: at 30 of 69 with analysts slipping it is already at 0.65 against a perfect
1.00 one row above.

**Signing the residual removes the crossing, and names the assumption that does it.**
`shortfall` is the same statistic read one-sided: audit where a stratum votes *low*, not
where it votes far. A blind spot is a failure to credit evidence and a reviewer who credits
less escalates less, so the corrupted tasks vote low whether they are ten of a stratum or
fifty. It never falls to an untargeted draw anywhere in the sweep, and it **ties the oracle
in 10 of the 12 cells**. Both misses are worth naming rather than rounding away: at 10 of
69 with slip it takes 0.50 against the bound's 0.55, one task short, and at 60 of 69 with
slip it takes 0.50 against 1.00, which is the real limit and is the subject of the
paragraph below.

That directional assumption is the scope condition and it is not new here: finding 22's
detector is already one-sided, for this reason and in these terms. A shared error that
*raised* the rate --- a fleet crediting something it should not --- needs the mirror of
this rule, and nothing in the aggregate says which of the two a fleet has. The honest
statement of what a deployment gets is therefore conditional in one place rather than two:
it no longer needs a nameable channel, and it still needs to know which way its analysts
are wrong.

**Where it does break.** At 60 of 69 corrupted, with analysts slipping at 0.15, `shortfall`
falls to 0.50 against the oracle's 1.00 --- nine of the sixty-nine remain clean, and a
rate estimated from a stratum that is seven-eighths corrupted is no longer a reference. That is
the same shape as every other limit on this page: the statistic needs a majority of the
stratum to be doing the right thing, which is a weaker requirement than the majority of the
*fleet* that findings 19 to 21 needed, and is a different quantity rather than a smaller
one.

!!! warning "This measures the class of failure that is unnameable, not the class that is unstructured"
    The slice here follows no partition the corpus records, which is what the open problem
    asked for. It is still a set of tasks that a shared standard corrupts *consistently*,
    and that consistency is what the residual reads. A shared error applied erratically ---
    the same analysts, the same habit, acting on it half the time --- would leave a weaker
    trace and this page does not price it. The claim that travels is about the handle
    being absent, not about the structure being absent.

!!! note "What this does to the open problem"
    The README's residue is withdrawn as stated. Detection does not require a public
    partition, and neither does the remedy of
    [finding 28](#28-the-open-problem-answered-detection-converts-into-coverage-not-into-correction):
    withholding by the one-sided residual buys what withholding by the named channel bought,
    on a fleet where no channel can be named. What replaces it is narrower and is a question
    about the world rather than about this generator --- whether a real watch floor's shared
    errors depress a rate consistently enough for a within-stratum residual to see them,
    which is a claim about operational data that no corpus generated here can settle.
