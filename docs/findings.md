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

## The findings, by what they are about

Grouped for navigation only. The numbering is chronological, so a group's members are
scattered through it, and nothing in the grouping is a claim about how they relate.

| | |
| --- | --- |
| **Attribution and policy** | [1. Leave-one-out cannot produce a governed label](#1-leave-one-out-attribution-cannot-produce-a-correct-governed-label) · [2. Bimodal on one policy ruling](#2-the-design-is-bimodal-on-one-policy-ruling) |
| **Triage baselines and scale** | [3. A corpus bug and a retracted finding](#3-a-corpus-bug-a-retracted-finding-and-a-real-benchmark-target) · [3b. Over-escalation is universal](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) · [4. Answerability against non-leakage](#4-answerability-and-surface-non-leakage-pull-against-each-other) |
| **Learning the rule** | [5. In-context learning does not close the gap](#5-in-context-learning-does-not-close-the-gap) · [6. Gradient learning does, on clean labels](#6-gradient-learning-does-close-the-gap-on-clean-labels) · [10. A fleet learns its analyst's standard](#10-a-fleet-learns-its-analysts-standard-not-the-worlds) |
| **What review costs** | [7. Review is abundant; correctness is not](#7-review-is-abundant-what-it-costs-is-correctness) · [8. Right and sloppy beats wrong and careful](#8-being-right-and-sloppy-beats-being-wrong-and-careful) |
| **Measurement design** | [9. Repeating one prompt measures the wrong thing](#9-a-measurement-that-repeats-one-prompt-measures-the-wrong-thing) · [17. Item difficulty does not separate the two](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst) |
| **Disclosure and identity** | [11. The stream still names the analyst](#11-the-gate-clears-every-item-and-the-stream-still-names-the-analyst) · [12. Reliability needs identity where it matters](#12-reliability-cannot-be-estimated-without-identity-where-it-matters) · [13. A tag can replace identity](#13-a-reliability-tag-can-replace-identity-and-the-leak-metric-cannot-tell-you-when) · [16. The cliff assumes independent fleets](#16-the-cliff-is-safe-only-because-the-fleets-were-drawn-independently) |
| **Cost of running it** | [14. What the agent costs on its hardware](#14-what-the-agent-costs-on-the-hardware-it-is-meant-to-run-on) · [15. The budget is spent on the wrong variable](#15-the-standard-privacy-mechanism-spends-the-budget-on-the-wrong-variable) · [19. What an authority of record costs](#19-an-authority-of-record-repairs-the-cliff-and-its-price-explodes) |
| **Estimating under aggregation** | [18. The cliff survives the protocol](#18-the-estimate-moves-under-secure-aggregation-and-the-cliff-does-not-move-with-it) · [19. An authority repairs it, at a price](#19-an-authority-of-record-repairs-the-cliff-and-its-price-explodes) · [20. Audit where the fleet splits](#20-audit-where-the-fleet-splits-and-the-prediction-that-said-otherwise) · [21. Where that policy stops working](#21-the-corpus-the-audit-policy-cannot-handle-built-on-purpose) |

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
| 3b | 40 | 0.000 | constant | unresolved (needs n>2000) | qwen2.5-3b (0.625) clears the majority floor (0.625) |
| 3b | 40 | 0.200 | constant | **resolved** | mistral-7b (0.425) is below the majority floor (0.625) |
| 5 | 600 | 0.009 | condition | unresolved (needs n>2000) | 8 shots (0.514) beats 0 shots (0.523) -- REFUTED at n=600 |
| 5 | 600 | 0.055 | condition | unresolved (needs n≥2000) | 2 shots (0.468) is worse than 0 shots (0.523) -- direction only |
| 5 | 600 | 0.486 | constant | **resolved** | 8 shots (0.514) is below the stated-rule ceiling (1.000) |
| 5 | 600 | 0.179 | constant | **resolved** | 8 shots (0.514) is below the majority floor (0.693) |
| 6 | 60 | 0.531 | condition | **resolved** | adapter (1.000) beats the base model (0.469) |
| 10 | 600 | 0.560 | condition | **resolved** | any-one adapter matches teacher (1.000) not world (0.440) |
| 10 | 600 | 0.112 | condition | **resolved** | inattentive adapter (0.902) beats its own teacher (0.790) |
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
| `authority_anchors` | 8 | **no** | n=8 is below 30; treat differences as provisional |
| `blind_spot` | 105 | yes | - |
| `consensus_reliability` | 200 | yes | - |
| `correlated_fleets` | 60 | yes | - |
| `decode_stability` | 30 | yes | - |
| `difficulty_confound` | 200 | yes | - |
| `edge_cost` | 199 | yes | - |
| `fleet_linkage` | 200 | yes | - |
| `label_fidelity` | 40 | yes | - |
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
| `tagged_aggregation` | 200 | yes | - |
| `triage_lift-llama3.1-8b` | 40 | **no** | accuracy 0.600 does not beat the majority floor 0.650: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `triage_lift-llama3.2-3b` | 40 | yes | - |
| `triage_lift-mistral-7b` | 40 | **no** | accuracy 0.487 does not beat the majority floor 0.641: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `triage_lift-qwen2.5-14b` | 40 | **no** | accuracy 0.625 does not beat the majority floor 0.650: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `triage_lift-qwen2.5-3b` | 40 | yes | - |
| `triage_lift-qwen2.5-7b` | 40 | **no** | accuracy 0.450 does not beat the majority floor 0.650: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |
| `triage_lift` | 40 | **no** | accuracy 0.450 does not beat the majority floor 0.650: this is not evidence of capability; recall is 1.000 while false positives exceed true positives: the model escalates indiscriminately, which scores well on recall alone |

**27 of 57** assessed artifacts are flagged. A flagged number may still be quoted as evidence that something *failed*, which is what the flag asserts; it may not be quoted as evidence of capability.

Exempt, because there is no sampling question to answer:

- `adapter_replication` -- compares assessed adapter artifacts against their own replicates; the question is whether two runs agree, which no sampling flag answers
- `external_gate_validation` -- carries its own permutation-null statistics per corpus
- `federation_eligibility` -- deterministic over the label lattice; nothing is sampled
- `fl_benchmarks` -- sizes the problem rather than settling it, is quoted nowhere in the manuscript, and reports a bootstrap interval per condition instead of a flag
- `fleet_sensitivity` -- a sweep over a nuisance parameter; reports invariants, samples nothing
- `gate_determinism` -- reports the gate's surface baseline at full precision on one machine; the result is the comparison against another machine, not the number
- `gate_determinism-cluster` -- the second machine of that comparison
- `power` -- prices hypothetical evaluation sizes; simulates outcomes rather than measuring any
- `teacher_fleet` -- aggregates assessed adapter artifacts; adds no measurement of its own
- `triage_lift` -- superseded by the per-model triage_lift-* artifacts, which are assessed
<!-- END GENERATED: measurement-health -->

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
eight, while precision falls from 0.368 to 0.380 -- it does not recover -- so the
examples are not teaching the rule, they are teaching the model to escalate more. At
eight shots the model escalates every single task, which is the clearest form this
failure has taken: a classifier that answers SIGNIFICANT to everything has a recall of
1.000 and has learned nothing. That is the same failure
[finding 3b](#3b-over-escalation-is-universal-and-scale-does-not-fix-it) found in
every model tested, and supplying examples does not repair it.

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

| Teacher | Standard | Targets matching world | Adapter vs **world** | Adapter vs **teacher** | Inherited error |
| --- | --- | --- | --- | --- | --- |
| by-the-book | 3 of 3 | 1.000 | 0.995 | 0.995 | -0.005 |
| inattentive | 3 of 3, slips 15% | 0.855 | **0.902** | 0.790 | **+0.046** |
| two-of-three | 2 of 3 | 0.728 | 0.730 | **0.982** | +0.002 |
| any-one | 1 of 3 | 0.439 | 0.440 | **1.000** | +0.001 |

Accuracy on **600** held-out tasks whose events are disjoint from training. The base
model scores accuracy 0.317 with 74 of 600 answers unparsable, so every row is a large
gain over not training at all. The last column is the adapter's agreement with the
world minus its teacher's, and it is the cleanest statement of the finding.

**A wrong standard is inherited almost exactly.** The two systematically-wrong
teachers hand over their error rate to within **0.002 and 0.001**: a teacher whose
targets agree with the world on 0.728 produces an adapter agreeing on 0.730, and one
at 0.439 produces 0.440. The adapter did not partially absorb the reviewer's rule or
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

**It survives a corpus the adapter has never seen any part of.** The result above
evaluates on held-out *events* from the same corpus. A stronger test evaluates on a
different corpus instantiation entirely, sharing no events, no vessels and no
renderings, which is what contamination-resistance guidance asks of a benchmark.
Repeating the sweep with the evaluation drawn from seed 101 (2h54m, same A100):

| Teacher | Targets vs world | Adapter vs world | Adapter vs teacher | Inherited |
| --- | --- | --- | --- | --- |
| by-the-book | 1.000 | 0.988 | 0.988 | -0.012 |
| inattentive | 0.858 | **0.943** | 0.835 | **+0.086** |
| two-of-three | 0.728 | 0.708 | 0.997 | -0.020 |
| any-one | 0.439 | 0.447 | 1.000 | +0.008 |

Both conclusions hold on a corpus with zero overlap. The systematically wrong teachers
are still tracked closely, now within about two points rather than half a point, and
the careless teacher is still the only one improved upon: 0.943 against its teacher's
0.835, a gap of 0.108 against a difference half-width of 0.035, resolved on its own
terms.

!!! warning "Do not read the two tables as a controlled comparison"
    The cross-corpus run trained on **1,740** tuples where the same-corpus run trained
    on **1,140**. That is not an oversight in either run but a consequence of the
    design: when evaluation comes from a different corpus, nothing has to be held out
    of the training one. So the differences *between* the tables, the filtering effect
    growing from +0.038 to +0.086 and the inheritance loosening from half a point to
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
| shed compartments | 7,053 | **0.205** [0.150, 0.265] | 10.0 |
| keep compartments | 1,000 | 0.000 [0.000, 0.000] | 16.0 |

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

Three things fall out of that table, and all of them are statements about this attack
rather than about the channel.

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

| keep | fabricate | Recovery | ε per indicator | **ε composed** | Label noise |
| --- | --- | --- | --- | --- | --- |
| 0.9 | 0.0 | 0.205 | ∞ | ∞ | 0.000 |
| 0.9 | 0.1 | 0.220 | 2.20 | 428.5 | 0.341 |
| 0.8 | 0.2 | 0.120 | 1.39 | 270.3 | 0.537 |
| 0.7 | 0.3 | 0.100 | 0.85 | 165.2 | 0.668 |
| **0.6** | **0.4** | **0.065** | **0.41** | **66.7** | **0.756** |

Recovery here is again what *this* attack achieves, so the column bounds leakage from
below; the epsilon columns, by contrast, bound it from above for any adversary, which
is exactly why a mechanism with a stated budget is worth more than a control validated
against one attack. That contrast is the reason this finding is worth having even
though its numbers are worse than the ladder's.

**The two epsilon columns are the finding.** Randomized response bounds the likelihood
ratio for *one* indicator, and the attack observes all of them: two clearances in this
fleet are separated by as many as **195 tasks**, so the guarantee against an adversary
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
0.220 against 0.205, which at 200 analysts is three people and is read as noise rather
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

    | Fleet | rate 0.1: independent | one culture | understatement | rate 0.3: independent | one culture | understatement |
    | --- | --- | --- | --- | --- | --- | --- |
    | 5 | 0.009 | 0.100 | 12x | 0.163 | 0.300 | 2x |
    | **9** (committed) | 0.001 | 0.100 | **112x** | 0.099 | 0.300 | 3x |
    | 15 | 0.000 | 0.100 | unbounded | 0.050 | 0.300 | 6x |
    | 25 | 0.000 | 0.100 | unbounded | 0.018 | 0.300 | 17x |
    | 51 | 0.000 | 0.100 | unbounded | 0.001 | 0.300 | **214x** |

    **The understatement grows monotonically with fleet size, at every rate.** The
    mechanism is not subtle: independent draws concentrate as the fleet grows, so
    P(wrong majority) falls toward zero, while a shared culture is a single coin
    whatever the headcount. By 15 analysts the independent probability at a 10% error
    rate rounds to zero at three decimals and the ratio stops being finite.

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

| Signature facts present | 0 | 1 | **2** | 3 |
| --- | --- | --- | --- | --- |
| Items | 21 | 63 | **52** | 64 |
| Class | routine | routine | **routine, near-boundary** | significant |

The 52 near-boundary items are also **exactly** the items a reviewer holding a
two-of-three standard gets wrong. "This item is ambiguous" and "this reviewer applies
the wrong rule" predict identical data.

**The control decides it.** Estimated difficulty by true overlap, under fleets that
differ only in composition:

| Fleet | ovl=0 | ovl=1 | **ovl=2** | ovl=3 | Spread | Converged |
| --- | --- | --- | --- | --- | --- | --- |
| **correct (control)** | 1.090 | 1.090 | **1.090** | 1.090 | **1.00** | yes, 1 iter |
| correct, 15% random slip | 2.227 | 1.677 | 1.701 | 2.038 | 1.33 | yes, 9 iters |
| 3 of 9 wrong standard | 0.942 | 0.942 | **2.563** | 0.942 | 2.72 | yes, 6 iters |
| 5 of 9 wrong standard | 0.933 | 0.933 | **4.398** | 0.933 | 4.72 | yes, 38 iters |

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
difficulty with no wrong standard anywhere, but the shape is inverted: it peaks at
overlap 0, *furthest* from the boundary, and stays nearly flat across the rest. Any
annotator error inflates apparent difficulty; only systematic error inflates it *at the
boundary*, and only systematic error inflates one band while leaving the others at the
control value. That difference is the one hopeful result here, and it is a shape rather
than a magnitude: the slip row's spread of 1.33 is smaller than 3-of-9's 2.72, but a
single summary number cannot tell you which band carries it.

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
    than first reported -- an inversion of 1.42x rather than 31x -- and that nothing
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

Finding 18 localised the cliff to non-identifiability, and
[finding 17](#17-adding-item-difficulty-does-not-separate-a-hard-case-from-a-wrong-analyst)
already showed no better estimator escapes it from the data alone. What breaks a
relabelling degeneracy is an **exogenous** label: a task whose disposition is asserted
by an authority rather than inferred from the fleet. That is the *authority of record*
the build order has owed since step 3, and this prices it.

**The scoring rule is the methodology.** An anchored task's label was handed over, so
counting it would measure how many answers the authority supplied rather than what they
bought. Every number below is computed **only over unanchored tasks**.

| Wrong of 9 | 0 | 5 | 12 | 50 | 80 | 100 | 150 | 180 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **5** | 0.660 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **6** | 0.660 | 0.667 | 0.663 | 0.658 | 0.638 | **1.000** | 1.000 | 1.000 |
| **7** | 0.660 | 0.667 | 0.663 | 0.658 | 0.638 | 0.667 | **1.000** | 1.000 |
| **9** | 0.660 | 0.667 | 0.663 | 0.658 | 0.638 | 0.667 | 0.680 | **1.000** |

**The price is not linear and it is not a curve.** It is a second threshold, and it
moves far faster than the fleet's error does:

<!-- BEGIN GENERATED: authority-price -->
| Wrong of 9 | Audited items needed | Share of the round |
| --- | --- | --- |
| 4 | 0 | 0.0% |
| 5 | 5 | 2.5% |
| 6 | 100 | 50.0% |
| 7 | 150 | 75.0% |
| 9 | 180 | 90.0% |

Threshold for 'repaired' is agreement ≥ 0.95 on unanchored tasks, over a corpus of 200.
<!-- END GENERATED: authority-price -->

At a bare majority an authority ruling on **five items in two hundred** restores the
estimate on the **93** that remain scorable. One analyst further and the same repair costs **half the
round**; two further, three quarters. The mechanism is visible in the M step: an
anchored task constrains every contributor's confusion matrix, but the unanchored
majority still outvotes it, so the anchors have to reach a share that dominates the
estimate rather than merely inform it.

!!! danger "A partial budget is briefly worse than none"
    The curve is not monotone. At 6 of 9 wrong, 80 anchors scores **0.638** against
    **0.660** with no anchors at all. A programme that funds an audit at a fraction of
    what the crossing requires does not buy a fraction of the benefit; it buys slightly
    less than nothing until it clears the threshold. This is the practical warning in
    the finding, and it is the reason to report the threshold rather than a rate.

!!! note "What the unanimity row is and is not"
    At 9 of 9 the fleet is unanimous, so there is no disagreement to estimate from and
    the estimator has nothing to work with. The 1.000 at 180 anchors is real but it is
    not assistance: the authority has ruled on 90% of the round, and what the remaining
    10% gets is the benefit of having learned that every contributor is inverted. An
    authority auditing nine items in ten has not been helped by a fleet. It is carried
    as the control that shows where the mechanism stops being a mechanism.

    Anchors are drawn uniformly and without regard to difficulty. An authority that
    audited the *hardest* items would score better and would be assuming the question:
    knowing which items are hard is knowing where the fleet is wrong, which is what the
    estimate was meant to establish. Uniform is the honest floor; a targeted policy can
    only beat it.

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
| 5 | 5 | **2** | 2 | 80 | 2 |
| 6 | 45 | **20** | 20 | 80 | 20 |
| 7 | 80 | **30** | 30 | 95 | 30 |

Items an authority must rule on to repair the estimate, out of **97 auditable** tasks (200 in the corpus). Lower is better; bold is the winning deployable policy. † `oracle` reads ground truth and is a bound rather than a method.
<!-- END GENERATED: audit-policy -->

**Uncertainty sampling wins, and it ties the oracle exactly.** Auditing the tasks the
fleet splits on repairs six-of-nine at **20** items against uniform's 45, and
seven-of-nine at **30** against 80. Scoring against the estimator's posterior instead
of the raw votes gives the identical answer at every cell.

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
    budget tested --- 20 of 20, and 30 of 30 --- which is why it matches the oracle
    exactly rather than approaching it. That is a property of a corpus whose difficulty
    structure is discrete and known, where the hard items and the reviewer's blind spot
    coincide by construction.
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
| Blind of 9 | `uniform` | `margin` | `posterior` | `consensus` | `oracle` |
| --- | --- | --- | --- | --- | --- |
| 0 | -- | -- | -- | -- | -- |
| 3 | -- | -- | -- | -- | -- |
| 5 | -- | -- | -- | -- | -- |
| 7 | -- | -- | -- | -- | -- |
| 8 | -- | -- | -- | -- | -- |
| 9 | -- | -- | -- | -- | -- |

Audited items needed to repair; `--` is not reached within 95. Below: the share of a 20-item audit landing on a genuinely corrupted task.

| Blind of 9 | `uniform` | `margin` | `posterior` | `consensus` | `oracle` |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.10 | 1.00 | 1.00 | 0.00 | 1.00 |
| 8 | 0.10 | 1.00 | 1.00 | 0.00 | 1.00 |
| 9 | 0.10 | 0.15 | 0.20 | 0.15 | 1.00 |
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
