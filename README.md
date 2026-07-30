# Pharos

A labeled fleet testbed for federated personalization with a governed disclosure boundary.

Pharos supplies the one thing no public corpus does: a body of reporting whose
objects carry real classification levels and cross-cutting compartments, so that
the personal/shared split in federated personalization can be **measured** rather
than asserted.

Design specs live in the
[Federated Analyst Fleets](https://github.com/ajbarea/kourai-khryseai) research
docs: `docs/research/federated-forge/pharos-testbed.md` for this testbed,
`index.md` for the system it serves.

## Why "Pharos"

The lighthouse at Alexandria. A watch station whose entire function was seeing
what was coming, and reporting it to whoever needed to know.

## What is here

Step 1 of the build order: the label algebra, the corpus generator, and the
acceptance gate that decides whether a generated corpus may be used at all.

| Module | Responsibility |
| --- | --- |
| `pharos.labels` | The product lattice: sensitivity, compartments, capacity. Joins, dominance, type-based declassification |
| `pharos.world` | The fictional maritime watch: channels, officer voices, and the fact vocabulary |
| `pharos.generate` | Deterministic corpus generation, reproducible from `(seed, config)` |
| `pharos.gate` | The shortcut gate: can plant membership be predicted without reading anything? |
| `pharos.manifest` | The citable record: version, seed, gate verdict, label histogram |

Everything is offline and deterministic. There are no model calls in step 1,
which is what makes the gate reproducible.

## Quickstart

```bash
make setup
make test
make gate
```

`make gate` generates a corpus, runs the gate, and exits non-zero when the corpus
is not usable.

## The label lattice

```text
label = (sensitivity, compartments, capacity)

sensitivity  : OPEN < INTERNAL < PROTECTED < RESTRICTED    total order, join = max
compartments : subset of {SENSOR, LIAISON, LEGAL, PARTNER} subset lattice, join = union
capacity     : ENUM | SCALAR | SPAN | FREETEXT             the form of a derived output
```

Sensitivity is a ladder; compartments are not. Two officers at `RESTRICTED` with
incomparable compartment sets dominate in neither direction, which is what makes
this a lattice test rather than a ladder test.

Capacity is in the lattice to answer **label creep**. An entry's label is the
join over every object that fed the turn, and a conservative join drives every
derived entry to the top, at which point nothing is releasable and federation
degrades to local-only learning. So `join` takes the output's capacity as a
required argument rather than joining input capacities: an enum verdict is an
enum verdict however sensitive its inputs were.

Declassification fails closed twice over. A capacity the policy does not name is
never released, and compartments survive by default, because shedding a
compartment discloses that the compartment had something to say.

## The shortcut gate

Planted ground truth invites a model to learn the artifact of insertion rather
than the property under test. In a federated fleet that is worse than one
inflated score: a shared adapter would propagate the shortcut to every
deployment, so convergence would look healthy while the fleet had learned an
insertion tell.

So a corpus has to earn its use. A probe gets only non-semantic features, ten of
them, no content words at all, and it has to fail.

Two decisions make this a real gate rather than a formality.

**Leave-one-center-out cross-validation, not a random split.** A random split
lets a surface feature present in both halves go unpunished. And a *single*
held-out center tests on a quarter of the corpus, where AUC sampling error is
wider than the pass band itself, so a single-fold gate reports its own noise as
leakage. An earlier version did exactly that: it passed two seeds and failed
three at values it could not distinguish from chance.

**The strongest probe sets the verdict, never an average.** A linear model and a
gradient-boosted tree both run, ranked by distance from chance, and the worse
news wins. A gate should assume the most capable attacker available.

## What the gate caught

Recorded because each fix is a permanent constraint on the vocabulary, not a
one-off patch. The generator has been rejected four times.

| Round | AUC | Cause | Fix |
| --- | --- | --- | --- |
| 1 | 0.737 | Significant facts had numeric slots, most fillers had none. `digit_ratio` alone hit 0.652 and decimal points inflated sentence counts | One timestamp and two integer numerals in every rendering; no decimals |
| 2 | 0.581 | Rendering length: pattern facts averaged 13.33 words against 12.64 for fillers, and plants always carry all three | Balanced word counts |
| 3 | 0.572 | Hand-balancing oscillated (+0.69, then -2.33, then +1.00). The leak was structural: plants were the only events with a deterministic triple, so **any** per-fact property became a class signal | Background draws a decoy triple, so both classes are one fixed triple plus two fillers and differ only in which triple |
| 4 | ~0.55 | Slot widths varied, so which slots a fact used was a digit signature. Pattern facts averaged 9.00 digits against 8.25 | Every non-time slot is a two-digit integer, so every rendering carries exactly nine digits |

Round 3 is the one worth internalizing: tuning a leaked property just moves the
leak. Removing the asymmetry that produced it removes the whole class.

## The gate is a calibration instrument, not a purity test

The gate originally demanded an AUC at chance, and that requirement could never
have been met. Working out why produced the most useful result in the project so
far.

**Content-defined ground truth cannot have a chance-level surface baseline.** A
plant is a plant because it carries the significant facts, so plants carry those
facts more often than background does, so the fact *mix* differs by class, so any
surface statistic of those facts carries some information. Measured here after
four rounds of normalization: every report holds exactly two fact sentences of
fourteen words and nine digits each, and plants still average 49.29 words against
49.63. The residual is the mix, and the mix is the definition.

Reaching a true chance baseline would require a vocabulary whose every rendering
is a surface twin of every other on character count, punctuation, and
capitalisation as well. That is achievable and it is not obviously worth it, since
the useful questions are answerable without it.

So the gate answers two questions instead of pronouncing on purity.

**Is the leak real?** Compare the observed statistic against a **permutation
null**, where labels are shuffled so no relationship survives. Measured: a null
mean of 0.50 with a standard deviation around 0.02 to 0.03. That confirms the gate
is unbiased, and it gives the band an empirical basis rather than an assumed one.
It also means the nominal 0.45 to 0.55 band is about two standard deviations,
which is reasonable.

**How large is it?** The observed AUC is the **surface baseline**: what a model
scores while reading nothing. Every downstream triage number is reported against
it, because an F1 is meaningless without knowing what shape alone already
achieves.

A corpus is therefore usable when its labels vary, its null has actually been
computed, and its baseline sits under the ceiling with room above it for a model
to demonstrate something. Current state across seven seeds: baselines 0.530 to
0.587 against nulls near 0.50, all significant, all under the 0.65 ceiling, all
usable.

## What was measured, and what broke

Three findings from building the labelling path end to end. All are reproducible
from the scripts named, all used `qwen2.5:7b-instruct` on an 8 GB RTX 3060 Ti, and
each one changed the design rather than confirming it.

### 1. Leave-one-out attribution cannot produce a correct governed label

`scripts/measure_label_fidelity.py`. Eight-source summarization turns, exact
leave-one-out: **62% source recall, and a wrong label on half of all turns, always
under-restrictive.** One turn moved from `RESTRICTED[LIAISON,PARTNER,SENSOR]` to
`PROTECTED[LEGAL]`, not merely laxer but incomparable.

The cause is corroboration. Leave-one-out asks which single source is
load-bearing, and a fact reported through several channels has none: drop any one
copy and the fact survives in the others, so no source is blamed and none of their
labels enters the join. Corroboration is not an edge case in this domain, it is
what channels are for. Leave-one-out is also the ceiling that cheaper estimators
approximate, so nothing faster repairs it.

The replacement costs nothing. Given what the output asserts, join the labels of
every source that **could** have asserted it. One detection pass, no ablation
sweep, and conservative by construction, so the error direction is creep rather
than leak.

### 2. The design is bimodal on one policy ruling

`scripts/measure_federation_eligibility.py`. Three aggregator ceilings, four
capacities. Turns average 2.88 compartments of 4, and seven of eight already sit
at the top of the level ladder, because a summary over eight sources joins nearly
everything.

| Declassification policy | FREETEXT | SPAN | SCALAR | ENUM |
| --- | --- | --- | --- | --- |
| keep compartments (fail-closed default) | 0-12% | 0-12% | 0-12% | 0-12% |
| drop compartments for low capacity | 0-12% | 0-12% | **100%** | **100%** |

So the question "may a low-capacity verdict shed the compartments of its sources?"
is not a detail. Answer no and the fleet is a set of unconnected local learners.
Answer yes and verdict-shaped outputs federate completely while prose never does,
which reproduces the design's split table from measurement rather than assertion.

### 3. A corpus bug, a retracted finding, and a real benchmark target

`scripts/measure_triage_lift.py`. This one went wrong before it went right, and the
sequence is the useful part.

The first measurement said the specialist could not do its own task: triage
accuracy 0.35 against a majority floor of 0.725, and a checklist prompt moved
accuracy up while collapsing recall to 0.167. The conclusion drawn was that a 7B
model cannot evaluate a three-way conjunction over facts split across reports.

**That conclusion was wrong, and the cause was a corpus bug.** Only **34% of
significant events actually rendered all three of their defining facts** into their
reports, because channels were chosen before coverage was checked and any shortfall
was padded from the whole vocabulary. Two thirds of the positive class was
unanswerable from its own prompt, and reports asserted facts their event did not
have. The model was being scored on evidence it was never shown, and its
escalate-on-anything behaviour was a reasonable response to partial evidence.

Worth noting: the shortcut gate could never have caught this. A surface probe tests
whether shape predicts the label, not whether the label is derivable from the
content. Semantic integrity needs its own checks, and they are now in the test
suite.

Generation now guarantees coverage: every fact of an event is rendered by some
channel that can carry it, no report asserts a fact outside its event, and the
per-report fact count stays constant so length carries no class signal. Verified
across seeds at 100% answerable and 0 indistinguishable.

On the corrected corpus:

| Setup | Accuracy | Majority | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Rule stated, plain prompt | 0.433 | 0.633 | 0.393 | 1.000 | 0.564 |
| **Rule stated, checklist prompt** | **1.000** | 0.633 | 1.000 | 1.000 | **1.000** |
| Rule withheld, plain prompt | 0.367 | 0.633 | 0.367 | 1.000 | 0.537 |
| Rule withheld, brief reasoning | 0.667 | 0.633 | 0.524 | 1.000 | 0.688 |

So the model does the conjunction perfectly **when the rule is given and the prompt
structures the check**. Which exposed a design error of its own: stating the rule in
the prompt leaves nothing for the fleet to learn, and learning the analyst's rule
from accept/revise/reject is the entire premise of personalization.

Withholding the rule gives the benchmark its proper shape. A ceiling of F1 1.000,
known to be reachable because a rule-given prompt reaches it, against a base of
0.537 to 0.688 that over-escalates at recall 1.000 and precision 0.37 to 0.52.
**That gap is the target**, and it is measurable rather than hoped for.

### 4. Answerability and surface non-leakage pull against each other

Fixing coverage raised the surface baseline from about 0.55 to 0.63-0.67, and
widening channel sets to remove the coverability filter did not bring it back down.
The cause is coverage itself: once every fact of an event must appear across a
fixed number of reports, report composition is tightly determined by the event's
fact set, and plants always carry the same triple.

This is a real tension rather than a bug, and it argues for treating the surface
baseline as a published property of a *correct* corpus rather than a defect to
drive to chance. A triage score is reported against it, and the ceiling on what
counts as usable has to accommodate what answerability costs.

### 5. In-context learning does not close the gap, so the adapter test is on the critical path

`scripts/measure_rule_learnability.py`. The design's premise is that a fleet learns
analytic craft from an analyst's accept, revise, and reject decisions. Withholding the
rule and supplying labelled examples instead is the cheap form of that question.

Rule never stated, examples drawn from events disjoint from the evaluation set, and
the example block class-balanced so it teaches the rule rather than the prior:

| Condition | Shots | F1 |
| --- | --- | --- |
| Zero-shot floor | 0 | 0.720 |
| Verdict-only examples | 2 / 4 / 8 | 0.615 / 0.640 / 0.636 |
| Examples with the officer's stated reason | 2 / 4 / 8 | 0.522 / 0.706 / 0.556 |
| **Rule stated, checklist prompt (ceiling)** | n/a | **1.000** |

**Examples close none of the gap**, and richer examples do not rescue it. A bare
verdict is roughly one bit, which is a poor teacher for "these three of fifteen facts
must co-occur", so supplying the officer's reasoning was the obvious next thing to
try. It moved nothing reliably.

Three caveats, because this result is easy to over-read:

- Twenty evaluation tasks per condition. Differences of about 0.1 sit inside the
  noise, so the ordering *between* conditions is not claimed, only that none of them
  reaches the ceiling.
- Eight shots is roughly 3,600 words of examples before the target case, so
  long-context dilution is an unseparated confound.
- **In-context learning is not gradient learning.** Eight examples in a prompt and a
  LoRA trained on thousands of tuples are different mechanisms, and the first failing
  does not establish that the second will.

The useful conclusion is about sequencing rather than viability. The cheap proxy for
the design's central premise came back negative-to-inconclusive, which means the
premise cannot be validated cheaply and the adapter experiment is no longer optional:
it is the thing that decides. What the ceiling establishes is where the bottleneck
sits. A model that reaches F1 1.000 when told the rule is not short of capability, it
is short of the rule, so rule *acquisition* is the whole question.

## Observability

Runs are analysable from their own output. Every measurement goes out as a
structured JSON log line carrying typed fields, so a surface baseline or a per-fold
AUC can be queried rather than parsed back out of a message:

~~~json
{"message": "gate.surface_baseline", "metric": "gate.surface_baseline",
 "value": 0.5867, "n_reports": 1200, "n_folds": 4}
~~~

OpenTelemetry is an **optional extra**, and deliberately so: Pharos has to run
offline and deterministically, so a missing collector or a missing dependency
degrades to silence rather than to an exception or a different number. There is a
test for exactly that. With  and  set,
spans and histograms export over OTLP and log lines gain  / , so
one identifier ties a generation to its gate and its permutation null.

~~~bash
export PHAROS_OTLP_ENDPOINT=http://localhost:4318
uv run --extra otel python -m pharos.cli gate
~~~

## Build order

Step 1, here, is the label algebra, generator, and gate. Still ahead:

2. **Tasks and scorers.** Task instances, the plant registry, and the four
   specialist scorers, with an adversarial-input pass over each scorer.
3. **Simulated analysts.** Persona-policy search, a simulator ensemble, and
   divergence reporting.

## License

MIT
