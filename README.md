<div align="center">

<a href="https://ajbarea.github.io/pharos/"><img src="docs/assets/pharos-hero.png" width="800" alt="Pharos Hero Image"></a>

# 🗼 Pharos

*A labeled fleet testbed for federated personalization with a governed disclosure boundary.*

[![CI](https://github.com/ajbarea/pharos/actions/workflows/ci.yml/badge.svg)](https://github.com/ajbarea/pharos/actions/workflows/ci.yml)
[![Documentation](https://github.com/ajbarea/pharos/actions/workflows/docs.yml/badge.svg)](https://github.com/ajbarea/pharos/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/ajbarea/pharos/graph/badge.svg?token=KC60fEY8dA)](https://codecov.io/gh/ajbarea/pharos)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Documentation](https://ajbarea.github.io/pharos/)** · **[Explorer](https://ajbarea.github.io/pharos/explorer/)**

</div>

---

Pharos supplies what public corpora do not: a body of reporting whose objects carry
real classification levels and cross-cutting compartments, so that two holders can be
**incomparable** rather than merely ranked, and the personal/shared split in
federated personalization can be **measured** rather than asserted.

## Why "Pharos"

The lighthouse at Alexandria. A watch station whose entire function was seeing what
was coming, and reporting it to whoever needed to know.

Real lighthouses use **sector lights**: different colours over different bearings, so
a mariner reading the colour knows which water is safe from where they stand. A label
here works the same way. Sensitivity says how far something travels, compartments say
along which bearings, and two holders in different sectors are incomparable rather
than ranked.

## Quickstart

```bash
make setup     # uv sync --all-groups
make test      # the suite, with a 92% branch-coverage floor
make gate      # generate a corpus and decide whether it is usable
```

`make gate` exits non-zero when the corpus is not usable, so it can block CI.

Python 3.12 to 3.14. Generation and gating need `numpy`, `scikit-learn` and the
OpenTelemetry API/SDK, which are runtime dependencies rather than an extra because a
published number has to be traceable to the run that produced it. A missing collector
degrades to silence, never to a different result. Everything else — the trainer, the
external validator, the explorer — is optional.

## What is here

**The corpus path**, offline and deterministic end to end:

| Module | Responsibility |
| --- | --- |
| `pharos.labels` | The product lattice: sensitivity, compartments, capacity. Joins, dominance, type-based declassification |
| `pharos.disclosure` | The release decision: release, needs-approval, or withhold, each with a reason |
| `pharos.world` | The fictional maritime watch: channels, officer voices, and the fact vocabulary |
| `pharos.scenario` | The world as configuration: load a different watch from TOML |
| `pharos.generate` | Deterministic corpus generation, reproducible from `(seed, config)` |
| `pharos.gate` | The shortcut gate: can plant membership be predicted without reading anything? |
| `pharos.manifest` | The citable record: version, seed, gate verdict, label histogram |

**Tasks and labelling:**

| Module | Responsibility |
| --- | --- |
| `pharos.tasks` | Task instances, and the governed label a verdict inherits from its sources |
| `pharos.analyst` | The reviewer as a specified policy: accept, revise, reject, and what each discloses |
| `pharos.detect` | Content-provenance labelling, the replacement for leave-one-out attribution |
| `pharos.attribute` | The only module that calls a model. Everything nondeterministic lives here |
| `pharos.models` | The model registry: what can be run, and what actually has been |
| `pharos.validity` | The conditions under which a score should not be quoted |
| `pharos.uncertainty` | Cluster-bootstrap intervals, and which estimand a deployment actually gets |

**The fleet, and what crosses between its nodes:**

| Module | Responsibility |
| --- | --- |
| `pharos.fleet` | Clearances, the contribution stream, and the attack that reads an analyst's compartments off it |
| `pharos.inference` | Truth inference from disagreeing contributors: Dawid-Skene, GLAD, CC-Rasch, and the variant that runs under a sum |
| `pharos.secagg` | Masked aggregation, and the server's view of a round as a value with no per-client field |
| `pharos.budget` | Privacy-budget composition over the distinguishing indicators, and the effective epsilon it buys |
| `pharos.fl` | Server-side aggregation rules, DP noise, and the attack surface they are measured against |
| `pharos.ledger` | The auditable decision record an edge node writes, and where it routes |
| `pharos.plants`, `pharos.scorers`, `pharos.adversarial` | The plant registry, the specialist scorers, and the perturbation primitives |

**Release and inspection:**

| Module | Responsibility |
| --- | --- |
| `pharos.provenance` | The stamp on every result: version, commit, and whether the tree was dirty |
| `pharos.export` | Writing a corpus out as JSON Lines, and hashing exactly what was written |
| `pharos.croissant` | Croissant metadata with the Responsible AI extension, emitted from the manifest |
| `pharos.telemetry` | Structured logs, spans, and the execution-context snapshot |
| `pharos.web` | The explorer: corpus, lattice, gate, a triage run, analyst review, and every exit priced, behind one page |
| `pharos.cli` | `gate`, `export`, `models`, `serve` |

**Model calls are confined to one module.** `pharos.attribute` is the only place a
model is contacted, so generation and gating stay reproducible and the acceptance
decision cannot drift with a model version.

That holds in practice, and the repeat we owed has now been run --- which corrected the
claim. The **corpus** is bit-identical on a WSL laptop and an RHEL 9 cluster node: the
SHA-256 of the serialized corpus agrees at every seed. The gate's **surface baseline**
is not. Two of seven seeds disagree, by 1.2e-05 and 6.5e-06, with numpy, scikit-learn,
scipy and the BLAS pinned to identical versions on both. The BLAS picks kernels by
processor --- AVX-512 on the Xeon, AVX2 on the Ryzen --- and that changes the reduction
order inside the probe fit.

The portable property is therefore the *verdict*, not the float: the largest
disagreement is four orders of magnitude below the gap between any baseline
(0.638--0.660) and the 0.720 acceptance ceiling, so no corpus changes status. Rerun it
with `scripts/measure_gate_determinism.py` and `cluster/gate-determinism.sbatch`, which
compare `float.hex()` and refuse outright unless both machines scored the same corpus.

## Documentation

| Page | What it covers |
| --- | --- |
| [Getting started](https://ajbarea.github.io/pharos/getting-started/) | Install, the everyday commands, reading a gate verdict, the explorer, observability |
| [Explorer](https://ajbarea.github.io/pharos/explorer/) | The corpus, the lattice, the gate, the analyst grid, and the argument the findings make, running with no backend |
| [The label lattice](https://ajbarea.github.io/pharos/reference/label-lattice/) | Dominance, joins, and why compartments make this a lattice rather than a ladder |
| [The shortcut gate](https://ajbarea.github.io/pharos/reference/gate/) | What the probe sees, the four rounds it rejected, and why it is a calibration instrument |
| [The release decision](https://ajbarea.github.io/pharos/reference/disclosure/) | Three dispositions, reason codes, purpose limitation, and the audited case table |
| [Reporting a noisy measurement](https://ajbarea.github.io/pharos/reference/uncertainty/) | Why a single pass is not a score, and what has to travel with one |
| [Corpus schema](https://ajbarea.github.io/pharos/reference/corpus-schema/) | The shape of an exported record |
| [Choosing a model](https://ajbarea.github.io/pharos/models/) | The registry, what `verified` means, and sweeping every installed model |
| [Releasing a corpus](https://ajbarea.github.io/pharos/releasing/) | Export, Croissant metadata, and provenance |
| [Running on a cluster](https://ajbarea.github.io/pharos/cluster/) | The RIT Research Computing path, and the traps that cost real time |
| [Findings](https://ajbarea.github.io/pharos/findings/) | Every measurement, its caveats, and the script that reproduces it |

[`RESEARCH.md`](RESEARCH.md) holds the survey behind that opening claim -- five
public corpora and where each stops, argued from the nearest candidates rather than
from an exhaustive search -- the public corpora Pharos should be used *alongside*,
and a verified citation for every external claim the design leans on.

Design specs live in the
[Federated Analyst Fleets](https://github.com/ajbarea/kourai-khryseai) research docs:
`docs/research/federated-forge/pharos-testbed.md` for this testbed, `index.md` for
the system it serves.

## What has been measured

Twenty-nine findings so far, each reproducible from a named script and each backed by a
committed artifact in `results/` that records the version, commit, platform, model,
and seed behind it. **They are provisional**: two of the first three did not survive
remeasurement at larger n, a third was retracted outright after a generator bug, and a
second generator defect in August 2026 moved every corpus-dependent number and withdrew
two more claims.
[Findings](https://ajbarea.github.io/pharos/findings/) carries the numbers, the
corrections, and the caveats.

| | Finding |
| --- | --- |
| 1 | Leave-one-out attribution cannot produce a correct governed label |
| 2 | Federation eligibility is bimodal on one policy ruling |
| 3 | A corpus bug, a retracted finding, and a real benchmark target |
| 3b | Over-escalation is universal, and scale does not fix it |
| 4 | Answerability and surface non-leakage pull against each other |
| 5 | The rule is not learnable from examples in the prompt |
| 6 | The rule *is* learnable by gradient descent, on clean labels |
| 7 | Review is abundant; what it costs is correctness, and the boundary needs an authority |
| 8 | Being right and sloppy beats being wrong and careful |
| 9 | A measurement that repeats one prompt measures the wrong thing |
| 10 | A fleet learns its analyst's standard, not the world's |
| 11 | The gate clears every item, and the stream still names the analyst |
| 12 | Reliability cannot be estimated without identity where it matters |
| 13 | A reliability tag can replace identity, and the leak metric cannot tell you when |
| 14 | What the agent costs on the hardware it is meant to run on |
| 15 | The standard privacy mechanism spends the budget on the wrong variable |
| 16 | The cliff is safe only because the fleets were drawn independently |
| 17 | Adding item difficulty does not separate a hard case from a wrong analyst |
| 18 | The estimate moves under secure aggregation, and the cliff does not move with it |
| 19 | An authority of record repairs the cliff, and its price explodes |
| 20 | Audit where the fleet splits, and the prediction that said otherwise |
| 21 | The corpus the audit policy cannot handle, built on purpose |
| 22 | The trace a blind spot leaves after it stops leaving disagreement |
| 23 | Once the detector names the channel, provenance finds the corrupted items that are findable at all |
| 24 | The crossing is a distribution over corpora, not a share |
| 25 | The cliff is not where we pointed the estimator, except at the crossing itself |
| 26 | Findings 20 to 23 were measured on one corpus, and one of their headlines was that corpus |
| 27 | Four guards could be inverted with the suite still green, and coverage called them all covered |
| 28 | The open problem, answered: detection converts into coverage, not into correction |
| 29 | The shape of the error is visible from the aggregate, and it is cheap to read wrong |
| 30 | The blind spot with no name is detected the same, and located by the same signal read one-sided |

The gate's calibration result is the one finding with support from outside this
generator: the same probe run against three public corpora exceeds its own
permutation null on every one of them.

```bash
make results     # regenerate the Ollama-backed measurements into results/
make review      # replay the committed verdicts past the analyst grid (no model)
make sweep       # target accuracy across the reviewer parameter grid (no model)
make power       # what each evaluation size can resolve (no model)
make linkage     # what the fleet's contribution stream leaks about analysts (no model)
make consensus   # whether reliability survives pooling contributors (no model)
make tagged      # whether a reliability tag can replace identity (no model)
make edge        # what the agent costs on laptop-class hardware (needs Ollama)
make budget      # what a privacy budget buys against the linkage channel (no model)
make correlated  # what the cliff costs when analysts are not independent (no model)
make difficulty  # whether item difficulty and a wrong standard are separable (no model)
make secure      # whether reliability can be estimated under secure aggregation (no model)
make authority   # what an authority of record costs, in audited items (no model)
make audit       # which items to rule on, and what a fallible authority buys (no model)
make blindspot   # where that policy stops working, built on purpose (no model)
make channel-bias # whether a unanimous blind spot leaves any trace at all (no model)
make selective-risk # what abstention buys once the audit budget buys nothing (no model)
make error-shape # whether the shape of the error is visible from the aggregate (no model)
make latent-blindspot # whether a blind spot with no nameable channel is findable (no model)

# Sensitivity: whether a finding survives a parameter nobody chose on principle
make fleet-sensitivity  # findings 12, 16 and 17 across fleets of 5 to 51 (no model)
make teacher-fleet      # whether adapters inherit their teachers, across 24 of them
```

## Build order

1. **Corpus and gate.** Done: the label algebra, the generator, and the acceptance
   gate.
2. **Tasks and scorers.** `pharos.tasks` carries the triage task and `pharos.detect`
   the labelling path. `pharos.plants`, `pharos.scorers` and `pharos.adversarial`
   supply the registry, the specialist scorers and the perturbation primitives. Still
   owed is the adversarial *evaluation*: the perturbations are tested to change the
   text they return while leaving ground truth alone, and nothing yet puts a perturbed
   prompt in front of a model and compares verdicts. A harness that appeared to do so
   was removed in August 2026 for reporting 1.0 at every seed by construction.
3. **Simulated analysts.** Done. This step existed to ask whether the rule survives
   being learned from a reviewer's decisions rather than from clean labels, and
   finding 10 answers it across 24 teachers crossing standard with carefulness: an
   adapter inherits a wrong standard almost exactly and cannot inherit a teacher's
   noise, so a wrong analyst yields a wrong agent while a merely careless one is
   partly repaired by training. Which of the two a teacher makes decides the sign:
   the correct standard is improved upon, the strictest is made worse. And fidelity
   to the teacher is not an acceptance criterion, because the adapter that reproduces
   its teacher perfectly scores below answering "escalate" to everything.
   `pharos.analyst` supplies the reviewer as a specified policy, and finding 7 reports
   what a review stream is worth: not scarce, but carrying the reviewer's standard
   rather than the world's,
   so a reviewer who over-escalates teaches targets below the majority floor. The
   compartment ruling *is* movable, by an authority rather than by the reviewer, at a
   cost of 52.5% of the stream.

4. **Fleet-level disclosure.** Done, and it produced the result the testbed exists
   for. A per-item gate does not compose: an attack reading no content recovers an
   analyst's compartment set from a stream of individually-approved items, at 0.820
   for the most highly cleared and 0.000 for everyone else (finding 11). Two escapes
   from the conflict that creates are now measured and closed. Estimating reliability
   from agreement needs no identity but fails once a wrong standard holds the majority
   (finding 12); a coarser reliability tag works, but only when tier is independent of
   clearance, and when it is not it discloses clearance on every record while the
   per-analyst metric reports zero (finding 13).

5. **Edge cost.** Done: finding 14 prices the agent on the 8 GB card the earlier
   findings already used. 57.1 MiB per personalization round, 3.7 s to wake a node,
   0.321 s per warm decision over 199 timed calls, 1.8 GB resident for the 3B model.

6. **Estimation under aggregation.** Done, and it answered the question this
   repository had carried as its open problem. Finding 10 needs contributor identity,
   finding 11 shows identity is the leak, and findings 12 and 13 rule out both ways of
   recovering the first from the second after the fact. The remaining direction was to
   compute the estimate *under* aggregation instead, and Dawid-Skene splits along
   exactly the seam a secure sum offers: its M step is per contributor and stays local,
   its E step is a product over contributors and so a sum in logs. Finding 18 reports
   the port as exact **to the fixed-point resolution the protocol quantizes at**
   (~1e-7 for nine contributors; measured at 3.8e-14 on this corpus, over ten
   wrong-standard compositions of one nine-analyst fleet, zero label disagreements) --
   and reports that **the cliff does not move with it**. That was the prediction, and it localises the failure to
   non-identifiability rather than to pooling or to a leak.

   What replaces the leak is smaller and task-shaped rather than person-shaped: a
   per-task mean needs its denominator, so the aggregate discloses how many analysts
   are cleared for each source-join, exactly and with no inference, on all 14 joins in
   the stream.

7. **The authority of record.** Done, and it is the mechanism step 3 owed. An
   exogenous label breaks the relabelling degeneracy no estimator escapes from the data
   alone. Finding 19 prices it, and the price is a second threshold rather than a rate:
   5 audited items in 200 repair a bare majority, 100 repair six of nine, 150 repair
   seven. A partial budget is briefly *worse* than none.

8. **Which items to rule on.** Done, and it refuted the prediction written into the
   script before the run. Finding 20 compares four deployable selection policies
   against an oracle bound, where deployable means reading only what the aggregator
   sees: per-task vote sums, contributor counts, and the estimator's posterior.
   Auditing where the fleet *splits* repairs six-of-nine at 20 items against uniform's
   45, and ties the oracle exactly. The prediction said the opposite would happen --
   that a confidently wrong majority would hide in unanimity -- and it was wrong,
   because reviewers differing by one escalation threshold diverge only on boundary
   items, which are exactly the items a fleet splits on. The inverted policy is not
   merely worse but *harmful*, driving agreement below the no-anchor baseline.

   Two things came free. Only **97 of 200** tasks are auditable at all, so finding 19
   spent about half its uniform budget on tasks that constrain nothing. And a fallible
   authority is robust: an auditor wrong one time in ten buys the same repair as a
   perfect one.

9. **Where that policy stops working.** Built, and it is finding 20's negative
   control. A reviewer who discounts a *channel* rather than misjudging a quantity
   corrupts a slice picked out by provenance instead of by difficulty -- 20 verdicts of
   200, every one on a task showing all three defining facts, which is the unambiguous
   end of the corpus. While a sighted minority remains the disagreement signal still
   points at them and every one of the policy's picks lands on a corrupted task. At
   unanimity the signal does not weaken, it ceases to exist, and the policy's hit rate
   falls to 0.15 against uniform's 0.10 -- chance, and `posterior` at 0.20 is no better
   in kind. The oracle still *selects* corrupted tasks at 1.000 on the same data, so
   what failed is the policy's ability to see them. (An earlier version added "so the
   tasks are still fixable"; that was withdrawn -- see the retraction in the findings
   page. Nothing in this experiment repairs anything.)

   Finding 20's advice is therefore **safe only inside its stated condition**, and
   falling outside it is not graceful degradation. A deployment cannot tell from the
   inside which regime it is in, because the observable that would distinguish them is
   the one that has gone.

10. **Detecting the regime.** Done, and it answers finding 21. Every signal up to that
   point was built from disagreement, and unanimity is the absence of disagreement --
   but the fleet's verdict *rate* is an observable too, and it can be conditioned on
   public structure. Testing whether the verdict rate is independent of a channel given
   the evidence count finds the discounted channel at the **attainable p-value floor of
   1/4201** at nine of nine blind, exactly where every audit policy scores at chance,
   with no other channel detected at any share and both negative controls silent.

   Two properties make it worth having. **Detection reaches down to one blind analyst
   in nine** on a noiseless fleet, and to four of nine at a realistic slip rate, so a
   house style is catchable before it becomes the majority, which is finding 16's
   warning. And it reads only per-task vote sums and public corpus structure, so it
   survives the secure-aggregation protocol of finding 18: no identity, no ground truth.

   What it cannot report is *extent*: the p-value floors, so the number of blind
   analysts is read off the stratified gap instead, which is linear in the share --
   -0.028 at one blind analyst through -0.250 at nine, in the ratio 1:9. An earlier
   version of this section reported a z-score, called its invariance across shares a
   finding, and quoted 8.1 against 1.8. That statistic was withdrawn on 2026-08-06: it
   is undefined exactly where the negative controls sit, because a noiseless fleet
   gives the null zero variance, so the controls could not have fired.

**What to do once it fires.** Answered by finding 28, and the answer is not a
correction. No budget repairs a label at unanimity, but a deployment can decline to
publish the slice the detector named: on the committed corpus that clears the corrupted
labels for a tenth of coverage, with no authority and no re-estimation, and it beats
every untargeted draw in all seven of the eight corpora that can host the experiment. It also inverts the rule a deployment
would otherwise reach for -- withholding where the fleet is least confident is *exact*
with one dissenter left and *at chance* at unanimity. The trade holds where the shared
error is most of the error and reverses where independent noise dominates, which is the
distinction the detector exists to draw.

**The open problem, withdrawn as stated, and what replaced it.** Finding 22 detects a
blind spot aligned with a known, public partition of the corpus, and this section used to
close by saying that a shared error following no observable partition would leave no trace
and that nothing here said how to find one. [Finding
30](https://ajbarea.github.io/pharos/findings/#30-the-blind-spot-with-no-name-is-detected-the-same-and-located-by-the-same-signal-read-one-sided)
built that error on purpose --- analysts who discount a set of individual reports rather
than a channel, drawn to corrupt the same 20 verdicts in the same evidence stratum --- and
neither half of the sentence held. The channel scan is silent on it, as expected. The index
of dispersion cannot tell it from the channel-keyed version at all: identical on
deterministic fleets, and inside the spread of five slice draws everywhere else. And
ranking tasks by the same statistic's per-task summands, read one-sided, recovers the
corrupted slice and ties an oracle with nothing named.

What is left is narrower and is a question about the world rather than about this
generator. The rule reads a *consistent depression* of a stratum's verdict rate, so it
needs the shared error to act the same way on the tasks it touches and to push in a known
direction --- the same one-sidedness finding 22's detector already assumes. Whether a real
watch floor's shared errors have that shape is a claim about operational data, and no
corpus generated here can settle it.

## License

MIT
