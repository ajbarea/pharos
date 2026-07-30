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

## Build order

Step 1, here, is the label algebra, generator, and gate. Still ahead:

2. **Tasks and scorers.** Task instances, the plant registry, and the four
   specialist scorers, with an adversarial-input pass over each scorer.
3. **Simulated analysts.** Persona-policy search, a simulator ensemble, and
   divergence reporting.

## License

MIT
