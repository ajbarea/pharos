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

## Known gap

**The corpus does not yet pass the gate.** Mean AUC sits around 0.55 to 0.58
against a band of 0.45 to 0.55.

The diagnosis is specific. Gradient boosting is clean at 0.51 to 0.53 while the
linear probe holds 0.54 to 0.58, and word count and digit width are now uniform
across renderings. That points at character-level lexical length: the same number
of words, made of different-length words. Closing it needs a character-count
normalization pass over the fact vocabulary, the same shape of fix as rounds 1
and 4.

Two things follow from this, both deliberate:

- `tests/test_gate.py` carries a **regression bound** at 0.60 that locks in the
  reduction from 0.737, plus an `xfail` for the 0.55 target so the gap is visible
  in test output rather than buried.
- `Manifest.usable` reports `False`. Nothing certifies this corpus, and the
  strict band has not been widened to manufacture a pass. A corpus tuned until
  its number looked good is exactly the failure the gate exists to prevent.

## Build order

Step 1, here, is the label algebra, generator, and gate. Still ahead:

2. **Tasks and scorers.** Task instances, the plant registry, and the four
   specialist scorers, with an adversarial-input pass over each scorer.
3. **Simulated analysts.** Persona-policy search, a simulator ensemble, and
   divergence reporting.

## License

MIT
