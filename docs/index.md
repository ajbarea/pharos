# Pharos

A labeled fleet testbed for federated personalization with a governed disclosure
boundary.

Pharos supplies what public corpora do not: a body of reporting whose objects
carry real classification levels **and** cross-cutting compartments, so that two
holders can be **incomparable** rather than merely ranked, and the personal/shared
split in federated personalization can be measured rather than asserted.

## The problem it exists for

Federated personalization splits what a model learns into a part that stays local
and a part that is shared. Deciding *which* knowledge goes where is a disclosure
question, and answering it requires data whose objects carry disclosure labels.

Public corpora do not. At best they carry a single sensitivity ladder, where every
label is comparable to every other and a join is a maximum. Real disclosure policy
is not a ladder. Two holders at the same level with different need-to-know
compartments dominate each other in neither direction, and it is precisely that
incomparability that makes a boundary hard to enforce and interesting to measure.

So Pharos generates a world where labels have that structure, and where ground
truth for the analytic task is defined by content rather than by an insertion
artifact.

## What is here

| Module | Responsibility |
| --- | --- |
| [`pharos.labels`](reference/label-lattice.md) | The product lattice: sensitivity, compartments, capacity. Joins, dominance, type-based declassification |
| `pharos.world` | The fictional maritime watch: channels, officer voices, and the fact vocabulary |
| `pharos.generate` | Deterministic corpus generation, reproducible from `(seed, config)` |
| [`pharos.gate`](reference/gate.md) | The shortcut gate: can plant membership be predicted without reading anything? |
| `pharos.manifest` | The citable record: version, seed, gate verdict, label histogram |
| `pharos.provenance` | The stamp on every result: version, commit, and whether the tree was dirty |
| [`pharos.export`](reference/corpus-schema.md) | Writing a corpus out, and hashing exactly what was written |
| [`pharos.croissant`](releasing.md) | Croissant metadata with the Responsible AI extension |

Everything in the generation and gating path is offline and deterministic. There
are no model calls in it, which is what makes the gate reproducible. Model calls
appear only in the measurement scripts under `scripts/`, and those record which
model produced each number.

## Why "Pharos"

The lighthouse at Alexandria. A watch station whose entire function was seeing
what was coming, and reporting it to whoever needed to know.

## Where the argument lives

This site is the **reference**: how the pieces work and how to run them. The
argument Pharos supports belongs to the manuscript, not here.
[Findings](findings.md) summarises what has been measured and points at the
scripts that reproduce each number. `RESEARCH.md` in the repository root holds
what is true *outside* the repo: the survey behind "no public corpus does this",
the public corpora Pharos is meant to be used alongside, and a verified citation
for every external claim the design leans on.

Design specs live in the Federated Analyst Fleets research docs
(`kourai-khryseai/docs/research/federated-forge/`): `pharos-testbed.md` for this
testbed, `index.md` for the system it serves.
