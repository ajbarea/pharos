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

Python 3.12 to 3.14. Generation and gating need only `numpy` and `scikit-learn`.

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

**Release and inspection:**

| Module | Responsibility |
| --- | --- |
| `pharos.provenance` | The stamp on every result: version, commit, and whether the tree was dirty |
| `pharos.export` | Writing a corpus out as JSON Lines, and hashing exactly what was written |
| `pharos.croissant` | Croissant metadata with the Responsible AI extension, emitted from the manifest |
| `pharos.telemetry` | Structured logs, spans, and the execution-context snapshot |
| `pharos.web` | The explorer: corpus, lattice, gate, a triage run, and analyst review behind one page |
| `pharos.cli` | `gate`, `export`, `models`, `serve` |

**Model calls are confined to one module.** `pharos.attribute` is the only place a
model is contacted, so generation and gating stay reproducible and the acceptance
decision cannot drift with a model version.

That holds in practice, not only in intention: the gate produces **bit-identical**
surface baselines on a WSL laptop and on an RHEL 9 cluster node with a different CPU
count, kernel, and libc -- 0.6547, 0.6588, and 0.6675 on seeds 1, 7, and 101.

## Documentation

| Page | What it covers |
| --- | --- |
| [Getting started](https://ajbarea.github.io/pharos/getting-started/) | Install, the everyday commands, reading a gate verdict, the explorer, observability |
| [Explorer](https://ajbarea.github.io/pharos/explorer/) | The corpus, the lattice, the gate, and the analyst grid, running with no backend |
| [The label lattice](https://ajbarea.github.io/pharos/reference/label-lattice/) | Dominance, joins, and why compartments make this a lattice rather than a ladder |
| [The shortcut gate](https://ajbarea.github.io/pharos/reference/gate/) | What the probe sees, the four rounds it rejected, and why it is a calibration instrument |
| [The release decision](https://ajbarea.github.io/pharos/reference/disclosure/) | Three dispositions, reason codes, purpose limitation, and the audited case table |
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

Eight findings so far, each reproducible from a named script and each backed by a
committed artifact in `results/` that records the version, commit, platform, model,
and seed behind it. **They are provisional**: two of the first three did not survive
remeasurement at larger n, and a third was retracted outright after a generator bug.
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

The gate's calibration result is the one finding with support from outside this
generator: the same probe run against three public corpora exceeds its own
permutation null on every one of them.

```bash
make results     # regenerate the Ollama-backed measurements into results/
make review      # replay the committed verdicts past the analyst grid (no model)
```

## Build order

1. **Corpus and gate.** Done: the label algebra, the generator, and the acceptance
   gate.
2. **Tasks and scorers.** `pharos.tasks` carries the triage task and `pharos.detect`
   the labelling path. Still owed are the plant registry, the remaining specialist
   scorers, and an adversarial-input pass over each.
3. **Simulated analysts.** `pharos.analyst` supplies the reviewer as a specified
   policy and finding 7 reports what a review stream is worth: not scarce, but
   carrying the reviewer's standard rather than the world's, so a reviewer who
   over-escalates teaches targets below the majority floor. The compartment ruling
   *is* movable, by an authority rather than by the reviewer, at a cost of 52.5% of
   the stream. Still owed is the experiment that decides the premise: **a learner
   trained on those decisions rather than on clean labels.** Finding 6 used the
   generator's ground truth; whether the rule survives a reviewer's standard is the
   question the fleet actually turns on. This step remains the critical path.

## License

MIT
