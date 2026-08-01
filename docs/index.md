---
title: Pharos
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![Pharos](assets/pharos-hero.png#only-dark){ alt="A lighthouse casting three coloured sector beams into the dark" }
![Pharos](assets/pharos-hero-light.png#only-light){ alt="A lighthouse casting three coloured sector beams across a pale sky" }

# Pharos

A labeled fleet testbed for federated personalization with a **governed disclosure boundary**.
{ .hero-subtitle }

<div class="hero-buttons" markdown>

[Get started](getting-started.md){ .md-button .md-button--primary }
[What has been measured](findings.md){ .md-button }

</div>

<div class="hero-beams" markdown>

<span class="beam-amber">Sensitivity</span> &nbsp;·&nbsp; <span class="beam-cyan">Compartments</span> &nbsp;·&nbsp; <span class="beam-magenta">Capacity</span>

</div>

</div>

<div class="scroll-hint" aria-hidden="true">
  <div class="scroll-chevron"></div>
</div>

<div class="landing-section" markdown>

## The problem it exists for { .section-title }

Federated personalization splits what a model learns into a part that stays local and a part that is shared. Deciding *which* knowledge goes where is a disclosure question, and answering it requires data whose objects carry disclosure labels.
{ .section-lead }

Public corpora do not. At best they carry a single sensitivity ladder, where every
label is comparable to every other and a join is a maximum. Real disclosure policy
is not a ladder. Two holders at the same level with different need-to-know
compartments dominate each other in neither direction, and it is precisely that
incomparability that makes a boundary hard to enforce and interesting to measure.

So Pharos generates a world where labels have that structure, and where ground
truth for the analytic task is defined by content rather than by an insertion
artifact.

</div>

<div class="landing-section" markdown>

## What is here { .section-title }

| Module | Responsibility |
| --- | --- |
| [`pharos.labels`](reference/label-lattice.md) | The product lattice: sensitivity, compartments, capacity. Joins, dominance, type-based declassification |
| `pharos.world` | The fictional maritime watch: channels, officer voices, and the fact vocabulary |
| `pharos.scenario` | The world as configuration: load a different watch from TOML |
| `pharos.generate` | Deterministic corpus generation, reproducible from `(seed, config)` |
| [`pharos.gate`](reference/gate.md) | The shortcut gate: can plant membership be predicted without reading anything? |
| `pharos.manifest` | The citable record: version, seed, gate verdict, label histogram |
| `pharos.tasks` | Task instances, and the governed label a verdict inherits from its sources |
| `pharos.detect` | Content-provenance labelling, the replacement for leave-one-out attribution |
| `pharos.attribute` | The only module that calls a model |
| [`pharos.models`](models.md) | The model registry: what can be run, and what actually has been |
| `pharos.validity` | The conditions under which a score should not be quoted |
| `pharos.provenance` | The stamp on every result: version, commit, and whether the tree was dirty |
| [`pharos.export`](reference/corpus-schema.md) | Writing a corpus out, and hashing exactly what was written |
| [`pharos.croissant`](releasing.md) | Croissant metadata with the Responsible AI extension |
| `pharos.telemetry` | Structured logs, spans, and the execution-context snapshot |
| `pharos.web` | The explorer: corpus, lattice, gate, and a triage run behind one page |
| `pharos.cli` | `gate`, `export`, `models`, `serve` |

Everything in the generation and gating path is offline and deterministic. There
are no model calls in it, which is what makes the gate reproducible. Model calls
appear only in the measurement scripts under `scripts/`, and those record which
model produced each number.

</div>

<div class="landing-section" markdown>

## Why "Pharos" { .section-title }

The lighthouse at Alexandria. A watch station whose entire function was seeing what was coming, and reporting it to whoever needed to know.
{ .section-lead }

The hero image is not only decoration. Real lighthouses use **sector lights**:
different colours over different bearings, so a mariner reading the colour knows
which water is safe from where they are standing. A label in Pharos works the same
way. Sensitivity says how far it travels, compartments say along which bearings, and
two holders in different sectors are incomparable rather than ranked.

</div>

<div class="landing-section" markdown>

## Where the argument lives { .section-title }

This site is the **reference**: how the pieces work and how to run them. The argument Pharos supports belongs to the manuscript, not here.
{ .section-lead }

[Findings](findings.md) summarises what has been measured and points at the scripts
that reproduce each number. `RESEARCH.md` in the repository root holds what is true
*outside* the repo: the survey behind "no public corpus does this", the public
corpora Pharos is meant to be used alongside, and a verified citation for every
external claim the design leans on.

Design specs live in the Federated Analyst Fleets research docs
(`kourai-khryseai/docs/research/federated-forge/`): `pharos-testbed.md` for this
testbed, `index.md` for the system it serves.

</div>
