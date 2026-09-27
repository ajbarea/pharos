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

<span class="pharos-chip pharos-chip-amber">Sensitivity</span> <span class="pharos-chip pharos-chip-cyan">Compartments</span> <span class="pharos-chip pharos-chip-magenta">Capacity</span> <span class="pharos-chip pharos-chip-rust">Velocity-FL Rust</span>

</div>

</div>

<div class="scroll-hint" aria-hidden="true">
  <div class="scroll-chevron"></div>
</div>

<div class="landing-section" markdown>

## What is Pharos? { .section-title }

Federated personalization splits what a model learns into **local knowledge** and **shared knowledge**. Deciding which stays local is a disclosure problem, and public corpora cannot test it: their labels form a single sensitivity ladder, while real disclosure policy is not a ladder.
{ .section-lead }

Pharos generates synthetic corpora whose labels carry a lattice, so two holders at the same level with different need-to-know compartments are **incomparable**, and measures privacy leakage, over-escalation and personalization against content-defined ground truth.
{ .section-lead }

</div>

<div class="landing-section" markdown>

## From seed to fleet { .section-title }

<div class="pipeline-flow" markdown>

<div class="pipeline-step" style="--step-accent: var(--pharos-amber)" markdown>
:material-dice-multiple-outline:{ .step-icon }
<span class="step-label">Generate</span>
</div>

<div class="pipeline-step" style="--step-accent: var(--pharos-cyan)" markdown>
:material-tag-multiple-outline:{ .step-icon }
<span class="step-label">Label</span>
</div>

<div class="pipeline-step" style="--step-accent: var(--pharos-magenta)" markdown>
:material-target:{ .step-icon }
<span class="step-label">Gate</span>
</div>

<div class="pipeline-step" style="--step-accent: var(--pharos-amber)" markdown>
:material-call-split:{ .step-icon }
<span class="step-label">Route</span>
</div>

<div class="pipeline-step" style="--step-accent: var(--pharos-cyan)" markdown>
:material-account-lock-outline:{ .step-icon }
<span class="step-label">Personalize</span>
</div>

<div class="pipeline-step" style="--step-accent: var(--pharos-magenta)" markdown>
:material-sigma:{ .step-icon }
<span class="step-label">Aggregate</span>
</div>

</div>

A seeded corpus, labelled on the lattice, checked by the shortcut gate, routed item by item to a personal adapter that never leaves its holder or to a shared fleet adapter, then aggregated by a robust rule. [The architecture](architecture.md) walks through each stage and every module.
{ .pipeline-caption }

</div>

<div class="landing-section" markdown>

## Explore { .section-title }

<div class="feature-grid">
  <a href="getting-started/" class="feature-card" style="--card-accent: var(--pharos-amber)">
    <span class="feature-icon" style="mask-image: url(assets/icons/rocket-launch-outline.svg)" aria-hidden="true"></span>
    <span class="feature-name">Getting started</span>
    <p>Install, generate a corpus, and read the gate's verdict.</p>
  </a>
  <a href="explorer/" class="feature-card" style="--card-accent: var(--pharos-cyan)">
    <span class="feature-icon" style="mask-image: url(assets/icons/compass-outline.svg)" aria-hidden="true"></span>
    <span class="feature-name">Visual Explorer</span>
    <p>The corpus, the lattice, gate rounds and triage review, in the browser.</p>
  </a>
  <a href="findings/" class="feature-card" style="--card-accent: var(--pharos-magenta)">
    <span class="feature-icon" style="mask-image: url(assets/icons/chart-box-outline.svg)" aria-hidden="true"></span>
    <span class="feature-name">Findings</span>
    <p>Every measured result, and the script that reproduces it.</p>
  </a>
  <a href="architecture/" class="feature-card" style="--card-accent: var(--pharos-amber)">
    <span class="feature-icon" style="mask-image: url(assets/icons/sitemap-outline.svg)" aria-hidden="true"></span>
    <span class="feature-name">Architecture</span>
    <p>The problem, the pipeline, and what each module does.</p>
  </a>
  <a href="reference/label-lattice/" class="feature-card" style="--card-accent: var(--pharos-cyan)">
    <span class="feature-icon" style="mask-image: url(assets/icons/book-open-variant-outline.svg)" aria-hidden="true"></span>
    <span class="feature-name">Reference</span>
    <p>The label lattice, the gate, the release decision and the governance kit.</p>
  </a>
  <a href="releasing/" class="feature-card" style="--card-accent: var(--pharos-magenta)">
    <span class="feature-icon" style="mask-image: url(assets/icons/package-variant-closed.svg)" aria-hidden="true"></span>
    <span class="feature-name">Releasing a corpus</span>
    <p>Export a release with its manifest and Croissant metadata.</p>
  </a>
</div>

</div>
