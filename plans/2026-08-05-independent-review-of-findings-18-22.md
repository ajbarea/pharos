# Independent review pass — findings 18-22 and the gate

Written 2026-08-05, to be run at the **start of a session**, not the end of one. The
reason is the whole point of the exercise: an author reviews what they intended, and
by the end of a long session I have re-read my own reasoning so many times that I am
the worst available reader of it. A cold context is the instrument.

## Why this needs to happen

Everything below was authored in one session, self-reviewed, and merged on a green
gate. Today alone, self-review missed:

- a `_stable_posterior` overflow present in **both** branches, which I had hand-checked
  and pronounced correct because I verified algebraic equality — the one property that
  cannot distinguish a stable form from an unstable one
- a scoring metric that measured its own denominator, making finding 21's repair claim
  false
- an orthogonality guard that could not fail, because its slack (1.5) always exceeded
  the gap it tested (1.25)
- a coverage gate that could not fail, unnoticed across every "gate green" claim I made
- three published thresholds that were reporting the anchor draw rather than the
  measurement
- finding 22's claim of early detection, which named shares the sweep had never
  visited: it measured 0, 5, 7 and 9, so its lowest non-zero point was already a
  majority of the fleet, and the prose claimed the range 1 through 9. Found by
  `/techne:docsync`, which asks only whether a claim traces to a measurement and does
  not need to understand the statistic to catch that it does not

Five of those six were found by a reader who was not me, by running the thing rather
than reading it, or by a check that ignored my reasoning entirely. That is the pattern
the review has to exploit. Note the shape of the last one: the claim was not wrong, it
was unmeasured, and re-running the sweep made it *stronger* (detection holds at one
blind analyst in nine). An unmeasured claim is not a claim that happens to be false; it
is one whose truth value nobody had, which is why "it turned out fine" is not evidence
the check was unnecessary.

## Scope

| Commit | What landed |
| --- | --- |
| `0b046ef` (#6) | Corrections from the first review pass, findings 18-21 |
| `4254a0a` (#7) | Coverage-gate fix, nested anchor draw, findings 19/20 corrections, finding 22 |
| `restamp-artifacts` (#8) | Artifact provenance re-stamp, finding 22 invariance tests |
| papers `#1` | Finding 22 into P3, two retractions propagated, LINEAGE + ALIGNMENT |

Review against `main` at the merge base of #6, **not** the `+/-` of each PR. Reviewing
a diff shows what changed; it does not show what the changed thing now does.

## How to run it

`/techne:elenchus` at `high`, which drives `/code-review` and adds the reproduce and
trace passes. It is token-heavy rather than billed, which is a reason not to fire it
casually, not a reason it needs a human to start it. The one thing that genuinely
cannot be self-started is `/code-review ultra`, the multi-agent cloud review; that is
user-triggered and billed. `ultra` is justified for the secagg and gate material if the
budget is there, and AJ has to start that one.

Failing either, dispatching independent reader agents directly covers the same five
questions, and is what was actually done on 2026-08-05.

Dispatch **two to three independent readers**, not one. They must not receive this
file's conclusions — hand them the code and the claims, not my account of them. Split
the five questions across them so no reader is asked to hold all five at once; the
questions are ordered by yield, not by affinity, so the split should follow method
(construct-and-run, trace-to-artifact, trace-call-sites) rather than question number.

## The five questions worth more than a general read

Ordered by where this project's bugs have actually come from.

### 1. Can each guard fail?

Every guard added in these commits, and for each: construct the input that trips it.
If the input cannot be constructed, the guard is decorative and the test asserting it
passes is worse than no test, because it reports coverage of a property nobody has.

Known instance: the orthogonality guard whose slack always exceeded the gap. Suspects
to check the same way — `CHANNEL_ENTANGLEMENT_SLACK`, `DETECTION_Z = 3.0`,
`MIN_PARTICIPANTS`, `CLIFF_GAP`, `REPAIRED = 0.95`, `--cov-fail-under=92`, and the
`should_fail_under` precision fix itself.

For the last one specifically: the fix is `precision = 2` in `pyproject.toml`. Confirm
by experiment, not by reading, that a run at 91.74% now exits non-zero. I have done
this once; a second reader doing it independently is cheap.

### 2. Does every published number come from an artifact?

Walk `docs/findings.md` and `papers/federated-forge/sections/*.tex` and, for each
numeral, find the field it was read from. A number with no field behind it is the
failure mode this project keeps hitting — the hand-typed 75-cell agreement grid, the
"20 of 20, 30 of 30" claim, the "half the round" phrasing that survived three
regenerations.

Nine blocks in `docs/` and seventeen tables in the paper are generated. Everything
*outside* those markers is prose written once beside a run, and prose does not
regenerate.

### 3. What else reports one draw as if it were a measurement?

The defect corrected in #7 was not local to `choose_anchors`. Ask of every seeded
quantity in `scripts/`: is this a sample, is it reported as a point, and would nesting
or re-seeding move it? `MASK_SEED`, `POLICY_SEED`, `ANCHOR_SEEDS`, `SEED = 7`, the
permutation nulls, `EQUIVALENCE_SEEDS`.

Finding 22's z is *exactly* invariant across blind shares. That is claimed as a
property of the construction rather than a coincidence. Verify the derivation
independently — if it is coincidence, the manuscript overclaims.

### 4. Trace every changed symbol across the whole repo

Not the diff — the callers. The makesense precedent is a feature that passed every test
and shipped dead because guidance landed on the inner function rather than the
registered wrapper.

Specifically: `choose_anchors` gained a raise and changed its distribution;
`evaluate` gained a `seed` parameter with a default; `Row` construction sites are a
known trap in this repo (`measure_secure_reliability.Row` carries a docstring warning
about exactly that, and a test still had to be caught by CI rather than by reading).

### 5. Are the parallel paths' guards mirrored?

Eight `sync_docs_tables` builders promise to refuse an empty artifact. Seven did;
`measurement_health` globbed instead and would have published "0 of 0 assessed
artifacts are flagged" over an empty directory. That was found by asserting the
promise, not by reading the code.

So: enumerate every family of near-identical functions across `scripts/` and check
that the guard *and its regression test* were mirrored to every member. Write the
input matrix and check every cell rather than spot-checking.

## What a good outcome looks like

Not "no issues." A review of five findings, a security-adjacent gate and two
retractions that returns clean is more likely to have been shallow than to have found
nothing. If it comes back empty, the next question is which of the five passes above
was not actually run.

## Explicitly out of scope

Style, naming, and comment density. The comments in this repo are deliberately long
and carry the reasons results were retracted; a reviewer trimming them is removing the
record. Point reviewers at behaviour.
