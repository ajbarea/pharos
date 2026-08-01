# The release decision

`pharos.disclosure`

Whether a derived output may leave, why, and when the honest answer is *ask
someone*. This page describes the decision; [the label lattice](label-lattice.md)
describes the algebra it runs on.

## Three dispositions, not two

The first version of this answered with a boolean:

```python
shared_eligible(label, ceiling, policy) -> bool
```

That binary quietly decided two findings. [Finding 2](../findings.md#2-the-design-is-bimodal-on-one-policy-ruling)
reported that eligibility is bimodal on the compartment ruling, and
[finding 7](../findings.md#7-review-is-abundant-what-it-costs-is-correctness)
reported that no fail-closed reviewer could make a blocked verdict releasable. Both
are true of a system with two doors, and both were measured on one that had exactly
two because that is how the function was written.

| Disposition | Meaning |
| --- | --- |
| `RELEASE` | Leaves with no further authority. The old boolean's true case. |
| `APPROVAL` | Blocked, but by a ruling somebody is entitled to make. |
| `WITHHOLD` | Blocked, and nobody may lift it. |

The middle one is the addition, and it is borrowed rather than invented: the
makesense policy engine distinguishes `auto_ok` / `needs_approval` / `blocked`, and
the Metro City Meteors governance suite five outcomes. Pharos had no counterpart to
the middle, and finding 7's release claim was the cost of that gap.

## Why a compartment shortfall is authorizable and a level is not

`APPROVAL` is offered for exactly one cause, and that is a design claim rather than
a convenience.

Finding 2 concluded that shedding a compartment is **a policy act rather than an
engineering problem**. A policy act is precisely what a human with authority can
perform and a rule cannot, so a compartment the ceiling does not hold is the one
objection an authority can answer.

Nothing else is like that. A level above the ceiling is not downgraded by being
asked. A capacity that can carry an instance verbatim is an objection to the *shape*
of the output, fixed by changing the shape rather than by finding someone senior. A
purpose the data's owners have ruled out is not a clearance question at all.

## Reason codes

Every decision carries one. A caller told only *blocked* cannot tell a compartment
it might get cleared for from a prohibition it will never get past -- which is the
same defect finding 7 measured on the review side, one layer down.

| Reason | Disposition | Fixed by |
| --- | --- | --- |
| `RELEASABLE` | `RELEASE` | -- |
| `COMPARTMENT_NOT_HELD` | `APPROVAL` | someone entitled to rule on the compartment |
| `LEVEL_ABOVE_CEILING` | `WITHHOLD` | a different ceiling, or a different ruling |
| `CAPACITY_NOT_DECLASSIFIABLE` | `WITHHOLD` | emitting a lower-capacity output |
| `PURPOSE_PROHIBITED` | `WITHHOLD` | nothing; the owners ruled it out |

Checks run **most-binding first**. A label failing on several counts reports the one
that cannot be lifted rather than the one that can, because naming the authorizable
objection would invite an escalation that must still be refused.

## Purpose is a separate axis

The lattice answers *who may read*. It does not answer what the reading may be
**for**, and those come apart:

```python
from pharos.disclosure import ProhibitedUse, Purpose, ReleasePolicy, decide

policy = ReleasePolicy(
    declassification=DROP_COMPARTMENTS,
    prohibited=frozenset({ProhibitedUse(Compartment.LEGAL, Purpose.FLEET_TRAINING)}),
)

decide(label, ceiling, policy, purpose=Purpose.FLEET_TRAINING).disposition  # withhold
decide(label, ceiling, policy, purpose=Purpose.INCIDENT_REVIEW).disposition  # release
```

Same label, same ruling, opposite outcomes. A prohibition that fired on every purpose
would be a clearance rule wearing a different name, and the case table asserts that
pair so it stays true.

## Two functions, and the difference matters

`decide` starts from a **source** label and applies a declassification ruling to it.
`admit` judges a label that has **already** been derived.

Confusing them is not harmless. Running a ruling over an already-released label
declassifies it twice, and a permissive reviewer would silently shed compartments
the proposer deliberately kept. That bug was written during this module's
development and caught by a test; `pharos.analyst` calls `admit`, because a proposal
handed to a reviewer has been through declassification already.

## The case table

`src/pharos/cases/disclosure.json` holds one worked example per gate, each with the
label, ceiling, ruling and purpose that produce it, its expected `(disposition,
reason)`, and a written pass criterion. Modelled on the Meteors governance suite,
where the subject under test is the governed decision rather than a model.

It is data rather than assertions buried in a test so a reader can audit the policy
without running anything. Three tests keep it honest: every case must be marked
`verified`, every reason code must be exercised by some case, and every case must
state both the gate it tests and its criterion.

```bash
uv run python -m pytest tests/test_disclosure.py -q
```

## What this does not do

There is no approval *workflow* here -- no queue, no authority, no record of a
ruling once made. `APPROVAL` is a statement about what kind of block this is, not a
mechanism for lifting it. Finding 7 measures the load such a mechanism would carry
(52.5% of the review stream under the fail-closed default); building it is step 3
work that has not been done.
