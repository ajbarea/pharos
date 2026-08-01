# The label lattice

```text
label = (sensitivity, compartments, capacity)

sensitivity  : OPEN < INTERNAL < PROTECTED < RESTRICTED    total order, join = max
compartments : subset of {SENSOR, LIAISON, LEGAL, PARTNER} subset lattice, join = union
capacity     : ENUM | SCALAR | SPAN | FREETEXT             the form of a derived output
```

Sensitivity is a ladder. Compartments are not. Two officers at `RESTRICTED` with
incomparable compartment sets dominate in neither direction, which is what makes
this a **lattice** test rather than a ladder test, and it is the property none of
the public corpora surveyed in `RESEARCH.md` supplies.

## Dominance

```python
from pharos.labels import Capacity, Compartment, Label, Sensitivity

holder = Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
item = Label(Sensitivity.INTERNAL, frozenset({Compartment.PARTNER}), Capacity.FREETEXT)

holder.dominates(item)  # False: outranks it, but lacks PARTNER
```

`dominates` requires **both** a level at least as high and a compartment set at
least as large. Outranking something is not sufficient to read it.

## Join

```python
from pharos.labels import join

join(source_labels, capacity=Capacity.ENUM)
```

The join takes the maximum sensitivity and the union of compartments, which is
the conservative thing to do and the only safe thing to do.

`capacity` is a **required keyword** rather than something joined from the
inputs, and that asymmetry is the answer to label creep. An entry's label is the
join over every object that fed the turn. If capacity were joined too, every
derived entry would climb to the top of the lattice, nothing would be releasable,
and federation would degrade to local-only learning. Capacity is a property of
the *form of the output*: an enum verdict is an enum verdict however sensitive
its inputs were.

## Declassification

```python
from pharos.labels import DeclassificationPolicy, declassify, shared_eligible

policy = DeclassificationPolicy()  # the fail-closed default
declassify(label, policy)
shared_eligible(label, release_ceiling, policy)
```

`DeclassificationPolicy` has three fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `declassifiable` | `{ENUM, SCALAR}` | Which capacities are eligible for release at all |
| `release_floor` | `OPEN` | The level an eligible output drops to |
| `drop_compartments` | `False` | Whether release also sheds compartments |

It fails closed twice over.

**A capacity the policy does not name is never released.** An unknown or
unlisted capacity returns the label unchanged rather than falling through to a
permissive branch.

**Compartments survive by default.** Shedding a compartment discloses that the
compartment had something to say, which is a disclosure in its own right. So
dropping one is a deliberate policy act, never an inference from low capacity.

`shared_eligible` is exactly dominance after declassification: an entry may train
a shared adapter when the aggregator's release ceiling dominates what the entry
declassifies to.

## Why `drop_compartments` is the load-bearing setting

That default is not a detail, and measurement says so. Across three aggregator
ceilings and four capacities at 40 turns, turns average 2.15 compartments of 4,
and most already sit high on the level ladder, because a summary over eight
sources joins nearly everything.

| Declassification policy | FREETEXT | SPAN | SCALAR | ENUM |
| --- | --- | --- | --- | --- |
| keep compartments (fail-closed default) | 0-38% | 0-38% | 0-38% | 0-38% |
| drop compartments for low capacity | 0-38% | 0-38% | **100%** | **100%** |

Answer "no, a low-capacity verdict may not shed its sources' compartments" and
the fleet is a set of unconnected local learners. Answer "yes" and verdict-shaped
outputs federate completely while prose never does.

Reproduce with `scripts/measure_federation_eligibility.py`. See
[Findings](../findings.md).
