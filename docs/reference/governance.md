# Governance: fleets, the aggregator's view, and what may act on it

`pharos.governance` is the kit findings 18 through 29 are measured with: a fleet of
simulated analysts, the aggregate a server actually holds, the rules that may select or
withhold from it, and the statistics that read its shape.

It is separate from the corpus and the gate on purpose. The corpus is what you measure
*on*; this is what you measure *with*, and it is the part you need if you want to price
your own selection rule rather than reproduce ours.

---

## The line this package draws

Every rule here is a function of a `ServerObservation`, and that type is the whole
constraint:

```python
from pharos.governance import ServerObservation, observe

view = observe(partitioned)  # per-task vote sums, contributor counts, EM posterior
view.votes["T-0007"]  # how many analysts called it significant
view.seen["T-0007"]  # how many reported on it at all
view.margin("T-0007")  # 0.0 is a dead heat, 1.0 is unanimity
```

| Field | What it is | Why a policy may read it |
| :--- | :--- | :--- |
| `votes`, `seen` | per-task sums | the secure-aggregation protocol already reveals them |
| `posterior` | the estimator's own output | the server holds it by construction |
| `evidence` | how much evidence a task shows | public corpus structure |
| `carries` | whether a task carries a named channel | public, and empty until a detector names one |

!!! danger "There is no per-analyst field, and that is the point"
    A rule that can see who said what is not a rule, it is an oracle wearing one's name.
    The absence is asserted on `__slots__` in the test suite rather than left to this
    page, because a measurement that quietly reads a per-analyst stream produces a
    publishable-looking number that no deployment can reproduce.

---

## Selecting, and withholding

Five policies plus a bound. `uniform` is the honest floor, `margin` and `posterior` are
uncertainty sampling against the votes and against the estimator, `consensus` is its
inversion, `channel` selects by provenance once a detector has named one, and `oracle`
reads ground truth and is never proposed.

```python
from pharos.governance import select, score, REPORT_BUDGET

held = select("channel", view, truth, REPORT_BUDGET, seed=4242)
cell = score(n_blind=9, slip_rate=0.0, policy="channel", withheld=held, pool=pool, wrong=wrong)
cell.risk  # errors among labels still published
cell.coverage  # how many labels are published at all
```

!!! warning "Read the two numbers together or not at all"
    Withholding removes errors by removing labels. Reporting `risk` without `coverage`
    turns deletion into an apparent repair, which is a mistake this project published
    twice before the `Outcome` split (`mechanical` versus `corrected`) made it visible.

`beats_every_draw` is the comparison to use against `uniform`: a targeted rule must beat
the *best* of the untargeted draws, not their median. A policy that beats the median of a
variable baseline has been credited with the draw.

---

## Reading the shape of the error

```python
from pharos.governance import dispersion

spread = dispersion(view, evidence, draws=2000, seed=7)
spread.index  # ~1 under independent error, above 1 when part of it is shared
spread.p_value  # against a binomial null simulated at each stratum's own rate
```

Within an evidence stratum a fleet applying one rule votes identically, so independent
slips are exactly binomial while a shared standard splits the stratum into two
deterministic groups. The index needs strictly less than a channel detector does: no
channel to name, no per-analyst stream, no ground truth, no healthy fleet to compare
against.

!!! info "Undefined is not clean"
    A fleet that never disagrees leaves no variance to compare against, and `index` is
    `None` rather than low. Reporting that as "no shared component" would assert
    independence on the strength of no evidence.

!!! danger "A simulated p-value floors at 1/(m+1)"
    That floor must sit **below** the alpha it is compared against, or the test reports
    "not significant" for every input including the ones it was built to catch. The
    measurement asserts the arithmetic at run time; a caller lowering `draws` must do the
    same.

---

## When a corpus cannot host the experiment

```python
from pharos.governance import assert_channel_usable, ChannelUnusableError

try:
    check = assert_channel_usable(tasks)
except ChannelUnusableError as refusal:
    ...  # this draw cannot support a blind-spot experiment
```

A blind-spot experiment needs a channel whose evidence is not entangled with item
difficulty; on some corpus draws none exists. The library raises, and each measurement
script translates that into exit code `3` — the code a sweep matches on to tell a draw
that *cannot host* an experiment from a draw that *crashed*. Those are different results,
and only one of them belongs in a denominator.
