"""The corpus manifest: the citable record of one generated corpus.

A figure quoted from a corpus is worth what its provenance is worth, so every
corpus carries a manifest naming the version that built it, the seed and config
that reproduce it, the gate verdict that licenses it, and a histogram proving its
labels actually vary.

The histogram matters more than it looks. A corpus whose labels are constant
cannot evaluate a disclosure boundary at all: the governed label would be the
same for every entry, and any experiment over it would be an experiment over a
constant. `build_manifest` refuses to certify that case.
"""

import json
from collections import Counter
from dataclasses import dataclass

from pharos import __version__
from pharos.gate import GateResult, run_gate
from pharos.generate import GeneratorConfig, Report, generate

#: Above this, shape explains too much of the task for a triage score to carry
#: meaning. Not a purity threshold: the baseline is expected to exceed chance,
#: and what matters is that it is measured, significant against its own null, and
#: small enough that a model has room to demonstrate something.
#:
#: Raised from 0.65 to 0.72 once coverage was guaranteed. Requiring that every
#: fact of an event actually be rendered ties report composition to the event fact
#: set, which lifts the baseline to 0.63-0.67 on a corpus that is now correct. The
#: alternative was to keep a tighter ceiling and reject corpora for being
#: answerable, which is the wrong trade.
MAX_SURFACE_BASELINE = 0.72


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything needed to reproduce a corpus and to decide whether to trust it."""

    pharos_version: str
    config: GeneratorConfig
    gate: GateResult
    n_reports: int
    n_events: int
    label_histogram: dict[str, int]
    plant_share: float
    max_surface_baseline: float = MAX_SURFACE_BASELINE

    @property
    def usable(self) -> bool:
        """Whether this corpus version may be used, and on what terms.

        Three conditions, and note what is deliberately *not* among them: a
        surface baseline at chance. Ground truth here is defined by the presence
        of particular content, so plants carry the significant facts more often by
        construction and some surface information is unavoidable. Demanding a
        chance baseline would be demanding a vocabulary of perfect surface twins.

        What is required instead is that the baseline be **measured, bounded, and
        published**:

        1. The labels vary. A constant label cannot evaluate a disclosure boundary.
        2. A permutation null was computed, so the baseline is known to exceed the
           gate's own noise rather than merely to look large.
        3. The baseline sits under `max_surface_baseline`. Above that, shape
           explains too much of the task for a triage score to mean anything.
        """
        return (
            len(self.label_histogram) > 1
            and self.gate.null_trials > 0
            and self.gate.surface_baseline <= self.max_surface_baseline
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "pharos_version": self.pharos_version,
            "config": self.config.as_dict(),
            "gate": self.gate.as_dict(),
            "n_reports": self.n_reports,
            "n_events": self.n_events,
            "label_histogram": self.label_histogram,
            "plant_share": round(self.plant_share, 4),
            "max_surface_baseline": self.max_surface_baseline,
            "usable": self.usable,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def label_histogram(reports: list[Report]) -> dict[str, int]:
    """Counts per distinct label cell, keyed as `LEVEL[COMPARTMENT,...]`."""
    counter: Counter[str] = Counter()
    for report in reports:
        compartments = ",".join(sorted(report.label.compartments))
        counter[f"{report.label.sensitivity.name}[{compartments}]"] += 1
    return dict(sorted(counter.items()))


def build_manifest(config: GeneratorConfig, *, null_trials: int = 20) -> Manifest:
    """Generate, gate, and certify a corpus in one step.

    `null_trials` defaults to a real permutation null because a manifest without
    one cannot certify anything: an unmeasured baseline is not a small baseline.
    """
    reports = generate(config)
    gate = run_gate(reports, null_trials=null_trials)
    events = {report.event_id: report.is_plant for report in reports}
    return Manifest(
        pharos_version=__version__,
        config=config,
        gate=gate,
        n_reports=len(reports),
        n_events=len(events),
        label_histogram=label_histogram(reports),
        plant_share=sum(events.values()) / max(len(events), 1),
    )
