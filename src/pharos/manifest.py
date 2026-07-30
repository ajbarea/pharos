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

    @property
    def usable(self) -> bool:
        """Whether this corpus version may be used at all.

        Two conditions, both hard. The gate must pass, and the labels must vary.
        """
        return self.gate.passed and len(self.label_histogram) > 1

    def as_dict(self) -> dict[str, object]:
        return {
            "pharos_version": self.pharos_version,
            "config": self.config.as_dict(),
            "gate": self.gate.as_dict(),
            "n_reports": self.n_reports,
            "n_events": self.n_events,
            "label_histogram": self.label_histogram,
            "plant_share": round(self.plant_share, 4),
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


def build_manifest(config: GeneratorConfig) -> Manifest:
    """Generate, gate, and certify a corpus in one step."""
    reports = generate(config)
    gate = run_gate(reports)
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
