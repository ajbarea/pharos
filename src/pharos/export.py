"""Writing a corpus out, and hashing what was written.

Pharos is a generator rather than a fixed corpus, so a released artifact is one
instantiation: `(version, commit, seed, config)` run to completion. That makes the
hash load-bearing in a way it is not for a hand-collected dataset. A reader who
reruns the generator at the recorded seed should get a byte-identical file, and the
digest recorded alongside is what lets them check that rather than trust it.

One row per report, JSON Lines, sorted keys. Sorted keys matter: an unordered dict
would hash differently between interpreter versions and quietly break the property
this module exists to provide.
"""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from pharos.generate import Report

#: Column order is fixed so the Croissant record set and the file agree.
CORPUS_FIELDS: tuple[str, ...] = (
    "report_id",
    "event_id",
    "report_type",
    "center_id",
    "voice",
    "vessel_name",
    "text",
    "sensitivity",
    "compartments",
    "capacity",
    "is_plant",
    "fact_ids",
)


def corpus_row(report: Report) -> dict[str, object]:
    """One report flattened to primitives, with its label unpacked into columns.

    The label is split rather than stringified so a consumer can filter on
    sensitivity or compartment without parsing a `LEVEL[A,B]` rendering back apart.
    """
    return {
        "report_id": report.report_id,
        "event_id": report.event_id,
        "report_type": str(report.report_type),
        "center_id": report.center.center_id,
        "voice": str(report.voice),
        "vessel_name": report.vessel_name,
        "text": report.text,
        "sensitivity": report.label.sensitivity.name,
        "compartments": sorted(str(c) for c in report.label.compartments),
        "capacity": report.label.capacity.name,
        "is_plant": report.is_plant,
        "fact_ids": list(report.fact_ids),
    }


def corpus_rows(reports: list[Report]) -> Iterator[dict[str, object]]:
    for report in reports:
        yield corpus_row(report)


def corpus_bytes(reports: list[Report]) -> bytes:
    """The corpus as it is written to disk. Separated so the hash is of exactly this."""
    lines = (json.dumps(row, sort_keys=True, ensure_ascii=False) for row in corpus_rows(reports))
    return ("\n".join(lines) + "\n").encode("utf-8")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_corpus(reports: list[Report], path: Path) -> tuple[int, str]:
    """Write the corpus to `path`, returning `(bytes written, sha256)`."""
    payload = corpus_bytes(reports)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload), sha256(payload)
