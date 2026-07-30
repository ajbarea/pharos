"""Detecting which facts a piece of text asserts.

Attribution needs to know what a summary actually says, and the corpus knows what
each report contains, so a detector bridges the two. It is keyword based on
purpose: a model-based detector would put a second opaque component inside the
measurement, and the thing being measured is already a model.

The detector is only trustworthy if it is measured, so `detector_accuracy` scores
it against the corpus itself, where ground truth is known by construction. Every
experiment reports that accuracy alongside its results, because an attribution
number computed through an unvalidated detector is uninterpretable.
"""

from dataclasses import dataclass

from pharos.generate import Report
from pharos.world import FACTS_BY_ID

#: Terms that identify a fact in free text. A fact counts as present when any of
#: its term groups matches, and a group matches when every term in it appears.
#: Groups exist because a summary may paraphrase: "draft below declared laden"
#: and "draft mismatch" are the same assertion in different words.
FACT_TERMS: dict[str, tuple[tuple[str, ...], ...]] = {
    "draft_mismatch": (("draft",), ("freeboard",)),
    "course_deviation": (
        ("course altered",),
        ("heading of",),
        ("track turned",),
        ("degrees off",),
        ("divergence",),
    ),
    "unlit_contact": (("unlit",), ("no lights",), ("without lights",)),
    "pilot_absent": (("pilot",),),
    "crew_substitution": (("engineer",), ("substitution",)),
    "certificate_unresolved": (("certificate",), ("registry",)),
    "berth_closure": (("berth",), ("maintenance",)),
    "demurrage_dispute": (("demurrage",),),
    "reflagging": (("reflag",), ("changed flag",), ("change flag",), ("flag changes",)),
    "ship_transfer_claim": (("transfer",), ("ship-to-ship",)),
    "shoal_activity": (("shoal",), ("small-craft",), ("small craft",)),
    "inspection_history": (("inspection",), ("port state",)),
    "routine_transit": (("knots",), ("outbound transit",)),
    "cargo_declared": (("cargo declared",), ("declared cargo",), ("manifest line",)),
    "weather_note": (("visibility",), ("sea state",)),
}

#: Substrings that appear in every report header or closing regardless of which
#: facts the report carries. Stripped before matching, because a hull type of
#: "general cargo" or a "flag Panama" line would otherwise register as an
#: assertion the report never made.
BOILERPLATE: tuple[str, ...] = (
    "general cargo",
    "refrigerated cargo",
    "bulk carrier",
    "product tanker",
    "container feeder",
    "flag panama",
    "flag liberia",
    "flag malta",
    "flag marshall islands",
    "flag cyprus",
    "watch center",
    "vessel report",
    "sensor track",
    "port notice",
    "crew manifest",
    "press item",
    "liaison tip",
    "detention record",
    "partner report",
)


def _strip_boilerplate(lowered: str) -> str:
    for phrase in BOILERPLATE:
        lowered = lowered.replace(phrase, " ")
    return lowered


def detect_facts(text: str) -> frozenset[str]:
    """The fact ids `text` asserts, by keyword match."""
    lowered = _strip_boilerplate(text.lower())
    found = {
        fact_id
        for fact_id, groups in FACT_TERMS.items()
        if any(all(term in lowered for term in group) for group in groups)
    }
    return frozenset(found)


@dataclass(frozen=True, slots=True)
class DetectorAccuracy:
    """How well the detector recovers facts from text whose facts are known."""

    n_reports: int
    recall: float
    precision: float

    @property
    def f1(self) -> float:
        if self.recall + self.precision == 0:
            return 0.0
        return 2 * self.recall * self.precision / (self.recall + self.precision)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_reports": self.n_reports,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
        }


def detector_accuracy(reports: list[Report]) -> DetectorAccuracy:
    """Score the detector against reports whose fact content is known.

    Recall is the share of facts a report contains that the detector finds.
    Precision is the share of facts it finds that the report actually contains.
    Both matter: a detector that under-reports makes attribution look sparse, and
    one that over-reports makes every source look load-bearing.
    """
    true_positive = 0
    detected_total = 0
    actual_total = 0
    for report in reports:
        actual = frozenset(report.fact_ids)
        detected = detect_facts(report.text)
        true_positive += len(actual & detected)
        detected_total += len(detected)
        actual_total += len(actual)
    return DetectorAccuracy(
        n_reports=len(reports),
        recall=true_positive / max(actual_total, 1),
        precision=true_positive / max(detected_total, 1),
    )


def unmatched_terms() -> tuple[str, ...]:
    """Fact ids in the world vocabulary that the detector has no terms for.

    A fact with no terms is invisible to attribution, which would silently
    understate how many sources fed a turn, so this is a hard check rather than a
    diagnostic.
    """
    return tuple(sorted(set(FACTS_BY_ID) - set(FACT_TERMS)))
