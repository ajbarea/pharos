"""Plant registry module for Pharos.

Formalizes the definition of planted intelligence facts and signals, moving away from
scattered hardcoded tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class FactDefinition:
    """An observable intelligence fact with surface term realizations."""

    fact_id: str
    description: str
    surface_terms: tuple[str, ...]


@dataclass(frozen=True)
class PlantSignature:
    """Formal conjunction rule defining what makes an event significant."""

    name: str
    required_facts: tuple[str, ...]
    decoy_patterns: tuple[tuple[str, ...], ...] = ()

    def is_triggered(self, present_facts: tuple[str, ...] | set[str] | list[str]) -> bool:
        """Evaluates if the present facts satisfy the full conjunction."""
        facts_set = set(present_facts)
        return all(f in facts_set for f in self.required_facts)


class PlantRegistry:
    """Central manager for registering and looking up plant signatures and facts."""

    def __init__(self) -> None:
        self._facts: dict[str, FactDefinition] = {}
        self._signatures: dict[str, PlantSignature] = {}

    def register_fact(
        self, fact_id: str, description: str, surface_terms: tuple[str, ...]
    ) -> FactDefinition:
        fact = FactDefinition(fact_id=fact_id, description=description, surface_terms=surface_terms)
        self._facts[fact_id] = fact
        return fact

    def register_signature(
        self,
        name: str,
        required_facts: tuple[str, ...],
        decoy_patterns: tuple[tuple[str, ...], ...] = (),
    ) -> PlantSignature:
        sig = PlantSignature(
            name=name, required_facts=required_facts, decoy_patterns=decoy_patterns
        )
        self._signatures[name] = sig
        return sig

    def get_fact(self, fact_id: str) -> FactDefinition | None:
        return self._facts.get(fact_id)

    def get_signature(self, name: str) -> PlantSignature | None:
        return self._signatures.get(name)

    @property
    def facts(self) -> Mapping[str, FactDefinition]:
        return self._facts

    @property
    def signatures(self) -> Mapping[str, PlantSignature]:
        return self._signatures


# Built-in Default Registry matching maritime-watch scenario
def default_maritime_registry() -> PlantRegistry:
    reg = PlantRegistry()
    reg.register_fact(
        "course_deviation",
        "Unexplained course deviation > 30 deg",
        ("deviation", "course change", "course offset"),
    )
    reg.register_fact(
        "draft_mismatch",
        "Draft depth mismatch with manifest",
        ("draft mismatch", "freeboard anomaly", "depth discrepancy"),
    )
    reg.register_fact(
        "unlit_contact",
        "Unlit vessel contact at night",
        ("unlit", "dark contact", "running lights off"),
    )

    reg.register_signature(
        name="maritime_watch_v1",
        required_facts=("course_deviation", "draft_mismatch", "unlit_contact"),
        decoy_patterns=(
            ("course_deviation", "draft_mismatch"),
            ("draft_mismatch", "unlit_contact"),
            ("course_deviation", "unlit_contact"),
        ),
    )
    return reg
