"""The model registry: what Pharos can be run against, and what it actually has been.

Every published Pharos number so far came from one model. That is a real limit on
what those numbers mean, and the fix is to make switching models trivial rather
than to hope nobody notices. This module is the switch.

Two ideas keep the registry honest.

**`verified` means smoke-tested, not plausible.** A spec is verified only once it
has actually answered a Pharos triage task and returned a parseable verdict. A
model nobody has run is listed as a candidate and says so. The distinction matters
because "supported models" lists in research code are usually aspirational, and a
reader cannot tell which entries were ever executed.

**Tags are checked against the daemon, not asserted.** Ollama library tags drift,
so the registry records what to ask for and `installed` reports what is actually
present. An unknown tag is not an error either: `resolve` wraps any raw string as
an ad-hoc spec, so a model the registry has never heard of still runs.

VRAM figures are approximate resident sizes for the quantization named, useful for
deciding what fits a given card, not exact.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, replace

DEFAULT_ENDPOINT_TAGS = "http://localhost:11434/api/tags"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One selectable model."""

    key: str
    tag: str
    family: str
    parameters: str
    quantization: str
    approx_vram_gb: float
    verified: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "tag": self.tag,
            "family": self.family,
            "parameters": self.parameters,
            "quantization": self.quantization,
            "approx_vram_gb": self.approx_vram_gb,
            "verified": self.verified,
            "note": self.note,
        }


#: Curated candidates, smallest first. `verified` is set only by having run the
#: model against a Pharos task; do not flip it on the strength of a model card.
REGISTRY: dict[str, ModelSpec] = {
    "llama3.2-3b": ModelSpec(
        key="llama3.2-3b",
        tag="llama3.2:3b-instruct-q4_K_M",
        family="Llama",
        parameters="3B",
        quantization="Q4_K_M",
        approx_vram_gb=2.3,
        verified=False,
        note="Smallest useful size class. Fits alongside other work on an 8 GB card.",
    ),
    "qwen2.5-3b": ModelSpec(
        key="qwen2.5-3b",
        tag="qwen2.5:3b-instruct",
        family="Qwen",
        parameters="3B",
        quantization="Q4_K_M",
        approx_vram_gb=2.0,
        verified=False,
        note="Same family as the reference model, one size class down.",
    ),
    "qwen2.5-7b": ModelSpec(
        key="qwen2.5-7b",
        tag="qwen2.5:7b-instruct",
        family="Qwen",
        parameters="7.6B",
        quantization="Q4_K_M",
        approx_vram_gb=4.7,
        verified=True,
        note="The reference model. Every published Pharos measurement used this.",
    ),
    "llama3.1-8b": ModelSpec(
        key="llama3.1-8b",
        tag="llama3.1:8b-instruct-q4_K_M",
        family="Llama",
        parameters="8B",
        quantization="Q4_K_M",
        approx_vram_gb=4.9,
        verified=False,
        note="Comparable size, different family. The natural cross-family check.",
    ),
    "mistral-7b": ModelSpec(
        key="mistral-7b",
        tag="mistral:7b-instruct",
        family="Mistral",
        parameters="7B",
        quantization="Q4_0",
        approx_vram_gb=4.1,
        verified=False,
        note="Third family at the reference size class.",
    ),
    "qwen2.5-14b": ModelSpec(
        key="qwen2.5-14b",
        tag="qwen2.5:14b-instruct",
        family="Qwen",
        parameters="14B",
        quantization="Q4_K_M",
        approx_vram_gb=9.0,
        verified=False,
        note="Exceeds 8 GB. Needs a larger card or CPU offload, which is slow.",
    ),
}

#: The registry key used when nothing is specified.
DEFAULT_KEY = "qwen2.5-7b"


def default_spec() -> ModelSpec:
    return REGISTRY[DEFAULT_KEY]


def resolve(name: str | None) -> ModelSpec:
    """A spec for `name`, which may be a registry key, a raw tag, or None.

    An unrecognised name is wrapped as an ad-hoc unverified spec rather than
    rejected, so the registry never becomes a gate on what can be run.
    """
    if name is None:
        return default_spec()
    if name in REGISTRY:
        return REGISTRY[name]
    for spec in REGISTRY.values():
        if spec.tag == name:
            return spec
    return ModelSpec(
        key=name,
        tag=name,
        family="unknown",
        parameters="unknown",
        quantization="unknown",
        approx_vram_gb=0.0,
        verified=False,
        note="Not in the registry. Passed through to the backend as given.",
    )


def installed(endpoint: str = DEFAULT_ENDPOINT_TAGS, timeout: float = 5.0) -> set[str]:
    """Tags the local Ollama daemon currently has, or an empty set when it is down.

    Degrades rather than raising: listing models is an informational act, and a
    stopped daemon should produce an honest "nothing installed" rather than a
    traceback in the middle of a status command.
    """
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return set()
    return {entry["name"] for entry in payload.get("models", []) if "name" in entry}


def catalog(endpoint: str = DEFAULT_ENDPOINT_TAGS) -> list[dict[str, object]]:
    """Every registry entry, annotated with whether it is installed right now."""
    present = installed(endpoint)
    return [{**spec.as_dict(), "installed": spec.tag in present} for spec in REGISTRY.values()]


def mark_verified(key: str) -> ModelSpec:
    """Return `key`'s spec with `verified` set. Used by the smoke test, not by hand."""
    return replace(REGISTRY[key], verified=True)
