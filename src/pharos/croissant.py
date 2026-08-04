"""Croissant metadata, including the Responsible AI extension.

Croissant is the MLCommons JSON-LD format for describing an ML-ready dataset, and
as of the 2026 NeurIPS Evaluations and Datasets track a dataset submission must
carry one, with RAI metadata inside it. Emitting it from the manifest rather than
maintaining it by hand is the only way it stays true: a hand-written metadata file
is a second source of truth that drifts from the generator on the first change.

A procedurally generated corpus stresses the format in a specific way. Croissant
assumes a `contentUrl` pointing at a file that already exists, but a Pharos corpus
does not exist until someone runs the generator. So the record identifies the
artifact by what reproduces it -- generator version, commit, seed, and config --
and carries the `sha256` of one instantiation, which is what turns "rerun this and
you get the same thing" from a claim into a check.

The RAI block is not boilerplate here. Pharos is synthetic, its classification
markings are fictional, and its surface baseline is a known and measured bias. Each
of those is a thing a reviewer would otherwise have to discover, so each is stated.

Spec: <https://docs.mlcommons.org/croissant/docs/croissant-spec.html>
RAI:  <https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html>
"""

import json
from typing import Any

from pharos.export import CORPUS_FIELDS
from pharos.manifest import Manifest
from pharos.provenance import code_provenance

CROISSANT_VERSION = "http://mlcommons.org/croissant/1.0"
RAI_VERSION = "http://mlcommons.org/croissant/RAI/1.0"

REPOSITORY = "https://github.com/ajbarea/pharos"

#: The official Croissant 1.0 @context, verbatim, plus the RAI prefix.
#:
#: Copied from the reference datasets rather than hand-assembled from the prose
#: spec. An abridged context validates as "not standard" and, worse, silently
#: fails to resolve terms it omits: an earlier hand-built version here dropped
#: `column`, so every field's `extract` resolved to nothing and mlcroissant
#: rejected the record. Keep this in sync with
#: <https://github.com/mlcommons/croissant/tree/main/datasets/1.0>.
CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": {"@id": "cr:equivalentProperty", "@type": "@vocab"},
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "wd": "https://www.wikidata.org/wiki/",
}

#: Column name to Croissant dataType. `compartments` and `fact_ids` are JSON arrays
#: within a line rather than scalar cells, so they are described as text.
FIELD_TYPES: dict[str, str] = {
    "report_id": "sc:Text",
    "event_id": "sc:Text",
    "report_type": "sc:Text",
    "center_id": "sc:Text",
    "voice": "sc:Text",
    "vessel_name": "sc:Text",
    "text": "sc:Text",
    "sensitivity": "sc:Text",
    "compartments": "sc:Text",
    "capacity": "sc:Text",
    "is_plant": "sc:Boolean",
    "fact_ids": "sc:Text",
}

FIELD_DESCRIPTIONS: dict[str, str] = {
    "report_id": "Stable identifier for one rendered report.",
    "event_id": "The world event this report describes. Many reports share one event.",
    "report_type": "The channel that carried the report, which is what confers its label.",
    "center_id": "The watch centre that filed it. The gate holds whole centres out.",
    "voice": "The officer voice that rendered it, one of a fixed set of styles.",
    "vessel_name": "The fictional vessel the report concerns.",
    "text": "The report body. The only field a specialist is meant to read.",
    "sensitivity": "Classification level: OPEN, INTERNAL, PROTECTED, or RESTRICTED.",
    "compartments": "JSON array of need-to-know compartments, a subset lattice.",
    "capacity": "Form of the entry: ENUM, SCALAR, SPAN, or FREETEXT.",
    "is_plant": "Whether this report belongs to a significant (planted) event.",
    "fact_ids": "JSON array of the fact identifiers this rendering asserts.",
}


def _rai_block(manifest: Manifest) -> dict[str, str]:
    """Responsible AI metadata.

    Written as claims a reviewer can check against the generator rather than as
    reassurance. The bias entry in particular reports a measured number, because
    the surface baseline is a known property of this corpus and hiding it would
    misrepresent what a triage score against it means.
    """
    baseline = manifest.gate.surface_baseline
    null_mean = manifest.gate.null_mean
    null_text = f"{null_mean:.4f}" if null_mean is not None else "not computed"
    return {
        "rai:dataCollection": (
            "Procedurally generated. No collection took place: reports are rendered from a "
            "fixed fact vocabulary by a seeded pseudo-random generator, so a corpus is "
            "reproducible from its version, commit, seed, and config alone."
        ),
        "rai:dataCollectionType": "Synthetic, procedurally generated. No human subjects.",
        "rai:dataCollectionTimeframe": (
            "Not applicable in the usual sense: nothing was collected over a period. A corpus "
            "is generated on demand and is a function of its seed and config, so the only "
            "date that carries meaning is the commit its provenance stamp names, and two "
            "corpora from the same commit and seed are identical whenever they were made."
        ),
        "rai:dataCollectionRawData": (
            "None. There is no source corpus, no scrape, and no derivation from any existing "
            "dataset. The world model and fact vocabulary are authored fiction."
        ),
        "rai:dataCollectionMissingData": (
            "None by construction. Generation guarantees that every fact of an event is "
            "rendered by some channel able to carry it, and this is asserted in the test "
            "suite. An earlier version did not, which invalidated a measurement and is "
            "recorded as a retracted finding."
        ),
        "rai:dataAnnotationProtocol": (
            "Labels are not annotations. Each report's label is conferred by the channel that "
            "carried it, assigned deterministically at generation time from the product "
            "lattice over sensitivity, compartments, and capacity. Ground truth for the "
            "triage task is likewise definitional: an event is significant exactly when it "
            "carries a fixed conjunction of facts."
        ),
        "rai:dataAnnotationPlatform": "None. Labels are assigned in code by pharos.labels.",
        "rai:dataAnnotationAnalysis": (
            "Every released corpus is scored by an acceptance gate before it may be used: a "
            "surface-only probe attempts to predict plant membership from length, punctuation, "
            "and channel alone, and is compared against a permutation null under the identical "
            f"procedure. This corpus scores {baseline:.4f} against a null of {null_text}. A "
            "corpus whose baseline is unmeasured, insignificant against its own null, or above "
            "the ceiling cannot support a triage claim and the gate exits non-zero. Measured "
            "scores additionally carry a validity assessment naming the conditions under which "
            "they should not be quoted."
        ),
        "rai:dataPreprocessingProtocol": (
            "None. Reports are consumed exactly as generated: no filtering, deduplication, "
            "normalisation, or tokenisation is applied between generation and release, so the "
            "SHA-256 in this record is the hash of what the generator emitted."
        ),
        "rai:dataManipulationProtocol": (
            "None after generation. The only transformation is serialisation to JSON Lines, "
            "which is byte-stable for a given corpus, and the hash is taken over exactly those "
            "bytes."
        ),
        "rai:dataImputationProtocol": (
            "None, and none is possible: there are no missing values to impute. Every field of "
            "every report is written at generation time, which the test suite asserts."
        ),
        "rai:machineAnnotationTools": (
            "pharos.generate for rendering, pharos.labels for the label algebra. No model is "
            "involved in producing the corpus or its labels."
        ),
        "rai:annotationsPerItem": (
            "One, assigned deterministically by the generating channel. No human annotation, "
            "so there is no inter-annotator disagreement to report."
        ),
        "rai:annotatorDemographics": "Not applicable. There are no human annotators.",
        "rai:personalSensitiveInformation": (
            "None. Vessels, officers, watch centres, and events are fictional. The "
            "classification levels and compartments are invented for this testbed and model "
            "no real classification system, programme, or organisation."
        ),
        "rai:dataBiases": (
            "A known and measured surface bias. Ground truth is defined by the presence of "
            f"particular content, so shape alone predicts the class above chance: {baseline:.4f} "
            f"AUC against a permutation null of {null_text}. This is published as the surface "
            "baseline rather than removed, because removing it would require a vocabulary of "
            "perfect surface twins and would trade away answerability. Every downstream score "
            "must be reported against this baseline, not against 0.5."
        ),
        "rai:dataLimitations": (
            "The evaluation this corpus supports is simulated end to end: synthetic documents, "
            "synthetic labels, and synthetic analysts. Behavioural claims carry that cap, and "
            "persona fidelity is validatable only out of domain against real human traces. The "
            "corpus does not model any real programme, organisation, or vessel, and must not be "
            "used to train or evaluate a classifier intended for deployment against real "
            "classified material."
        ),
        "rai:dataUseCases": (
            "Evaluating whether a personal/shared disclosure boundary in federated "
            "personalization is enforceable and auditable, and measuring what enforcing it "
            "costs. Suitable for label-propagation, declassification-policy, and "
            "federation-eligibility experiments that need labels with genuine lattice "
            "structure rather than a single sensitivity ladder."
        ),
        "rai:dataSocialImpact": (
            "The intended effect is to let disclosure claims in federated personalization be "
            "measured rather than asserted, since no public corpus carries the label structure "
            "required to test one. The corresponding risk is over-reading: results here are "
            "evidence about a mechanism, not evidence that the mechanism is safe on real data."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "Versioned with the generator. Each corpus is identified by generator version, "
            "git commit, seed, and config, and a released instantiation carries the sha256 of "
            "the exact bytes. Regenerating at the recorded seed reproduces them."
        ),
    }


def croissant_record(
    manifest: Manifest,
    *,
    corpus_filename: str = "corpus.jsonl",
    sha256: str | None = None,
    content_url: str | None = None,
    date_published: str | None = None,
) -> dict[str, Any]:
    """A Croissant 1.0 record describing one instantiation of a Pharos corpus.

    `date_published` is omitted rather than defaulted to today, so calling this
    twice on one manifest returns equal records. The CLI supplies the date when it
    writes a release; a test comparing two records does not want a clock in them.
    """
    provenance = code_provenance()
    commit = provenance.get("git_commit") or "unknown"
    config = manifest.config

    file_object: dict[str, Any] = {
        "@type": "cr:FileObject",
        "@id": corpus_filename,
        "name": corpus_filename,
        "description": (
            "One instantiation of the corpus, JSON Lines, one report per line, keys sorted."
        ),
        "encodingFormat": "application/jsonlines",
        "contentUrl": content_url or f"{REPOSITORY}#regenerate-with-pharos-cli-export",
    }
    if sha256 is not None:
        file_object["sha256"] = sha256

    fields = [
        {
            "@type": "cr:Field",
            "@id": f"report/{name}",
            "name": name,
            "description": FIELD_DESCRIPTIONS[name],
            "dataType": FIELD_TYPES[name],
            "source": {
                "fileObject": {"@id": corpus_filename},
                "extract": {"column": name},
            },
        }
        for name in CORPUS_FIELDS
    ]

    record: dict[str, Any] = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        # Both, as a list. RAI conformance is declared exactly one way -- by naming the RAI
        # URI in `dct:conformsTo` -- and this record used to put it in `rai:version`, which
        # is not a property the specification defines. A consumer checking conformance the
        # way the spec says to would have read this dataset as core Croissant with some
        # unrecognised extra keys, which is the one thing the RAI extension exists to avoid.
        "conformsTo": [CROISSANT_VERSION, RAI_VERSION],
        "name": "pharos",
        "description": (
            "A labeled fleet testbed for federated personalization with a governed disclosure "
            "boundary. Maritime watch reporting whose objects carry classification levels and "
            "cross-cutting compartments, so that the personal/shared split in federated "
            "personalization can be measured rather than asserted. Procedurally generated and "
            "reproducible from its seed."
        ),
        "url": REPOSITORY,
        "license": "https://opensource.org/licenses/MIT",
        "version": manifest.pharos_version,
        "citeAs": (
            f"Pharos {manifest.pharos_version} (commit {commit}), corpus seed {config.seed}, "
            f"{manifest.n_events} events, surface baseline "
            f"{manifest.gate.surface_baseline:.4f}."
        ),
        "creator": {"@type": "sc:Person", "name": "A. J. Barea"},
        "keywords": [
            "federated learning",
            "personalization",
            "information flow control",
            "synthetic corpus",
            "disclosure policy",
        ],
        "isAccessibleForFree": True,
        **_rai_block(manifest),
        "cr:generatorConfig": manifest.as_dict()["config"],
        "cr:codeProvenance": provenance,
        "distribution": [file_object],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "report",
                "name": "report",
                "description": "One rendered report and the label its channel confers.",
                "key": {"@id": "report/report_id"},
                "field": fields,
            }
        ],
    }
    if date_published is not None:
        record["datePublished"] = date_published
    return record


def to_json(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False)
