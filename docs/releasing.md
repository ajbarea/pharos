# Releasing a Corpus & Metadata

Export a dataset release along with metadata and manifest verification:

```bash
uv run python -m pharos.cli export --seed 7 --events 400 --out export/
```

This exports three primary artifacts:

| Output Artifact | Description | Schema / Reference |
| :--- | :--- | :--- |
| `corpus.jsonl` | The exported event corpus | [Corpus Schema](reference/corpus-schema.md) |
| `manifest.json` | Certified record: seed, config, gate verdict, label distribution | `pharos.manifest` |
| `croissant.json` | MLCommons Croissant metadata with Responsible AI extension | `pharos.croissant` |

!!! danger "Enforced Gate Verification"
    The export CLI **refuses to write output** if the generated corpus fails its shortcut gate check. An un-gated or invalid dataset cannot be released.

---

## Reproducibility & Cryptographic Hashing

Because Pharos is a procedural dataset generator, every release is uniquely specified by `(version, commit, seed, config)`. 

!!! info "SHA-256 Digest Certification"
    The cryptographic `sha256` hash in `croissant.json` MUST equal the exact hash of `corpus.jsonl`. This allows downstream consumers to verify bit-identical reproducibility.

---

## MLCommons Croissant & Responsible AI Metadata

Pharos automatically emits [Croissant JSON-LD](https://docs.mlcommons.org/croissant/docs/croissant-spec.html) metadata conforming to NeurIPS dataset requirements, enriched with the [Responsible AI (RAI)](https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html) extension.

### Responsible AI Block Fields

* **`rai:dataBiases`**: Documents surface baselines and permutation null distributions.
* **`rai:dataLimitations`**: Declares that data is synthetically generated for privacy/governance research and must not be used to train operational classifiers.
* **`rai:personalSensitiveInformation`**: Confirms security labels are synthetic lattice constructions and do not model real organizations or clearance systems.
* **`rai:dataCollectionMissingData`**: Documents event coverage guarantees and historical dataset corrections.

---

## Automated Metadata Validation

Validate Croissant JSON-LD metadata against official MLCommons schemas:

```bash
# Install validation dependencies
uv sync --group croissant

# Run validation suite
uv run pytest tests/test_croissant_validation.py
```

!!! tip "Preventing Schema Drift"
    Metadata generation is driven directly by `pharos.croissant` and programmatically validated with `mlcroissant` in CI, preventing hand-authored spec drift.

---

## Provenance Block Structure

Every exported measurement and manifest records audit provenance metadata:

```json
{
  "pharos_version": "0.1.0",
  "git_commit": "1ec0d01a2b3c",
  "git_dirty": false,
  "generated_at": "2026-07-30T19:12:04+00:00",
  "python": "3.13.1",
  "platform": "Linux-6.6.87-x86_64",
  "executable": ".venv/bin/python3",
  "seed": 7,
  "model": "qwen2.5:7b-instruct"
}
```

The first seven keys are in every artifact this repository has written. The rest are
conditional, and their absence is information: `seed` appears where a run is seeded,
while `model`, `model_key` and `endpoint` appear only where a model was called --- so a
measurement that claims to be model-free and carries a `model` is a measurement that is
not what it says. Individual scripts add their own parameters beside these, such as the
draw or fleet count a sweep ran over.

!!! note "Deterministic Manifests vs. Measurement Stamps"
    Measurement result artifacts log timestamps and `git_dirty` flags. Dataset `manifest.json` files **omit timestamps** to ensure two manifests built from identical seeds yield identical hashes.

