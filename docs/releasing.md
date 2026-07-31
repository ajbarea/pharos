# Releasing a corpus

```bash
uv run python -m pharos.cli export --seed 7 --events 400 --out export/
```

Writes three files:

| File | What it is |
| --- | --- |
| `corpus.jsonl` | The corpus itself. See [corpus schema](reference/corpus-schema.md) |
| `manifest.json` | The record that certifies it: seed, config, gate verdict, label histogram |
| `croissant.json` | Croissant metadata with the Responsible AI extension |

The command **refuses to write anything** when the corpus fails its own gate. A
citable artifact whose own gate rejected it is worse than no artifact.

## Why a generator needs a hash

Pharos is a generator rather than a fixed corpus, so a released artifact is one
instantiation: `(version, commit, seed, config)` run to completion. That makes the
digest load-bearing in a way it is not for a hand-collected dataset. A reader who
reruns the generator at the recorded seed should get a byte-identical file, and
the `sha256` recorded alongside is what lets them **check** that rather than
trust it.

The export test asserts the property directly: the digest in `croissant.json`
equals the hash of the `corpus.jsonl` written next to it.

## Croissant, and what a generated dataset does to it

[Croissant](https://docs.mlcommons.org/croissant/docs/croissant-spec.html) is the
MLCommons JSON-LD format for describing an ML-ready dataset. As of the 2026
NeurIPS Evaluations and Datasets track, a dataset submission must carry one, with
[Responsible AI](https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html)
metadata inside it.

The record is generated from the manifest rather than maintained by hand. A
separate metadata file is a second source of truth, and it drifts from the
generator on the first change.

A procedurally generated corpus stresses the format in one specific way. Croissant
assumes a `contentUrl` pointing at a file that already exists, and a Pharos corpus
does not exist until someone runs the generator. So the record identifies the
artifact by what reproduces it, and carries `cr:generatorConfig` and
`cr:codeProvenance` alongside the digest.

## Validate, do not read

```bash
uv sync --group croissant
uv run pytest tests/test_croissant_validation.py
```

Hand-verifying JSON-LD against a prose specification does not work, and this is
not a hypothetical.

The first version of `pharos.croissant` was written straight from the
specification page and looked correct: every property the spec names, in the
right shapes, RAI block complete. It was **invalid**. The hand-assembled
`@context` omitted `column`, so every field's `extract` resolved to nothing and
the record described a file with no readable columns. Reading it could not catch
that. One call to `mlcroissant` did, immediately.

The context is now copied verbatim from the MLCommons reference datasets, and the
validator runs in the suite.

## The Responsible AI block

Not boilerplate, and worth reading before citing a number from this corpus. Four
entries carry real content:

**`rai:dataBiases`** reports the measured surface baseline with its permutation
null, because that bias is a known property of the corpus and every downstream
score must be reported against it rather than against 0.5.

**`rai:dataLimitations`** states the honest ceiling: the evaluation this corpus
supports is simulated end to end, so behavioural claims carry that cap, and the
corpus must not be used to train or evaluate a classifier intended for deployment
against real classified material.

**`rai:personalSensitiveInformation`** records that the classification levels and
compartments are invented for this testbed and model no real classification
system, programme, or organisation.

**`rai:dataCollectionMissingData`** records that coverage is guaranteed by
construction and asserted in the test suite, and that an earlier version did not
guarantee it, which invalidated a measurement. That is in the metadata rather than
only in a changelog because it is the kind of thing a reader of the *data* needs.

## Provenance on every artifact

Every measurement result carries a `provenance` block:

```json
{
  "pharos_version": "0.1.0",
  "git_commit": "1ec0d01a2b3c",
  "git_dirty": false,
  "generated_at": "2026-07-30T19:12:04+00:00",
  "python": "3.13.1",
  "platform": "Linux-6.6.87-x86_64",
  "model": "qwen2.5:7b-instruct",
  "seed": 7
}
```

`git_dirty` is reported rather than forbidden. A dirty measurement is often the
honest state of an experiment in progress, and the useful thing is that a reader
can see it was dirty rather than be handed a commit that does not describe the
code that ran.

The corpus manifest carries `code_provenance` **without** the clock, so two
manifests built from one seed still compare equal. A timestamp there would break
the reproducibility the manifest exists to certify.
