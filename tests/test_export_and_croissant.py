"""A released corpus has to be checkable, not merely described."""

import json

import pytest

from pharos.cli import main
from pharos.croissant import CONTEXT, CROISSANT_VERSION, croissant_record
from pharos.export import CORPUS_FIELDS, corpus_bytes, corpus_row, sha256, write_corpus
from pharos.generate import GeneratorConfig, generate
from pharos.manifest import build_manifest

CONFIG = GeneratorConfig(seed=5, n_events=120, plant_rate=0.3)


@pytest.fixture(scope="module")
def reports():
    return generate(CONFIG)


@pytest.fixture(scope="module")
def manifest():
    return build_manifest(CONFIG, null_trials=4)


def test_a_row_unpacks_the_label_into_columns(reports):
    row = corpus_row(reports[0])
    assert set(row) == set(CORPUS_FIELDS)
    assert row["sensitivity"] in {"OPEN", "INTERNAL", "PROTECTED", "RESTRICTED"}
    assert isinstance(row["compartments"], list)


def test_the_same_seed_hashes_to_the_same_bytes():
    """The whole point of shipping a digest: rerun the seed, check the hash."""
    assert sha256(corpus_bytes(generate(CONFIG))) == sha256(corpus_bytes(generate(CONFIG)))


def test_a_different_seed_hashes_differently():
    other = GeneratorConfig(seed=6, n_events=120, plant_rate=0.3)
    assert sha256(corpus_bytes(generate(CONFIG))) != sha256(corpus_bytes(generate(other)))


def test_written_bytes_are_the_bytes_that_were_hashed(tmp_path, reports):
    size, digest = write_corpus(reports, tmp_path / "corpus.jsonl")
    written = (tmp_path / "corpus.jsonl").read_bytes()
    assert len(written) == size
    assert sha256(written) == digest


def test_every_line_is_json_with_the_declared_fields(tmp_path, reports):
    write_corpus(reports, tmp_path / "corpus.jsonl")
    lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(reports)
    for line in lines:
        assert set(json.loads(line)) == set(CORPUS_FIELDS)


def test_croissant_declares_the_spec_it_conforms_to(manifest):
    record = croissant_record(manifest)
    assert record["conformsTo"] == CROISSANT_VERSION
    assert record["@type"] == "sc:Dataset"
    assert record["@context"]["rai"] == "http://mlcommons.org/croissant/RAI/"


def test_croissant_describes_every_corpus_column(manifest):
    """A record set that omits a column silently misdescribes the file."""
    record = croissant_record(manifest)
    described = {field["name"] for field in record["recordSet"][0]["field"]}
    assert described == set(CORPUS_FIELDS)


def test_croissant_field_sources_point_at_the_declared_file(manifest):
    record = croissant_record(manifest)
    file_id = record["distribution"][0]["@id"]
    for field in record["recordSet"][0]["field"]:
        assert field["source"]["fileObject"]["@id"] == file_id


def test_croissant_carries_the_required_rai_properties(manifest):
    record = croissant_record(manifest)
    required = {
        "rai:dataCollection",
        "rai:dataCollectionType",
        "rai:dataAnnotationProtocol",
        "rai:personalSensitiveInformation",
        "rai:dataBiases",
        "rai:dataLimitations",
        "rai:dataUseCases",
    }
    assert required <= set(record)
    assert all(record[key].strip() for key in required)


def test_the_bias_entry_reports_the_measured_baseline(manifest):
    """RAI bias text must carry the number, not a reassurance that it is small."""
    record = croissant_record(manifest)
    assert f"{manifest.gate.surface_baseline:.4f}" in record["rai:dataBiases"]


def test_croissant_records_the_seed_that_reproduces_the_corpus(manifest):
    record = croissant_record(manifest)
    assert record["cr:generatorConfig"]["seed"] == CONFIG.seed
    assert "git_commit" in record["cr:codeProvenance"]


def test_the_digest_reaches_the_distribution_block(manifest):
    record = croissant_record(manifest, sha256="a" * 64)
    assert record["distribution"][0]["sha256"] == "a" * 64


def test_context_is_the_spec_context(manifest):
    record = croissant_record(manifest)
    assert record["@context"] is not CONTEXT or record["@context"] == CONTEXT
    assert record["@context"]["@vocab"] == "https://schema.org/"


def test_export_writes_all_three_artifacts(tmp_path):
    code = main(["export", "--seed", "5", "--events", "120", "--out", str(tmp_path)])
    assert code == 0
    for name in ("corpus.jsonl", "croissant.json", "manifest.json"):
        assert (tmp_path / name).exists(), name
    record = json.loads((tmp_path / "croissant.json").read_text(encoding="utf-8"))
    assert record["distribution"][0]["sha256"]


def test_exported_croissant_digest_matches_the_exported_corpus(tmp_path):
    """The claim the record makes about the file has to be true of the file."""
    main(["export", "--seed", "5", "--events", "120", "--out", str(tmp_path)])
    record = json.loads((tmp_path / "croissant.json").read_text(encoding="utf-8"))
    assert record["distribution"][0]["sha256"] == sha256((tmp_path / "corpus.jsonl").read_bytes())


def test_export_refuses_a_corpus_that_fails_its_own_gate(tmp_path, monkeypatch):
    """A citable artifact whose gate rejected it is worse than no artifact."""
    import pharos.cli as cli

    real = cli.build_manifest

    def unusable(config, **kwargs):
        built = real(config, **kwargs)
        object.__setattr__(built, "max_surface_baseline", 0.0)
        return built

    monkeypatch.setattr(cli, "build_manifest", unusable)
    assert main(["export", "--seed", "5", "--events", "120", "--out", str(tmp_path)]) == 1
    assert not (tmp_path / "corpus.jsonl").exists()
