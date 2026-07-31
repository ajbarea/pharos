"""The Croissant record is checked by the official validator, not by reading it.

Hand-verifying JSON-LD against a prose spec does not work. The first version of
`pharos.croissant` was written straight from the specification page and looked
correct: it carried every property the spec names, in the right shapes. It was
still invalid, because the abridged `@context` omitted `column`, so every field's
`extract` resolved to nothing. Only `mlcroissant` caught that, and it took one call.

So the record is validated by the tool that consumers will use on it.
"""

import pytest

from pharos.cli import main

mlc = pytest.importorskip(
    "mlcroissant",
    reason="install the 'croissant' dependency group to validate metadata",
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    assert main(["export", "--seed", "5", "--events", "120", "--out", str(out)]) == 0
    return out


def test_the_record_validates_against_the_official_validator(exported):
    dataset = mlc.Dataset(jsonld=str(exported / "croissant.json"))
    assert dataset.metadata.name == "pharos"


def test_the_validator_sees_every_declared_field(exported):
    dataset = mlc.Dataset(jsonld=str(exported / "croissant.json"))
    (record_set,) = dataset.metadata.record_sets
    assert len(record_set.fields) == 12


def test_the_validator_resolves_the_distribution(exported):
    """The failure that motivated this file: a file object nothing pointed at."""
    dataset = mlc.Dataset(jsonld=str(exported / "croissant.json"))
    assert [file.id for file in dataset.metadata.distribution] == ["corpus.jsonl"]
