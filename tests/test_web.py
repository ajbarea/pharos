"""The explorer must show what the Python API says, not a second implementation.

The model endpoint is not exercised here. It calls a live backend, which is not a
thing a test suite should require, and its failure mode is covered instead: an
unreachable backend must surface as a clear 502 rather than a traceback.
"""

import importlib.util

import pytest

pytest.importorskip("fastapi", reason="install the 'ui' dependency group")

# starlette's TestClient needs an httpx-family client, and which one moves: plain
# `httpx` is deprecated in favour of `httpx2`. Naming one of them here is how this
# whole file silently skipped once, so accept either and skip only if both are gone.
if not any(importlib.util.find_spec(name) for name in ("httpx2", "httpx")):
    pytest.skip("install the 'ui' dependency group", allow_module_level=True)

from fastapi.testclient import TestClient

from pharos.labels import Capacity, Compartment, Label, Sensitivity
from pharos.web import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_the_page_is_served_and_is_self_contained(client):
    """No CDN: a testbed that runs offline must not have a front door that does not."""
    body = client.get("/").text
    assert "<title>Pharos explorer</title>" in body
    for remote in ("https://cdn", "http://cdn", "unpkg.com", "jsdelivr", "googleapis"):
        assert remote not in body, f"page reaches out to {remote}"


def test_vocabulary_comes_from_the_enums(client):
    payload = client.get("/api/vocabulary").json()
    assert payload["sensitivity"] == [s.name for s in Sensitivity]
    assert payload["capacity"] == [c.name for c in Capacity]
    assert set(payload["compartments"]) == {str(c) for c in Compartment}


def test_models_endpoint_lists_the_registry(client):
    payload = client.get("/api/models").json()
    assert payload["default"] == "qwen2.5-7b"
    assert {"key", "tag", "verified", "installed"} <= set(payload["models"][0])


def test_corpus_is_deterministic_across_requests(client):
    first = client.get("/api/corpus?seed=3&events=20").json()
    second = client.get("/api/corpus?seed=3&events=20").json()
    assert first == second


def test_corpus_reports_carry_unpacked_labels(client):
    payload = client.get("/api/corpus?seed=3&events=20").json()
    label = payload["reports"][0]["label"]
    assert set(label) == {"sensitivity", "compartments", "capacity", "rendered"}
    assert len(payload["label_histogram"]) > 1


def test_corpus_refuses_an_unbounded_request(client):
    """A curious click must not be able to start a minutes-long job."""
    assert client.get("/api/corpus?events=99999").status_code == 400


def test_dominance_reports_incomparability(client):
    """The case the whole lattice exists for, and the one a one-way answer hides."""
    body = {
        "holder": {"sensitivity": "PROTECTED", "capacity": "FREETEXT", "compartments": ["SENSOR"]},
        "item": {"sensitivity": "PROTECTED", "capacity": "FREETEXT", "compartments": ["LIAISON"]},
    }
    payload = client.post("/api/dominance", json=body).json()
    assert payload["holder_may_read_item"] is False
    assert payload["item_may_read_holder"] is False
    assert payload["incomparable"] is True


def test_dominance_agrees_with_the_library(client):
    holder = Label(Sensitivity.RESTRICTED, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    item = Label(Sensitivity.INTERNAL, frozenset({Compartment.SENSOR}), Capacity.FREETEXT)
    body = {
        "holder": {"sensitivity": "RESTRICTED", "capacity": "FREETEXT", "compartments": ["SENSOR"]},
        "item": {"sensitivity": "INTERNAL", "capacity": "FREETEXT", "compartments": ["SENSOR"]},
    }
    payload = client.post("/api/dominance", json=body).json()
    assert payload["holder_may_read_item"] == holder.dominates(item)


def test_dominance_fails_closed_on_an_unknown_component(client):
    body = {"holder": {"sensitivity": "NOT_A_LEVEL"}, "item": {}}
    assert client.post("/api/dominance", json=body).status_code == 400


def test_dominance_fails_closed_on_a_missing_side(client):
    assert client.post("/api/dominance", json={"holder": {}}).status_code == 400


def test_gate_reports_its_own_reduced_settings(client):
    payload = client.get("/api/gate?seed=7&events=60&null_trials=3").json()
    assert 0.0 <= payload["surface_baseline"] <= 1.0
    assert payload["null_mean"] is not None
    # The endpoint runs smaller than the published configuration, and has to say so
    # rather than let a casual reader quote it as the headline number.
    assert "noisier" in payload["note"]


def test_triage_surfaces_a_dead_backend_as_502(client):
    """An unreachable model is an expected state, not a crash."""
    body = {"model": "qwen2.5-7b", "seed": 7, "index": 0, "endpoint": "http://127.0.0.1:1/nope"}
    response = client.post("/api/triage", json=body)
    assert response.status_code == 502
    assert "model call failed" in response.json()["detail"]
