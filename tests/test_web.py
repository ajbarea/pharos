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


def test_the_page_invents_no_values(client):
    """Every number on the page comes from an endpoint, so the page has no use for chance.

    A visualizer tab once printed a `Math.random()` string as a "SHA-256 digest" beside a
    typed-in latency, on a page whose header says everything runs against the Python.
    """
    assert "Math.random" not in client.get("/").text


def test_every_tab_names_a_panel_and_every_panel_has_a_tab(client):
    """Panel switching reads `aria-controls`, so a tab without its panel is a dead tab.

    The browser test checks that each panel opens; this checks the wiring it relies on,
    on every interpreter, without a browser.
    """
    from html.parser import HTMLParser
    from typing import override

    class Roles(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.controls: list[str] = []
            self.panels: list[str] = []

        @override
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            found = dict(attrs)
            if found.get("role") == "tab":
                self.controls.append(found.get("aria-controls") or "")
            elif found.get("role") == "tabpanel":
                self.panels.append(found.get("id") or "")

    roles = Roles()
    roles.feed(client.get("/").text)
    assert roles.controls, "found no tabs; the parser or the page changed"
    assert sorted(roles.controls) == sorted(roles.panels)


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


def test_the_lattice_table_agrees_with_the_library_on_every_pair(client):
    """The page draws the whole lattice from this table, so every cell has to be right."""
    from pharos.web import _cell_key, _compartment_subsets

    table = client.get("/api/lattice").json()
    cells = {
        _cell_key(s.name, subset): Label(
            s, frozenset(Compartment[c] for c in subset), Capacity.FREETEXT
        )
        for s in Sensitivity
        for subset in _compartment_subsets()
    }
    assert set(table["pairs"]) == set(cells)
    for holder_key, holder in cells.items():
        for item_key, item in cells.items():
            cell = table["pairs"][holder_key][item_key]
            assert cell["holder_may_read_item"] == holder.dominates(item)
            assert cell["item_may_read_holder"] == item.dominates(holder)


def test_a_refusal_names_exactly_what_dominance_checks(client):
    """The page explains a refusal from these fields, so they must match the verdict.

    No shortfall on either condition must mean the read is allowed, and any shortfall
    must mean it is refused, on every ordered pair of cells.
    """
    table = client.get("/api/lattice").json()
    for row in table["pairs"].values():
        for cell in row.values():
            for allowed, short in (
                (cell["holder_may_read_item"], cell["holder_short"]),
                (cell["item_may_read_holder"], cell["item_short"]),
            ):
                assert allowed == (not short["level"] and not short["compartments"])


def test_the_lattice_table_answers_what_the_dominance_endpoint_answers(client):
    """Two routes to one decision, checked on the incomparable case and a comparable one."""
    table = client.get("/api/lattice").json()
    for holder, item in [
        (("PROTECTED", ["SENSOR"]), ("PROTECTED", ["LIAISON"])),
        (("RESTRICTED", ["SENSOR"]), ("INTERNAL", [])),
    ]:
        body = {
            "holder": {"sensitivity": holder[0], "compartments": holder[1], "capacity": "SPAN"},
            "item": {"sensitivity": item[0], "compartments": item[1], "capacity": "ENUM"},
        }
        direct = client.post("/api/dominance", json=body).json()
        cell = table["pairs"][f"{holder[0]}|{','.join(holder[1])}"][
            f"{item[0]}|{','.join(item[1])}"
        ]
        assert cell["incomparable"] == direct["incomparable"]
        assert cell["holder_may_read_item"] == direct["holder_may_read_item"]
        capacity = body[table["capacity_follows"]]["capacity"]
        assert table["labels"][f"{cell['join']}|{capacity}"] == direct["join"]


def test_review_all_is_each_task_reviewed_in_turn(client):
    """One call for the whole matrix, and it must be the same answers, task by task."""
    every = client.get("/api/review/all", params={"seed": 7, "verdict": False}).json()["tasks"]
    assert every
    for index in (0, 1, len(every) - 1):
        one = client.get("/api/review", params={"seed": 7, "index": index, "verdict": False}).json()
        assert every[index] == one


def test_review_shows_the_reports_the_reviewers_read(client):
    payload = client.get("/api/review", params={"seed": 7, "index": 3}).json()
    corpus = client.get("/api/corpus", params={"seed": 7, "events": 40}).json()
    texts = {r["report_id"]: r["text"] for r in corpus["reports"]}
    assert payload["sources"]
    for source in payload["sources"]:
        assert texts[source["report_id"]] == source["text"]
        assert set(source["label"]) == {"sensitivity", "compartments", "capacity", "rendered"}


def test_gate_reports_the_threshold_it_actually_tests_against(client):
    """Significance is the baseline against the null's 95th percentile, so ship that."""
    payload = client.get("/api/gate?seed=7&events=60&null_trials=3").json()
    assert payload["null_p95"] is not None
    assert payload["leak_is_significant"] == (payload["surface_baseline"] > payload["null_p95"])
    assert len(payload["band"]) == 2
    for probe, folds in payload["per_fold_auc"].items():
        assert len(folds) == len(payload["scored_centers"]), probe


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


def test_review_shows_every_reviewer_and_what_it_disclosed(client):
    payload = client.get("/api/review", params={"seed": 7, "index": 0, "verdict": True}).json()

    assert payload["evidence_needed"] == 3
    assert len(payload["evidence_shown"]) == (
        3 if payload["truth_significant"] else len(payload["evidence_shown"])
    )
    names = [r["analyst"] for r in payload["reviewers"]]
    assert "by-the-book" in names and "unexplained" in names

    by_name = {r["analyst"]: r for r in payload["reviewers"]}
    # The by-the-book reviewer applies the world's own rule, so its corrected
    # verdict is the truth whenever it supplies one.
    control = by_name["by-the-book"]
    if control["corrected_verdict"] is not None:
        assert control["corrected_verdict"] == payload["truth_significant"]
    # An objection with no stated grounds is the whole point of that row.
    silent = by_name["unexplained"]
    if silent["action"] != "accept":
        assert silent["grounds"] == []


def test_review_reports_whether_a_correction_can_actually_leave(client):
    """A reviewer can object correctly and hand back something still blocked."""
    blocked = None
    for index in range(20):
        payload = client.get("/api/review", params={"index": index, "verdict": False}).json()
        if not payload["proposal_releasable"]:
            blocked = payload
            break
    assert blocked is not None, "no unreleasable proposal in the first 20 tasks"

    by_name = {r["analyst"]: r for r in blocked["reviewers"]}
    assert by_name["by-the-book"]["correction_releasable"] is False
    assert by_name["releaser"]["correction_releasable"] is True


def test_every_request_parameter_is_bounded(client):
    """The cost bound the module comment claims, checked on every parameter that feeds it.

    `MAX_EVENTS` was the whole of it, and gate cost is events times null trials -- so the
    documented promise that "a curious click cannot start a minutes-long job" was true of
    one parameter and false of the other. A limit nobody has seen refuse a request is not
    yet a limit, so each is exercised here.
    """
    from pharos.web import MAX_EVENTS, MAX_LIMIT, MAX_NULL_TRIALS

    assert client.get(f"/api/gate?events={MAX_EVENTS + 1}").status_code == 400
    assert client.get(f"/api/gate?null_trials={MAX_NULL_TRIALS + 1}").status_code == 400
    assert client.get("/api/gate?events=0").status_code == 400
    assert client.get("/api/corpus?events=-5").status_code == 400
    assert client.get(f"/api/corpus?limit={MAX_LIMIT + 1}").status_code == 400
    assert client.get("/api/review?seed=-1").status_code == 400
    # And the bound is a bound rather than a ban: the documented defaults still work.
    assert client.get("/api/gate?events=40&null_trials=2").status_code == 200
    assert client.get("/api/corpus?events=40&limit=12").status_code == 200
    assert client.get("/api/review?seed=7&index=3").status_code == 200


def test_a_bad_field_in_the_json_body_is_a_bad_request_not_a_crash(client):
    """`int("abc")` inside an endpoint is a 500 with a traceback for a plainly bad request.

    The triage endpoint coerces its own fields because they arrive in a JSON body rather
    than a query string, so FastAPI does not do it. That coercion had no error path.
    """
    response = client.post("/api/triage", json={"seed": "abc"})
    assert response.status_code == 400
    assert "seed" in response.json()["detail"]

    negative = client.post("/api/triage", json={"index": -1})
    assert negative.status_code == 400
