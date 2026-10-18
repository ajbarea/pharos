"""The explorer as a reader meets it: the page itself, run in a browser.

`test_web.py` calls every endpoint the page uses, and `test_escapes.py` checks that the
freeze carries the argument and that the page routes it. None of that runs the page,
which is how a tab whose panel never opened, a model tab that threw on every answer and
button text at 1.7:1 all shipped (#86). These drive Chromium against a frozen explorer,
the same two files GitHub Pages serves.

The docs `build` job runs them against the copy it is about to publish, by setting
`PHAROS_EXPLORER_DIR=docs/explorer`, so what passes here is what deploys. Without that
variable the fixture freezes its own copy, which calls every model-free endpoint over
the frozen input space and is slow. The CI matrix deselects the `browser` marker: it
installs no browser, and the docs job already runs these.
"""

import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, override
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.browser

pytest.importorskip("playwright.sync_api", reason="install the 'browser' dependency group")
pytest.importorskip("fastapi", reason="install the 'ui' dependency group")

from playwright.sync_api import Browser, Page, Route, ViewportSize, sync_playwright  # noqa: E402
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import Request as PlaywrightRequest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

WIDE: ViewportSize = {"width": 1920, "height": 1080}
PHONE: ViewportSize = {"width": 390, "height": 844}

#: WCAG 2.2 SC 1.4.3, text below the large-text size.
TEXT_CONTRAST = 4.5

# Reads the computed colours of every visible, enabled button, resolving a transparent
# button against the first opaque ancestor. A semi-transparent fill has no single
# contrast ratio, so it is reported as `null` and fails rather than being guessed at.
BUTTON_CONTRAST = """() => {
  const parse = (s) => s.match(/[\\d.]+/g).map(Number);
  const opaque = (c) => c.length === 3 || c[3] === 1;
  const lum = (c) => {
    const f = (v) => ((v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  };
  const ratio = (a, b) => {
    const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };
  const backdrop = (el) => {
    for (let e = el; e; e = e.parentElement) {
      const c = parse(getComputedStyle(e).backgroundColor);
      if (opaque(c)) return c;
      if (c[3] > 0) return null;
    }
    return [255, 255, 255];
  };
  return [...document.querySelectorAll('button')]
    .filter((b) => !b.disabled && b.offsetParent !== null)
    .map((b) => {
      const bg = backdrop(b);
      return {
        button: b.id || b.textContent.trim(),
        ratio: bg === null ? null : ratio(parse(getComputedStyle(b).color), bg),
      };
    });
}"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Keep the request log out of the test output."""


@pytest.fixture(scope="module")
def explorer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    given = os.environ.get("PHAROS_EXPLORER_DIR")
    if given:
        path = Path(given)
        assert (path / "index.html").is_file(), f"{path} holds no index.html; run make explorer"
        assert (path / "bundle.json").is_file(), f"{path} holds no bundle.json; run make explorer"
        return path
    out = tmp_path_factory.mktemp("explorer")
    subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts/build_static_explorer.py"), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def bundle(explorer_dir: Path) -> dict[str, Any]:
    return json.loads((explorer_dir / "bundle.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def explorer_url(explorer_dir: Path) -> Iterator[str]:
    handler = functools.partial(_QuietHandler, directory=str(explorer_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch()
        except PlaywrightError as exc:
            # The docs job installs Chromium, and `make explorer-check` names a frozen
            # explorer to test. In either, a skip would pass by testing nothing.
            if os.environ.get("CI") or os.environ.get("PHAROS_EXPLORER_DIR"):
                raise
            pytest.skip(
                f"Chromium is not installed; run `uv run playwright install chromium`: {exc}"
            )
        yield chromium
        chromium.close()


@pytest.fixture
def open_page(browser: Browser, explorer_url: str) -> Iterator[Callable[..., Page]]:
    """Open the explorer, failing the test on any page error or console error.

    A missing `bundle.json` is how the page detects a live server, so the 404 a live-mode
    test arranges for it is expected and is the one console error let through.
    """
    contexts = []
    errors: list[str] = []

    def console(message: Any) -> None:
        if message.type == "error" and not message.location.get("url", "").endswith("/bundle.json"):
            errors.append(f"console: {message.text}")

    def open_(
        viewport: ViewportSize = WIDE,
        color_scheme: Literal["light", "dark"] = "light",
        setup: Callable[[Page], None] | None = None,
        hash: str = "",
    ) -> Page:
        context = browser.new_context(viewport=viewport, color_scheme=color_scheme)
        contexts.append(context)
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", console)
        if setup:
            setup(page)
        page.goto(explorer_url + hash)
        return page

    yield open_
    for context in contexts:
        context.close()
    assert not errors, "\n".join(errors)


def _ready(page: Page) -> None:
    """Wait for the page to load its data and paint its first view."""
    page.wait_for_function("document.body.dataset.ready === 'true'")


def _tab_ids(page: Page) -> list[str]:
    ids = page.eval_on_selector_all('[role="tab"]', "tabs => tabs.map(t => t.id)")
    return [str(tab_id) for tab_id in ids]


def _open(page: Page, tab_id: str) -> str:
    """Select a tab and wait for its view to paint its answer; return the panel id."""
    page.click(f"#{tab_id}")
    panel_id = str(page.get_attribute(f"#{tab_id}", "aria-controls"))
    page.wait_for_function(
        """id => {
          const answer = document.querySelector('#' + id + ' .answer');
          return answer && answer.offsetParent !== null && answer.textContent.trim().length > 0;
        }""",
        arg=panel_id,
    )
    # A view that throws paints its message where the answer goes. Waiting for any
    # answer would accept it, which is how a broken render could pass.
    error = page.eval_on_selector(
        f"#{panel_id}", "p => p.querySelector('.answer.error')?.textContent"
    )
    assert error is None, f"{tab_id} failed to render: {error}"
    return panel_id


def test_a_returning_reader_gets_the_bundle_a_redeploy_published(browser, explorer_dir, tmp_path):
    """A redeploy keeps the bundle's URL, so a reader's cached copy must not outlive it.

    The page once fetched it with `force-cache`, which never asks the server again: a
    returning reader got the new page with the old bundle, and three views threw.
    """
    site = tmp_path / "site"
    shutil.copytree(explorer_dir, site)
    bundle_file = site / "bundle.json"
    deployed = json.loads(bundle_file.read_text(encoding="utf-8"))
    dropped = str(deployed["seeds"][-1])

    handler = functools.partial(_QuietHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    context = browser.new_context(viewport=WIDE)
    try:
        page = context.new_page()
        page.goto(url)
        _ready(page)
        assert dropped in page.inner_text("#seed-control")

        redeployed = dict(deployed, seeds=deployed["seeds"][:-1])
        bundle_file.write_text(json.dumps(redeployed), encoding="utf-8")
        later = time.time() + 60
        os.utime(bundle_file, (later, later))

        page.goto(url)
        _ready(page)
        assert dropped not in page.inner_text("#seed-control"), "the page kept the old bundle"
    finally:
        context.close()
        server.shutdown()
        server.server_close()


def test_every_tab_opens_its_own_panel_and_only_that_one(open_page):
    page = open_page()
    _ready(page)
    for tab_id in _tab_ids(page):
        panel_id = _open(page, tab_id)
        shown = page.eval_on_selector_all(
            '[role="tabpanel"]',
            "panels => panels.filter(p => p.getBoundingClientRect().height > 0).map(p => p.id)",
        )
        assert shown == [panel_id], f"{tab_id} shows {shown}"
        assert page.inner_text(f"#{panel_id}").strip(), f"{panel_id} opened empty"
        selected = page.eval_on_selector_all(
            '[role="tab"][aria-selected="true"]', "tabs => tabs.map(t => t.id)"
        )
        assert selected == [tab_id]


def test_the_exits_panel_shows_every_exit_the_bundle_carries(open_page, bundle):
    page = open_page()
    _ready(page)
    page.click("#tab-btn-escapes")
    expected = len(bundle["escapes"]["tension"]) + len(bundle["escapes"]["escapes"])
    cards = page.locator("#tab-escapes article.escape")
    assert cards.count() == expected
    assert all(cards.nth(i).is_visible() for i in range(expected))


def test_arrow_keys_walk_every_tab_and_select_where_focus_lands(open_page):
    """No tab is disabled, the offline model tab included: its view explains itself.

    A disabled tab cannot take focus, which is how stepping onto one once selected a
    tab the keyboard was not on.
    """
    page = open_page()
    _ready(page)
    enabled = _tab_ids(page)
    assert page.locator('[role="tab"]:disabled').count() == 0

    def state() -> dict[str, Any]:
        return page.evaluate(
            """() => ({
              focused: document.activeElement.id,
              selected: [...document.querySelectorAll('[role="tab"][aria-selected="true"]')]
                .map(t => t.id),
            })"""
        )

    page.focus(f"#{enabled[0]}")
    for expected in [*enabled[1:], enabled[0]]:
        page.keyboard.press("ArrowRight")
        assert state() == {"focused": expected, "selected": [expected]}
    page.keyboard.press("ArrowLeft")
    assert state() == {"focused": enabled[-1], "selected": [enabled[-1]]}
    page.keyboard.press("Home")
    assert state() == {"focused": enabled[0], "selected": [enabled[0]]}
    page.keyboard.press("End")
    assert state() == {"focused": enabled[-1], "selected": [enabled[-1]]}


@pytest.mark.parametrize("color_scheme", ["light", "dark"])
def test_every_button_label_is_readable_in_both_schemes(open_page, color_scheme):
    page = open_page(color_scheme=color_scheme)
    _ready(page)
    failures = []
    for tab_id in _tab_ids(page):
        _open(page, tab_id)
        failures.extend(
            f"{tab_id} / {row['button']}: {row['ratio']}"
            for row in page.evaluate(BUTTON_CONTRAST)
            if row["ratio"] is None or row["ratio"] < TEXT_CONTRAST
        )
    assert not failures, f"below {TEXT_CONTRAST}:1 in {color_scheme}: {sorted(set(failures))}"


def test_no_tab_scrolls_sideways_at_phone_width(open_page):
    page = open_page(viewport=PHONE)
    _ready(page)
    for tab_id in _tab_ids(page):
        _open(page, tab_id)
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0, f"{tab_id} scrolls {overflow}px sideways at {PHONE['width']}px"


def test_live_mode_renders_a_model_answer(open_page, monkeypatch):
    """The model tab only runs against a live server, so no offline check reaches it.

    The page is served as the live app would serve it: no bundle, and every `/api/` call
    answered by the real FastAPI app. Only the model call is stubbed.
    """
    from fastapi.testclient import TestClient

    import pharos.web

    monkeypatch.setattr(pharos.web, "generate_text", lambda prompt, **_: "SIGNIFICANT")
    client = TestClient(pharos.web.create_app())

    def forward(route: Route, request: PlaywrightRequest) -> None:
        parts = urlsplit(request.url)
        response = client.request(
            request.method,
            parts.path + (f"?{parts.query}" if parts.query else ""),
            content=request.post_data_buffer,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        route.fulfill(
            status=response.status_code,
            body=response.content,
            content_type=response.headers.get("content-type"),
        )

    def as_live_server(page: Page) -> None:
        page.route("**/bundle.json", lambda route: route.fulfill(status=404))
        page.route("**/api/**", forward)

    page = open_page(setup=as_live_server)
    _ready(page)
    page.wait_for_selector("#t-model option", state="attached")
    page.click("#tab-btn-triage")
    page.click("#t-go")
    page.wait_for_function("() => !document.getElementById('t-go').disabled")
    assert "Model said SIGNIFICANT" in page.inner_text("#t-out")


def test_every_view_answers_on_arrival_without_a_click(open_page, bundle):
    """The bundle already holds every answer, so no view should open on an empty form."""
    page = open_page()
    _ready(page)
    seed = str(page.eval_on_selector('input[name="seed"]:checked', "r => r.value"))
    corpus = bundle["corpus"][seed]
    significant = sum(1 for row in corpus["export_rows"] if row["is_plant"])
    answers = {
        tab_id: page.inner_text(f"#{_open(page, tab_id)} .answer") for tab_id in _tab_ids(page)
    }
    assert f"{corpus['n_reports']} reports" in answers["tab-btn-corpus"]
    assert f"{significant} of them significant" in answers["tab-btn-corpus"]
    assert str(bundle["gate"][seed]["surface_baseline"]) in answers["tab-btn-gate"]
    counts = bundle["escapes"]["counts"]
    assert f"{counts['closes']} exits close" in answers["tab-btn-escapes"]


def test_every_view_uses_the_width_of_a_wide_screen(open_page):
    """At 1920 the views were pinned to the left 1100 pixels and the rest sat empty."""
    page = open_page()
    _ready(page)
    for tab_id in _tab_ids(page):
        panel_id = _open(page, tab_id)
        box = page.evaluate(
            """id => {
              const canvas = document.querySelector('#' + id + ' .canvas').getBoundingClientRect();
              return {right: canvas.right, width: canvas.width, viewport: innerWidth};
            }""",
            panel_id,
        )
        assert box["right"] >= box["viewport"] - 1, f"{tab_id} stops short of the right edge"
        assert box["width"] >= 0.75 * box["viewport"], f"{tab_id} canvas is {box['width']}px"


def test_a_link_reopens_the_view_it_was_copied_from(open_page, bundle):
    page = open_page()
    _ready(page)
    seed = str(bundle["seeds"][-1])
    page.click(f'input[name="seed"][value="{seed}"]', force=True)
    _open(page, "tab-btn-lattice")
    page.click('#l-canvas .node[data-cell="RESTRICTED|LEGAL,SENSOR"]')
    page.wait_for_function("location.hash.includes('object=RESTRICTED')")
    link = page.url

    # A fresh page from the link alone, as a reader who was sent it would open it.
    again = open_page(hash="#" + link.split("#", 1)[1])
    _ready(again)
    assert again.get_attribute("#tab-btn-lattice", "aria-selected") == "true"
    assert again.eval_on_selector('input[name="seed"]:checked', "r => r.value") == seed
    assert again.input_value("#i-sens") == "RESTRICTED"
    ticked = again.eval_on_selector_all(
        "#i-comps input:checked", "boxes => boxes.map(b => b.value)"
    )
    assert sorted(ticked) == ["LEGAL", "SENSOR"]


def test_the_corpus_lists_every_report_and_a_cell_narrows_it(open_page, bundle):
    page = open_page()
    _ready(page)
    seed = str(page.eval_on_selector('input[name="seed"]:checked', "r => r.value"))
    corpus = bundle["corpus"][seed]
    _open(page, "tab-btn-corpus")
    assert page.locator("#c-canvas table.reports tbody tr").count() == corpus["n_reports"]

    cell = page.locator('#c-canvas .cell[data-class="all"]:not([disabled])').first
    key = str(cell.get_attribute("data-cell"))
    sensitivity, compartments = key.split("|")
    expected = sum(
        1
        for row in corpus["export_rows"]
        if row["sensitivity"] == sensitivity
        and ",".join(sorted(row["compartments"])) == compartments
    )
    cell.click()
    page.wait_for_function(
        "n => document.querySelectorAll('#c-canvas table.reports tbody tr').length === n",
        arg=expected,
    )


def test_review_opens_on_the_first_task_where_the_reviewers_act_differently(open_page, bundle):
    page = open_page()
    _ready(page)
    seed = str(page.eval_on_selector('input[name="seed"]:checked', "r => r.value"))
    tasks = bundle["review"][seed]["true"]
    first = next(t for t in tasks if len({r["action"] for r in t["reviewers"]}) > 1)
    _open(page, "tab-btn-review")
    assert first["task_id"] in page.inner_text("#r-detail-h")


def test_the_gate_draws_the_threshold_the_gate_tests_against(open_page, bundle):
    """Significance is the baseline against the null's 95th percentile, not its spread."""
    page = open_page()
    _ready(page)
    seed = str(page.eval_on_selector('input[name="seed"]:checked', "r => r.value"))
    gate = bundle["gate"][seed]
    _open(page, "tab-btn-gate")
    assert page.get_attribute("#g-canvas .threshold", "data-value") == str(gate["null_p95"])
    assert page.get_attribute("#g-canvas .baseline", "data-value") == str(gate["surface_baseline"])


def test_the_longest_label_fits_a_phone(open_page):
    """Every compartment held is the longest label the lattice can print."""
    longest = "RESTRICTED|LEGAL,LIAISON,PARTNER,SENSOR|FREETEXT"
    page = open_page(viewport=PHONE, hash=f"#tab=lattice&holder={longest}&object={longest}")
    _ready(page)
    _open(page, "tab-btn-lattice")
    assert "RESTRICTED[LEGAL,LIAISON,PARTNER,SENSOR]" in page.inner_text("#l-canvas .verdicts")
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 0, f"the lattice scrolls {overflow}px sideways at {PHONE['width']}px"


@pytest.mark.parametrize(
    ("holder", "item"),
    [
        # Same level, different compartments: the case the lattice exists for.
        ("PROTECTED|SENSOR", "PROTECTED|LIAISON"),
        # Different levels, still incomparable: the holder is higher and lacks one.
        ("RESTRICTED|SENSOR", "PROTECTED|LIAISON,SENSOR"),
        # Comparable: the holder dominates.
        ("RESTRICTED|LEGAL", "INTERNAL|"),
    ],
)
def test_the_lattice_says_what_blocks_each_read(open_page, bundle, holder, item):
    """The sentence is built from the shortfall Python reports, so it is checked against it."""
    page = open_page(hash=f"#tab=lattice&holder={holder}|ENUM&object={item}|ENUM")
    _ready(page)
    answer = page.inner_text(f"#{_open(page, 'tab-btn-lattice')} .answer")
    cell = bundle["dominance"]["pairs"][holder][item]
    if cell["holder_may_read_item"]:
        assert answer.startswith("The holder may read the object.")
        return
    for who, other, short in (
        ("holder", "object", cell["holder_short"]),
        ("object", "holder", cell["item_short"]),
    ):
        if short["compartments"]:
            assert f"the {who} lacks {' and '.join(short['compartments'])}" in answer
        if short["level"]:
            assert f"the {who}\u2019s level is below the {other}\u2019s" in answer


@pytest.mark.parametrize("color_scheme", ["light", "dark"])
def test_the_cues_that_carry_an_answer_clear_three_to_one(open_page, color_scheme):
    """WCAG 1.4.11: a lit lattice cell and the gate's null and chance band must be visible."""
    page = open_page(color_scheme=color_scheme)
    _ready(page)
    _open(page, "tab-btn-lattice")
    lattice = page.evaluate(
        BUTTON_CONTRAST.split("return [...document")[0]
        + """return {
          lit: ratio(parse(getComputedStyle(document.querySelector('.node.reads')).borderTopColor),
                     parse(getComputedStyle(document.querySelector('.node:not(.reads):not(.holder)')).backgroundColor)),
        };
      }"""
    )
    _open(page, "tab-btn-gate")
    gate = page.evaluate(
        BUTTON_CONTRAST.split("return [...document")[0]
        + """const page = parse(getComputedStyle(document.body).backgroundColor);
        return {
          edge: ratio(parse(getComputedStyle(document.querySelector('.chance-edge')).stroke), page),
          spread: ratio(parse(getComputedStyle(document.querySelector('.spread')).stroke), page),
        };
      }"""
    )
    for name, value in {**lattice, **gate}.items():
        assert value >= 3, f"{name} is {value:.2f}:1 in {color_scheme}"


def test_a_hash_naming_no_view_falls_back_to_the_first(open_page):
    """`tab=btn-corpus` once matched a tab button's id and left the page blank."""
    page = open_page(hash="#tab=btn-corpus&seed=abc&holder=|||")
    _ready(page)
    assert page.get_attribute("#tab-btn-corpus", "aria-selected") == "true"
    assert page.inner_text("#c-canvas .answer").strip()


def test_each_grid_is_a_single_tab_stop(open_page):
    """The review matrix put 360 buttons between the tabs and the task it opened."""
    page = open_page()
    _ready(page)
    for tab_id, grid in (
        ("tab-btn-review", "#r-canvas .matrix"),
        ("tab-btn-lattice", "#l-canvas .lattice"),
    ):
        _open(page, tab_id)
        stops = page.eval_on_selector_all(
            f"{grid} button", "bs => bs.filter(b => b.tabIndex >= 0).length"
        )
        assert stops == 1, f"{grid} has {stops} tab stops"
    _open(page, "tab-btn-review")
    page.focus("#r-canvas .colhead[tabindex='0']")
    before = page.inner_text("#r-detail-h")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "t => document.querySelector('#r-detail-h').textContent !== t", arg=before
    )
    assert page.evaluate("document.activeElement.classList.contains('colhead')")


def test_live_gate_names_only_the_centers_it_scored(open_page, monkeypatch):
    """A center with one class has no fold; its neighbour's AUC once wore its name."""
    from fastapi.testclient import TestClient

    import pharos.web

    client = TestClient(pharos.web.create_app())
    expected = client.get("/api/gate?seed=19&events=30").json()
    skipped = set(expected["held_out_centers"]) - set(expected["scored_centers"])
    assert skipped, "seed 19 at 30 events no longer skips a fold; pick another input"

    def forward(route: Route, request: PlaywrightRequest) -> None:
        parts = urlsplit(request.url)
        response = client.request(
            request.method,
            parts.path + (f"?{parts.query}" if parts.query else ""),
            content=request.post_data_buffer,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        route.fulfill(
            status=response.status_code,
            body=response.content,
            content_type=response.headers.get("content-type"),
        )

    def as_live_server(page: Page) -> None:
        page.route("**/bundle.json", lambda route: route.fulfill(status=404))
        page.route("**/api/**", forward)

    page = open_page(setup=as_live_server, hash="#tab=gate&seed=19&events=30")
    _ready(page)
    page.click("#g-go")
    page.wait_for_selector("#g-canvas .chart")
    titles = " ".join(
        page.eval_on_selector_all("#g-canvas .fold title", "ts => ts.map(t => t.textContent)")
    )
    for center in skipped:
        assert center not in titles
    for center in expected["scored_centers"]:
        assert center in titles
