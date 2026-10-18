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
import subprocess
import sys
import threading
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
            # The docs job installs Chromium. There, a skip would pass by absence.
            if os.environ.get("CI"):
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
    ) -> Page:
        context = browser.new_context(viewport=viewport, color_scheme=color_scheme)
        contexts.append(context)
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", console)
        if setup:
            setup(page)
        page.goto(explorer_url)
        return page

    yield open_
    for context in contexts:
        context.close()
    assert not errors, "\n".join(errors)


def _ready(page: Page) -> None:
    """Wait for the offline build to finish wiring itself up."""
    page.wait_for_selector("#offline-banner:not([hidden])")
    page.wait_for_selector("#escapes-list article", state="attached")


def _enabled_tab_ids(page: Page) -> list[str]:
    ids = page.eval_on_selector_all(
        '[role="tab"]', "tabs => tabs.filter(t => !t.disabled).map(t => t.id)"
    )
    return [str(tab_id) for tab_id in ids]


def test_every_tab_opens_its_own_panel_and_only_that_one(open_page):
    page = open_page()
    _ready(page)
    for tab_id in _enabled_tab_ids(page):
        page.click(f"#{tab_id}")
        panel_id = page.get_attribute(f"#{tab_id}", "aria-controls")
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


def test_arrow_keys_walk_the_enabled_tabs_and_select_where_focus_lands(open_page):
    page = open_page()
    _ready(page)
    enabled = _enabled_tab_ids(page)
    # The offline build disables the model tab. Without a disabled tab to step over,
    # this test could not tell skipping it from never meeting it.
    assert len(enabled) < page.locator('[role="tab"]').count()

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
    # The download buttons exist only once a corpus is on screen.
    page.click("#c-go")
    page.wait_for_selector("#c-dl-jsonl", state="visible")
    failures = []
    for tab_id in _enabled_tab_ids(page):
        page.click(f"#{tab_id}")
        failures.extend(
            f"{tab_id} / {row['button']}: {row['ratio']}"
            for row in page.evaluate(BUTTON_CONTRAST)
            if row["ratio"] is None or row["ratio"] < TEXT_CONTRAST
        )
    assert not failures, f"below {TEXT_CONTRAST}:1 in {color_scheme}: {sorted(set(failures))}"


def test_no_tab_scrolls_sideways_at_phone_width(open_page):
    page = open_page(viewport=PHONE)
    _ready(page)
    for tab_id in _enabled_tab_ids(page):
        page.click(f"#{tab_id}")
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
    page.wait_for_selector("#t-model option", state="attached")
    page.click("#tab-btn-triage")
    page.click("#t-go")
    page.wait_for_function("() => !document.getElementById('t-go').disabled")
    assert "Model said SIGNIFICANT" in page.inner_text("#t-out")
