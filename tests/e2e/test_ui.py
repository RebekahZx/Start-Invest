"""Browser test of the web UI against a live API. Run with: RUN_E2E=1 pytest -m e2e"""

import os
import socket
import threading
import time

import pytest
import uvicorn

from start_invest.api import create_app

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("RUN_E2E") != "1", reason="set RUN_E2E=1 to run browser tests"),
]


@pytest.fixture(scope="module")
def base_url(trained):
    out, _ = trained
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(create_app(str(out / "bundle.joblib")), port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def page(base_url):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None)
        pg = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(base_url)
        pg.wait_for_selector("#rows tr")
        pg.errors = errors
        yield pg
        browser.close()


def test_feed_renders_live(page):
    assert "Live API" in page.inner_text("#mode")
    assert page.locator("#rows tr").count() == 25
    assert page.locator("#rows td.why .strip").count() == 25
    assert not page.errors


def test_drawer_shows_both_explanations(page):
    page.click("#rows tr:nth-child(1)")
    page.wait_for_selector("#d-odds .wf")
    assert page.locator("#d-body .wf").count() == 2  # ranker + evaluator waterfalls
    page.keyboard.press("Escape")
    assert page.get_attribute("#drawer", "aria-hidden") == "true"


def test_shortlist_and_export(page):
    page.click("#rows tr:nth-child(2) .star")
    page.click("#tab-shortlist")
    assert page.locator("#sl-rows tr").count() == 1
    with page.expect_download() as dl:
        page.click("#export")
    assert dl.value.suggested_filename.endswith("-shortlist.csv")
    page.click("#tab-feed")


def test_filters_and_empty_state(page):
    page.fill("#q", "zzzz-no-such-company")
    assert page.is_visible("#feed-empty")
    page.click("#clear-filters")
    assert page.locator("#rows tr").count() == 25


def test_score_form(page):
    page.click("#tab-score")
    page.click("#score-btn")
    page.wait_for_selector("#score-out .wf")
    assert "%" in page.inner_text("#score-out .oddsline")
