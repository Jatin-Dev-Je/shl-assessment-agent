"""
catalog/scraper.py
──────────────────
Scrapes the SHL Individual Test Solutions catalog using Playwright.
SHL catalog is JavaScript-rendered — httpx/BS4 cannot access it.

Two-phase scrape:
  Phase 1: Paginate catalog listing → name, url, test_type,
           remote_testing, adaptive
  Phase 2: Visit each detail page   → description, job_levels,
           duration, languages

Usage:
    python catalog/scraper.py
    make scrape
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
OUTPUT_PATH = Path("catalog/catalog.json")
PAGE_SIZE = 12
MAX_PAGES = 40
DETAIL_DELAY = 1.5


def _make_context(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        extra_http_headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        },
    )
    return browser, context


def _parse_listing_page(page: Page, start: int) -> list[dict[str, Any]]:
    url = f"{CATALOG_URL}?start={start}&type=1"
    print(f"\nPage start={start} → {url}")

    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
    except Exception as e:
        print(f"  Navigation error: {e}")
        return []

    page.wait_for_timeout(int(random.uniform(2500, 4000)))

    rows = (
        page.query_selector_all("table tbody tr")
        or page.query_selector_all(".product-catalogue__row")
        or page.query_selector_all("[class*='catalogue'][class*='row']")
        or page.query_selector_all("tr[data-entity-id]")
    )

    print(f"  Rows found: {len(rows)}")

    if not rows:
        html = page.content()
        debug_path = Path("catalog/debug.html")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(html, encoding="utf-8")
        print("  No rows found — saved catalog/debug.html")
        return []

    assessments: list[dict[str, Any]] = []

    for row in rows:
        try:
            link = row.query_selector("a")
            if not link:
                continue
            name = link.inner_text().strip()
            href = link.get_attribute("href")
            if not name or not href:
                continue
            full_url = f"{BASE_URL}{href}" if href.startswith("/") else href
            cells = row.query_selector_all("td")
            test_type = cells[-1].inner_text().strip() if len(cells) > 1 else ""
            remote_testing = bool(
                row.query_selector("[class*='remote']")
                or row.query_selector("[title*='remote' i]")
            )
            adaptive = bool(
                row.query_selector("[class*='adaptive']")
                or row.query_selector("[title*='adaptive' i]")
            )
            assessments.append({
                "name": name,
                "url": full_url,
                "test_type": test_type,
                "remote_testing": remote_testing,
                "adaptive": adaptive,
                "description": "",
                "job_levels": [],
                "languages": [],
                "duration": "",
            })
            print(f"    + {name} [{test_type}]")
        except Exception as e:
            print(f"  Row error: {e}")
            continue

    return assessments


def scrape_listing(context) -> list[dict[str, Any]]:
    all_assessments: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    page = context.new_page()

    for page_num in range(MAX_PAGES):
        start = page_num * PAGE_SIZE
        page_assessments = _parse_listing_page(page, start)

        if not page_assessments:
            print("  Stopping — no rows returned.")
            break

        new_items = [a for a in page_assessments if a["url"] not in seen_urls]

        if not new_items:
            print("  Stopping — no new items.")
            break

        for a in new_items:
            seen_urls.add(a["url"])
        all_assessments.extend(new_items)
        print(f"  +{len(new_items)} new (total: {len(all_assessments)})")

    page.close()
    return all_assessments


def _scrape_detail(page: Page, url: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "description": "",
        "job_levels": [],
        "languages": [],
        "duration": "",
        "remote_testing": False,
        "adaptive": False,
    }

    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)
    except Exception as e:
        print(f"    Detail fetch error: {e}")
        return result

    try:
        desc_el = (
            page.query_selector(".product-catalogue-training-test__description")
            or page.query_selector("[class*='description']")
            or page.query_selector("article p")
            or page.query_selector("main p")
        )
        if desc_el:
            result["description"] = desc_el.inner_text().strip()[:1000]

        known_levels = [
            "Director", "Entry-Level", "Executive",
            "Front Line Manager", "General Population", "Graduate",
            "Manager", "Mid-Professional",
            "Professional Individual Contributor", "Supervisor",
        ]
        page_text = page.inner_text("body").lower()
        result["job_levels"] = [l for l in known_levels if l.lower() in page_text]

        dur_el = page.query_selector("[class*='duration']")
        if dur_el:
            result["duration"] = dur_el.inner_text().strip()[:100]
        else:
            for line in page.inner_text("body").splitlines():
                if "minute" in line.lower() and any(c.isdigit() for c in line):
                    result["duration"] = line.strip()[:100]
                    break

        lang_el = page.query_selector("[class*='language']")
        if lang_el:
            lang_text = lang_el.inner_text().strip()
            result["languages"] = [l.strip() for l in lang_text.split(",") if l.strip()][:20]

        result["remote_testing"] = bool(
            page.query_selector("[class*='remote']") or "remote testing" in page_text
        )
        result["adaptive"] = bool(
            page.query_selector("[class*='adaptive']") or "adaptive" in page_text
        )

    except Exception as e:
        print(f"    Detail parse error: {e}")

    return result


def enrich_with_details(context, assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    total = len(assessments)
    page = context.new_page()

    for i, assessment in enumerate(assessments):
        print(f"  [{i+1}/{total}] {assessment['name'][:60]}")
        details = _scrape_detail(page, assessment["url"])
        merged = {**assessment}
        merged["description"] = details["description"] or assessment["description"]
        merged["job_levels"] = details["job_levels"] or assessment["job_levels"]
        merged["languages"] = details["languages"] or assessment["languages"]
        merged["duration"] = details["duration"] or assessment["duration"]
        if not merged["remote_testing"]:
            merged["remote_testing"] = details["remote_testing"]
        if not merged["adaptive"]:
            merged["adaptive"] = details["adaptive"]
        enriched.append(merged)
        time.sleep(DETAIL_DELAY)

    page.close()
    return enriched


def save_catalog(assessments: list[dict[str, Any]], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved {len(assessments)} assessments → {path}")


def load_catalog(path: Path = OUTPUT_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"catalog.json not found at {path}. Run: make scrape")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"catalog.json at {path} is empty. Run: make scrape")
    return data


def scrape_catalog() -> list[dict[str, Any]]:
    with sync_playwright() as p:
        browser, context = _make_context(p)
        try:
            print("=== Phase 1: Scraping catalog listing ===")
            assessments = scrape_listing(context)
            if not assessments:
                print("ERROR: No assessments found. Check catalog/debug.html")
                return []
            print(f"\n=== Phase 2: Enriching {len(assessments)} detail pages ===")
            enriched = enrich_with_details(context, assessments)
        finally:
            context.close()
            browser.close()
    return enriched


if __name__ == "__main__":
    try:
        assessments = scrape_catalog()
    except Exception as e:
        print(f"\nScrape failed: {e}")
        assessments = []

    print(f"\nTotal scraped: {len(assessments)}")
    if assessments:
        save_catalog(assessments)
    else:
        print("Nothing scraped — check catalog/debug.html")