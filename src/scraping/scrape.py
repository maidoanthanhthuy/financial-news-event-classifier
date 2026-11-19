"""Scrape CafeF and VnExpress listing pages into a CSV; safe to stop and rerun.

Usage: python -m src.scraping.scrape [--max-pages N]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.scraping.config import (
    CAFEF_BASE,
    CAFEF_PAGE_STEP,
    CAFEF_SECTIONS,
    CHECKPOINT_JSON,
    DELAY_SECONDS,
    END_DATE,
    OUTPUT_CSV,
    START_DATE,
    USER_AGENT,
    VNEXPRESS_BASE,
    VNEXPRESS_PAGES,
    VNEXPRESS_SECTIONS,
)
from src.scraping.parse import (
    even_windows,
    new_rows,
    parse_cafef_list,
    parse_vnexpress_list,
    parse_vnexpress_pubdate,
)


def fetch(url: str) -> str:
    """Wait, then download a page with our User-Agent."""
    time.sleep(DELAY_SECONDS)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def load_checkpoint(path: str = CHECKPOINT_JSON) -> dict:
    """Read the checkpoint (task key -> next page or 'done')."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save_checkpoint(state: dict, path: str = CHECKPOINT_JSON) -> None:
    """Write the checkpoint to disk."""
    Path(path).write_text(json.dumps(state, indent=2))


def load_seen_urls(path: str = OUTPUT_CSV) -> set[str]:
    """Return urls already saved in the output CSV."""
    p = Path(path)
    return set(pd.read_csv(p, usecols=["url"])["url"]) if p.exists() else set()


def append_rows(df: pd.DataFrame, path: str = OUTPUT_CSV) -> None:
    """Append rows to the output CSV, writing the header only once."""
    df.to_csv(path, mode="a", header=not Path(path).exists(), index=False)


def save_page(rows: list[dict], seen: set[str]) -> int:
    """Save new rows from one page and return how many were added."""
    df = new_rows(rows, seen, START_DATE, END_DATE)
    append_rows(df)
    seen.update(df["url"])
    return len(df)


def article_time(url: str) -> str | None:
    """Fetch a VnExpress article and return its publish time (None on error)."""
    try:
        return parse_vnexpress_pubdate(fetch(url))
    except requests.RequestException as err:
        print(f"skip {url}: {err}")
        return None


def scrape_cafef(
    section: str, zone_id: int, state: dict, seen: set, max_pages: int | None
) -> None:
    """Sample every CAFEF_PAGE_STEP-th timeline page until older than START_DATE."""
    key = f"cafef:{section}"
    pages_done = 0
    while state.get(key) != "done" and pages_done != max_pages:
        page = state.get(key, 1)
        html = fetch(f"{CAFEF_BASE}/timelinelist/{zone_id}/{page}.chn")
        rows = parse_cafef_list(html, section)
        added = save_page(rows, seen)
        oldest = min((r["published_at"][:10] for r in rows), default="")
        state[key] = "done" if oldest < START_DATE else page + CAFEF_PAGE_STEP
        save_checkpoint(state)
        pages_done += 1
        print(f"{key} page {page}: {len(rows)} items, {added} new, oldest {oldest}")


def scrape_vnexpress(
    section: str, cate_id: int, state: dict, seen: set, max_pages: int | None
) -> None:
    """Take page 1 of each of VNEXPRESS_PAGES equal date windows; open each article."""
    windows = even_windows(START_DATE, END_DATE, VNEXPRESS_PAGES)
    pages_done = 0
    for i, (start_ts, end_ts) in enumerate(windows, start=1):
        key = f"vnexpress:{section}:{i}"
        if state.get(key) == "done" or pages_done == max_pages:
            continue
        url = (
            f"{VNEXPRESS_BASE}/category/day/cateid/{cate_id}/fromdate/"
            f"{start_ts}/todate/{end_ts}/allcate/{cate_id}/page/1"
        )
        rows = parse_vnexpress_list(fetch(url), section)
        rows = [r for r in rows if r["url"] not in seen]
        for r in rows:
            r["published_at"] = article_time(r["url"])
        added = save_page(rows, seen)
        state[key] = "done"
        save_checkpoint(state)
        pages_done += 1
        print(f"{key} window {i}/{len(windows)}: {added} new")


def main() -> None:
    """Run all scrapers, resuming from the checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="listing pages per section in this run (for testing)",
    )
    args = parser.parse_args()

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint()
    seen = load_seen_urls()
    for section, zone_id in CAFEF_SECTIONS.items():
        scrape_cafef(section, zone_id, state, seen, args.max_pages)
    for section, cate_id in VNEXPRESS_SECTIONS.items():
        scrape_vnexpress(section, cate_id, state, seen, args.max_pages)


if __name__ == "__main__":
    main()
