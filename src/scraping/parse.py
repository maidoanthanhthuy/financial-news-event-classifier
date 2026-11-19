"""Pure functions that turn listing-page HTML into news rows."""

from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from src.scraping.config import CAFEF_BASE, COLUMNS, VN_TZ


def parse_cafef_list(
    html: str, section: str, base_url: str = CAFEF_BASE, tz: str = VN_TZ
) -> list[dict]:
    """Extract news rows from a CafeF timeline page."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for item in soup.select("div.tlitem.box-category-item"):
        link = item.select_one("h3 a")
        time_tag = item.select_one("span.time")
        if link is None or time_tag is None or not time_tag.get("title"):
            continue
        sapo = item.select_one("p.sapo")
        published = pd.Timestamp(time_tag["title"]).tz_localize(tz)
        rows.append(
            {
                "source": "cafef",
                "section": section,
                "title": link.get_text(strip=True),
                "sapo": sapo.get_text(strip=True) if sapo else "",
                "published_at": published.isoformat(),
                "url": urljoin(base_url, link["href"]),
            }
        )
    return rows


def parse_vnexpress_list(html: str, section: str) -> list[dict]:
    """Extract news rows (without publish time) from a VnExpress archive page."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for item in soup.select("article.item-news"):
        link = item.select_one(".title-news a")
        if link is None:
            continue
        sapo = item.select_one("p.description")
        rows.append(
            {
                "source": "vnexpress",
                "section": section,
                "title": link.get_text(strip=True),
                "sapo": sapo.get_text(strip=True) if sapo else "",
                "published_at": None,
                "url": link["href"],
            }
        )
    return rows


def parse_vnexpress_pubdate(html: str) -> str | None:
    """Return the ISO publish time from a VnExpress article page, or None."""
    meta = BeautifulSoup(html, "lxml").select_one('meta[name="pubdate"]')
    return meta["content"] if meta else None


def even_windows(
    start: str, end: str, n: int, tz: str = VN_TZ
) -> list[tuple[int, int]]:
    """Split [start, end] (local dates) into n equal windows of (from_unix, to_unix)."""
    begin = pd.Timestamp(start).tz_localize(tz)
    finish = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(tz)
    edges = [int(t.timestamp()) for t in pd.date_range(begin, finish, periods=n + 1)]
    return [(a, b - 1) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def new_rows(
    rows: list[dict], seen_urls: set[str], start: str, end: str
) -> pd.DataFrame:
    """Keep rows with an unseen url and a publish date within [start, end]."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df[~df["url"].isin(seen_urls)].drop_duplicates(subset="url")
    day = df["published_at"].str[:10]
    return df[(day >= start) & (day <= end)].reset_index(drop=True)
