from src.scraping.parse import (
    even_windows,
    new_rows,
    parse_cafef_list,
    parse_vnexpress_list,
    parse_vnexpress_pubdate,
)

CAFEF_HTML = """
<div class="tlitem box-category-item" data-id="1">
  <h3><a href="/tin-a-1.chn">Tin A</a></h3>
  <span class="time time-ago" title="2025-03-10T09:52:00">x</span>
  <p class="sapo box-category-sapo">Sapo A</p>
</div>
<div class="tlitem box-category-item" data-id="2">
  <h3><a href="/tin-b-2.chn">Tin B</a></h3>
</div>
"""

VNEXPRESS_HTML = """
<article class="item-news item-news-common">
  <h3 class="title-news"><a href="https://vnexpress.net/tin-c-1.html">Tin C</a></h3>
  <p class="description"><a href="https://vnexpress.net/tin-c-1.html">Sapo C</a></p>
</article>
<article class="item-news"><div>ad</div></article>
"""


def test_parse_cafef_list_reads_fields_and_skips_items_without_time():
    rows = parse_cafef_list(CAFEF_HTML, "doanh-nghiep")
    assert rows == [
        {
            "source": "cafef",
            "section": "doanh-nghiep",
            "title": "Tin A",
            "sapo": "Sapo A",
            "published_at": "2025-03-10T09:52:00+07:00",
            "url": "https://cafef.vn/tin-a-1.chn",
        }
    ]


def test_parse_vnexpress_list_skips_items_without_title():
    rows = parse_vnexpress_list(VNEXPRESS_HTML, "vi-mo")
    assert len(rows) == 1
    assert rows[0]["title"] == "Tin C"
    assert rows[0]["sapo"] == "Sapo C"
    assert rows[0]["published_at"] is None


def test_parse_vnexpress_pubdate():
    html = '<head><meta name="pubdate" content="2024-11-15T17:40:58+07:00"/></head>'
    assert parse_vnexpress_pubdate(html) == "2024-11-15T17:40:58+07:00"
    assert parse_vnexpress_pubdate("<head></head>") is None


def test_even_windows_split_range_equally_in_vietnam_time():
    windows = even_windows("2024-11-01", "2024-11-30", 2)
    assert windows == [
        (1730394000, 1731689999),  # 2024-11-01 00:00 .. 11-15 23:59:59 +07
        (1731690000, 1732985999),  # 2024-11-16 00:00 .. 11-30 23:59:59 +07
    ]


def test_new_rows_drops_seen_duplicates_out_of_range_and_missing_time():
    base = {"source": "s", "section": "x", "title": "t", "sapo": ""}
    rows = [
        {**base, "url": "a", "published_at": "2025-01-01T08:00:00+07:00"},
        {**base, "url": "a", "published_at": "2025-01-01T08:00:00+07:00"},
        {**base, "url": "b", "published_at": "2025-01-02T08:00:00+07:00"},
        {**base, "url": "c", "published_at": "2023-01-01T08:00:00+07:00"},
        {**base, "url": "d", "published_at": None},
    ]
    df = new_rows(rows, {"b"}, "2024-11-01", "2026-10-31")
    assert df["url"].tolist() == ["a"]
