"""Settings for the news scrapers."""

START_DATE = "2024-11-01"  # inclusive, Vietnam time
END_DATE = "2026-10-31"  # inclusive, Vietnam time
VN_TZ = "Asia/Ho_Chi_Minh"

USER_AGENT = "financial-news-research/0.1 (student research project)"
DELAY_SECONDS = 2.5  # pause before every request

OUTPUT_CSV = "data/raw/news.csv"
CHECKPOINT_JSON = "data/raw/checkpoint.json"
COLUMNS = ["source", "section", "title", "sapo", "published_at", "url"]

CAFEF_BASE = "https://cafef.vn"
CAFEF_SECTIONS = {  # section slug -> zone id
    "doanh-nghiep": 18836,
    "thi-truong-chung-khoan": 18831,
}
CAFEF_PAGE_STEP = 10  # sample 1 timeline page in every N (pages 1, 1+N, 1+2N, ...)

VNEXPRESS_BASE = "https://vnexpress.net"
VNEXPRESS_SECTIONS = {  # section slug -> category id
    "vi-mo": 1003184,
    "ngan-hang": 1003175,
    "doanh-nghiep": 1003170,
}
VNEXPRESS_PAGES = 12  # split the date range into N equal windows, 1 page each
