"""Settings for the event study data."""

from src.classify.config import INPUT_CSV, LABELED_CSV, TEST_BLIND_CSV

RAW_CSV = INPUT_CSV
TRAIN_CSV = LABELED_CSV
TEST_CSV = TEST_BLIND_CSV
NEWS_EVENTS_CSV = "data/processed/news_events.csv"
NEWS_EVENTS_LOG = "data/processed/news_events_log.txt"
EVENTS_CSV = "data/processed/events.csv"
DROP_LABELS = ["other", "macro_policy"]  # macro news has no single-firm reaction
# 5 trading days ~ 7 calendar days; replace with a trading calendar later
MIN_GAP_DAYS = 7
OUTPUT_COLS = [
    "id",
    "source",
    "published_at",
    "url",
    "title",
    "ticker",
    "label",
    "label_source",
]
SECONDS_PER_LLM_CALL = 2.5  # mean classify_news time on 20 test items
USD_PER_LLM_CALL = 0.000225  # LLM v2 run on the test set: $0.0225 / 100 items

# Daily prices from the KBS API (unofficial; same calls as vn-macro-assistant)
KBS_BASE_URL = "https://kbbuddywts.kbsec.com.vn/iis-server/investment"
KBS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
KBS_ROWS_KEY = "data_day"
KBS_MIN_ROW_RATIO = 0.9  # rows / weekdays below this -> refetch year by year
HTTP_RETRIES = 2
HTTP_PAUSE_SECONDS = 1.0
HTTP_TIMEOUT_SECONDS = 30.0
INDEX_SYMBOL = "VNINDEX"
PRICE_START = "2024-09-01"
PRICE_END = "2026-11-10"
PRICES_RAW_DIR = "data/raw/prices"
PRICES_DB = "data/processed/prices.duckdb"
LATE_START_CUTOFF = "2024-11-01"  # tickers whose prices start after this

# Abnormal returns (src/event_study/event_returns.sql)
WINDOWS = [1, 3, 5]  # k: close of session -1 to close of session k-1
MARKET_CLOSE = "15:00:00"  # news after this local time counts from the next day
EVENT_RETURNS_CSV = "data/processed/event_returns.csv"

# Summary by label (src/event_study/summary.py)
EVENT_SUMMARY_CSV = "data/processed/event_summary.csv"  # committed; read by the app
CONFIDENCE = 0.95
ROBUST_EXCLUDE = ["VIC"]  # most frequent ticker; rerun without it
