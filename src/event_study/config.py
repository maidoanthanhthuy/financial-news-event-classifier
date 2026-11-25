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
