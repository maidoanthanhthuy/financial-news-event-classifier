"""Settings for LLM pre-labeling."""

MODEL = "gpt-6-luna"  # does not accept temperature=0; uses the default
LABELS = (
    "business_results",
    "dividends_issuance",
    "macro_policy",
    "ma_investment",
    "legal_personnel",
    "other",
)

INPUT_CSV = "data/raw/news.csv"
OUTPUT_CSV = "data/processed/news_prelabeled.csv"
GUIDE_PATH = "docs/labeling_guide.md"
SEED = 42  # order in which news items are labeled; also used for review sampling

REVIEW_CSV = "data/labeled/review.csv"
LABELED_CSV = "data/labeled/news_labeled.csv"
REVIEW_PER_LABEL = 100  # news items sampled per label (excluding "other")

# USD per 1M tokens (aggregator sites, Nov 2026; check OpenAI's pricing page)
PRICE_INPUT = 0.10
PRICE_OUTPUT = 0.50
