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
SEED = 42  # order in which news items are labeled

# USD per 1M tokens (aggregator sites, Nov 2026; check OpenAI's pricing page)
PRICE_INPUT = 0.10
PRICE_OUTPUT = 0.50
