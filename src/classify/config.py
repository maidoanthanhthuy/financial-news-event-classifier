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
JOIN_KEYS = ["source", "published_at", "title"]  # unique per news item

REVIEW_OTHER_CSV = "data/labeled/review_other.csv"
OTHER_REVIEW_N = 60  # LLM "other" items sampled for review (6th label)

TEST_BLIND_CSV = "data/labeled/test_blind.csv"
TEST_LLM_CSV = "data/processed/test_llm_labels.csv"  # draft labels, older guide
TEST_LLM_V2_CSV = "data/processed/test_llm_v2.csv"  # classify_news, current guide
TEST_PER_LABEL = 20  # blind test items per label (excluding "other")

# USD per 1M tokens (aggregator sites, Nov 2026; check OpenAI's pricing page)
PRICE_INPUT = 0.10
PRICE_OUTPUT = 0.50

VN30_ALIASES_CSV = "data/reference/vn30_aliases.csv"  # blank ticker = blocked name
TICKER_SEP = ";"
TICKER_CHECK_CSV = "data/processed/ticker_check.csv"
TICKER_CHECK_N = 20  # train items sampled with and without predicted tickers

EXPLAIN_PROMPT_PATH = "docs/explain_prompt.md"
EXPLAIN_SAMPLES_CSV = "data/processed/explain_samples.csv"
EXPLAIN_PER_LABEL = 2  # train items per event label in the explanation trial
EXPLAIN_REASONING_EFFORT = (
    "none"  # lowest level gpt-6-luna accepts (0 reasoning tokens)
)

OTHER_LABEL = "other"
OTHER_EXPLANATION = (
    "Tin này không thuộc 5 loại sự kiện doanh nghiệp mà công cụ theo dõi."
)
PIPELINE_TRIAL_N = 4  # event items in the pipeline trial (plus 1 "other" item)
