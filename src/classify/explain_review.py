"""Run the pipeline on the blind test set and sample explanations for review.

Usage: python -m src.classify.explain_review
    -> data/processed/test_pipeline.csv (all items)
    -> data/processed/explain_review.csv (sample to grade by hand)
"""

import pandas as pd

from src.classify.config import (
    EXPLAIN_REVIEW_CSV,
    EXPLAIN_REVIEW_N,
    OTHER_LABEL,
    SEED,
    TEST_BLIND_CSV,
    TEST_PIPELINE_CSV,
    TICKER_SEP,
)
from src.classify.explain import explain_news
from src.classify.llm_label import classify_news, cost_usd
from src.classify.pipeline import build_graph, count_usage, run_pipeline
from src.classify.tickers import load_aliases

REVIEW_COLS = ["correct", "easy", "no_advice", "note"]


def review_sample(
    df: pd.DataFrame, n: int, seed: int, other_label: str = OTHER_LABEL
) -> pd.DataFrame:
    """n random rows not labeled other, with empty review columns added."""
    sample = df[df["label"] != other_label].sample(n=n, random_state=seed)
    cols = ["id", "title", "sapo", "label", "main_tickers", "explanation"]
    return sample[cols].assign(**{c: "" for c in REVIEW_COLS}).reset_index(drop=True)


def main() -> None:
    """Run the pipeline on every test item, save all outputs and a review sample."""
    test = pd.read_csv(TEST_BLIND_CSV)
    usage = []
    graph = build_graph(
        count_usage(classify_news, usage),
        count_usage(explain_news, usage),
        load_aliases(),
    )
    rows = []
    for i, row in enumerate(test.itertuples(index=False), start=1):
        out = run_pipeline(row.title, row.sapo, graph)
        rows.append(
            {
                "id": row.id,
                "title": row.title,
                "sapo": row.sapo,
                "label": out["label"],
                "reason": out["reason"],
                "tickers": TICKER_SEP.join(out["tickers"]),
                "main_tickers": TICKER_SEP.join(out["main_tickers"]),
                "explanation": out["explanation"],
            }
        )
        print(f"{i}/{len(test)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TEST_PIPELINE_CSV, index=False)
    review_sample(df, EXPLAIN_REVIEW_N, SEED).to_csv(EXPLAIN_REVIEW_CSV, index=False)

    cost = sum(cost_usd(i, o) for i, o in usage)
    n_other = (df["label"] == OTHER_LABEL).sum()
    print(f"items {len(df)}, labeled other {n_other}, LLM calls {len(usage)}")
    print(f"cost ${cost:.4f}; wrote {TEST_PIPELINE_CSV} and {EXPLAIN_REVIEW_CSV}")


if __name__ == "__main__":
    main()
