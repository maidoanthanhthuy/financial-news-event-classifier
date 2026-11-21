"""Sample LLM labels for review, merge reviewed labels, build a blind test set.

Usage:
    python -m src.classify.review sample   # write data/labeled/review.csv
    python -m src.classify.review merge    # write data/labeled/news_labeled.csv
    python -m src.classify.review test     # write blind test set + hidden LLM labels
"""

import argparse

import pandas as pd

from src.classify.config import (
    JOIN_KEYS,
    LABELED_CSV,
    LABELS,
    OUTPUT_CSV,
    REVIEW_CSV,
    REVIEW_PER_LABEL,
    SEED,
    TEST_BLIND_CSV,
    TEST_LLM_CSV,
    TEST_PER_LABEL,
)

REVIEW_COLUMNS = [
    "id",
    "source",
    "published_at",
    "title",
    "sapo",
    "llm_label",
    "reason",
    "human_label",
]
FINAL_COLUMNS = ["id", "source", "published_at", "url", "title", "sapo", "final_label"]
BLIND_COLUMNS = ["id", "title", "sapo", "human_label", "tickers"]
TEST_LLM_COLUMNS = ["id", "source", "published_at", "url", "llm_label", "reason"]


def sample_for_review(
    prelabeled: pd.DataFrame,
    per_label: int = REVIEW_PER_LABEL,
    seed: int = SEED,
    skip_label: str = "other",
) -> pd.DataFrame:
    """Pick per_label random items for each label except skip_label."""
    pool = prelabeled[prelabeled["label"] != skip_label]
    picked = pool.groupby("label").sample(n=per_label, random_state=seed)
    out = picked.rename(columns={"label": "llm_label"}).reset_index(drop=True)
    out["id"] = range(1, len(out) + 1)
    out["human_label"] = ""
    return out[REVIEW_COLUMNS]


def add_url(
    df: pd.DataFrame, prelabeled: pd.DataFrame, keys: list = JOIN_KEYS
) -> pd.DataFrame:
    """Add the url column from prelabeled, matching rows on keys."""
    out = df.merge(prelabeled[keys + ["url"]], on=keys, how="left", validate="1:1")
    if out["url"].isna().any():
        raise ValueError(f"No url for id {out.loc[out['url'].isna(), 'id'].tolist()}")
    return out


def merge_review(
    review: pd.DataFrame, labels: tuple = LABELS, drop_label: str = "other"
) -> pd.DataFrame:
    """Use human_label when filled, else llm_label; drop items marked drop_label."""
    human = review["human_label"].fillna("").astype(str).str.strip()
    bad = (human != "") & ~human.isin(labels)
    if bad.any():
        raise ValueError(f"Unknown human_label at id {review.loc[bad, 'id'].tolist()}")
    out = review.assign(final_label=human.where(human != "", review["llm_label"]))
    out = out[out["final_label"] != drop_label]
    return out[FINAL_COLUMNS].reset_index(drop=True)


def sample_blind_test(
    prelabeled: pd.DataFrame,
    exclude: pd.DataFrame,
    per_label: int = TEST_PER_LABEL,
    seed: int = SEED,
    keys: list = JOIN_KEYS,
    skip_label: str = "other",
) -> pd.DataFrame:
    """Pick per_label random items per label, skipping skip_label and excluded rows."""
    seen = (
        prelabeled[keys]
        .merge(exclude[keys].drop_duplicates(), how="left", indicator=True)["_merge"]
        .eq("both")
        .to_numpy()
    )
    pool = prelabeled[~seen & (prelabeled["label"] != skip_label)]
    picked = pool.groupby("label").sample(n=per_label, random_state=seed)
    picked = picked.sample(frac=1, random_state=seed)  # shuffle: ids must not hint
    out = picked.rename(columns={"label": "llm_label"}).reset_index(drop=True)
    out["id"] = range(1, len(out) + 1)
    out["human_label"] = ""
    out["tickers"] = ""
    return out


def main() -> None:
    """Run the 'sample', 'merge' or 'test' step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["sample", "merge", "test"])
    args = parser.parse_args()

    if args.step == "sample":
        review = sample_for_review(pd.read_csv(OUTPUT_CSV))
        review.to_csv(REVIEW_CSV, index=False)
        print(f"wrote {len(review)} rows to {REVIEW_CSV}")
        print(review["llm_label"].value_counts().to_string())
    elif args.step == "merge":
        review = add_url(pd.read_csv(REVIEW_CSV), pd.read_csv(OUTPUT_CSV))
        final = merge_review(review)
        final.to_csv(LABELED_CSV, index=False)
        print(f"wrote {len(final)} rows to {LABELED_CSV}")
        print(final["final_label"].value_counts().to_string())
    else:
        test = sample_blind_test(pd.read_csv(OUTPUT_CSV), pd.read_csv(REVIEW_CSV))
        test[BLIND_COLUMNS].to_csv(TEST_BLIND_CSV, index=False)
        test[TEST_LLM_COLUMNS].to_csv(TEST_LLM_CSV, index=False)
        print(f"wrote {len(test)} rows to {TEST_BLIND_CSV} and {TEST_LLM_CSV}")


if __name__ == "__main__":
    main()
