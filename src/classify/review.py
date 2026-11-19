"""Sample LLM labels for human review, then merge the reviewed labels.

Usage:
    python -m src.classify.review sample   # write data/labeled/review.csv
    python -m src.classify.review merge    # write data/labeled/news_labeled.csv
"""

import argparse

import pandas as pd

from src.classify.config import (
    LABELED_CSV,
    LABELS,
    OUTPUT_CSV,
    REVIEW_CSV,
    REVIEW_PER_LABEL,
    SEED,
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
FINAL_COLUMNS = ["id", "source", "published_at", "title", "sapo", "final_label"]


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


def main() -> None:
    """Run the 'sample' or 'merge' step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["sample", "merge"])
    args = parser.parse_args()

    if args.step == "sample":
        review = sample_for_review(pd.read_csv(OUTPUT_CSV))
        review.to_csv(REVIEW_CSV, index=False)
        print(f"wrote {len(review)} rows to {REVIEW_CSV}")
        print(review["llm_label"].value_counts().to_string())
    else:
        final = merge_review(pd.read_csv(REVIEW_CSV))
        final.to_csv(LABELED_CSV, index=False)
        print(f"wrote {len(final)} rows to {LABELED_CSV}")
        print(final["final_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
