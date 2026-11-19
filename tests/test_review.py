import pandas as pd
import pytest

from src.classify.review import FINAL_COLUMNS, merge_review, sample_for_review


def make_prelabeled() -> pd.DataFrame:
    labels = ["business_results"] * 4 + ["macro_policy"] * 4 + ["other"] * 4
    return pd.DataFrame(
        {
            "source": "cafef",
            "section": "x",
            "title": [f"t{i}" for i in range(12)],
            "sapo": "s",
            "published_at": "2025-01-01T08:00:00+07:00",
            "url": [f"u{i}" for i in range(12)],
            "label": labels,
            "reason": "r",
        }
    )


def test_sample_for_review_takes_n_per_label_and_skips_other():
    df = make_prelabeled()
    review = sample_for_review(df, per_label=2, seed=0)
    assert review["llm_label"].value_counts().to_dict() == {
        "business_results": 2,
        "macro_policy": 2,
    }
    assert review["id"].tolist() == [1, 2, 3, 4]
    assert (review["human_label"] == "").all()
    assert review.equals(sample_for_review(df, per_label=2, seed=0))
    assert "label" in df.columns  # input unchanged


def test_merge_review_prefers_human_label_and_drops_other():
    review = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "source": "cafef",
            "published_at": "2025-01-01",
            "title": ["a", "b", "c"],
            "sapo": "s",
            "llm_label": ["business_results", "macro_policy", "ma_investment"],
            "reason": "r",
            "human_label": [None, " other ", "legal_personnel"],
        }
    )
    final = merge_review(review)
    assert final.columns.tolist() == FINAL_COLUMNS
    assert final["id"].tolist() == [1, 3]
    assert final["final_label"].tolist() == ["business_results", "legal_personnel"]


def test_merge_review_rejects_unknown_human_label():
    review = pd.DataFrame(
        {
            "id": [7],
            "source": "cafef",
            "published_at": "2025-01-01",
            "title": "a",
            "sapo": "s",
            "llm_label": "macro_policy",
            "reason": "r",
            "human_label": "macro",
        }
    )
    with pytest.raises(ValueError, match=r"\[7\]"):
        merge_review(review)
