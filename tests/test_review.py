import pandas as pd
import pytest

from src.classify.review import (
    FINAL_COLUMNS,
    add_url,
    merge_review,
    sample_blind_test,
    sample_for_review,
)


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
            "url": ["ua", "ub", "uc"],
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


def test_add_url_matches_on_keys():
    df = make_prelabeled().iloc[[3, 0]].drop(columns="url").assign(id=[1, 2])
    out = add_url(df, make_prelabeled())
    assert out["url"].tolist() == ["u3", "u0"]
    with pytest.raises(ValueError, match="No url"):
        add_url(df.assign(title=["t3", "missing"]), make_prelabeled())


def test_sample_blind_test_skips_reviewed_and_other():
    df = make_prelabeled()
    exclude = df.iloc[[0, 1, 4]]  # two business_results, one macro_policy
    test = sample_blind_test(df, exclude, per_label=2, seed=0)
    assert test["llm_label"].value_counts().to_dict() == {
        "business_results": 2,
        "macro_policy": 2,
    }
    assert set(test["title"]) & set(exclude["title"]) == set()
    assert test["id"].tolist() == [1, 2, 3, 4]
    assert test["llm_label"].tolist() != sorted(test["llm_label"])  # shuffled
    assert (test[["human_label", "tickers"]] == "").all().all()
    assert test.equals(sample_blind_test(df, exclude, per_label=2, seed=0))
