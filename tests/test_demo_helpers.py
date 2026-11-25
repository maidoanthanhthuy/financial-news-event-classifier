import pandas as pd

from src.demo.helpers import label_name, past_reaction, pick_samples


def _summary(t_pvalue, mean=-0.005):
    return pd.DataFrame(
        {
            "sample": ["all", "without_VIC"],
            "label": ["ma_investment", "ma_investment"],
            "k": [3, 3],
            "n": [49, 29],
            "mean": [mean, -0.0078],
            "pct_positive": [0.3673, 0.31],
            "t_pvalue": [t_pvalue, 0.038],
        }
    )


def test_label_name():
    assert label_name("ma_investment", {"ma_investment": "M&A"}) == "M&A"
    assert label_name("unknown", {}) == "unknown"


def test_pick_samples_keeps_id_order():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["a", "b", "c"],
            "sapo": ["", "", ""],
            "url": ["u1", "u2", "u3"],
            "final_label": ["x", "y", "z"],
        }
    )
    assert pick_samples(df, [3, 1])["title"].tolist() == ["c", "a"]


def test_past_reaction_not_significant():
    text = past_reaction(_summary(0.13), "ma_investment", k=3, sample="all")
    assert "49 tin" in text
    assert "37%" in text
    assert "Không có xu hướng rõ ràng" in text


def test_past_reaction_significant_gives_direction():
    text = past_reaction(_summary(0.01, mean=-0.012), "ma_investment", k=3)
    assert "giảm 1,2%" in text


def test_past_reaction_skipped_labels_and_missing():
    s = _summary(0.13)
    assert past_reaction(s, "other", skip=["other"]) is None
    assert past_reaction(s, "business_results") is None
