import numpy as np
import pandas as pd
import pytest

from src.event_study.summary import describe_ar, event_summary, summarize_by_label


def test_describe_ar_values():
    s = describe_ar(pd.Series([0.01, 0.02, 0.03, -0.01, np.nan]))
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(0.0125)
    assert s["median"] == pytest.approx(0.015)
    assert s["pct_positive"] == pytest.approx(0.75)
    assert s["ci_low"] < s["mean"] < s["ci_high"]
    assert 0 <= s["t_pvalue"] <= 1
    assert 0 <= s["wilcoxon_pvalue"] <= 1


def test_describe_ar_small_samples():
    assert describe_ar(pd.Series([], dtype=float))["n"] == 0
    one = describe_ar(pd.Series([0.02]))
    assert one["mean"] == pytest.approx(0.02)
    assert np.isnan(one["t_pvalue"])


def _returns():
    return pd.DataFrame(
        {
            "ticker": ["VIC", "FPT", "HPG", "VIC", "FPT", "HPG"],
            "label": ["a", "a", "a", "b", "b", "b"],
            "k": [1, 1, 1, 1, 1, 1],
            "abnormal_return": [0.05, 0.01, -0.02, 0.0, 0.02, 0.03],
        }
    )


def test_summarize_by_label_one_row_per_group():
    out = summarize_by_label(_returns())
    assert out[["label", "k"]].values.tolist() == [["a", 1], ["b", 1]]
    assert out["n"].tolist() == [3, 3]


def test_event_summary_without_excluded_ticker():
    returns = _returns()
    out = event_summary(returns, exclude=["VIC"])
    assert out["sample"].unique().tolist() == ["all", "without_VIC"]
    robust = out[out["sample"] == "without_VIC"]
    assert robust["n"].tolist() == [2, 2]
    assert len(returns) == 6
