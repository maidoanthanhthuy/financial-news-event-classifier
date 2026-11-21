import pandas as pd
import pytest

from src.classify.evaluate import (
    confusion,
    join_test_labels,
    per_label_scores,
    summary_scores,
)

LABELS = ["a", "b", "c"]
TRUE = pd.Series(["a", "a", "b", "b"])
PRED = pd.Series(["a", "b", "b", "b"])


def test_confusion_counts_and_keeps_unused_labels():
    cm = confusion(TRUE, PRED, LABELS)
    assert cm.loc["a"].tolist() == [1, 1, 0]
    assert cm.loc["b"].tolist() == [0, 2, 0]
    assert cm.loc["c"].tolist() == [0, 0, 0]


def test_per_label_scores_values():
    s = per_label_scores(TRUE, PRED, LABELS)
    assert s.loc["a", "precision"] == 1.0
    assert s.loc["a", "recall"] == 0.5
    assert s.loc["b", "precision"] == pytest.approx(2 / 3)
    assert s.loc["b", "f1"] == pytest.approx(0.8)
    assert s.loc["c"].tolist() == [0.0, 0.0, 0.0, 0]
    assert not s.isna().any().any()


def test_summary_scores():
    out = summary_scores(TRUE, PRED, ["a", "b"])
    assert out["accuracy"] == 0.75
    assert out["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_join_test_labels_splits_other():
    blind = pd.DataFrame({"id": [1, 2, 3], "human_label": ["a", "other", "b"]})
    llm = pd.DataFrame({"id": [3, 1, 2], "llm_label": ["b", "a", "a"], "reason": "r"})
    main_df, other_df = join_test_labels(blind, llm)
    assert main_df["id"].tolist() == [1, 3]
    assert main_df["llm_label"].tolist() == ["a", "b"]
    assert other_df["id"].tolist() == [2]
