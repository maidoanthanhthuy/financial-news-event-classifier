import pandas as pd
import pytest

from src.classify.evaluate import (
    compare_runs,
    confusion,
    join_predictions,
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


def test_join_predictions_aligns_on_id_and_fills_missing():
    truth = pd.DataFrame({"id": [1, 2, 3], "human_label": ["a", "b", "c"]})
    preds = pd.DataFrame({"id": [3, 1], "label": ["c", None]})
    out = join_predictions(truth, preds, "label")
    assert out["pred"].tolist() == ["none", "none", "c"]
    assert "pred" not in truth.columns  # input unchanged


def test_compare_runs_scores_all_labels_and_without_other():
    df = pd.DataFrame(
        {"human_label": ["a", "b", "other", "other"], "pred": ["a", "b", "a", "other"]}
    )
    out = compare_runs({"run": df}, ["a", "b", "other"])
    assert out.loc["run", "acc_3"] == 0.75
    assert out.loc["run", "acc_2_no_other"] == 1.0
    assert out.loc["run", "macro_f1_3"] == pytest.approx((2 / 3 + 1 + 2 / 3) / 3)


def test_per_label_scores_counts_predictions_outside_labels_as_misses():
    true = pd.Series(["a", "a"])
    pred = pd.Series(["a", "other"])
    s = per_label_scores(true, pred, ["a"])
    assert s.loc["a", "recall"] == 0.5
    assert s.loc["a", "support"] == 2
