import numpy as np
import pandas as pd
import pytest

from src.finetune.data import (
    build_text,
    length_stats,
    prepare_dataset,
    stratified_split,
    underthesea_segment,
)
from src.finetune.predict import softmax, to_result
from src.finetune.train import best_epoch, compute_metrics, summarize_seeds


def test_build_text_joins_and_handles_missing():
    assert build_text(" Tin A ", "Sapo A") == "Tin A. Sapo A"
    assert build_text("Tin A", float("nan")) == "Tin A"
    assert build_text(None, "") == ""


def test_underthesea_segment_joins_compound_words():
    out = underthesea_segment("Ngân hàng Nhà nước tăng lãi suất")
    assert "Ngân_hàng" in out and "lãi_suất" in out


def test_prepare_dataset_maps_labels_and_segments():
    df = pd.DataFrame(
        {"id": [1, 2], "title": ["T1", "T2"], "sapo": ["S1", None], "lab": ["b", "a"]}
    )
    out = prepare_dataset(
        df, "lab", {"a": 0, "b": 1}, segment_fn=lambda x: x.replace(" ", "_")
    )
    assert out["text"].tolist() == ["T1._S1", "T2"]
    assert out["label"].tolist() == [1, 0]
    assert out["id"].tolist() == [1, 2]
    assert "text" not in df.columns


def test_stratified_split_keeps_label_shares_and_is_reproducible():
    df = pd.DataFrame(
        {"text": list("abcdefghijklmnopqrst"), "label": [0] * 10 + [1] * 10}
    )
    train, val = stratified_split(df, val_size=0.2, seed=1)
    assert len(train) == 16 and len(val) == 4
    assert val["label"].value_counts().to_dict() == {0: 2, 1: 2}
    assert not set(train["text"]) & set(val["text"])
    assert val["text"].tolist() == stratified_split(df, 0.2, seed=1)[1]["text"].tolist()


def test_length_stats():
    s = length_stats([10, 20, 30, 200], max_length=128)
    assert s["max"] == 200
    assert s["share_over_128"] == 0.25


def test_compute_metrics_from_logits():
    logits = np.array([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1]])
    out = compute_metrics((logits, np.array([0, 1, 1])))
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert 0 <= out["macro_f1"] <= 1


def test_best_epoch_picks_highest_metric():
    log = [
        {"loss": 1.0},
        {"eval_macro_f1": 0.6, "epoch": 1},
        {"eval_macro_f1": 0.8, "epoch": 2},
    ]
    assert best_epoch(log)["epoch"] == 2


def test_summarize_seeds():
    res = [{"seed": 1, "macro_f1": 0.7}, {"seed": 2, "macro_f1": 0.9}]
    s = summarize_seeds(res)
    assert s["mean"] == pytest.approx(0.8)
    assert s["std"] == pytest.approx(np.std([0.7, 0.9], ddof=1))
    assert s["best_seed"] == 2


def test_softmax_and_to_result():
    p = softmax([1000.0, 1000.0])
    assert p.tolist() == [0.5, 0.5]
    out = to_result([0.0, 3.0], {0: "a", 1: "b"})
    assert out["label"] == "b"
    assert sum(out["probs"].values()) == pytest.approx(1.0)
