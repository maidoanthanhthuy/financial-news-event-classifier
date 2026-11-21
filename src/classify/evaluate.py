"""Compare predicted labels with human labels on the blind test set.

Usage:
    python -m src.classify.evaluate   # 6-label scores for test_llm_v2.csv + comparison
"""

import pandas as pd

from src.classify.config import LABELS, TEST_BLIND_CSV, TEST_LLM_CSV, TEST_LLM_V2_CSV


def confusion(y_true: pd.Series, y_pred: pd.Series, labels: list) -> pd.DataFrame:
    """Confusion matrix: rows are true labels, columns are predicted labels."""
    cm = pd.crosstab(y_true, y_pred).reindex(index=labels, columns=labels)
    return cm.fillna(0).astype(int)


def per_label_scores(
    y_true: pd.Series, y_pred: pd.Series, labels: list
) -> pd.DataFrame:
    """Precision, recall, F1 and support for each label (0 when undefined)."""
    cm = confusion(y_true, y_pred, labels)
    tp = pd.Series([cm.loc[lab, lab] for lab in labels], index=labels)
    # count from the raw series so predictions outside `labels` still count
    pred_n = y_pred.value_counts().reindex(labels, fill_value=0)
    true_n = y_true.value_counts().reindex(labels, fill_value=0)
    precision = (tp / pred_n).fillna(0.0)
    recall = (tp / true_n).fillna(0.0)
    f1 = (2 * precision * recall / (precision + recall)).fillna(0.0)
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": true_n}
    )


def summary_scores(y_true: pd.Series, y_pred: pd.Series, labels: list) -> dict:
    """Accuracy and macro-F1 over the given labels."""
    scores = per_label_scores(y_true, y_pred, labels)
    accuracy = float((y_true.to_numpy() == y_pred.to_numpy()).mean())
    return {"accuracy": accuracy, "macro_f1": float(scores["f1"].mean())}


def join_predictions(
    truth: pd.DataFrame, preds: pd.DataFrame, pred_col: str, missing: str = "none"
) -> pd.DataFrame:
    """Join human labels with predictions on id; pred column 'pred', gaps = missing."""
    p = preds[["id", pred_col]].rename(columns={pred_col: "pred"})
    out = truth.merge(p, on="id", how="left", validate="1:1")
    return out.assign(pred=out["pred"].fillna(missing))


def compare_runs(
    runs: dict, labels: list, other: str = "other", true_col: str = "human_label"
) -> pd.DataFrame:
    """Accuracy and macro-F1 per run, on all labels and on items not labeled other."""
    event_labels = [lab for lab in labels if lab != other]
    rows = {}
    for name, df in runs.items():
        sub = df[df[true_col] != other]
        all6 = summary_scores(df[true_col], df["pred"], labels)
        ev5 = summary_scores(sub[true_col], sub["pred"], event_labels)
        rows[name] = {
            f"acc_{len(labels)}": all6["accuracy"],
            f"macro_f1_{len(labels)}": all6["macro_f1"],
            f"acc_{len(event_labels)}_no_{other}": ev5["accuracy"],
            f"macro_f1_{len(event_labels)}_no_{other}": ev5["macro_f1"],
        }
    return pd.DataFrame(rows).T


def main() -> None:
    """Print 6-label scores of the current LLM run and compare with the draft run."""
    labels = list(LABELS)
    truth = pd.read_csv(TEST_BLIND_CSV)
    v2 = join_predictions(truth, pd.read_csv(TEST_LLM_V2_CSV), "label")
    draft = join_predictions(truth, pd.read_csv(TEST_LLM_CSV), "llm_label")
    y_true, y_pred = v2["human_label"], v2["pred"]
    pd.set_option("display.width", 200, "display.max_colwidth", 90)

    print(f"items: {len(v2)}; labels: {len(labels)}")
    print(f"{summary_scores(y_true, y_pred, labels)}\n")
    print(per_label_scores(y_true, y_pred, labels).round(3), "\n")
    print("confusion (rows = human, cols = LLM):")
    print(confusion(y_true, y_pred, labels).to_string(), "\n")
    wrong = v2[y_true != y_pred]
    print(f"mismatches ({len(wrong)}):")
    print(wrong[["id", "human_label", "pred", "title"]].to_string(index=False))
    print("\ncomparison (draft = older guide, v2 = current guide):")
    runs = {"draft": draft, "v2": v2}
    print(compare_runs(runs, labels).round(3).to_string())


if __name__ == "__main__":
    main()
