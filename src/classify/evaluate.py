"""Compare predicted labels with human labels.

Usage:
    python -m src.classify.evaluate   # human labels in the blind test set vs. LLM
"""

import pandas as pd

from src.classify.config import LABELS, TEST_BLIND_CSV, TEST_LLM_CSV


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
    pred_n, true_n = cm.sum(axis=0), cm.sum(axis=1)
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


def join_test_labels(
    blind: pd.DataFrame, llm: pd.DataFrame, drop_label: str = "other"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join human and LLM labels on id; split off items humans marked drop_label."""
    df = blind.merge(llm[["id", "llm_label", "reason"]], on="id", validate="1:1")
    dropped = df["human_label"] == drop_label
    return df[~dropped].reset_index(drop=True), df[dropped].reset_index(drop=True)


def main() -> None:
    """Print LLM vs. human scores on the blind test set."""
    labels = [lab for lab in LABELS if lab != "other"]
    main_df, other_df = join_test_labels(
        pd.read_csv(TEST_BLIND_CSV), pd.read_csv(TEST_LLM_CSV)
    )
    y_true, y_pred = main_df["human_label"], main_df["llm_label"]
    pd.set_option("display.width", 200, "display.max_colwidth", 90)

    print(f"items scored: {len(main_df)}; human 'other' (excluded): {len(other_df)}")
    print(f"{summary_scores(y_true, y_pred, labels)}\n")
    print(per_label_scores(y_true, y_pred, labels).round(3), "\n")
    print("confusion (rows = human, cols = LLM):")
    print(confusion(y_true, y_pred, labels).to_string(), "\n")
    wrong = main_df[y_true != y_pred]
    print(f"mismatches ({len(wrong)}):")
    print(wrong[["id", "human_label", "llm_label", "title"]].to_string(index=False))
    print("\nexcluded human 'other' items, LLM label:")
    print(other_df[["id", "llm_label", "title"]].to_string(index=False))


if __name__ == "__main__":
    main()
