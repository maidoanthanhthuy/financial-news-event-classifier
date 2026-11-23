"""Score the published PhoBERT model on the blind test set (CPU).

Usage: python -m src.finetune.test_phobert   # writes data/processed/test_phobert.csv
"""

import pandas as pd

from src.classify.config import LABELS
from src.classify.evaluate import (
    confusion,
    join_predictions,
    per_label_scores,
    summary_scores,
)
from src.finetune.config import HF_REPO_ID, TEST_CSV, TEST_PHOBERT_CSV
from src.finetune.predict import predict_frame


def main() -> None:
    """Predict the test set, save predictions and print scores and timing."""
    labels = list(LABELS)
    truth = pd.read_csv(TEST_CSV)
    preds = predict_frame(truth, HF_REPO_ID).assign(model=HF_REPO_ID)
    preds.to_csv(TEST_PHOBERT_CSV, index=False)
    df = join_predictions(truth, preds, "label")
    y_true, y_pred = df["human_label"], df["pred"]
    pd.set_option("display.width", 200, "display.max_colwidth", 90)

    secs = preds["seconds"]
    print(f"wrote {len(preds)} rows to {TEST_PHOBERT_CSV}")
    print(
        f"seconds per item (CPU): mean {secs.mean():.3f}, median {secs.median():.3f},"
        f" max {secs.max():.3f}, total {secs.sum():.1f}\n"
    )
    print(f"{summary_scores(y_true, y_pred, labels)}\n")
    print(per_label_scores(y_true, y_pred, labels).round(3), "\n")
    print("confusion (rows = human, cols = PhoBERT):")
    print(confusion(y_true, y_pred, labels).to_string(), "\n")
    wrong = df[y_true != y_pred]
    print(f"mismatches ({len(wrong)}):")
    print(wrong[["id", "human_label", "pred", "title"]].to_string(index=False))


if __name__ == "__main__":
    main()
