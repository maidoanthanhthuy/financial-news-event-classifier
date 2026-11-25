"""Pure helpers for the Streamlit demo."""

import pandas as pd

from src.demo.config import (
    ALPHA,
    LABEL_VI,
    NO_HISTORY_LABELS,
    SUMMARY_K,
    SUMMARY_SAMPLE,
)


def label_name(label: str, names: dict = LABEL_VI) -> str:
    """Vietnamese name of a label (the label itself if unknown)."""
    return names.get(label, label)


def pick_samples(labeled: pd.DataFrame, ids: list) -> pd.DataFrame:
    """Rows of the labeled set with the given ids, in the order of ids."""
    rows = labeled.set_index("id").loc[ids].reset_index()
    return rows[["id", "title", "sapo", "url", "final_label"]]


def past_reaction(
    summary: pd.DataFrame,
    label: str,
    k: int = SUMMARY_K,
    sample: str = SUMMARY_SAMPLE,
    alpha: float = ALPHA,
    skip: list = NO_HISTORY_LABELS,
) -> str | None:
    """One Vietnamese sentence on past abnormal returns for this label, or None."""
    if label in skip:
        return None
    match = (summary["label"] == label) & (summary["k"] == k)
    row = summary[match & (summary["sample"] == sample)]
    if row.empty:
        return None
    r = row.iloc[0]
    text = (
        f"Tin loại này trong quá khứ: {int(r['n'])} tin, "
        f"{r['pct_positive']:.0%} tin có giá tăng hơn thị trường (VN-Index) "
        f"sau {k} phiên."
    )
    if pd.isna(r["t_pvalue"]) or r["t_pvalue"] >= alpha:
        return text + " Không có xu hướng rõ ràng."
    direction = "tăng" if r["mean"] > 0 else "giảm"
    pct = f"{abs(r['mean']) * 100:.1f}".replace(".", ",")
    return text + f" Trung bình giá {direction} {pct}% so với thị trường."
