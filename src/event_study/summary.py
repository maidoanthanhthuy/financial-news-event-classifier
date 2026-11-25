"""Summary statistics of abnormal returns by label and window.

Usage:
    python -m src.event_study.summary

Reads event_returns.csv, writes event_summary.csv (all events and without the
ROBUST_EXCLUDE tickers), prints the table.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.event_study.config import (
    CONFIDENCE,
    EVENT_RETURNS_CSV,
    EVENT_SUMMARY_CSV,
    ROBUST_EXCLUDE,
)


def describe_ar(values: pd.Series, confidence: float = CONFIDENCE) -> dict:
    """Mean, median, sd, share > 0, t-based CI and t/Wilcoxon p-values vs. 0."""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    out = {
        "n": n,
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "pct_positive": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "t_pvalue": np.nan,
        "wilcoxon_pvalue": np.nan,
    }
    if n == 0:
        return out
    out.update(mean=x.mean(), median=np.median(x), pct_positive=(x > 0).mean())
    if n < 2:
        return out
    sd = x.std(ddof=1)
    half = stats.t.ppf((1 + confidence) / 2, n - 1) * sd / np.sqrt(n)
    out.update(
        std=sd,
        ci_low=x.mean() - half,
        ci_high=x.mean() + half,
        t_pvalue=stats.ttest_1samp(x, 0.0).pvalue,
        wilcoxon_pvalue=stats.wilcoxon(x).pvalue if np.any(x != 0) else np.nan,
    )
    return out


def summarize_by_label(
    returns: pd.DataFrame, confidence: float = CONFIDENCE
) -> pd.DataFrame:
    """One row per label and k with describe_ar statistics."""
    rows = [
        {"label": label, "k": k, **describe_ar(g["abnormal_return"], confidence)}
        for (label, k), g in returns.groupby(["label", "k"])
    ]
    return pd.DataFrame(rows)


def event_summary(
    returns: pd.DataFrame,
    exclude: list = ROBUST_EXCLUDE,
    confidence: float = CONFIDENCE,
) -> pd.DataFrame:
    """Summary for all events and without the excluded tickers (column sample)."""
    full = summarize_by_label(returns, confidence).assign(sample="all")
    rest = returns[~returns["ticker"].isin(exclude)]
    robust = summarize_by_label(rest, confidence).assign(
        sample="without_" + "_".join(exclude)
    )
    out = pd.concat([full, robust], ignore_index=True)
    return out[["sample"] + [c for c in out.columns if c != "sample"]]


def plot_mean_ar(summary: pd.DataFrame, sample: str = "all") -> plt.Figure:
    """Bar chart of mean AR (%) by label with CI bars, one panel per k."""
    df = summary[summary["sample"] == sample]
    ks = sorted(df["k"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), ks, strict=True):
        d = df[df["k"] == k].sort_values("label")
        mean = d["mean"] * 100
        err = [mean - d["ci_low"] * 100, d["ci_high"] * 100 - mean]
        ax.bar(d["label"], mean, yerr=err, capsize=4, color="#4C78A8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"k = {k}")
        ax.tick_params(axis="x", rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    np.atleast_1d(axes)[0].set_ylabel("Mean abnormal return (%)")
    fig.suptitle(f"Mean abnormal return by label, {CONFIDENCE:.0%} CI ({sample})")
    fig.tight_layout()
    return fig


def main() -> None:
    """Write event_summary.csv and print it."""
    out = event_summary(pd.read_csv(EVENT_RETURNS_CSV))
    out.to_csv(EVENT_SUMMARY_CSV, index=False)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
