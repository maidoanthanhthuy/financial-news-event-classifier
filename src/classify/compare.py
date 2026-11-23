"""Compare two classifiers on the same items: McNemar test and bootstrap CIs.

Usage: python -m src.classify.compare   # PhoBERT (seed 44) vs. LLM v2 on the test set
"""

from math import comb

import numpy as np
import pandas as pd

from src.classify.config import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    LABELS,
    TEST_BLIND_CSV,
    TEST_LLM_V2_CSV,
    TEST_PHOBERT_CSV,
)


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact (binomial) McNemar test on paired correct/wrong outcomes."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}


def macro_f1_codes(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Macro-F1 over k label codes (0 for a label with no true or predicted items)."""
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    pred_n, true_n = cm.sum(axis=0), cm.sum(axis=1)
    denom = pred_n + true_n
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def bootstrap_macro_f1(
    y_true, pred_a, pred_b, labels: list, n_boot: int, seed: int
) -> dict:
    """Point estimates and 95% percentile CIs for macro-F1 of a, b and a - b."""
    code = {lab: i for i, lab in enumerate(labels)}
    t, a, b = (np.array([code[x] for x in s]) for s in (y_true, pred_a, pred_b))
    k, n = len(labels), len(t)
    rng = np.random.default_rng(seed)
    stats = np.empty((n_boot, 2))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = macro_f1_codes(t[idx], a[idx], k), macro_f1_codes(t[idx], b[idx], k)
    diff = stats[:, 0] - stats[:, 1]
    point = macro_f1_codes(t, a, k), macro_f1_codes(t, b, k)

    def ci(x: np.ndarray) -> tuple:
        return tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))

    return {
        "a": point[0],
        "a_ci": ci(stats[:, 0]),
        "b": point[1],
        "b_ci": ci(stats[:, 1]),
        "diff": point[0] - point[1],
        "diff_ci": ci(diff),
        "share_diff_above_0": float((diff > 0).mean()),
    }


def main() -> None:
    """Print McNemar and bootstrap results for PhoBERT (a) vs. LLM v2 (b)."""
    truth = pd.read_csv(TEST_BLIND_CSV)[["id", "human_label"]]
    pho = pd.read_csv(TEST_PHOBERT_CSV)[["id", "label"]].rename(columns={"label": "a"})
    llm = pd.read_csv(TEST_LLM_V2_CSV)[["id", "label"]].rename(columns={"label": "b"})
    df = truth.merge(pho, on="id", validate="1:1").merge(llm, on="id", validate="1:1")
    y = df["human_label"]
    print(f"items: {len(df)} (a = PhoBERT seed 44, b = LLM v2)")
    print("McNemar:", mcnemar_exact(df["a"] == y, df["b"] == y))
    print(
        f"both correct: {((df['a'] == y) & (df['b'] == y)).sum()},",
        f"both wrong: {((df['a'] != y) & (df['b'] != y)).sum()}",
    )
    res = bootstrap_macro_f1(
        y, df["a"], df["b"], list(LABELS), BOOTSTRAP_N, BOOTSTRAP_SEED
    )
    for key, val in res.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
