"""Extract VN30 tickers from news text with an alias dictionary.

Usage:
    python -m src.classify.tickers         # score extract_tickers on the blind test set
    python -m src.classify.tickers check   # sample train items for a manual check
"""

import re
import sys
import unicodedata

import pandas as pd

from src.classify.config import (
    LABELED_CSV,
    SEED,
    TEST_BLIND_CSV,
    TICKER_CHECK_CSV,
    TICKER_CHECK_N,
    TICKER_SEP,
    VN30_ALIASES_CSV,
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def find_matches(text: str, aliases: list) -> list:
    """All (start, end, ticker) alias hits; all-caps aliases are case-sensitive."""
    hits = []
    for ticker, alias in aliases:
        alias = _nfc(alias)
        flags = 0 if alias.isupper() else re.IGNORECASE
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        for m in re.finditer(pattern, text, flags):
            hits.append((m.start(), m.end(), ticker))
    return hits


def extract_tickers(title: str, sapo: str, aliases: list) -> list:
    """VN30 tickers in title + sapo, longest alias wins; blank-ticker aliases block."""
    text = _nfc(f"{title or ''}\n{sapo or ''}")
    hits = sorted(find_matches(text, aliases), key=lambda h: (h[0] - h[1], h[0]))
    taken, kept = [], []
    for start, end, ticker in hits:
        if all(end <= s or start >= e for s, e in taken):
            taken.append((start, end))
            kept.append((start, ticker))
    found = [t for _, t in sorted(kept) if t]
    return list(dict.fromkeys(found))


def main_tickers(title: str, sapo: str, aliases: list) -> list:
    """Tickers from extract_tickers whose name or ticker appears in the title."""
    in_title = set(extract_tickers(title, "", aliases))
    return [t for t in extract_tickers(title, sapo, aliases) if t in in_title]


def split_tickers(value, sep: str = TICKER_SEP) -> set:
    """Set of tickers from a 'A;B' cell; empty set for NaN or blank."""
    if not isinstance(value, str):
        return set()
    return {t.strip() for t in value.split(sep) if t.strip()}


def ticker_scores(true: pd.Series, pred: pd.Series, sep: str = TICKER_SEP) -> dict:
    """Micro precision/recall over (item, ticker) pairs and exact-match item count."""
    t_sets = [split_tickers(v, sep) for v in true]
    p_sets = [split_tickers(v, sep) for v in pred]
    tp = sum(len(t & p) for t, p in zip(t_sets, p_sets, strict=True))
    n_pred = sum(len(p) for p in p_sets)
    n_true = sum(len(t) for t in t_sets)
    return {
        "precision": tp / n_pred if n_pred else 0.0,
        "recall": tp / n_true if n_true else 0.0,
        "exact": sum(t == p for t, p in zip(t_sets, p_sets, strict=True)),
        "items": len(t_sets),
    }


def sample_for_check(
    df: pd.DataFrame, pred_col: str, n: int, seed: int
) -> pd.DataFrame:
    """n random rows with a non-empty pred_col and n with an empty one."""
    has = df[pred_col].fillna("") != ""
    parts = [df[has].sample(n=n, random_state=seed)]
    parts.append(df[~has].sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def load_aliases(path: str = VN30_ALIASES_CSV) -> list:
    """Read (ticker, alias) pairs; blank ticker becomes ''."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return list(zip(df["ticker"].str.strip(), df["alias"].str.strip(), strict=True))


def check() -> None:
    """Write a sample of train items with predicted tickers for a manual check."""
    aliases = load_aliases()
    df = pd.read_csv(LABELED_CSV)
    pred = [
        TICKER_SEP.join(extract_tickers(t, s, aliases))
        for t, s in zip(df["title"], df["sapo"], strict=True)
    ]
    df = df.assign(predicted_tickers=pred, correct_tickers="")
    n_has = (df["predicted_tickers"] != "").sum()
    print(f"train items: {len(df)}; with tickers: {n_has}")
    out = sample_for_check(df, "predicted_tickers", TICKER_CHECK_N, SEED)
    cols = ["id", "title", "sapo", "predicted_tickers", "correct_tickers"]
    out[cols].to_csv(TICKER_CHECK_CSV, index=False)
    print(f"wrote {len(out)} rows to {TICKER_CHECK_CSV}")


def main() -> None:
    """Print ticker extraction scores and mismatches on the blind test set."""
    aliases = load_aliases()
    df = pd.read_csv(TEST_BLIND_CSV)
    pred = [
        TICKER_SEP.join(extract_tickers(t, s, aliases))
        for t, s in zip(df["title"], df["sapo"], strict=True)
    ]
    df = df.assign(pred=pred, tickers=df["tickers"].fillna(""))
    print(ticker_scores(df["tickers"], df["pred"]))
    pairs = zip(df["tickers"], df["pred"], strict=True)
    wrong = df[[split_tickers(a) != split_tickers(b) for a, b in pairs]]
    pd.set_option("display.width", 200, "display.max_colwidth", 100)
    print(f"\nmismatches ({len(wrong)}):")
    print(wrong[["id", "tickers", "pred", "title"]].to_string(index=False))


if __name__ == "__main__":
    check() if sys.argv[1:] == ["check"] else main()
