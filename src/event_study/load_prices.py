"""Load raw KBS price files into DuckDB and check coverage.

Usage:
    python -m src.event_study.load_prices

Reads data/raw/prices/manifest.json, writes table daily_prices(date, ticker,
close) to data/processed/prices.duckdb (replaced on each run), prints checks.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.event_study.config import (
    EVENTS_CSV,
    INDEX_SYMBOL,
    KBS_ROWS_KEY,
    LATE_START_CUTOFF,
    PRICES_DB,
    PRICES_RAW_DIR,
)


def parse_kbs_rows(
    payload: dict, ticker: str, rows_key: str = KBS_ROWS_KEY
) -> pd.DataFrame:
    """Turn one KBS payload into rows (date, ticker, close); close is adjusted."""
    rows = payload.get(rows_key) or []
    return pd.DataFrame(
        {
            "date": pd.to_datetime([str(r["t"])[:10] for r in rows]),
            "ticker": ticker,
            "close": pd.to_numeric([r["c"] for r in rows]).astype(float),
        }
    )


def combine_prices(frames: list) -> pd.DataFrame:
    """Concatenate price frames, drop duplicate ticker+date rows, sort."""
    cols = ["date", "ticker", "close"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)[cols]
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """First date, last date and number of sessions per ticker."""
    return prices.groupby("ticker")["date"].agg(first="min", last="max", n="count")


def missing_vs_index(prices: pd.DataFrame, index_symbol: str = INDEX_SYMBOL) -> dict:
    """Per ticker, index trading dates missing within the ticker's own date range."""
    index_dates = set(prices.loc[prices["ticker"] == index_symbol, "date"])
    out = {}
    for ticker, g in prices[prices["ticker"] != index_symbol].groupby("ticker"):
        lo, hi = g["date"].min(), g["date"].max()
        expected = {d for d in index_dates if lo <= d <= hi}
        missing = sorted(expected - set(g["date"]))
        if missing:
            out[ticker] = missing
    return out


def late_starters(
    cov: pd.DataFrame, events: pd.DataFrame, cutoff: str = LATE_START_CUTOFF
) -> pd.DataFrame:
    """Tickers whose prices start after cutoff, with event counts (all, early)."""
    late = cov[cov["first"] > pd.Timestamp(cutoff)]
    ev_date = pd.to_datetime(events["published_at"].str[:10])
    rows = []
    for ticker, first in late["first"].items():
        mine = events["ticker"] == ticker
        rows.append(
            {
                "ticker": ticker,
                "first": first.date(),
                "events": int(mine.sum()),
                "before_first_price": int((mine & (ev_date < first)).sum()),
            }
        )
    return pd.DataFrame(
        rows, columns=["ticker", "first", "events", "before_first_price"]
    )


def read_raw(raw_dir: str = PRICES_RAW_DIR) -> pd.DataFrame:
    """Read every file listed in the manifest into one price table."""
    root = Path(raw_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    frames = [
        parse_kbs_rows(json.loads((root / f).read_bytes()), symbol)
        for symbol, entry in manifest["symbols"].items()
        for f in entry["files"]
    ]
    return combine_prices(frames)


def write_db(prices: pd.DataFrame, db_path: str = PRICES_DB) -> None:
    """Replace table daily_prices in the DuckDB file."""
    with duckdb.connect(db_path) as con:
        con.register("prices_df", prices)
        con.execute(
            "CREATE OR REPLACE TABLE daily_prices AS "
            "SELECT CAST(date AS DATE) AS date, ticker, close FROM prices_df"
        )


def main() -> None:
    """Load prices into DuckDB and print coverage checks."""
    prices = read_raw()
    write_db(prices)
    print(f"daily_prices: {len(prices)} rows -> {PRICES_DB}")
    cov = coverage(prices)
    print(cov.to_string())
    missing = missing_vs_index(prices)
    print("\nIndex dates missing per ticker:")
    for ticker, dates in missing.items():
        print(f"  {ticker}: {len(dates)} -> {[d.date().isoformat() for d in dates]}")
    if not missing:
        print("  none")
    extra = set(prices["date"]) - set(
        prices.loc[prices["ticker"] == INDEX_SYMBOL, "date"]
    )
    print(f"Dates with stock prices but no index price: {len(extra)}")
    print(f"\nTickers starting after {LATE_START_CUTOFF} (events from {EVENTS_CSV}):")
    print(late_starters(cov, pd.read_csv(EVENTS_CSV)).to_string(index=False))


if __name__ == "__main__":
    main()
