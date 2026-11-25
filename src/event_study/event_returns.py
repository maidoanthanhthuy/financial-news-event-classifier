"""Abnormal returns around each event, computed in DuckDB (event_returns.sql).

Usage:
    python -m src.event_study.event_returns

Loads events.csv into prices.duckdb as table events, writes table event_returns
and data/processed/event_returns.csv, prints counts per window.
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.event_study.config import (
    EVENT_RETURNS_CSV,
    EVENTS_CSV,
    INDEX_SYMBOL,
    MARKET_CLOSE,
    PRICES_DB,
    WINDOWS,
)

SQL_PATH = Path(__file__).with_name("event_returns.sql")
EVENT_COLS = ["id", "ticker", "label", "label_source", "published_at"]


def compute_event_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    sql: str,
    index_symbol: str = INDEX_SYMBOL,
    market_close: str = MARKET_CLOSE,
    windows: list = WINDOWS,
) -> pd.DataFrame:
    """Run the event-return SQL on in-memory copies of events and prices."""
    with duckdb.connect() as con:
        con.register("events", events[EVENT_COLS])
        con.register("daily_prices", prices[["date", "ticker", "close"]])
        params = {
            "index_symbol": index_symbol,
            "market_close": market_close,
            "windows": list(windows),
        }
        return con.execute(sql, params).df()


def main() -> None:
    """Compute event returns, save them to DuckDB and CSV, print counts."""
    events = pd.read_csv(EVENTS_CSV)
    with duckdb.connect(PRICES_DB) as con:
        prices = con.sql("SELECT * FROM daily_prices").df()
        out = compute_event_returns(events, prices, SQL_PATH.read_text())
        con.register("events_df", events)
        con.execute("CREATE OR REPLACE TABLE events AS SELECT * FROM events_df")
        con.register("out_df", out)
        con.execute("CREATE OR REPLACE TABLE event_returns AS SELECT * FROM out_df")
    out.to_csv(EVENT_RETURNS_CSV, index=False)

    kept = out["id"].nunique()
    print(f"events: {len(events)}, with prices after listing: {kept}")
    counts = out.groupby("k")["abnormal_return"].agg(computed="count", total="size")
    print(counts.to_string())
    by_label = out.pivot_table(
        index="label", columns="k", values="abnormal_return", aggfunc="count"
    )
    print(by_label.to_string())


if __name__ == "__main__":
    main()
