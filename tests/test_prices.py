import pandas as pd

from src.event_study.fetch_prices import looks_truncated, summarize_payload
from src.event_study.load_prices import (
    combine_prices,
    coverage,
    late_starters,
    missing_vs_index,
    parse_kbs_rows,
)

PAYLOAD = {
    "symbol": "FPT",
    "data_day": [
        {"t": "2026-01-06 07:00", "o": "1", "c": "51000.0", "v": "10"},
        {"t": "2026-01-05 07:00", "o": 1, "c": 50000, "v": 10},
    ],
}


def _prices(rows):
    df = pd.DataFrame(rows, columns=["date", "ticker", "close"])
    return df.assign(date=pd.to_datetime(df["date"]))


def test_summarize_payload():
    s = summarize_payload(PAYLOAD)
    assert s == {"n_rows": 2, "first_date": "2026-01-05", "last_date": "2026-01-06"}
    assert summarize_payload(None)["n_rows"] == 0


def test_looks_truncated():
    monday, friday = pd.Timestamp("2026-01-05").date(), pd.Timestamp("2026-01-09")
    assert looks_truncated(3, monday, friday.date())
    assert not looks_truncated(5, monday, friday.date())


def test_parse_kbs_rows_handles_string_numbers():
    out = parse_kbs_rows(PAYLOAD, "FPT")
    assert list(out.columns) == ["date", "ticker", "close"]
    assert out["close"].tolist() == [51000.0, 50000.0]
    assert out["close"].dtype == float
    assert out["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-06", "2026-01-05"]
    assert parse_kbs_rows({"data_day": []}, "FPT").empty


def test_combine_prices_drops_duplicates_and_sorts():
    a = parse_kbs_rows(PAYLOAD, "FPT")
    out = combine_prices([a, a.iloc[:1]])
    assert len(out) == 2
    assert out["date"].is_monotonic_increasing


def test_missing_vs_index_within_ticker_range():
    prices = _prices(
        [
            ("2026-01-05", "VNINDEX", 1.0),
            ("2026-01-06", "VNINDEX", 1.0),
            ("2026-01-07", "VNINDEX", 1.0),
            ("2026-01-08", "VNINDEX", 1.0),
            ("2026-01-06", "AAA", 1.0),  # starts late: 01-05 is not missing
            ("2026-01-08", "AAA", 1.0),
            ("2026-01-05", "BBB", 1.0),
            ("2026-01-08", "BBB", 1.0),
        ]
    )
    out = missing_vs_index(prices, "VNINDEX")
    assert [d.day for d in out["AAA"]] == [7]
    assert [d.day for d in out["BBB"]] == [6, 7]


def test_late_starters_counts_events():
    prices = _prices([("2024-09-04", "OLD", 1.0), ("2025-05-13", "NEW", 1.0)])
    events = pd.DataFrame(
        {
            "ticker": ["NEW", "NEW", "OLD"],
            "published_at": [
                "2025-01-02T09:00:00+07:00",
                "2025-06-01T09:00:00+07:00",
                "2025-01-02T09:00:00+07:00",
            ],
        }
    )
    out = late_starters(coverage(prices), events, cutoff="2024-11-01")
    assert out["ticker"].tolist() == ["NEW"]
    assert out[["events", "before_first_price"]].values.tolist() == [[2, 1]]
