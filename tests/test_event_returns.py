import pandas as pd
import pytest

from src.event_study.event_returns import SQL_PATH, compute_event_returns

# Two trading weeks: Mon 2026-01-05 .. Fri 01-09, Mon 01-12 .. Fri 01-16
DATES = list(pd.bdate_range("2026-01-05", "2026-01-16"))


def _prices():
    rows = [(d, "VNINDEX", 100.0 + i) for i, d in enumerate(DATES)]
    rows += [(d, "AAA", 10.0 * (i + 1)) for i, d in enumerate(DATES)]
    # BBB has no price on Thu 01-08
    rows += [(d, "BBB", 50.0) for d in DATES if d != pd.Timestamp("2026-01-08")]
    # CCC lists on Mon 01-12
    rows += [(d, "CCC", 20.0) for d in DATES if d >= pd.Timestamp("2026-01-12")]
    df = pd.DataFrame(rows, columns=["date", "ticker", "close"])
    return df.assign(date=df["date"].dt.date)


def _events(rows):
    df = pd.DataFrame(rows, columns=["id", "ticker", "published_at"])
    return df.assign(label="a", label_source="human")


def _run(rows):
    out = compute_event_returns(
        _events(rows), _prices(), SQL_PATH.read_text(), "VNINDEX", "15:00:00", [1, 3, 5]
    )
    return out.set_index(["id", "k"])


def _day(ts):
    return pd.Timestamp(ts).date()


def test_values_for_a_weekday_event():
    out = _run([(1, "AAA", "2026-01-06T10:00:00+07:00")])
    r = out.loc[(1, 1)]
    assert _day(r["event_date"]) == _day("2026-01-06")
    assert _day(r["start_date"]) == _day("2026-01-05")
    assert r["stock_return"] == pytest.approx(20 / 10 - 1)
    assert r["index_return"] == pytest.approx(101 / 100 - 1)
    assert r["abnormal_return"] == pytest.approx(1.0 - 0.01)
    # k=5 ends at session 4, after the weekend
    assert _day(out.loc[(1, 5), "end_date"]) == _day("2026-01-12")


def test_weekend_and_after_close_move_to_next_session():
    out = _run(
        [
            (1, "AAA", "2026-01-10T09:00:00+07:00"),  # Saturday
            (2, "AAA", "2026-01-09T15:30:00+07:00"),  # Friday after close
            (3, "AAA", "2026-01-09T15:00:00+07:00"),  # exactly 15:00: same day
        ]
    )
    assert _day(out.loc[(1, 1), "event_date"]) == _day("2026-01-12")
    assert _day(out.loc[(1, 1), "start_date"]) == _day("2026-01-09")
    assert _day(out.loc[(2, 1), "event_date"]) == _day("2026-01-12")
    assert _day(out.loc[(3, 1), "event_date"]) == _day("2026-01-09")


def test_missing_price_leaves_only_that_window_empty():
    out = _run([(1, "BBB", "2026-01-06T10:00:00+07:00")])
    assert out.loc[(1, 1), "abnormal_return"] == pytest.approx(0 - 0.01)
    assert pd.isna(out.loc[(1, 3), "abnormal_return"])  # needs Thu 01-08
    assert pd.notna(out.loc[(1, 5), "abnormal_return"])


def test_window_past_calendar_end_is_empty():
    out = _run([(1, "AAA", "2026-01-16T10:00:00+07:00")])
    assert pd.notna(out.loc[(1, 1), "abnormal_return"])
    assert pd.isna(out.loc[(1, 3), "abnormal_return"])


def test_pre_listing_news_dropped_and_listing_day_empty():
    out = _run(
        [
            (1, "CCC", "2026-01-07T10:00:00+07:00"),  # before listing
            (2, "CCC", "2026-01-12T10:00:00+07:00"),  # listing day: no session -1
        ]
    )
    assert out.index.get_level_values("id").unique().tolist() == [2]
    assert out["abnormal_return"].isna().all()
    assert len(out) == 3
