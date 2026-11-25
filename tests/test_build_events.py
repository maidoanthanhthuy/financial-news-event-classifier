import pandas as pd
import pytest

from src.event_study.build_events import (
    add_ids,
    attach_human_labels,
    drop_close_repeats,
    estimate,
    select_events,
    single_ticker_news,
)

ALIASES = [("FPT", "FPT"), ("VIC", "Vingroup"), ("HPG", "Hòa Phát")]


def test_add_ids_does_not_modify_input():
    raw = pd.DataFrame({"title": ["a", "b"]})
    out = add_ids(raw)
    assert out["id"].tolist() == [1, 2]
    assert "id" not in raw.columns


def test_single_ticker_news_keeps_exactly_one_title_ticker():
    news = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["FPT lãi lớn", "FPT và Vingroup", "Thị trường", "Hòa Phát"],
            "sapo": ["Vingroup cũng vậy", "", "FPT", ""],
        }
    )
    out = single_ticker_news(news, ALIASES)
    assert out["id"].tolist() == [1, 4]
    assert out["ticker"].tolist() == ["FPT", "HPG"]


def test_attach_human_labels_by_url_and_title_sapo():
    news = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "url": ["u1", "u2", "u3"],
            "title": ["t1", "t2", "t3"],
            "sapo": ["s1", "s2", "s3"],
        }
    )
    train = pd.DataFrame({"id": [9], "url": ["u1"], "final_label": ["a"]})
    test = pd.DataFrame(
        {"id": [9], "title": ["t2"], "sapo": ["s2"], "human_label": ["b"]}
    )
    out = attach_human_labels(news, train, test)
    assert out["label"].tolist()[:2] == ["a", "b"]
    assert pd.isna(out["label"].iloc[2])
    assert out["label_source"].tolist()[:2] == ["human", "human"]
    assert pd.isna(out["label_source"].iloc[2])
    assert out["id"].tolist() == [1, 2, 3]


def test_estimate():
    e = estimate(100, 0.001, 3.0)
    assert e["usd"] == pytest.approx(0.1)
    assert e["minutes"] == pytest.approx(5.0)


def _events(rows):
    return pd.DataFrame(rows, columns=["id", "published_at", "ticker", "label"])


def test_drop_close_repeats_keeps_first_and_resets_after_gap():
    news = _events(
        [
            (1, "2026-01-01T09:00:00+07:00", "FPT", "a"),
            (2, "2026-01-03T09:00:00+07:00", "FPT", "a"),  # 2 days after 1
            (3, "2026-01-08T09:00:00+07:00", "FPT", "a"),  # 7 days after 1
            (4, "2026-01-02T09:00:00+07:00", "FPT", "b"),  # other label
            (5, "2026-01-02T09:00:00+07:00", "VIC", "a"),  # other ticker
        ]
    )
    out = drop_close_repeats(news, min_gap_days=7)
    assert sorted(out["id"]) == [1, 3, 4, 5]
    assert len(news) == 5


def test_select_events_drops_labels():
    news = _events(
        [
            (1, "2026-01-01T09:00:00+07:00", "FPT", "macro_policy"),
            (2, "2026-01-01T09:00:00+07:00", "FPT", "other"),
            (3, "2026-01-01T09:00:00+07:00", "FPT", "a"),
        ]
    )
    out = select_events(news, drop_labels=["other", "macro_policy"])
    assert out["id"].tolist() == [3]
