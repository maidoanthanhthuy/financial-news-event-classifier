import pandas as pd
import pytest

from src.classify.tickers import (
    extract_tickers,
    main_tickers,
    split_tickers,
    ticker_scores,
)

ALIASES = [
    ("GAS", "GAS"),
    ("GAS", "PV Gas"),
    ("FPT", "FPT"),
    ("", "FPT Shop"),
    ("", "FPT Long Châu"),
    ("MSN", "Masan"),
    ("MCH", "Masan Consumer"),
    ("HPG", "Hòa Phát"),
    ("HPG", "Hoà Phát"),
    ("VIC", "Vingroup"),
    ("TCX", "TCX"),
    ("TCX", "TCBS"),
    ("", "The CrownX"),
    ("", "The CrownX (TCX)"),
]


def test_ticker_needs_uppercase_and_word_boundary():
    assert extract_tickers("Giá gas tăng mạnh", "", ALIASES) == []
    assert extract_tickers("GASOLINE và GAS", "", ALIASES) == ["GAS"]
    assert extract_tickers("Cổ phiếu GAS tăng", "", ALIASES) == ["GAS"]


def test_name_alias_is_case_insensitive():
    assert extract_tickers("pv gas lãi lớn", "", ALIASES) == ["GAS"]


def test_longest_alias_wins():
    assert extract_tickers("Masan Consumer chia cổ tức", "", ALIASES) == ["MCH"]
    assert extract_tickers("Masan và Masan Consumer", "", ALIASES) == ["MSN", "MCH"]


def test_blocked_subsidiary_names():
    assert extract_tickers("FPT Shop mở thêm cửa hàng FPT Long Châu", "", ALIASES) == []
    assert extract_tickers("FPT Shop và FPT", "", ALIASES) == ["FPT"]


def test_both_tone_mark_styles_and_unicode_forms():
    nfd = "Hòa Phát"  # decomposed Unicode
    assert extract_tickers("Hoà Phát", "", ALIASES) == ["HPG"]
    assert extract_tickers(nfd, "", ALIASES) == ["HPG"]


def test_uses_sapo_dedups_and_handles_missing():
    out = extract_tickers("Vingroup", "Vingroup cùng FPT", ALIASES)
    assert out == ["VIC", "FPT"]
    assert extract_tickers(None, None, ALIASES) == []


def test_split_tickers():
    assert split_tickers("VIC; STB") == {"VIC", "STB"}
    assert split_tickers(float("nan")) == set()
    assert split_tickers("") == set()


def test_ticker_scores():
    true = pd.Series(["VIC;STB", "", "FPT"])
    pred = pd.Series(["VIC", "HPG", "FPT"])
    s = ticker_scores(true, pred)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["exact"] == 1
    assert s["items"] == 3


def test_sample_for_check_balances_and_is_reproducible():
    from src.classify.tickers import sample_for_check

    df = pd.DataFrame({"id": range(10), "p": ["VIC", ""] * 5})
    a = sample_for_check(df, "p", 3, seed=1)
    assert len(a) == 6
    assert (a["p"] != "").sum() == 3
    assert a["id"].tolist() == sample_for_check(df, "p", 3, seed=1)["id"].tolist()


def test_crownx_abbreviation_is_not_tcx():
    sapo = "Quỹ đầu tư vào The CrownX (TCX) để mua cổ phần của Masan."
    assert extract_tickers("", sapo, ALIASES) == ["MSN"]
    assert extract_tickers("TCBS tăng vốn", "", ALIASES) == ["TCX"]


def test_main_tickers_keeps_only_title_tickers():
    title = "Xanh SM tăng vốn, FPT mở rộng"
    sapo = "Nhiều công ty như Vingroup và FPT tăng vốn."
    assert extract_tickers(title, sapo, ALIASES) == ["FPT", "VIC"]
    assert main_tickers(title, sapo, ALIASES) == ["FPT"]
    assert main_tickers("Thị trường hôm nay", sapo, ALIASES) == []
