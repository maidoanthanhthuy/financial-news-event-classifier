import pandas as pd
import pytest
from pydantic import ValidationError

from src.classify.llm_label import NewsLabel, build_messages, cost_usd, pending_rows


def test_build_messages_puts_guide_in_system_and_news_in_user():
    msgs = build_messages("Tin A", "Sapo A", "GUIDE")
    assert msgs[0] == {"role": "system", "content": "GUIDE"}
    assert msgs[1]["role"] == "user"
    assert "Tin A" in msgs[1]["content"] and "Sapo A" in msgs[1]["content"]


def test_cost_usd():
    assert cost_usd(1_000_000, 2_000_000, 0.10, 0.50) == pytest.approx(1.10)


def test_pending_rows_skips_done_and_is_reproducible():
    news = pd.DataFrame({"url": list("abcdef"), "title": list("ABCDEF")})
    first = pending_rows(news, {"b", "d"}, seed=0)
    again = pending_rows(news, {"b", "d"}, seed=0)
    assert sorted(first["url"]) == ["a", "c", "e", "f"]
    assert first["url"].tolist() == again["url"].tolist()
    assert news["url"].tolist() == list("abcdef")  # input unchanged


def test_news_label_rejects_unknown_label():
    assert NewsLabel(label="other", reason="daily market report").label == "other"
    with pytest.raises(ValidationError):
        NewsLabel(label="weather", reason="x")
