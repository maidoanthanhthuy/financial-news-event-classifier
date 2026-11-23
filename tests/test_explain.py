from types import SimpleNamespace

import pandas as pd

from src.classify.explain import build_explain_messages, explain_news, sample_per_label


def fake_client(content):
    """Client stub whose create() returns `content` with fixed token usage."""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=200, completion_tokens=50),
    )
    create = lambda **kwargs: resp  # noqa: E731
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_build_explain_messages_includes_all_fields():
    msgs = build_explain_messages("Tin A", "Sapo A", "ma_investment", ["FPT"], "P")
    assert msgs[0] == {"role": "system", "content": "P"}
    user = msgs[1]["content"]
    for part in ["Tin A", "Sapo A", "ma_investment", "FPT"]:
        assert part in user


def test_build_explain_messages_without_tickers():
    msgs = build_explain_messages("T", "S", "other", [], "P")
    assert "Tickers: none" in msgs[1]["content"]


def test_explain_news_returns_text_and_tokens():
    out = explain_news(
        "T", "S", "x", [], client=fake_client("  Giải thích.  "), prompt="P"
    )
    assert out == {
        "explanation": "Giải thích.",
        "input_tokens": 200,
        "output_tokens": 50,
    }


def test_explain_news_handles_empty_content():
    out = explain_news("T", "S", "x", [], client=fake_client(None), prompt="P")
    assert out["explanation"] == ""


def test_sample_per_label_counts_and_reproducible():
    df = pd.DataFrame({"id": range(9), "lab": list("aaabbbccc")})
    a = sample_per_label(df, "lab", ["a", "b"], 2, seed=0)
    assert a["lab"].tolist() == ["a", "a", "b", "b"]
    assert (
        a["id"].tolist()
        == sample_per_label(df, "lab", ["a", "b"], 2, seed=0)["id"].tolist()
    )
