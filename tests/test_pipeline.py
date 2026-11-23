from src.classify.pipeline import build_graph, run_pipeline

ALIASES = [("FPT", "FPT"), ("VIC", "Vingroup")]


def make_graph(label, calls):
    """Graph whose LLM steps are fakes that record their arguments."""

    def fake_classify(title, sapo):
        calls.append(("classify", title, sapo))
        return {"label": label, "reason": "r"}

    def fake_explain(title, sapo, label, tickers):
        calls.append(("explain", label, tickers))
        return {"explanation": "giải thích"}

    return build_graph(fake_classify, fake_explain, ALIASES, "other", "FIXED")


def test_event_branch_explains_with_main_tickers():
    calls = []
    out = run_pipeline("FPT lãi lớn", "Vingroup cũng vậy", make_graph("x", calls))
    assert out["label"] == "x" and out["reason"] == "r"
    assert out["tickers"] == ["FPT", "VIC"]
    assert out["main_tickers"] == ["FPT"]
    assert out["explanation"] == "giải thích"
    assert calls[-1] == ("explain", "x", ["FPT"])


def test_other_branch_skips_explain():
    calls = []
    out = run_pipeline("FPT lãi lớn", "", make_graph("other", calls))
    assert out["label"] == "other"
    assert out["explanation"] == "FIXED"
    assert out["tickers"] == ["FPT"]
    assert [c[0] for c in calls] == ["classify"]


def test_count_usage_records_tokens():
    from src.classify.pipeline import count_usage

    usage = []
    fn = count_usage(lambda x: {"input_tokens": 3, "output_tokens": x}, usage)
    assert fn(5)["output_tokens"] == 5
    assert usage == [(3, 5)]


def test_review_sample_skips_other_and_adds_empty_columns():
    import pandas as pd

    from src.classify.explain_review import REVIEW_COLS, review_sample

    df = pd.DataFrame(
        {
            "id": range(6),
            "title": list("abcdef"),
            "sapo": list("ABCDEF"),
            "label": ["x", "other", "y", "other", "x", "y"],
            "main_tickers": [""] * 6,
            "explanation": ["e"] * 6,
        }
    )
    out = review_sample(df, 3, seed=0)
    assert len(out) == 3 and "other" not in set(out["label"])
    assert list(out.columns[-4:]) == REVIEW_COLS
    assert (out[REVIEW_COLS] == "").all().all()
    assert out["id"].tolist() == review_sample(df, 3, seed=0)["id"].tolist()
