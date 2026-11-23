"""LangGraph pipeline: classify -> tickers -> explain (skipped for "other").

Usage: python -m src.classify.pipeline   # trial on 5 train items (real API calls)
"""

import time
from collections.abc import Callable
from functools import lru_cache
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.classify.config import (
    LABELED_CSV,
    OTHER_EXPLANATION,
    OTHER_LABEL,
    PIPELINE_TRIAL_N,
    SEED,
)
from src.classify.explain import explain_news
from src.classify.llm_label import classify_news, cost_usd
from src.classify.tickers import extract_tickers, load_aliases, main_tickers


class NewsState(TypedDict, total=False):
    """Data passed between pipeline nodes."""

    title: str
    sapo: str
    label: str
    reason: str
    tickers: list
    main_tickers: list
    explanation: str


def build_graph(
    classify_fn: Callable,
    explain_fn: Callable,
    aliases: list,
    other_label: str = OTHER_LABEL,
    other_explanation: str = OTHER_EXPLANATION,
):
    """Compile the pipeline graph with the given LLM functions and alias list."""

    def classify(state: NewsState) -> dict:
        res = classify_fn(state["title"], state["sapo"])
        return {"label": res["label"], "reason": res["reason"]}

    def tickers(state: NewsState) -> dict:
        t, s = state["title"], state["sapo"]
        return {
            "tickers": extract_tickers(t, s, aliases),
            "main_tickers": main_tickers(t, s, aliases),
        }

    def explain(state: NewsState) -> dict:
        res = explain_fn(
            state["title"], state["sapo"], state["label"], state["main_tickers"]
        )
        return {"explanation": res["explanation"]}

    def skip(state: NewsState) -> dict:
        return {"explanation": other_explanation}

    def route(state: NewsState) -> str:
        return "skip" if state["label"] == other_label else "explain"

    g = StateGraph(NewsState)
    g.add_node("classify", classify)
    g.add_node("tickers", tickers)
    g.add_node("explain", explain)
    g.add_node("skip", skip)
    g.add_edge(START, "classify")
    g.add_edge("classify", "tickers")
    g.add_conditional_edges("tickers", route, ["explain", "skip"])
    g.add_edge("explain", END)
    g.add_edge("skip", END)
    return g.compile()


@lru_cache(maxsize=1)
def default_graph():
    """Graph with the real OpenAI calls and the VN30 alias file."""
    return build_graph(classify_news, explain_news, load_aliases())


def run_pipeline(title: str, sapo: str, graph=None) -> dict:
    """Run one news item through the pipeline and return the final state."""
    graph = graph or default_graph()
    return graph.invoke({"title": title, "sapo": sapo})


def count_usage(fn: Callable, usage: list) -> Callable:
    """Wrap an LLM function so each call appends (input, output) tokens to usage."""

    def wrapper(*args):
        res = fn(*args)
        usage.append((res["input_tokens"], res["output_tokens"]))
        return res

    return wrapper


def main() -> None:
    """Run the real pipeline on train items and print results, time and cost."""
    df = pd.read_csv(LABELED_CSV)
    is_other = df["final_label"] == OTHER_LABEL
    items = pd.concat(
        [
            df[~is_other].sample(n=PIPELINE_TRIAL_N, random_state=SEED),
            df[is_other].sample(n=1, random_state=SEED),
        ]
    )
    usage = []
    graph = build_graph(
        count_usage(classify_news, usage),
        count_usage(explain_news, usage),
        load_aliases(),
    )
    for row in items.itertuples(index=False):
        usage.clear()
        start = time.perf_counter()
        out = run_pipeline(row.title, row.sapo, graph)
        secs = time.perf_counter() - start
        cost = sum(cost_usd(i, o) for i, o in usage)
        print(f"[{row.id}] human={row.final_label} pred={out['label']}")
        print(f"T: {row.title}")
        print(f"reason: {out['reason']}")
        print(f"tickers={out['tickers']} main_tickers={out['main_tickers']}")
        print(f"=> {out['explanation']}")
        print(f"{secs:.1f}s, {len(usage)} LLM calls, ${cost:.5f}\n")


if __name__ == "__main__":
    main()
