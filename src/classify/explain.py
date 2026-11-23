"""Explain a news item in simple Vietnamese for beginner investors.

Usage: python -m src.classify.explain   # trial on train items -> explain_samples.csv
"""

from pathlib import Path

import pandas as pd
from openai import OpenAI

from src.classify.config import (
    EXPLAIN_PER_LABEL,
    EXPLAIN_PROMPT_PATH,
    EXPLAIN_REASONING_EFFORT,
    EXPLAIN_SAMPLES_CSV,
    LABELED_CSV,
    LABELS,
    MODEL,
    SEED,
)
from src.classify.llm_label import cost_usd, get_client
from src.classify.tickers import load_aliases, main_tickers


def build_explain_messages(
    title: str, sapo: str, label: str, tickers: list, prompt: str
) -> list[dict]:
    """Chat messages: prompt as system, news fields as user."""
    user = (
        f"Title: {title}\nSapo: {sapo}\nEvent label: {label}\n"
        f"Tickers: {', '.join(tickers) if tickers else 'none'}"
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user},
    ]


def explain_news(
    title: str,
    sapo: str,
    label: str,
    tickers: list,
    client: OpenAI | None = None,
    prompt: str | None = None,
    model: str = MODEL,
    reasoning_effort: str = EXPLAIN_REASONING_EFFORT,
) -> dict:
    """Return a short Vietnamese explanation and token counts.

    Pass tickers from main_tickers (tickers named in the title).
    """
    client = client or get_client()
    prompt = load_prompt() if prompt is None else prompt
    resp = client.chat.completions.create(
        model=model,
        messages=build_explain_messages(title, sapo, label, tickers, prompt),
        reasoning_effort=reasoning_effort,
    )
    return {
        "explanation": (resp.choices[0].message.content or "").strip(),
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def sample_per_label(
    df: pd.DataFrame, label_col: str, labels: list, n: int, seed: int
) -> pd.DataFrame:
    """n random rows for each label in `labels`, in label order."""
    parts = [df[df[label_col] == lab].sample(n=n, random_state=seed) for lab in labels]
    return pd.concat(parts, ignore_index=True)


def load_prompt(path: str = EXPLAIN_PROMPT_PATH) -> str:
    """Read the explanation system prompt."""
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    """Explain EXPLAIN_PER_LABEL train items per event label and save them."""
    labels = [lab for lab in LABELS if lab != "other"]
    df = sample_per_label(
        pd.read_csv(LABELED_CSV), "final_label", labels, EXPLAIN_PER_LABEL, SEED
    )
    client, prompt, aliases = get_client(), load_prompt(), load_aliases()
    rows = []
    for row in df.itertuples(index=False):
        tickers = main_tickers(row.title, row.sapo, aliases)
        res = explain_news(
            row.title, row.sapo, row.final_label, tickers, client, prompt
        )
        rows.append(
            {
                "id": row.id,
                "label": row.final_label,
                "tickers": ";".join(tickers),
                "title": row.title,
                "sapo": row.sapo,
            }
            | res
        )
        print(f"{len(rows)}/{len(df)}", end="\r")
    out = pd.DataFrame(rows).assign(model=MODEL)
    out.to_csv(EXPLAIN_SAMPLES_CSV, index=False)
    cost = cost_usd(out["input_tokens"].sum(), out["output_tokens"].sum())
    print(f"wrote {len(out)} rows to {EXPLAIN_SAMPLES_CSV}; cost ${cost:.4f}")


if __name__ == "__main__":
    main()
