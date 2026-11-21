"""Pre-label news with an OpenAI model; safe to stop and rerun.

Usage: python -m src.classify.llm_label [--limit N]
"""

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.classify.config import (
    GUIDE_PATH,
    INPUT_CSV,
    LABELS,
    MODEL,
    OUTPUT_CSV,
    PRICE_INPUT,
    PRICE_OUTPUT,
    SEED,
)


class NewsLabel(BaseModel):
    """Structured answer expected from the model."""

    label: Literal[LABELS]
    reason: str = Field(description="Short reason in English, under 15 words")


def build_messages(title: str, sapo: str, guide: str) -> list[dict]:
    """Build the chat messages for one news item."""
    return [
        {"role": "system", "content": guide},
        {"role": "user", "content": f"Title: {title}\nSapo: {sapo}"},
    ]


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    price_input: float = PRICE_INPUT,
    price_output: float = PRICE_OUTPUT,
) -> float:
    """Return the API cost in USD for the given token counts."""
    return (input_tokens * price_input + output_tokens * price_output) / 1_000_000


def pending_rows(
    news: pd.DataFrame, done_urls: set[str], seed: int = SEED
) -> pd.DataFrame:
    """Return not-yet-labeled news in a fixed shuffled order."""
    todo = news[~news["url"].isin(done_urls)]
    return todo.sample(frac=1, random_state=seed).reset_index(drop=True)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return a shared OpenAI client (API key read from .env)."""
    load_dotenv()
    return OpenAI()


def load_guide(path: str = GUIDE_PATH) -> str:
    """Read the labeling guide used as the system prompt."""
    return Path(path).read_text(encoding="utf-8")


def classify_news(
    title: str,
    sapo: str,
    client: OpenAI | None = None,
    guide: str | None = None,
    model: str = MODEL,
) -> dict:
    """Classify one news item into one of 6 labels; return label, reason, tokens."""
    client = client or get_client()
    guide = load_guide() if guide is None else guide
    resp = client.chat.completions.parse(
        model=model,
        messages=build_messages(title, sapo, guide),
        response_format=NewsLabel,
    )
    parsed = resp.choices[0].message.parsed
    return {
        "label": parsed.label if parsed else None,
        "reason": parsed.reason if parsed else None,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def main() -> None:
    """Label pending news and append each result to the output CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="items in this run")
    args = parser.parse_args()

    client, guide = get_client(), load_guide()
    out = Path(OUTPUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out, usecols=["url"])["url"]) if out.exists() else set()

    todo = pending_rows(pd.read_csv(INPUT_CSV), done).head(args.limit)
    total_in = total_out = 0
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        result = classify_news(row.title, row.sapo, client, guide)
        total_in += result["input_tokens"]
        total_out += result["output_tokens"]
        if result["label"] is None:
            print(f"skip (no parsed answer): {row.url}")
            continue
        record = row._asdict() | result
        pd.DataFrame([record]).to_csv(
            out, mode="a", header=not out.exists(), index=False
        )
        print(f"{i}/{len(todo)} {result['label']}: {row.title[:60]}")

    cost = cost_usd(total_in, total_out)
    print(f"tokens in={total_in} out={total_out}, cost ${cost:.4f}")


if __name__ == "__main__":
    main()
