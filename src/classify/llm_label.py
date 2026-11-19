"""Pre-label news with an OpenAI model; safe to stop and rerun.

Usage: python -m src.classify.llm_label [--limit N]
"""

import argparse
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


def label_one(client: OpenAI, title: str, sapo: str, guide: str) -> tuple:
    """Ask the model for one label; return (NewsLabel or None, usage)."""
    resp = client.chat.completions.parse(
        model=MODEL,
        messages=build_messages(title, sapo, guide),
        response_format=NewsLabel,
    )
    return resp.choices[0].message.parsed, resp.usage


def main() -> None:
    """Label pending news and append each result to the output CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="items in this run")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()
    guide = Path(GUIDE_PATH).read_text(encoding="utf-8")
    out = Path(OUTPUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out, usecols=["url"])["url"]) if out.exists() else set()

    todo = pending_rows(pd.read_csv(INPUT_CSV), done).head(args.limit)
    total_in = total_out = 0
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        result, usage = label_one(client, row.title, row.sapo, guide)
        total_in += usage.prompt_tokens
        total_out += usage.completion_tokens
        if result is None:
            print(f"skip (no parsed answer): {row.url}")
            continue
        record = row._asdict() | {
            "label": result.label,
            "reason": result.reason,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
        pd.DataFrame([record]).to_csv(
            out, mode="a", header=not out.exists(), index=False
        )
        print(f"{i}/{len(todo)} {result.label}: {row.title[:60]}")

    cost = cost_usd(total_in, total_out)
    print(f"tokens in={total_in} out={total_out}, cost ${cost:.4f}")


if __name__ == "__main__":
    main()
