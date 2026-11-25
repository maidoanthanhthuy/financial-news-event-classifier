"""Build the event list: raw news tied to exactly one main VN30 ticker, labeled.

Usage:
    python -m src.event_study.build_events plan   # counts and estimates, no API
    python -m src.event_study.build_events run    # label and write; safe to rerun
    python -m src.event_study.build_events events # filter and de-duplicate events
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.classify.llm_label import classify_news, cost_usd, get_client, load_guide
from src.classify.tickers import load_aliases, main_tickers
from src.event_study.config import (
    DROP_LABELS,
    EVENTS_CSV,
    MIN_GAP_DAYS,
    NEWS_EVENTS_CSV,
    NEWS_EVENTS_LOG,
    OUTPUT_COLS,
    RAW_CSV,
    SECONDS_PER_LLM_CALL,
    TEST_CSV,
    TRAIN_CSV,
    USD_PER_LLM_CALL,
)


def add_ids(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy of raw news with id = 1-based row number."""
    return raw.assign(id=range(1, len(raw) + 1))


def single_ticker_news(news: pd.DataFrame, aliases: list) -> pd.DataFrame:
    """Rows whose main_tickers has exactly one ticker, with a ticker column."""
    found = [
        main_tickers(t, s, aliases)
        for t, s in zip(news["title"], news["sapo"], strict=True)
    ]
    keep = [len(f) == 1 for f in found]
    tickers = [f[0] for f, k in zip(found, keep, strict=True) if k]
    return news[keep].assign(ticker=tickers)


def attach_human_labels(
    news: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Add label/label_source from human labels: train by url, test by title+sapo."""
    tr = train[["url", "final_label"]].rename(columns={"final_label": "h_train"})
    te = test[["title", "sapo", "human_label"]].rename(
        columns={"human_label": "h_test"}
    )
    out = news.merge(tr, on="url", how="left", validate="m:1")
    out = out.merge(te, on=["title", "sapo"], how="left", validate="m:1")
    label = out["h_train"].fillna(out["h_test"])
    source = label.notna().map({True: "human", False: None})
    return out.drop(columns=["h_train", "h_test"]).assign(
        label=label, label_source=source
    )


def estimate(n_calls: int, usd_per_call: float, seconds_per_call: float) -> dict:
    """Estimated cost (USD) and time (minutes) for n sequential LLM calls."""
    return {
        "usd": n_calls * usd_per_call,
        "minutes": n_calls * seconds_per_call / 60,
    }


def drop_close_repeats(
    events: pd.DataFrame, min_gap_days: int = MIN_GAP_DAYS
) -> pd.DataFrame:
    """Keep the first of same ticker+label news less than min_gap_days apart."""
    date = pd.to_datetime(events["published_at"].str[:10])
    df = events.assign(_date=date).sort_values(["_date", "id"])
    keep = []
    last_kept: dict = {}
    for t, lab, d in zip(df["ticker"], df["label"], df["_date"], strict=True):
        key = (t, lab)
        ok = key not in last_kept or (d - last_kept[key]).days >= min_gap_days
        keep.append(ok)
        if ok:
            last_kept[key] = d
    return df[keep].drop(columns="_date")


def select_events(
    news: pd.DataFrame,
    drop_labels: list = DROP_LABELS,
    min_gap_days: int = MIN_GAP_DAYS,
) -> pd.DataFrame:
    """Drop unwanted labels, then same ticker+label repeats within min_gap_days."""
    return drop_close_repeats(news[~news["label"].isin(drop_labels)], min_gap_days)


def prepare() -> pd.DataFrame:
    """Read raw news and labeled sets; return single-ticker news with human labels."""
    news = single_ticker_news(add_ids(pd.read_csv(RAW_CSV)), load_aliases())
    return attach_human_labels(news, pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV))


def log(message: str, path: str = NEWS_EVENTS_LOG) -> None:
    """Append a timestamped line to the log file and print it."""
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def plan() -> None:
    """Print counts and the cost/time estimate without calling the API."""
    raw_n = len(pd.read_csv(RAW_CSV))
    df = prepare()
    n_llm = int(df["label"].isna().sum())
    print(f"raw news: {raw_n}")
    print(f"exactly one main ticker: {len(df)}")
    print(f"human labels: {len(df) - n_llm}; need LLM: {n_llm}")
    print(f"tickers: {df['ticker'].value_counts().to_dict()}")
    est = estimate(n_llm, USD_PER_LLM_CALL, SECONDS_PER_LLM_CALL)
    print(f"estimate: ${est['usd']:.3f}, {est['minutes']:.0f} min")


def run() -> None:
    """Write human-labeled rows, then label the rest with the LLM; resumable."""
    out = Path(NEWS_EVENTS_CSV)
    df = prepare()
    done = set(pd.read_csv(out, usecols=["id"])["id"]) if out.exists() else set()
    todo = df[~df["id"].isin(done)]
    human = todo[todo["label_source"] == "human"]
    if len(human):
        human[OUTPUT_COLS].to_csv(out, mode="a", header=not out.exists(), index=False)
    llm = todo[todo["label_source"] != "human"].sort_values("id")
    log(f"start: {len(done)} done, {len(human)} human added, {len(llm)} for LLM")

    client, guide = get_client(), load_guide()
    total_in = total_out = skipped = 0
    for i, row in enumerate(llm.itertuples(index=False), start=1):
        res = classify_news(row.title, row.sapo, client, guide)
        total_in += res["input_tokens"]
        total_out += res["output_tokens"]
        if res["label"] is None:
            skipped += 1
            continue
        rec = row._asdict() | {"label": res["label"], "label_source": "llm"}
        pd.DataFrame([rec])[OUTPUT_COLS].to_csv(
            out, mode="a", header=not out.exists(), index=False
        )
        print(f"{i}/{len(llm)}", end="\r")
    cost = cost_usd(total_in, total_out)
    log(
        f"end: {len(llm) - skipped} LLM labels, {skipped} unparsed, "
        f"tokens in={total_in} out={total_out}, cost ${cost:.4f}"
    )


def events() -> None:
    """Write the filtered event list and print counts per label."""
    news = pd.read_csv(NEWS_EVENTS_CSV)
    kept = news[~news["label"].isin(DROP_LABELS)]
    out = select_events(news)
    out.to_csv(EVENTS_CSV, index=False)
    counts = pd.DataFrame(
        {
            "before": kept["label"].value_counts(),
            "after": out["label"].value_counts(),
        }
    )
    print(counts.fillna(0).astype(int))
    print(f"events: {len(kept)} -> {len(out)}, tickers: {out['ticker'].nunique()}")


def main() -> None:
    """Dispatch the plan, run or events command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "run", "events"])
    args = parser.parse_args()
    {"plan": plan, "run": run, "events": events}[args.command]()


if __name__ == "__main__":
    main()
