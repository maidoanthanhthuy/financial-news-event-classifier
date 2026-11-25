"""Fetch raw daily prices from the KBS API for the event study tickers.

Usage:
    python -m src.event_study.fetch_prices
    python -m src.event_study.fetch_prices --symbols FPT VNINDEX

Each response is saved byte for byte to data/raw/prices/<SYMBOL>.json. Existing
files are never overwritten, so a re-run only fetches what is missing.
manifest.json in the same folder logs every call.
"""

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.event_study.config import (
    HTTP_PAUSE_SECONDS,
    HTTP_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    INDEX_SYMBOL,
    KBS_BASE_URL,
    KBS_HEADERS,
    KBS_MIN_ROW_RATIO,
    KBS_ROWS_KEY,
    NEWS_EVENTS_CSV,
    PRICE_END,
    PRICE_START,
    PRICES_RAW_DIR,
)


def kbs_url(kind: str, symbol: str, base_url: str = KBS_BASE_URL) -> str:
    """KBS daily history URL; kind is 'stocks' or 'index'."""
    return f"{base_url}/{kind}/{symbol}/data_day"


def kbs_params(start: date, end: date) -> dict:
    """KBS query parameters (dates as DD-MM-YYYY)."""
    return {"sdate": start.strftime("%d-%m-%Y"), "edate": end.strftime("%d-%m-%Y")}


def summarize_payload(payload: object, rows_key: str = KBS_ROWS_KEY) -> dict:
    """Row count and first/last date of a KBS payload ('t' is 'YYYY-MM-DD 07:00')."""
    rows = payload.get(rows_key) if isinstance(payload, dict) else None
    rows = rows if isinstance(rows, list) else []
    dates = sorted(str(r["t"])[:10] for r in rows if isinstance(r, dict) and "t" in r)
    return {
        "n_rows": len(rows),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
    }


def looks_truncated(
    n_rows: int, start: date, end: date, min_ratio: float = KBS_MIN_ROW_RATIO
) -> bool:
    """True if there are fewer rows than min_ratio x weekdays in [start, end]."""
    weekdays = sum(
        (start + timedelta(days=i)).weekday() < 5 for i in range((end - start).days + 1)
    )
    return n_rows < min_ratio * weekdays


def year_ranges(start: date, end: date) -> list:
    """Split [start, end] into calendar-year pieces."""
    return [
        (max(start, date(y, 1, 1)), min(end, date(y, 12, 31)))
        for y in range(start.year, end.year + 1)
    ]


def get_with_retry(session: requests.Session, url: str, params: dict) -> dict:
    """GET a URL with retries and a pause after each try."""
    error, http_status = None, None
    for attempt in range(1, HTTP_RETRIES + 2):
        try:
            resp = session.get(
                url, params=params, headers=KBS_HEADERS, timeout=HTTP_TIMEOUT_SECONDS
            )
            http_status = resp.status_code
            resp.raise_for_status()
            return {
                "content": resp.content,
                "payload": resp.json(),
                "error": None,
                "http_status": http_status,
                "attempts": attempt,
            }
        except (requests.RequestException, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            time.sleep(HTTP_PAUSE_SECONDS)
    return {
        "content": None,
        "payload": None,
        "error": error,
        "http_status": http_status,
        "attempts": HTTP_RETRIES + 1,
    }


def get_or_fetch(
    session: requests.Session, url: str, start: date, end: date, path: Path
) -> tuple:
    """Reuse the file at path, or fetch and save it; return (status, summary, call)."""
    if path.exists():
        return "existing", summarize_payload(json.loads(path.read_bytes())), None
    params = kbs_params(start, end)
    res = get_with_retry(session, url, params)
    summary = summarize_payload(res["payload"])
    if res["error"]:
        status = "error"
    elif summary["n_rows"] == 0:
        status = "empty"
    else:
        status = "ok"
        with path.open("xb") as f:  # never overwrite
            f.write(res["content"])
    call = {
        "file": path.name if status == "ok" else None,
        "url": url,
        "params": params,
        "status": status,
        "http_status": res["http_status"],
        "attempts": res["attempts"],
        "error": res["error"],
        **summary,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return status, summary, call


def fetch_symbol(
    session: requests.Session,
    symbol: str,
    kind: str,
    start: date,
    end: date,
    out_dir: Path,
) -> tuple:
    """Fetch one symbol; refetch year by year if the full call looks capped."""
    url = kbs_url(kind, symbol)
    status, full, call = get_or_fetch(
        session, url, start, end, out_dir / f"{symbol}.json"
    )
    calls = [call] if call else []
    ok = status in ("ok", "existing")
    entry = {
        "kind": kind,
        "status": "ok" if ok else status,
        "chunked": False,
        "files": [f"{symbol}.json"] if ok else [],
        **full,
    }
    if not ok or not looks_truncated(full["n_rows"], start, end):
        return entry, calls
    files, pieces, failed = [], [], False
    for s, e in year_ranges(start, end):
        name = f"{symbol}_{s.year}.json"
        st, piece, call = get_or_fetch(session, url, s, e, out_dir / name)
        calls += [call] if call else []
        pieces.append(piece)
        files += [name] if st in ("ok", "existing") else []
        failed = failed or st == "error"
    firsts = [p["first_date"] for p in pieces if p["first_date"]]
    lasts = [p["last_date"] for p in pieces if p["last_date"]]
    entry.update(
        status="error" if failed else "ok",
        chunked=True,
        files=files,
        full_call_rows=full["n_rows"],
        n_rows=sum(p["n_rows"] for p in pieces),
        first_date=min(firsts, default=None),
        last_date=max(lasts, default=None),
    )
    return entry, calls


def write_manifest(path: Path, manifest: dict) -> None:
    """Write the manifest atomically (temp file, then rename)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    """Fetch all tickers in news_events.csv plus the index."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", help="only these symbols")
    args = parser.parse_args()
    tickers = sorted(pd.read_csv(NEWS_EVENTS_CSV)["ticker"].unique())
    jobs = [(t, "stocks") for t in tickers] + [(INDEX_SYMBOL, "index")]
    if args.symbols:
        jobs = [j for j in jobs if j[0] in set(args.symbols)]
    start, end = date.fromisoformat(PRICE_START), date.fromisoformat(PRICE_END)

    out_dir = Path(PRICES_RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "source": "KBS",
            "endpoint": f"{KBS_BASE_URL}/{{kind}}/{{symbol}}/data_day",
            "start": PRICE_START,
            "end": PRICE_END,
            "price_adjustment": "adjusted close (checked in vn-macro-assistant "
            "against CafeF, 2026-11-11); volume not adjusted",
            "units": {"stocks": "VND", "index": "index points"},
            "symbols": {},
            "calls": [],
        }
    )
    session = requests.Session()
    for symbol, kind in jobs:
        entry, calls = fetch_symbol(session, symbol, kind, start, end, out_dir)
        manifest["calls"].extend(calls)
        manifest["symbols"][symbol] = entry
        write_manifest(manifest_path, manifest)
        print(
            f"{symbol:8} {entry['status']:6} {entry['n_rows']:4} rows "
            f"{entry['first_date']} -> {entry['last_date']}"
            + (" (year by year)" if entry["chunked"] else "")
        )
    bad = [s for s, e in manifest["symbols"].items() if e["status"] != "ok"]
    if bad:
        print(f"Not OK: {' '.join(bad)} -> re-run with --symbols {' '.join(bad)}")


if __name__ == "__main__":
    main()
