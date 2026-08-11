"""
probe_news.py
-------------
Diagnostic: how far back does the Finnhub API key actually return company news?

The free tier advertises limited historical coverage, and the exact cutoff is not
documented reliably. This probes one ticker across a series of past months and
reports where the data stops, so the study window can be set from evidence rather
than assumption.

Run BEFORE launching the full experiment suite.

Usage
-----
    export FINNHUB_KEY="your_key"
    python probe_news.py
    python probe_news.py --ticker NVDA --months 36
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta

import requests

URL = "https://finnhub.io/api/v1/company-news"


def month_window(months_ago: int) -> tuple[str, str]:
    """Return (from, to) ISO date strings for the calendar month N months back."""
    today = date.today()
    year = today.year
    month = today.month - months_ago
    while month <= 0:
        month += 12
        year -= 1
    start = date(year, month, 1)
    end = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    return start.isoformat(), end.isoformat()


def probe(ticker: str, months: int, key: str, sleep_sec: float) -> None:
    print(f"\nProbing Finnhub company-news for {ticker}, {months} months back.")
    print("Respecting the 60 req/min free-tier limit, so this takes a moment.\n")
    print(f"{'Month':<12} {'Articles':>9}   Status")
    print("-" * 46)

    earliest_with_data = None
    consecutive_empty = 0

    for m in range(months):
        frm, to = month_window(m)
        label = frm[:7]

        try:
            resp = requests.get(
                URL,
                params={"symbol": ticker, "from": frm, "to": to, "token": key},
                timeout=20,
            )
        except Exception as exc:
            print(f"{label:<12} {'-':>9}   request failed: {exc}")
            time.sleep(sleep_sec)
            continue

        if resp.status_code in (401, 403):
            print(f"\nAPI key rejected (HTTP {resp.status_code}).")
            print("Check that FINNHUB_KEY is set correctly and the key is active.")
            sys.exit(1)

        if resp.status_code == 429:
            print(f"{label:<12} {'-':>9}   rate limited; waiting 60s")
            time.sleep(60)
            continue

        if resp.status_code != 200:
            print(f"{label:<12} {'-':>9}   HTTP {resp.status_code}")
            time.sleep(sleep_sec)
            continue

        try:
            n = len(resp.json())
        except Exception:
            n = 0

        if n > 0:
            earliest_with_data = frm
            consecutive_empty = 0
            print(f"{label:<12} {n:>9}   ok")
        else:
            consecutive_empty += 1
            print(f"{label:<12} {0:>9}   empty")
            if consecutive_empty >= 4:
                print("\nFour consecutive empty months — treating this as the cutoff.")
                break

        time.sleep(sleep_sec)

    print("\n" + "=" * 46)
    if earliest_with_data:
        print(f"Earliest month returning data: {earliest_with_data[:7]}")
        print("\nSuggested config window (leave a small margin):")
        print(f"  study_start: \"{earliest_with_data}\"")
        print(f"  study_end:   \"{date.today().isoformat()}\"")
    else:
        print("No news returned for any month. Either the key lacks news access,")
        print("or the free tier does not cover this endpoint for your account.")
    print("=" * 46)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL", help="Ticker to probe (default AAPL)")
    ap.add_argument("--months", type=int, default=30, help="How many months back to test")
    ap.add_argument("--sleep", type=float, default=1.1, help="Seconds between requests")
    args = ap.parse_args()

    key = os.environ.get("FINNHUB_KEY", "").strip()
    if not key:
        print("FINNHUB_KEY is not set in the environment.")
        print('Run:  export FINNHUB_KEY="your_key_here"')
        sys.exit(1)

    print(f"Key detected: {key[:6]}…{key[-4:]}  ({len(key)} chars)")
    probe(args.ticker, args.months, key, args.sleep)


if __name__ == "__main__":
    main()
