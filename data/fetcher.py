"""
data/fetcher.py
---------------
Downloads and caches price data (yfinance) and news headlines (Finnhub)
for the full asset universe defined in config.yaml.

Caching strategy
----------------
  * Price data  → {cache_dir}/{ticker}_prices.parquet
  * News data   → {cache_dir}/{ticker}_news.parquet
  * force_refetch=true in config.yaml bypasses the cache for that run.

Lookahead safety
----------------
  * Prices are fetched with auto_adjust=True (handles splits/dividends).
  * News timestamps are stored in UTC; downstream alignment is handled in
    sentiment.py with a strict t-1 rule (see no_future_leakage flag).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_prices(
    ticker:       str,
    start:        str,
    end:          str,
    cache_dir:    str,
    force:        bool = False,
    max_retries:  int  = 3,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily OHLCV data for *ticker* between *start* and
    *end* (inclusive), sourced from yfinance.

    Columns returned (lowercase):
      open, high, low, close, volume

    Index: DatetimeIndex (tz-naive, date only).
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"{ticker}_prices.parquet"

    if not force and cache_file.exists():
        log.info(f"[fetcher] Loading cached prices: {cache_file}")
        df = pd.read_parquet(cache_file)
        # Only return rows within the requested window
        df.index = pd.to_datetime(df.index)
        return df.loc[start:end]

    log.info(f"[fetcher] Downloading prices for {ticker} ({start} → {end})")
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"yfinance failed for {ticker}: {exc}") from exc
            log.warning(f"[fetcher] Attempt {attempt} failed, retrying…")
            time.sleep(2 ** attempt)

    if raw.empty:
        raise ValueError(f"No price data returned for {ticker} ({start}→{end}).")

    # Normalise column names — yfinance ≥0.2 returns MultiIndex when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(ticker, axis=1, level=1)
    raw.columns = [c.lower() for c in raw.columns]

    # Keep only OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    df = raw[keep].copy()
    df.index = pd.to_datetime(df.index).normalize()   # strip intraday time component
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    df.to_parquet(cache_file)
    log.info(f"[fetcher] Cached {len(df)} rows → {cache_file}")
    return df


def fetch_vix(
    start:     str,
    end:       str,
    cache_dir: str,
    force:     bool = False,
) -> pd.Series:
    """
    Return a daily Series of VIX close prices.
    VIX ticker is '^VIX' on yfinance.
    """
    df = fetch_prices("^VIX", start, end, cache_dir, force=force)
    return df["close"].rename("vix")


# ---------------------------------------------------------------------------
# News fetching (Finnhub)
# ---------------------------------------------------------------------------

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"


def fetch_news(
    ticker:      str,
    start:       str,
    end:         str,
    api_key:     str,
    cache_dir:   str,
    force:       bool = False,
    max_articles: int = 2000,
    sleep_sec:   float = 1.1,
) -> pd.DataFrame:
    """
    Fetch company news headlines from Finnhub for *ticker* over [start, end].

    Returns a DataFrame with columns:
      datetime (UTC, tz-aware), headline, summary, source, ticker

    If api_key is empty (''), returns an empty DataFrame so the pipeline
    degrades gracefully to no-news sentiment.

    Caching: results are stored as a parquet; subsequent calls load from disk
    unless force=True.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"{ticker}_news.parquet"

    if not force and cache_file.exists():
        log.info(f"[fetcher] Loading cached news: {cache_file}")
        df = pd.read_parquet(cache_file)
        # Filter to requested window (Finnhub datetime col is UTC-aware)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts   = pd.Timestamp(end,   tz="UTC")
        return df[(df["datetime"] >= start_ts) & (df["datetime"] < end_ts)]

    if not api_key:
        log.warning(f"[fetcher] No Finnhub API key — skipping news for {ticker}.")
        return pd.DataFrame(columns=["datetime", "headline", "summary", "source", "ticker"])

    log.info(f"[fetcher] Fetching Finnhub news for {ticker} ({start} → {end})")

    # Finnhub free tier allows ~60 req/min; we chunk by month to stay safe
    date_ranges = _monthly_chunks(start, end)
    rows: List[dict] = []

    for chunk_start, chunk_end in date_ranges:
        params = {
            "symbol": ticker,
            "from":   chunk_start,
            "to":     chunk_end,
            "token":  api_key,
        }
        try:
            resp = requests.get(FINNHUB_NEWS_URL, params=params, timeout=15)
            if resp.status_code == 429:
                log.warning(
                    f"[fetcher] RATE LIMITED by Finnhub ({chunk_start}→{chunk_end}). "
                    f"Backing off 60s and retrying once."
                )
                time.sleep(60)
                resp = requests.get(FINNHUB_NEWS_URL, params=params, timeout=15)
            if resp.status_code in (401, 403):
                log.error(
                    f"[fetcher] Finnhub rejected the API key (HTTP {resp.status_code}). "
                    f"Check FINNHUB_KEY. Aborting news fetch for {ticker}."
                )
                break
            resp.raise_for_status()
            articles = resp.json()
        except Exception as exc:
            log.warning(f"[fetcher] Finnhub request failed ({chunk_start}→{chunk_end}): {exc}")
            articles = []

        for art in articles[:max_articles]:
            ts = pd.Timestamp(art.get("datetime", 0), unit="s", tz="UTC")
            rows.append({
                "datetime": ts,
                "headline": art.get("headline", ""),
                "summary":  art.get("summary",  ""),
                "source":   art.get("source",   ""),
                "ticker":   ticker,
            })

        time.sleep(sleep_sec)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["datetime", "headline", "summary", "source", "ticker"])
    else:
        df.drop_duplicates(subset=["datetime", "headline"], inplace=True)
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

    df.to_parquet(cache_file)
    log.info(f"[fetcher] Cached {len(df)} news articles → {cache_file}")

    # Filter to requested window
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")
    return df[(df["datetime"] >= start_ts) & (df["datetime"] < end_ts)]


# ---------------------------------------------------------------------------
# Multi-asset loader — builds aligned price panel + VIX
# ---------------------------------------------------------------------------

def load_price_panel(
    tickers:   List[str],
    start:     str,
    end:       str,
    vix_ticker: str,
    cache_dir: str,
    force:     bool = False,
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """
    Download/load prices for every ticker in *tickers* plus VIX.

    Returns
    -------
    prices : dict[ticker → DataFrame(open, high, low, close, volume)]
    vix    : Series of daily VIX close, aligned to the union of trading days
    """
    prices: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        prices[t] = fetch_prices(t, start, end, cache_dir, force=force)
        log.info(f"[fetcher] {t}: {len(prices[t])} trading days")

    vix = fetch_vix(start, end, cache_dir, force=force)
    return prices, vix


def load_news_panel(
    tickers:   List[str],
    start:     str,
    end:       str,
    api_key:   str,
    cache_dir: str,
    force:     bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Download/load news headlines for every ticker.

    Returns dict[ticker → DataFrame(datetime, headline, summary, source, ticker)]
    """
    news: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        news[t] = fetch_news(t, start, end, api_key, cache_dir, force=force)
        log.info(f"[fetcher] {t}: {len(news[t])} news articles")
    return news


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_chunks(start: str, end: str) -> List[Tuple[str, str]]:
    """
    Split [start, end] into month-sized sub-intervals for the Finnhub API.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    chunks = []
    cur = s
    while cur < e:
        nxt = (cur + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(days=1)
        nxt = min(nxt, e)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return chunks


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")

    tickers    = cfg["tickers"]
    vix_ticker = cfg.get("vix_ticker", "^VIX")
    cache_dir  = cfg["data"]["cache_dir"]
    api_key    = cfg["data"].get("finnhub_key", "") or os.environ.get("FINNHUB_KEY", "")
    force      = cfg["data"].get("force_refetch", False)

    # Determine full date span across all walk-forward folds
    folds      = cfg["walk_forward"]
    full_start = min(f["train_start"] for f in folds)
    full_end   = max(f["test_end"]    for f in folds)

    log.info(f"Fetching data: {full_start} → {full_end}")
    prices, vix = load_price_panel(tickers, full_start, full_end, vix_ticker, cache_dir, force=force)
    news        = load_news_panel(tickers, full_start, full_end, api_key, cache_dir, force=force)

    print("\nPrice summary:")
    for t, df in prices.items():
        print(f"  {t}: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} days)")
    print(f"  VIX: {vix.index[0].date()} → {vix.index[-1].date()}  ({len(vix)} days)")
    print("\nNews summary:")
    for t, df in news.items():
        print(f"  {t}: {len(df)} articles")
