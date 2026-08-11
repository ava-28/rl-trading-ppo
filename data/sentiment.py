"""
data/sentiment.py
-----------------
Sentiment scoring for the research-grade trading system.

Backends
--------
  finbert : ProsusAI/finbert — transformer-based financial sentiment
  vader   : VADER compound score (lightweight fallback, no GPU needed)
  none    : returns 0.0 everywhere (clean ablation baseline)

Lookahead safety (CRITICAL)
----------------------------
News published on day t is only used to form the signal for day t+1.
The `no_future_leakage` flag in config.yaml enforces this via a .shift(1)
on the aggregated daily score before merging into the feature matrix.

Aggregation
-----------
  ewm  : exponentially weighted mean with configurable half-life (default 3d)
  mean : simple daily average
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SentimentBackend = Literal["finbert", "vader", "none"]


# ---------------------------------------------------------------------------
# Sentence-level scoring
# ---------------------------------------------------------------------------

def _score_finbert(
    texts:       List[str],
    model_name:  str    = "ProsusAI/finbert",
    batch_size:  int    = 16,
    device:      str    = "cpu",
    cache_dir:   Optional[str] = None,
) -> List[float]:
    """
    Score a list of strings with FinBERT.

    Returns a list of floats in [-1, 1]:
      +1 = positive, -1 = negative, 0 = neutral.
    Weighted by softmax probabilities:
      score = P(positive) - P(negative)
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers is required for FinBERT.  "
            "Install with: pip install transformers torch"
        ) from exc

    if not texts:
        return []

    # Load pipeline once (cached across calls via module-level singleton)
    _pipe = _get_finbert_pipe(model_name, device, cache_dir)

    scores: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # truncate to avoid exceeding max token length
        batch = [t[:512] for t in batch]
        try:
            results = _pipe(batch, truncation=True, max_length=512)
            for res in results:
                pos = next((r["score"] for r in res if r["label"].lower() == "positive"), 0.0)
                neg = next((r["score"] for r in res if r["label"].lower() == "negative"), 0.0)
                scores.append(float(pos - neg))
        except Exception as exc:
            log.warning(f"[sentiment] FinBERT batch failed: {exc}; using 0.0 for batch")
            scores.extend([0.0] * len(batch))

    return scores


# Module-level singleton for the FinBERT pipeline (avoids re-loading)
_FINBERT_PIPE = None

def _get_finbert_pipe(model_name: str, device: str, cache_dir: Optional[str]):
    global _FINBERT_PIPE
    if _FINBERT_PIPE is None:
        log.info(f"[sentiment] Loading FinBERT: {model_name}")
        from transformers import pipeline
        kwargs: dict = {
            "model": model_name,
            "tokenizer": model_name,
            "task": "text-classification",
            "top_k": None,   # return all labels
        }
        if cache_dir:
            kwargs["model_kwargs"] = {"cache_dir": cache_dir}
        # device: 0 = first GPU, -1 = CPU, "mps" = Apple Silicon
        if device.startswith("cuda"):
            kwargs["device"] = 0
        elif device == "mps":
            kwargs["device"] = "mps"
        else:
            kwargs["device"] = -1
        _FINBERT_PIPE = pipeline(**kwargs)
    return _FINBERT_PIPE


def _score_vader(texts: List[str]) -> List[float]:
    """Score texts with VADER compound score (−1 to +1)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError(
            "vaderSentiment is required as fallback.  "
            "Install with: pip install vaderSentiment"
        ) from exc
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_daily(
    news_df:    pd.DataFrame,
    scores:     List[float],
    aggregation: Literal["ewm", "mean"] = "ewm",
    ewm_halflife: int = 3,
    max_articles: int = 5,
    trading_calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Aggregate article-level scores into a daily sentiment time series.

    Parameters
    ----------
    news_df          : DataFrame with 'datetime' column (UTC-aware)
    scores           : matching list of sentence-level scores
    aggregation      : 'ewm' or 'mean'
    ewm_halflife     : EWM decay parameter (days)
    max_articles     : cap number of articles per day used in averaging
    trading_calendar : if supplied, reindex to these dates so the series
                       aligns perfectly with the price panel

    Returns
    -------
    pd.Series with date index (tz-naive), filled forward for missing days
    """
    if len(news_df) == 0 or not scores:
        if trading_calendar is not None:
            return pd.Series(0.0, index=trading_calendar, name="sentiment_score")
        return pd.Series(dtype=float, name="sentiment_score")

    tmp = news_df.copy()
    tmp["score"] = scores
    # Convert to local date (drop timezone for merging with price index)
    tmp["date"] = tmp["datetime"].dt.tz_localize(None).dt.normalize()

    # Cap articles per day and average
    daily = (
        tmp.sort_values("score", ascending=False)
           .groupby("date")
           .head(max_articles)
           .groupby("date")["score"]
           .mean()
    )
    daily.index = pd.to_datetime(daily.index)

    if aggregation == "ewm":
        # EWM over the daily series (fills in a persistence effect)
        daily = daily.ewm(halflife=ewm_halflife, adjust=True).mean()

    # Forward-fill gaps (weekends, holidays)
    if trading_calendar is not None:
        daily = daily.reindex(trading_calendar).fillna(method="ffill").fillna(0.0)
    else:
        daily = daily.fillna(0.0)

    return daily.rename("sentiment_score")


# ---------------------------------------------------------------------------
# Full per-ticker pipeline
# ---------------------------------------------------------------------------

def build_sentiment_series(
    news_df:        pd.DataFrame,
    backend:        SentimentBackend    = "finbert",
    model_name:     str                 = "ProsusAI/finbert",
    aggregation:    Literal["ewm","mean"] = "ewm",
    ewm_halflife:   int                 = 3,
    max_articles:   int                 = 5,
    no_future_leakage: bool             = True,
    trading_calendar: Optional[pd.DatetimeIndex] = None,
    device:         str                 = "cpu",
) -> pd.Series:
    """
    End-to-end sentiment series for one ticker.

    Parameters
    ----------
    no_future_leakage : if True, shift the daily score forward by 1 trading
                        day so that news from day t is only used as a signal
                        at day t+1.

    Returns
    -------
    pd.Series with date index (tz-naive), values in [-1, 1].
    """
    if backend == "none" or news_df.empty:
        if trading_calendar is not None:
            return pd.Series(0.0, index=trading_calendar, name="sentiment_score")
        return pd.Series(dtype=float, name="sentiment_score")

    headlines = news_df["headline"].fillna("").tolist()

    # Score
    if backend == "finbert":
        scores = _score_finbert(headlines, model_name=model_name, device=device)
    elif backend == "vader":
        scores = _score_vader(headlines)
    else:
        raise ValueError(f"Unknown sentiment backend: {backend!r}")

    # Aggregate to daily
    daily = aggregate_daily(
        news_df,
        scores,
        aggregation      = aggregation,
        ewm_halflife     = ewm_halflife,
        max_articles     = max_articles,
        trading_calendar = trading_calendar,
    )

    # Causal shift: news from t → signal at t+1
    if no_future_leakage:
        daily = daily.shift(1).fillna(0.0)

    return daily


def build_sentiment_panel(
    news_panel:     Dict[str, pd.DataFrame],
    trading_calendars: Dict[str, pd.DatetimeIndex],
    backend:        SentimentBackend    = "finbert",
    model_name:     str                 = "ProsusAI/finbert",
    aggregation:    Literal["ewm","mean"] = "ewm",
    ewm_halflife:   int                 = 3,
    max_articles:   int                 = 5,
    no_future_leakage: bool             = True,
    device:         str                 = "cpu",
) -> Dict[str, pd.Series]:
    """
    Build per-ticker daily sentiment series for all assets.

    Returns dict[ticker → pd.Series]
    """
    panel: Dict[str, pd.Series] = {}
    for ticker, news_df in news_panel.items():
        cal = trading_calendars.get(ticker)
        log.info(f"[sentiment] Scoring {ticker} ({len(news_df)} articles, backend={backend})")
        panel[ticker] = build_sentiment_series(
            news_df,
            backend            = backend,
            model_name         = model_name,
            aggregation        = aggregation,
            ewm_halflife       = ewm_halflife,
            max_articles       = max_articles,
            no_future_leakage  = no_future_leakage,
            trading_calendar   = cal,
            device             = device,
        )
    return panel


# ---------------------------------------------------------------------------
# Inject sentiment into feature panel
# ---------------------------------------------------------------------------

def inject_sentiment(
    feat_panel:  Dict[str, pd.DataFrame],
    sent_panel:  Dict[str, pd.Series],
) -> Dict[str, pd.DataFrame]:
    """
    Overwrite the 'sentiment_score' column in each asset's feature DataFrame
    with the corresponding daily sentiment series.

    Aligns on the date index; missing dates are filled with 0.0.
    """
    updated: Dict[str, pd.DataFrame] = {}
    for ticker, feat_df in feat_panel.items():
        df = feat_df.copy()
        if ticker in sent_panel:
            sent = sent_panel[ticker].reindex(df.index).fillna(0.0)
            df["sentiment_score"] = sent.values
        updated[ticker] = df
    return updated
