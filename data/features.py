"""
data/features.py
----------------
Feature engineering for the research-grade multi-asset trading system.

For each asset we compute 11 features (all strictly causal — computed from
data available at time t to produce a signal used at t+1):

  1.  ret_1d          — 1-day log return
  2.  ret_2d          — 2-day log return
  3.  ret_3d          — 3-day log return
  4.  ret_4d          — 4-day log return
  5.  ret_5d          — 5-day log return
  6.  rsi_14          — 14-day RSI  (Wilder smoothing)
  7.  ma_ratio        — close / MA20  (deviation from short trend)
  8.  ma_cross        — MA20 / MA50  (trend regime indicator)
  9.  realvol_10      — 10-day realised volatility (std of log returns)
  10. vol_ratio       — today volume / 20-day avg volume
  11. sentiment_score — filled in by data/sentiment.py (default 0.0)

Plus one shared feature:
  12. vix_norm — normalised VIX close

Normalisation is applied *online* using expanding statistics over the
training window (zscore or minmax), preventing any lookahead bias.

The public API is build_feature_matrix(), which returns a dict mapping
each ticker to a ready-to-use DataFrame, plus a vix Series.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURES_PER_STOCK = 11   # must match env/multi_asset_env.py constant


# ---------------------------------------------------------------------------
# Low-level indicator helpers
# ---------------------------------------------------------------------------

def _log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Strictly causal log return: log(close[t] / close[t-period])."""
    return np.log(close / close.shift(period))


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI with Wilder (exponential) smoothing.
    Returns a Series in [0, 100]; NaN for the first *window* rows.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing = EWM with span = 2*window - 1 (equiv. alpha = 1/window)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _realvol(close: pd.Series, window: int = 10) -> pd.Series:
    """Annualised realised volatility (rolling std of log returns × √252)."""
    log_ret = _log_returns(close, 1)
    return log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(252)


def _vol_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume divided by rolling mean volume (volume momentum proxy)."""
    avg_vol = volume.rolling(window=window, min_periods=window).mean()
    return volume / avg_vol.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Per-stock raw feature construction
# ---------------------------------------------------------------------------

def build_raw_features(
    price_df:     pd.DataFrame,
    lookback:     int = 5,
    rsi_window:   int = 14,
    ma_short:     int = 20,
    ma_long:      int = 50,
    vol_window:   int = 10,
) -> pd.DataFrame:
    """
    Compute the 11 raw (un-normalised) features for a single asset.

    Parameters
    ----------
    price_df : DataFrame with columns open, high, low, close, volume
    Returns  : DataFrame with FEATURES_PER_STOCK columns, same index.
    """
    close  = price_df["close"]
    volume = price_df.get("volume", pd.Series(np.nan, index=price_df.index))

    feats: Dict[str, pd.Series] = {}

    # --- Lookback returns (ret_1d … ret_Nd) ---
    for lag in range(1, lookback + 1):
        feats[f"ret_{lag}d"] = _log_returns(close, lag)

    # --- RSI ---
    feats["rsi_14"] = _rsi(close, window=rsi_window) / 100.0   # scale to [0, 1]

    # --- MA ratio and crossover ---
    ma_s = _sma(close, ma_short)
    ma_l = _sma(close, ma_long)
    feats["ma_ratio"] = close / ma_s.replace(0, np.nan) - 1.0  # deviation from MA20
    feats["ma_cross"] = ma_s / ma_l.replace(0, np.nan) - 1.0   # MA20 / MA50 spread

    # --- Realised volatility ---
    feats["realvol_10"] = _realvol(close, window=vol_window)

    # --- Volume ratio ---
    feats["vol_ratio"] = _vol_ratio(volume, window=ma_short)

    # --- Sentiment placeholder (filled in by sentiment.py) ---
    feats["sentiment_score"] = pd.Series(0.0, index=price_df.index)

    df_feats = pd.DataFrame(feats, index=price_df.index)
    assert df_feats.shape[1] == FEATURES_PER_STOCK, (
        f"Expected {FEATURES_PER_STOCK} features, got {df_feats.shape[1]}"
    )
    return df_feats


# ---------------------------------------------------------------------------
# Normalisation — purely causal (expanding window over training set)
# ---------------------------------------------------------------------------

NormMode = Literal["zscore", "minmax", "none"]


def normalise_expanding(
    df:   pd.DataFrame,
    mode: NormMode = "zscore",
    eps:  float    = 1e-8,
) -> pd.DataFrame:
    """
    Apply expanding-window normalisation to every column independently.

    For zscore : x_norm = (x - μ_expanding) / (σ_expanding + ε)
    For minmax : x_norm = (x - min_expanding) / (max_expanding - min_expanding + ε)

    Expanding statistics are computed up to and including row t, so there is
    absolutely no lookahead bias — statistics from future rows are never seen.

    This function should only be called on the *training* slice.  For test
    data, use normalise_with_stats() with the stats fit on the training slice.
    """
    if mode == "none":
        return df.copy()

    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    if mode == "zscore":
        exp_mean = df.expanding(min_periods=1).mean()
        exp_std  = df.expanding(min_periods=1).std().fillna(0)
        result   = (df - exp_mean) / (exp_std + eps)
    elif mode == "minmax":
        exp_min = df.expanding(min_periods=1).min()
        exp_max = df.expanding(min_periods=1).max()
        result  = (df - exp_min) / (exp_max - exp_min + eps)
    else:
        raise ValueError(f"Unknown normalisation mode: {mode!r}")

    return result


def fit_normalise_stats(
    df:   pd.DataFrame,
    mode: NormMode = "zscore",
) -> dict:
    """
    Compute normalisation statistics from the *entire* training DataFrame
    (to be applied to test data without further updating).

    Returns a dict with keys 'mean'/'std' (zscore) or 'min'/'max' (minmax).
    """
    if mode == "zscore":
        return {"mean": df.mean(), "std": df.std().replace(0, 1e-8), "mode": "zscore"}
    elif mode == "minmax":
        return {"min": df.min(), "max": df.max(), "mode": "minmax"}
    return {"mode": "none"}


def normalise_with_stats(df: pd.DataFrame, stats: dict, eps: float = 1e-8) -> pd.DataFrame:
    """Apply pre-fit normalisation stats to new (test) data."""
    mode = stats.get("mode", "zscore")
    if mode == "none":
        return df.copy()
    elif mode == "zscore":
        return (df - stats["mean"]) / (stats["std"] + eps)
    elif mode == "minmax":
        rng = stats["max"] - stats["min"] + eps
        return (df - stats["min"]) / rng
    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# VIX normalisation
# ---------------------------------------------------------------------------

def normalise_vix(
    vix:     pd.Series,
    stats:   Optional[dict] = None,
    mode:    NormMode = "zscore",
    eps:     float    = 1e-8,
) -> Tuple[pd.Series, dict]:
    """
    Normalise the VIX series.

    If *stats* is None, fit on the full series (call during training).
    Otherwise apply the pre-fit stats (call during testing).

    Returns (normalised_series, stats_dict).
    """
    df = vix.to_frame("vix")
    if stats is None:
        stats = fit_normalise_stats(df, mode=mode)
    norm_df = normalise_with_stats(df, stats, eps=eps)
    return norm_df["vix"], stats


# ---------------------------------------------------------------------------
# Full multi-asset feature builder
# ---------------------------------------------------------------------------

def build_feature_panel(
    prices:        Dict[str, pd.DataFrame],
    vix:           pd.Series,
    norm_mode:     NormMode = "zscore",
    lookback:      int      = 5,
    rsi_window:    int      = 14,
    ma_short:      int      = 20,
    ma_long:       int      = 50,
    vol_window:    int      = 10,
    train_end:     Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.Series, Dict[str, dict], dict]:
    """
    Build the complete feature panel for all assets.

    Parameters
    ----------
    prices     : dict[ticker → OHLCV DataFrame]
    vix        : raw VIX close Series
    norm_mode  : 'zscore' | 'minmax' | 'none'
    train_end  : last date of the training window (YYYY-MM-DD).
                 Normalisation stats are fit on data up to this date.
                 If None, fit on the full series.

    Returns
    -------
    feat_panel  : dict[ticker → normalised feature DataFrame]
    vix_norm    : normalised VIX Series
    feat_stats  : dict[ticker → normalisation stats dict]
    vix_stats   : normalisation stats for VIX
    """
    feat_panel: Dict[str, pd.DataFrame] = {}
    feat_stats: Dict[str, dict]         = {}

    for ticker, price_df in prices.items():
        raw = build_raw_features(
            price_df,
            lookback   = lookback,
            rsi_window = rsi_window,
            ma_short   = ma_short,
            ma_long    = ma_long,
            vol_window = vol_window,
        )

        if norm_mode == "none":
            feat_panel[ticker] = raw
            feat_stats[ticker] = {"mode": "none"}
        else:
            # Fit stats on training portion only
            train_mask = (
                raw.index <= pd.Timestamp(train_end) if train_end else slice(None)
            )
            train_raw = raw.loc[train_mask] if train_end else raw

            stats = fit_normalise_stats(train_raw, mode=norm_mode)
            feat_stats[ticker] = stats
            feat_panel[ticker] = normalise_with_stats(raw, stats)

        log.debug(f"[features] {ticker}: {feat_panel[ticker].shape} feature matrix")

    # VIX
    vix_train = (
        vix.loc[vix.index <= pd.Timestamp(train_end)] if train_end else vix
    )
    vix_df_train = vix_train.to_frame("vix")
    vix_stats    = fit_normalise_stats(vix_df_train, mode=norm_mode)
    vix_norm_df  = normalise_with_stats(vix.to_frame("vix"), vix_stats)
    vix_norm     = vix_norm_df["vix"]

    return feat_panel, vix_norm, feat_stats, vix_stats


# ---------------------------------------------------------------------------
# Observation noise injection (training-time augmentation)
# ---------------------------------------------------------------------------

def add_obs_noise(
    obs:       np.ndarray,
    std:       float = 0.01,
    rng:       Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add i.i.d. Gaussian noise to an observation vector.

    Parameters
    ----------
    obs  : 1-D numpy array (the state vector)
    std  : noise standard deviation (set to 0 to disable)
    rng  : optional numpy random Generator for reproducibility

    This mimics real-world measurement uncertainty and reduces overfitting
    to exact feature magnitudes.
    """
    if std <= 0.0:
        return obs
    noise_gen = rng if rng is not None else np.random.default_rng()
    return obs + noise_gen.normal(0.0, std, size=obs.shape).astype(obs.dtype)


# ---------------------------------------------------------------------------
# Standalone test / quick check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from data.fetcher import load_price_panel

    tickers   = cfg["tickers"]
    folds     = cfg["walk_forward"]
    start     = min(f["train_start"] for f in folds)
    end       = max(f["test_end"]    for f in folds)
    cache_dir = cfg["data"]["cache_dir"]
    vix_tick  = cfg.get("vix_ticker", "^VIX")

    prices, vix = load_price_panel(tickers, start, end, vix_tick, cache_dir)

    feat_panel, vix_norm, _, _ = build_feature_panel(
        prices,
        vix,
        norm_mode = cfg["features"]["normalize"],
        lookback  = cfg["features"]["lookback_returns"],
        train_end = folds[0]["train_end"],
    )

    for t, df in feat_panel.items():
        n_nan = df.isna().sum().sum()
        print(f"  {t}: shape={df.shape}  NaNs={n_nan}")
    print(f"  VIX norm: shape={vix_norm.shape}  NaNs={vix_norm.isna().sum()}")
