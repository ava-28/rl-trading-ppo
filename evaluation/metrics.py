"""
evaluation/metrics.py
---------------------
All performance metrics for the research-grade trading system.

Metrics computed
----------------
  total_return       : cumulative return over the evaluation period
  annualized_return  : CAGR using actual number of trading days
  sharpe_ratio       : annualised Sharpe (excess return / vol, rf=0)
  sortino_ratio      : annualised Sortino (downside deviation denominator)
  max_drawdown       : peak-to-trough drawdown fraction (most negative)
  calmar_ratio       : annualized_return / |max_drawdown|
  win_rate           : fraction of trades that were profitable
  turnover           : total traded value / mean portfolio value
  num_trades         : total number of BUY + SELL executions

All metrics are computed from the equity curve and trade log returned by
the environment after a test episode.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Equity-curve helpers
# ---------------------------------------------------------------------------

def equity_to_returns(equity: pd.Series) -> pd.Series:
    """Daily log returns from an equity curve."""
    return np.log(equity / equity.shift(1)).dropna()


def running_drawdown(equity: pd.Series) -> pd.Series:
    """Return a series of drawdown fractions (≤ 0) at each point."""
    peak = equity.cummax()
    return (equity - peak) / peak.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def total_return(equity: pd.Series) -> float:
    """(final − initial) / initial."""
    if len(equity) < 2:
        return 0.0
    return float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0])


def annualized_return(equity: pd.Series) -> float:
    """
    CAGR: (final/initial)^(252/T) − 1
    where T is the number of trading days.
    """
    T = len(equity) - 1
    if T <= 0:
        return 0.0
    total_ret = (equity.iloc[-1] / equity.iloc[0])
    if total_ret <= 0:
        return -1.0
    return float(total_ret ** (TRADING_DAYS_PER_YEAR / T) - 1.0)


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio.
    rf is expressed as annual rate; converted to per-day: rf/252.
    Returns 0 if volatility is near zero.
    """
    rets = equity_to_returns(equity)
    if len(rets) < 2:
        return 0.0
    excess = rets - risk_free_rate / TRADING_DAYS_PER_YEAR
    std    = rets.std()
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sortino ratio.
    Uses only downside returns in the denominator.
    """
    rets = equity_to_returns(equity)
    if len(rets) < 2:
        return 0.0
    excess     = rets - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside   = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    down_std   = downside.std()
    if down_std < 1e-10:
        return 0.0
    return float(excess.mean() / down_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (a negative number in [-1, 0])."""
    if len(equity) < 2:
        return 0.0
    dd = running_drawdown(equity)
    return float(dd.min())


def calmar_ratio(equity: pd.Series) -> float:
    """
    Calmar = annualized_return / |max_drawdown|.
    Returns 0 if max drawdown is near zero.
    """
    ann_ret = annualized_return(equity)
    mdd     = abs(max_drawdown(equity))
    if mdd < 1e-10:
        return 0.0
    return float(ann_ret / mdd)


def win_rate(trade_log: pd.DataFrame, prices: Dict[str, pd.DataFrame]) -> float:
    """
    Fraction of completed round-trip trades that were profitable.

    A round trip is: BUY at price p₁ → SELL at price p₂.
    Win if p₂ > p₁ (ignoring commissions for this metric).

    If trade_log is empty, returns 0.0.
    """
    if trade_log is None or len(trade_log) == 0:
        return 0.0

    wins   = 0
    totals = 0

    for ticker in trade_log["ticker"].unique() if "ticker" in trade_log.columns else []:
        ticker_trades = trade_log[trade_log["ticker"] == ticker].copy()
        ticker_trades = ticker_trades.sort_values("date")

        buy_stack: List[float] = []
        for _, row in ticker_trades.iterrows():
            if row["action"] == "BUY":
                buy_stack.append(float(row["price"]))
            elif row["action"] == "SELL" and buy_stack:
                buy_price = buy_stack.pop(0)   # FIFO
                totals += 1
                if float(row["price"]) > buy_price:
                    wins += 1

    return float(wins / totals) if totals > 0 else 0.0


def turnover(
    trade_log:      pd.DataFrame,
    equity:         pd.Series,
) -> float:
    """
    Total traded value / mean portfolio value.
    Measures how actively the strategy trades (higher = more active).
    """
    if trade_log is None or len(trade_log) == 0:
        return 0.0
    total_traded = float((trade_log["qty"] * trade_log["price"]).sum())
    mean_pv      = float(equity.mean())
    if mean_pv < 1e-6:
        return 0.0
    return total_traded / mean_pv


# ---------------------------------------------------------------------------
# Compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    equity:    pd.Series,
    trade_log: Optional[pd.DataFrame],
    prices:    Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, float]:
    """
    Compute the full set of evaluation metrics.

    Parameters
    ----------
    equity    : daily portfolio value series
    trade_log : DataFrame with columns date, ticker, action, qty, price
    prices    : raw price panel (needed for win_rate round-trip matching)

    Returns
    -------
    dict with one float per metric
    """
    if len(equity) < 2:
        return {m: 0.0 for m in [
            "total_return", "annualized_return", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "win_rate", "turnover", "num_trades",
        ]}

    tl = trade_log if trade_log is not None else pd.DataFrame()

    return {
        "total_return":      total_return(equity),
        "annualized_return": annualized_return(equity),
        "sharpe_ratio":      sharpe_ratio(equity),
        "sortino_ratio":     sortino_ratio(equity),
        "max_drawdown":      max_drawdown(equity),
        "calmar_ratio":      calmar_ratio(equity),
        "win_rate":          win_rate(tl, prices or {}),
        "turnover":          turnover(tl, equity),
        "num_trades":        float(len(tl)),
    }


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seed_metrics(
    per_seed: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Given a list of per-seed metric dicts, compute mean and std for each metric.

    Returns dict[metric_name → {"mean": float, "std": float}]
    """
    keys = list(per_seed[0].keys())
    result: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [d[k] for d in per_seed if k in d]
        arr  = np.array(vals, dtype=float)
        result[k] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return result


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    pct_keys  = {"total_return", "annualized_return", "max_drawdown", "win_rate", "turnover"}
    ratio_keys = {"sharpe_ratio", "sortino_ratio", "calmar_ratio"}
    int_keys  = {"num_trades"}

    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k in pct_keys:
            print(f"  {k:<22}: {v:+.2%}")
        elif k in ratio_keys:
            print(f"  {k:<22}: {v:+.3f}")
        elif k in int_keys:
            print(f"  {k:<22}: {int(v)}")
        else:
            print(f"  {k:<22}: {v:.4f}")
    print(f"{'='*50}\n")


def print_multi_seed_metrics(
    aggregated: Dict[str, Dict[str, float]],
    title:      str = "Multi-Seed Summary",
):
    pct_keys   = {"total_return", "annualized_return", "max_drawdown", "win_rate", "turnover"}
    ratio_keys = {"sharpe_ratio", "sortino_ratio", "calmar_ratio"}

    print(f"\n{'='*60}")
    print(f"  {title}  (mean ± std)")
    print(f"{'='*60}")
    for k, v in aggregated.items():
        mean, std = v["mean"], v["std"]
        if k in pct_keys:
            print(f"  {k:<22}: {mean:+.2%} ± {std:.2%}")
        elif k in ratio_keys:
            print(f"  {k:<22}: {mean:+.3f} ± {std:.3f}")
        else:
            print(f"  {k:<22}: {mean:.4f} ± {std:.4f}")
    print(f"{'='*60}\n")
