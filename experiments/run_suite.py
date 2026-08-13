"""
experiments/run_suite.py
------------------------
Master experiment runner for the research-grade multi-asset trading system.

Orchestrates the full pipeline:
  1.  Load config.yaml
  2.  Fetch / cache price data and news (yfinance + Finnhub)
  3.  Build feature panel (technical indicators + VIX, per fold normalisation)
  4.  Build sentiment panel (FinBERT or VADER, causal alignment)
  5.  For each walk-forward fold × each ablation configuration × each seed:
        a.  Slice train / test data
        b.  Train PPO agent
        c.  Evaluate on out-of-sample test window
        d.  Compute all evaluation metrics
  6.  Aggregate results (mean ± std across seeds; then across folds)
  7.  Write CSV / JSON result files
  8.  Generate visualisation suite (equity curves, drawdowns, heatmaps)

Usage
-----
  # Run the full suite with default config:
  python -m experiments.run_suite

  # Override config file:
  python -m experiments.run_suite --config path/to/config.yaml

  # Run only specific folds / ablations (for quick debugging):
  python -m experiments.run_suite --folds fold_1 --ablations full no_sentiment

  # Dry-run: load data, build features, don't train:
  python -m experiments.run_suite --dry-run

Output directory (results_dir from config)
  results_v2/
    checkpoints/           # best agent checkpoints per fold × ablation × seed
    equity_curves/         # per-seed equity CSV files
    trade_logs/            # per-seed trade log CSV files
    metrics_raw.csv        # all (fold, ablation, seed) rows
    metrics_summary.json   # nested {fold_id → {ablation_id → {metric → {mean, std}}}}
    ablation_table.csv     # pivot table: ablation_id × metric
    plots/                 # all generated PNG figures
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# ---- Ensure project root is on sys.path when running as a script ----
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.fetcher   import load_price_panel, load_news_panel
from data.features  import build_feature_panel, fit_normalise_stats, normalise_with_stats
from data.sentiment import build_sentiment_panel, inject_sentiment
from training.walk_forward import load_folds, get_fold_data, slice_prices
from training.trainer      import run_single_seed, set_seed
from evaluation.metrics    import compute_all_metrics, aggregate_seed_metrics, print_multi_seed_metrics
from evaluation.ablation   import load_ablation_configs
from evaluation.visualizer import generate_all_plots

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def run_data_pipeline(cfg: dict) -> tuple:
    """
    Fetch and prepare raw price data, VIX, and news.

    Returns
    -------
    prices    : dict[ticker → OHLCV DataFrame]  (full date range)
    vix       : pd.Series  (full date range)
    news      : dict[ticker → news DataFrame]   (full date range)
    """
    tickers    = cfg["tickers"]
    vix_ticker = cfg.get("vix_ticker", "^VIX")
    cache_dir  = cfg["data"]["cache_dir"]
    api_key    = cfg["data"].get("finnhub_key", "") or os.environ.get("FINNHUB_KEY", "")
    force      = cfg["data"].get("force_refetch", False)

    # Determine the full date window needed across all folds
    folds      = cfg["walk_forward"]
    full_start = min(f["train_start"] for f in folds)
    full_end   = max(f["test_end"]    for f in folds)

    log.info(f"[data] Fetching {len(tickers)} tickers + VIX: {full_start} → {full_end}")
    prices, vix = load_price_panel(
        tickers, full_start, full_end, vix_ticker, cache_dir, force=force
    )

    # The price-only study has no use for news; skip the API round-trip entirely
    # rather than fetching ~500 empty monthly chunks at 1.1s apiece.
    if cfg["data"].get("skip_news", False):
        log.info("[data] skip_news=true — no news fetched (price-only study).")
        news = {t: pd.DataFrame(
            columns=["datetime", "headline", "summary", "source", "ticker"]
        ) for t in tickers}
        return prices, vix, news

    log.info("[data] Fetching news…")
    news = load_news_panel(tickers, full_start, full_end, api_key, cache_dir, force=force)

    return prices, vix, news


# ---------------------------------------------------------------------------
# Equity curve builder from recorded episode data
# ---------------------------------------------------------------------------

def build_equity_series(equity_log: List[dict]) -> pd.Series:
    """Convert the list of {date, portfolio_value} dicts to a Series."""
    if not equity_log:
        return pd.Series(dtype=float)
    df = pd.DataFrame(equity_log)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df["portfolio_value"]


# ---------------------------------------------------------------------------
# Incremental checkpointing
#
# The suite previously wrote results only after all runs finished, so a crash
# at run 27 of 30 lost everything. Each (fold, ablation, seed) is now appended
# to metrics_incremental.csv the moment it completes, and any combination
# already present is skipped on restart. A crash costs one run, not the suite.
# ---------------------------------------------------------------------------

INCREMENTAL_CSV = "metrics_incremental.csv"


def _incremental_path(results_dir: str) -> str:
    return os.path.join(results_dir, INCREMENTAL_CSV)


def load_completed_runs(results_dir: str) -> Dict[tuple, dict]:
    """Map (fold_id, ablation, seed) -> metrics dict for runs already finished."""
    path = _incremental_path(results_dir)
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log.warning(f"[resume] could not read {path}: {exc}")
        return {}
    done = {}
    for _, row in df.iterrows():
        key = (str(row["fold_id"]), str(row["ablation"]), int(row["seed"]))
        done[key] = row.to_dict()
    if done:
        log.info(f"[resume] found {len(done)} completed run(s) in {path}")
    return done


def append_completed_run(results_dir: str, metrics: dict) -> None:
    """Append one finished run's metrics. Written immediately, flushed to disk."""
    path = _incremental_path(results_dir)
    pd.DataFrame([metrics]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )


def _load_saved_equity(results_dir: str, fold_id: str, abl_id: str, seed: int) -> pd.Series:
    """Recover a skipped run's equity curve from disk so plots still work."""
    p = os.path.join(results_dir, "equity_curves", f"{fold_id}_{abl_id}_seed{seed}_equity.csv")
    if not os.path.exists(p):
        return pd.Series(dtype=float)
    try:
        s = pd.read_csv(p, index_col=0)
        s.index = pd.to_datetime(s.index)
        return s.iloc[:, 0]
    except Exception:
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Single fold × ablation × multi-seed run
# ---------------------------------------------------------------------------

def run_fold_ablation(
    fold_id:        str,
    train_feat:     Dict[str, pd.DataFrame],
    train_vix:      pd.Series,
    train_prices:   Dict[str, pd.DataFrame],
    test_feat:      Dict[str, pd.DataFrame],
    test_vix:       pd.Series,
    test_prices:    Dict[str, pd.DataFrame],
    cfg:            dict,
    ablation_cfg:   dict,
    results_dir:    str,
) -> dict:
    """
    Run multi-seed training + evaluation for one (fold, ablation) pair.

    Returns a consolidated result dict.
    """
    abl_id     = ablation_cfg["id"]
    seeds      = cfg["training"]["seeds"]
    ckpt_dir   = os.path.join(results_dir, "checkpoints", fold_id, abl_id)
    equity_dir = os.path.join(results_dir, "equity_curves")
    trade_dir  = os.path.join(results_dir, "trade_logs")

    for d in [ckpt_dir, equity_dir, trade_dir]:
        os.makedirs(d, exist_ok=True)

    per_seed_metrics:   List[Dict[str, float]] = []
    per_seed_equity:    List[pd.Series]        = []
    per_seed_histories: List[List[dict]]       = []

    completed = load_completed_runs(results_dir)

    for seed in seeds:
        # ---- Resume: skip anything already finished in a previous invocation ----
        key = (str(fold_id), str(abl_id), int(seed))
        if key in completed:
            log.info(f"\n  [{fold_id} | {abl_id} | seed={seed}]  already complete — skipping")
            per_seed_metrics.append(completed[key])
            per_seed_equity.append(_load_saved_equity(results_dir, fold_id, abl_id, seed))
            per_seed_histories.append([])
            continue

        log.info(f"\n  [{fold_id} | {abl_id} | seed={seed}]")
        try:
            agent, test_result, history = run_single_seed(
                seed           = seed,
                fold_id        = fold_id,
                train_feat     = train_feat,
                train_vix      = train_vix,
                train_prices   = train_prices,
                test_feat      = test_feat,
                test_vix       = test_vix,
                test_prices    = test_prices,
                config         = cfg,
                ablation_cfg   = ablation_cfg,
                checkpoint_dir = ckpt_dir,
            )
        except Exception as exc:
            log.error(f"  FAILED: {exc}", exc_info=True)
            continue

        # Build equity series
        equity = build_equity_series(test_result.get("equity_curve", []))
        trade_log = test_result.get("trade_log", pd.DataFrame())

        # Compute full metric set from equity curve + trade log
        if len(equity) >= 2:
            metrics = compute_all_metrics(equity, trade_log, test_prices)
        else:
            metrics = {
                "total_return": test_result.get("total_return", 0.0),
                "final_value":  test_result.get("final_value",  0.0),
                "num_trades":   float(len(trade_log)),
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
                "max_drawdown": 0.0, "calmar_ratio":  0.0,
                "win_rate": 0.0,     "turnover":      0.0,
            }

        metrics["seed"]     = float(seed)
        metrics["fold_id"]  = fold_id
        metrics["ablation"] = abl_id

        per_seed_metrics.append(metrics)
        per_seed_equity.append(equity)
        per_seed_histories.append(history)

        # Persist this run immediately, before starting the next one.
        append_completed_run(results_dir, metrics)

        # Persist equity curve and trade log
        if not equity.empty:
            eq_path = os.path.join(equity_dir, f"{fold_id}_{abl_id}_seed{seed}_equity.csv")
            equity.to_csv(eq_path, header=True)

        if not trade_log.empty:
            tl_path = os.path.join(trade_dir, f"{fold_id}_{abl_id}_seed{seed}_trades.csv")
            trade_log.to_csv(tl_path, index=False)

        log.info(
            f"    total_return={metrics.get('total_return',0):+.2%}  "
            f"sharpe={metrics.get('sharpe_ratio',0):+.3f}  "
            f"max_dd={metrics.get('max_drawdown',0):+.2%}  "
            f"trades={int(metrics.get('num_trades',0))}"
        )

    # Aggregate across seeds
    scalar_metrics = [
        {k: v for k, v in m.items() if isinstance(v, float) and k not in ("seed",)}
        for m in per_seed_metrics
    ]
    aggregated = aggregate_seed_metrics(scalar_metrics) if scalar_metrics else {}

    return {
        "fold_id":           fold_id,
        "ablation_id":       abl_id,
        "per_seed_metrics":  per_seed_metrics,
        "aggregated":        aggregated,
        "per_seed_equity":   per_seed_equity,
        "per_seed_histories":per_seed_histories,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    t0 = time.time()

    # ---- Config ----
    cfg = load_config(args.config)
    results_dir = cfg.get("results_dir", "results_v2")
    os.makedirs(results_dir, exist_ok=True)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level   = log_level,
        format  = "%(asctime)s %(levelname)-7s %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(results_dir, "run.log"), mode="w"),
        ],
    )

    log.info("=" * 70)
    log.info("  Research-Grade PPO Trading — Experiment Suite")
    log.info("=" * 70)
    log.info(f"  Config     : {args.config}")
    log.info(f"  Results    : {results_dir}")
    log.info(f"  Tickers    : {cfg['tickers']}")
    log.info(f"  Seeds      : {cfg['training']['seeds']}")
    log.info("=" * 70)

    # ---- 1. Data pipeline ----
    prices, vix, news = run_data_pipeline(cfg)

    if args.dry_run:
        log.info("[dry-run] Data loaded. Exiting before training.")
        return

    # ---- 2. Folds ----
    all_folds = load_folds(cfg)
    if args.folds:
        all_folds = [f for f in all_folds if f.fold_id in args.folds]
        log.info(f"[suite] Running folds: {[f.fold_id for f in all_folds]}")

    # ---- 3. Ablation configs ----
    ablation_cfgs = load_ablation_configs(cfg)
    if args.ablations:
        ablation_cfgs = [a for a in ablation_cfgs if a["id"] in args.ablations]
        log.info(f"[suite] Running ablations: {[a['id'] for a in ablation_cfgs]}")

    feat_cfg = cfg.get("features", {})
    norm_mode = feat_cfg.get("normalize", "zscore")

    # ---- 4. Main loop: fold × ablation ----
    all_results:  List[dict]          = []
    all_raw_rows: List[dict]          = []
    # {fold_id: {abl_id: [equity_series_per_seed]}}
    fold_equity_map: Dict[str, Dict[str, List[pd.Series]]] = {}
    # {abl_id: {hist: [hist_per_seed_across_folds]}}
    history_map: Dict[str, List[List[dict]]] = {}

    for fold in all_folds:
        log.info(f"\n{'━'*70}")
        log.info(f"  FOLD: {fold.fold_id}  "
                 f"train={fold.train_start}→{fold.train_end}  "
                 f"test={fold.test_start}→{fold.test_end}")
        log.info(f"{'━'*70}")

        fold_equity_map[fold.fold_id] = {}

        # Build feature panel normalised on the TRAINING window of this fold
        feat_panel, vix_norm, feat_stats, vix_stats = build_feature_panel(
            prices    = prices,
            vix       = vix,
            norm_mode = norm_mode,
            lookback  = feat_cfg.get("lookback_returns",  5),
            rsi_window= feat_cfg.get("rsi_window",       14),
            ma_short  = feat_cfg.get("ma_short",         20),
            ma_long   = feat_cfg.get("ma_long",          50),
            vol_window= feat_cfg.get("vol_window",       10),
            train_end = fold.train_end,
        )

        # Slice raw prices for this fold (needed inside the env for mark-to-market)
        train_prices_fold = slice_prices(prices, fold.train_start, fold.train_end)
        test_prices_fold  = slice_prices(prices, fold.test_start,  fold.test_end)

        # Build sentiment panel if any ablation requires it
        need_sentiment = any(a.get("use_sentiment", True) for a in ablation_cfgs)
        sent_backend   = cfg.get("sentiment", {}).get("backend", "vader")

        if need_sentiment and sent_backend != "none":
            trading_calendars = {
                t: feat_panel[t].index for t in cfg["tickers"]
            }
            sent_panel = build_sentiment_panel(
                news_panel          = news,
                trading_calendars   = trading_calendars,
                backend             = sent_backend,
                model_name          = cfg["sentiment"].get("model", "ProsusAI/finbert"),
                aggregation         = cfg["sentiment"].get("aggregation", "ewm"),
                ewm_halflife        = cfg["sentiment"].get("ewm_halflife", 3),
                max_articles        = cfg["sentiment"].get("max_articles_per_day", 5),
                no_future_leakage   = cfg["sentiment"].get("no_future_leakage", True),
                device              = cfg["training"].get("device", "cpu"),
            )
            feat_panel_with_sent = inject_sentiment(feat_panel, sent_panel)
        else:
            feat_panel_with_sent = feat_panel

        # Slice feature panel into train / test windows
        (train_feat, train_vix, train_prices_aligned,
         test_feat,  test_vix,  test_prices_aligned) = get_fold_data(
            fold       = fold,
            feat_panel = feat_panel_with_sent,
            vix_norm   = vix_norm,
            prices     = prices,
        )

        for abl_cfg in ablation_cfgs:
            abl_id = abl_cfg["id"]
            log.info(f"\n  ▶  Ablation: {abl_id}")

            result = run_fold_ablation(
                fold_id        = fold.fold_id,
                train_feat     = train_feat,
                train_vix      = train_vix,
                train_prices   = train_prices_aligned,
                test_feat      = test_feat,
                test_vix       = test_vix,
                test_prices    = test_prices_aligned,
                cfg            = cfg,
                ablation_cfg   = abl_cfg,
                results_dir    = results_dir,
            )

            all_results.append(result)

            # Accumulate equity for visualisation
            fold_equity_map[fold.fold_id].setdefault(abl_id, []).extend(
                result["per_seed_equity"]
            )
            # Accumulate histories
            history_map.setdefault(abl_id, []).extend(
                result["per_seed_histories"]
            )

            # Flatten per-seed rows for CSV
            for row in result["per_seed_metrics"]:
                all_raw_rows.append(row)

            # Print fold × ablation summary
            agg = result["aggregated"]
            if agg:
                log.info(
                    f"  [{fold.fold_id} | {abl_id}]  "
                    f"return={agg.get('total_return',{}).get('mean',0):+.2%}"
                    f"±{agg.get('total_return',{}).get('std',0):.2%}  "
                    f"sharpe={agg.get('sharpe_ratio',{}).get('mean',0):+.3f}"
                    f"±{agg.get('sharpe_ratio',{}).get('std',0):.3f}"
                )

    # ---- 5. Save raw metrics CSV ----
    if all_raw_rows:
        raw_df = pd.DataFrame(all_raw_rows)
        raw_path = os.path.join(results_dir, "metrics_raw.csv")
        raw_df.to_csv(raw_path, index=False)
        log.info(f"\n[suite] Saved raw metrics → {raw_path}")

    # ---- 6. Aggregate across folds ----
    summary_by_abl: Dict[str, Any] = {}
    for result in all_results:
        abl_id = result["ablation_id"]
        summary_by_abl.setdefault(abl_id, []).extend(result["per_seed_metrics"])

    final_summary: Dict[str, Any] = {}
    for abl_id, seed_rows in summary_by_abl.items():
        scalar_rows = [
            {k: v for k, v in r.items() if isinstance(v, float) and k not in ("seed",)}
            for r in seed_rows
        ]
        final_summary[abl_id] = aggregate_seed_metrics(scalar_rows)

    summary_path = os.path.join(results_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2, default=str)
    log.info(f"[suite] Saved summary → {summary_path}")

    # ---- 7. Ablation comparison table ----
    metric_cols = [
        "total_return", "annualized_return", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "calmar_ratio",
        "win_rate", "turnover", "num_trades",
    ]
    rows = []
    for abl_id, agg in final_summary.items():
        row = {"ablation": abl_id}
        for col in metric_cols:
            if col in agg:
                row[f"{col}_mean"] = round(agg[col]["mean"], 5)
                row[f"{col}_std"]  = round(agg[col]["std"],  5)
        rows.append(row)

    table_df = pd.DataFrame(rows).set_index("ablation")
    table_path = os.path.join(results_dir, "ablation_table.csv")
    table_df.to_csv(table_path)
    log.info(f"[suite] Saved ablation table → {table_path}")

    # ---- 8. Console summary ----
    log.info("\n" + "=" * 70)
    log.info("  RESULTS SUMMARY  (mean ± std across seeds + folds)")
    log.info("=" * 70)
    for abl_id, agg in final_summary.items():
        ret  = agg.get("total_return",  {})
        sr   = agg.get("sharpe_ratio",  {})
        mdd  = agg.get("max_drawdown",  {})
        log.info(
            f"  {abl_id:<18}  "
            f"return={ret.get('mean',0):+.2%}±{ret.get('std',0):.2%}  "
            f"sharpe={sr.get('mean',0):+.3f}±{sr.get('std',0):.3f}  "
            f"maxDD={mdd.get('mean',0):+.2%}±{mdd.get('std',0):.2%}"
        )
    log.info("=" * 70)

    # ---- 9. Visualisations ----
    plot_dir = os.path.join(results_dir, "plots")
    try:
        # Build benchmark from SPY prices if available
        spy_prices = prices.get("SPY")
        benchmark = None
        if spy_prices is not None:
            benchmark = spy_prices["close"].rename("SPY")

        generate_all_plots(
            ablation_results = all_results,
            fold_equity_map  = fold_equity_map,
            save_dir         = plot_dir,
            benchmark        = benchmark,
        )
        log.info(f"[suite] Plots saved → {plot_dir}/")
    except Exception as exc:
        log.warning(f"[suite] Visualisation failed (non-fatal): {exc}")

    elapsed = time.time() - t0
    log.info(f"\n[suite] Done in {elapsed/60:.1f} minutes.  Results at: {results_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-grade PPO trading — full experiment suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to experiment config YAML file",
    )
    p.add_argument(
        "--folds", nargs="+", default=None,
        metavar="FOLD_ID",
        help="Subset of fold IDs to run (e.g. fold_1 fold_2)",
    )
    p.add_argument(
        "--ablations", nargs="+", default=None,
        metavar="ABL_ID",
        help="Subset of ablation IDs to run (e.g. full no_sentiment)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Load and process data but skip training",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
