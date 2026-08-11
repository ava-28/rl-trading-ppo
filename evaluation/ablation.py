"""
evaluation/ablation.py
----------------------
Ablation study runner for the research-grade trading system.

The six ablation configurations (defined in config.yaml):
  full          — all components enabled
  no_sentiment  — sentiment signal disabled (sentiment_score forced to 0)
  no_short      — long-only trading (SELL only closes longs, no short selling)
  naive_reward  — raw return reward only (drawdown and turnover penalties off)
  no_noise      — observation noise disabled during training
  random_prices — price series shuffled (sanity check: destroys temporal structure)

Usage
-----
  from evaluation.ablation import run_ablation_suite
  results = run_ablation_suite(fold_id, train_data, test_data, config)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from training.trainer import run_multi_seed
from evaluation.metrics import compute_all_metrics, aggregate_seed_metrics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation config parsing
# ---------------------------------------------------------------------------

def load_ablation_configs(config: dict) -> List[dict]:
    """
    Return the list of ablation configuration dicts from config.yaml.
    Each dict has keys: id, use_sentiment, use_short_selling, use_risk_reward,
    action_noise, shuffle_prices (optional).
    """
    ablations = config.get("ablations", [])
    if not ablations:
        log.warning("[ablation] No ablation configs found; using 'full' only.")
        ablations = [{"id": "full", "use_sentiment": True, "use_short_selling": True,
                      "use_risk_reward": True, "action_noise": True}]
    return ablations


# ---------------------------------------------------------------------------
# Single ablation × single fold
# ---------------------------------------------------------------------------

def run_single_ablation(
    ablation_cfg:   dict,
    fold_id:        str,
    train_feat:     Dict[str, pd.DataFrame],
    train_vix:      pd.Series,
    train_prices:   Dict[str, pd.DataFrame],
    test_feat:      Dict[str, pd.DataFrame],
    test_vix:       pd.Series,
    test_prices:    Dict[str, pd.DataFrame],
    config:         dict,
    checkpoint_dir: str,
) -> dict:
    """
    Run multi-seed training + evaluation for one ablation × one fold.

    Returns a dict with:
      ablation_id      : str
      fold_id          : str
      per_seed_metrics : list[dict]
      mean_metrics     : dict[str → float]
      std_metrics      : dict[str → float]
      aggregated       : dict[str → {mean, std}]  — from aggregate_seed_metrics
    """
    abl_id = ablation_cfg.get("id", "unknown")
    abl_ckpt_dir = os.path.join(checkpoint_dir, f"{fold_id}_{abl_id}")

    log.info(f"\n{'─'*60}")
    log.info(f"  Ablation: {abl_id}  |  Fold: {fold_id}")
    log.info(f"{'─'*60}")

    result = run_multi_seed(
        fold_id        = fold_id,
        train_feat     = train_feat,
        train_vix      = train_vix,
        train_prices   = train_prices,
        test_feat      = test_feat,
        test_vix       = test_vix,
        test_prices    = test_prices,
        config         = config,
        ablation_cfg   = ablation_cfg,
        checkpoint_dir = abl_ckpt_dir,
    )

    result["ablation_id"] = abl_id
    return result


# ---------------------------------------------------------------------------
# Full ablation suite across all folds
# ---------------------------------------------------------------------------

def run_ablation_suite(
    folds:         List[dict],   # list of {fold_id, train_feat, …, test_prices}
    config:        dict,
    checkpoint_dir: str,
    ablation_ids:   Optional[List[str]] = None,   # filter to subset; None = all
) -> List[dict]:
    """
    Run the complete ablation × fold matrix.

    Parameters
    ----------
    folds         : list of dicts, each with:
                    fold_id, train_feat, train_vix, train_prices,
                    test_feat, test_vix, test_prices
    ablation_ids  : if not None, only run ablations whose id is in this list

    Returns
    -------
    list of result dicts, one per (ablation_id, fold_id) combination
    """
    ablation_cfgs = load_ablation_configs(config)
    if ablation_ids is not None:
        ablation_cfgs = [a for a in ablation_cfgs if a["id"] in ablation_ids]

    all_results: List[dict] = []

    for abl_cfg in ablation_cfgs:
        for fold in folds:
            result = run_single_ablation(
                ablation_cfg   = abl_cfg,
                fold_id        = fold["fold_id"],
                train_feat     = fold["train_feat"],
                train_vix      = fold["train_vix"],
                train_prices   = fold["train_prices"],
                test_feat      = fold["test_feat"],
                test_vix       = fold["test_vix"],
                test_prices    = fold["test_prices"],
                config         = config,
                checkpoint_dir = checkpoint_dir,
            )
            all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Aggregate across folds for each ablation
# ---------------------------------------------------------------------------

def summarise_ablation_results(
    results: List[dict],
) -> Dict[str, Dict[str, Any]]:
    """
    For each ablation ID, aggregate metrics across all folds (mean over folds,
    then report mean ± std across seeds within each fold, then mean ± std across
    folds).

    Returns dict[ablation_id → {metric → {mean, std}}]
    """
    # Group by ablation_id
    by_ablation: Dict[str, List[dict]] = {}
    for r in results:
        abl_id = r["ablation_id"]
        by_ablation.setdefault(abl_id, []).append(r)

    summary: Dict[str, Dict[str, Any]] = {}
    for abl_id, abl_results in by_ablation.items():
        # Each abl_result has per_seed_metrics (list of seed metric dicts)
        # Flatten all seeds across all folds
        all_seed_metrics: List[Dict[str, float]] = []
        for fold_result in abl_results:
            for seed_r in fold_result.get("per_seed_metrics", []):
                # Extract scalar metrics only (no trade_log / equity_curve)
                scalar = {
                    k: v for k, v in seed_r.items()
                    if isinstance(v, (int, float)) and k not in ("seed",)
                }
                all_seed_metrics.append(scalar)

        if not all_seed_metrics:
            continue

        summary[abl_id] = aggregate_seed_metrics(all_seed_metrics)

    return summary


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def results_to_dataframe(
    summary: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert the ablation summary into a tidy DataFrame for display / export.

    Rows   = ablation variants
    Cols   = metrics (mean ± std formatted as strings)
    """
    default_metrics = [
        "total_return", "annualized_return", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "calmar_ratio",
        "win_rate", "turnover", "num_trades",
    ]
    cols = metrics or default_metrics

    pct_keys   = {"total_return", "annualized_return", "max_drawdown",
                  "win_rate", "turnover"}
    ratio_keys = {"sharpe_ratio", "sortino_ratio", "calmar_ratio"}

    rows = []
    for abl_id, metric_dict in summary.items():
        row: Dict[str, str] = {"ablation": abl_id}
        for col in cols:
            if col not in metric_dict:
                row[col] = "—"
                continue
            mean = metric_dict[col]["mean"]
            std  = metric_dict[col]["std"]
            if col in pct_keys:
                row[col] = f"{mean:+.2%} ± {std:.2%}"
            elif col in ratio_keys:
                row[col] = f"{mean:+.3f} ± {std:.3f}"
            else:
                row[col] = f"{mean:.1f} ± {std:.1f}"
        rows.append(row)

    return pd.DataFrame(rows).set_index("ablation")
