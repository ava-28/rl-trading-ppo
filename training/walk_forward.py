"""
training/walk_forward.py
------------------------
Walk-forward split manager for the research-grade trading system.

Walk-forward validation
-----------------------
Instead of a single train/test split, we use K rolling folds where each
fold trains on a window [train_start, train_end] and evaluates on the
immediately following out-of-sample window [test_start, test_end].

This mirrors how a real systematic strategy would be deployed and validated:
  - No test data is ever seen during training.
  - Each fold is an independent experiment.
  - Results are averaged across folds to get a robust performance estimate.

Fold structure (from config.yaml)
----------------------------------
  fold_1: train 2020–2022 → test 2023
  fold_2: train 2021–2023 → test 2024
  fold_3: train 2022–2024 → test 2025

The test windows are non-overlapping, covering three distinct market regimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Fold:
    """Represents one walk-forward experiment fold."""
    fold_id:     str
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str


def load_folds(config: dict) -> List[Fold]:
    """
    Parse the walk_forward section of config.yaml into a list of Fold objects.
    """
    folds = []
    for entry in config.get("walk_forward", []):
        folds.append(Fold(
            fold_id     = entry["id"],
            train_start = entry["train_start"],
            train_end   = entry["train_end"],
            test_start  = entry["test_start"],
            test_end    = entry["test_end"],
        ))
    if not folds:
        raise ValueError("No walk_forward folds found in config.yaml.")
    log.info(f"[walk_forward] Loaded {len(folds)} folds: {[f.fold_id for f in folds]}")
    return folds


def slice_panel(
    feat_panel: Dict[str, pd.DataFrame],
    start:      str,
    end:        str,
) -> Dict[str, pd.DataFrame]:
    """
    Return a new panel containing only rows within [start, end] (inclusive).
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return {
        ticker: df.loc[(df.index >= s) & (df.index <= e)].copy()
        for ticker, df in feat_panel.items()
    }


def slice_series(series: pd.Series, start: str, end: str) -> pd.Series:
    """Slice a time series to [start, end] (inclusive)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return series.loc[(series.index >= s) & (series.index <= e)].copy()


def slice_prices(
    prices: Dict[str, pd.DataFrame],
    start:  str,
    end:    str,
) -> Dict[str, pd.DataFrame]:
    """Slice the raw price panel."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return {
        t: df.loc[(df.index >= s) & (df.index <= e)].copy()
        for t, df in prices.items()
    }


def align_panel_lengths(
    feat_panel:    Dict[str, pd.DataFrame],
    vix_norm:      pd.Series,
    prices:        Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], pd.Series, Dict[str, pd.DataFrame]]:
    """
    Align all DataFrames in *feat_panel* to the common intersection of dates.
    Also aligns vix_norm and prices to the same date index.

    This handles minor ticker-specific gaps (e.g., different listing dates,
    halted trading days) by taking the intersection of all trading calendars.
    """
    # Find common trading days across all assets
    common = None
    for df in feat_panel.values():
        idx = df.index
        common = idx if common is None else common.intersection(idx)

    if common is None or len(common) == 0:
        raise ValueError("No common trading days found across the asset panel.")

    # Slice everything to the common index
    feat_aligned  = {t: df.loc[common].copy() for t, df in feat_panel.items()}
    vix_aligned   = vix_norm.reindex(common).ffill().fillna(0.0)
    price_aligned = {
        t: df.loc[df.index.intersection(common)].copy()
        for t, df in prices.items()
    }

    log.debug(f"[walk_forward] Aligned panel: {len(common)} common trading days")
    return feat_aligned, vix_aligned, price_aligned


def get_fold_data(
    fold:       Fold,
    feat_panel: Dict[str, pd.DataFrame],   # full normalised feature panel
    vix_norm:   pd.Series,                 # full normalised VIX
    prices:     Dict[str, pd.DataFrame],   # full raw price panel
) -> Tuple[
    Dict[str, pd.DataFrame],  # train_feat
    pd.Series,                # train_vix
    Dict[str, pd.DataFrame],  # train_prices
    Dict[str, pd.DataFrame],  # test_feat
    pd.Series,                # test_vix
    Dict[str, pd.DataFrame],  # test_prices
]:
    """
    Slice the full data panels into train and test windows for a given fold.

    NOTE: The feature panel passed in must already be normalised using stats
    fit on the *training window only*.  Re-normalisation per fold must happen
    upstream (in run_suite.py) before calling this function, because the
    normalisation statistics should never include any test data.
    """
    train_feat   = slice_panel(feat_panel, fold.train_start, fold.train_end)
    train_vix    = slice_series(vix_norm,  fold.train_start, fold.train_end)
    train_prices = slice_prices(prices,    fold.train_start, fold.train_end)

    test_feat    = slice_panel(feat_panel, fold.test_start, fold.test_end)
    test_vix     = slice_series(vix_norm,  fold.test_start, fold.test_end)
    test_prices  = slice_prices(prices,    fold.test_start, fold.test_end)

    # Align each split independently
    train_feat, train_vix, train_prices = align_panel_lengths(
        train_feat, train_vix, train_prices
    )
    test_feat, test_vix, test_prices = align_panel_lengths(
        test_feat, test_vix, test_prices
    )

    log.info(
        f"[{fold.fold_id}] train={len(train_vix)}d "
        f"({fold.train_start}→{fold.train_end})  "
        f"test={len(test_vix)}d ({fold.test_start}→{fold.test_end})"
    )
    return train_feat, train_vix, train_prices, test_feat, test_vix, test_prices
