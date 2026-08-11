"""
evaluation/visualizer.py
------------------------
Publication-quality visualisations for the research-grade trading system.

Plots produced
--------------
  1. equity_curves_with_bands()
     — Mean equity curve ± 1σ band across seeds, one line per ablation.
     — Benchmark (buy-and-hold SPY) overlaid.

  2. drawdown_curves()
     — Running drawdown curves for each ablation, with shaded area.

  3. return_distributions()
     — Violin + strip plot of final returns across seeds for each ablation.

  4. training_convergence()
     — Episode vs. mean return during training (with EWM smoothing).

  5. ablation_comparison_table()
     — Heatmap of metrics across ablation variants (seaborn style).

All functions save to *save_dir* and optionally return the Figure.
Matplotlib backend is set to 'Agg' so it works without a display.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works on HPC / servers)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Colour palette (colour-blind friendly)
PALETTE = [
    "#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
    "#FF9800", "#009688", "#607D8B",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, save_dir: str, filename: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info(f"[visualizer] Saved → {path}")
    plt.close(fig)


def _apply_style(ax: plt.Axes):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------------
# 1. Equity curves with confidence bands
# ---------------------------------------------------------------------------

def equity_curves_with_bands(
    equity_by_ablation: Dict[str, List[pd.Series]],   # {ablation_id: [series_per_seed]}
    benchmark:          Optional[pd.Series] = None,   # buy-and-hold baseline
    save_dir:           str = "results_v2/plots",
    title:              str = "Equity Curves (mean ± 1σ across seeds)",
) -> plt.Figure:
    """
    Plot mean equity curve ± 1σ band for each ablation.

    Parameters
    ----------
    equity_by_ablation : dict mapping ablation_id to a list of equity curve Series
                         (one per seed).  Series indexed by date or integer step.
    benchmark          : optional baseline equity curve to plot as dashed black line
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (abl_id, curves) in enumerate(equity_by_ablation.items()):
        if not curves:
            continue
        # Align all curves to the same length (min length)
        min_len = min(len(c) for c in curves)
        arr = np.stack([c.values[:min_len] for c in curves], axis=0)   # (seeds, T)
        # Normalise to initial value = 1
        arr = arr / arr[:, 0:1]

        x = np.arange(min_len)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0, ddof=0)

        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, mean,          color=color, label=abl_id, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    if benchmark is not None:
        bm = benchmark.values / benchmark.values[0]
        ax.plot(np.arange(len(bm)), bm, "k--", linewidth=1.2, label="Buy & Hold SPY", alpha=0.7)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Normalised Portfolio Value")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", fontsize=9)
    _apply_style(ax)
    fig.tight_layout()

    _savefig(fig, save_dir, "equity_curves.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Drawdown curves
# ---------------------------------------------------------------------------

def drawdown_curves(
    equity_by_ablation: Dict[str, List[pd.Series]],
    save_dir:           str = "results_v2/plots",
    title:              str = "Drawdown Curves",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (abl_id, curves) in enumerate(equity_by_ablation.items()):
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        color   = PALETTE[i % len(PALETTE)]

        # Compute drawdown for each seed
        dd_list = []
        for c in curves:
            vals = c.values[:min_len].astype(float)
            peak = np.maximum.accumulate(vals)
            dd   = (vals - peak) / peak
            dd_list.append(dd)

        arr  = np.stack(dd_list, axis=0)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0, ddof=0)
        x    = np.arange(min_len)

        ax.plot(x, mean, color=color, label=abl_id, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="lower left", fontsize=9)
    _apply_style(ax)
    fig.tight_layout()

    _savefig(fig, save_dir, "drawdown_curves.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Return distribution (violin + strip)
# ---------------------------------------------------------------------------

def return_distributions(
    returns_by_ablation: Dict[str, List[float]],   # {ablation_id: [total_return per seed]}
    save_dir:            str = "results_v2/plots",
    title:               str = "Return Distribution Across Seeds",
) -> plt.Figure:
    abl_ids = list(returns_by_ablation.keys())
    n       = len(abl_ids)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 6))

    data   = [returns_by_ablation[k] for k in abl_ids]
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    # Violin plot
    parts = ax.violinplot(data, positions=range(n), showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)

    # Individual seed dots
    for i, vals in enumerate(data):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=colors[i], s=25, zorder=3, alpha=0.8)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(abl_ids, rotation=20, ha="right")
    ax.set_ylabel("Total Return")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    _apply_style(ax)
    fig.tight_layout()

    _savefig(fig, save_dir, "return_distributions.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Training convergence
# ---------------------------------------------------------------------------

def training_convergence(
    history_by_ablation: Dict[str, List[List[dict]]],   # {abl_id: [[ep_dict per ep] per seed]}
    save_dir:            str = "results_v2/plots",
    title:               str = "Training Convergence (mean ± 1σ)",
    ewm_span:            int = 20,
) -> plt.Figure:
    """
    Parameters
    ----------
    history_by_ablation : dict mapping ablation_id to a list of per-seed histories,
                          where each history is a list of dicts with keys 'ep' and
                          'total_return'.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (abl_id, seed_histories) in enumerate(history_by_ablation.items()):
        if not seed_histories:
            continue

        min_len = min(len(h) for h in seed_histories)
        returns = np.array([
            [ep["total_return"] for ep in h[:min_len]]
            for h in seed_histories
        ])   # (seeds, episodes)

        # EWM smoothing per seed
        smoothed = np.array([
            pd.Series(row).ewm(span=ewm_span).mean().values
            for row in returns
        ])

        x    = np.arange(min_len)
        mean = smoothed.mean(axis=0)
        std  = smoothed.std(axis=0, ddof=0)
        color = PALETTE[i % len(PALETTE)]

        ax.plot(x, mean, color=color, label=abl_id, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Total Return (smoothed)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", fontsize=9)
    _apply_style(ax)
    fig.tight_layout()

    _savefig(fig, save_dir, "training_convergence.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Ablation comparison heatmap
# ---------------------------------------------------------------------------

def ablation_heatmap(
    summary:   Dict[str, Dict[str, Dict[str, float]]],   # {abl_id: {metric: {mean, std}}}
    metrics:   Optional[List[str]] = None,
    save_dir:  str = "results_v2/plots",
    title:     str = "Ablation Study — Mean Metrics",
) -> plt.Figure:
    """
    Heatmap of metric means across ablation variants.

    Rows   = ablation variants
    Cols   = metrics
    Colour = value (positive = green, negative = red for return-like metrics)
    """
    default_metrics = [
        "total_return", "annualized_return", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate",
    ]
    cols = metrics or default_metrics
    rows = list(summary.keys())

    # Build mean matrix
    mat = np.zeros((len(rows), len(cols)))
    for i, abl_id in enumerate(rows):
        for j, metric in enumerate(cols):
            mat[i, j] = summary[abl_id].get(metric, {}).get("mean", 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.5), max(4, len(rows) * 0.8)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)

    # Annotate cells
    pct_keys = {"total_return", "annualized_return", "max_drawdown", "win_rate"}
    for i in range(len(rows)):
        for j in range(len(cols)):
            v    = mat[i, j]
            text = f"{v:+.1%}" if cols[j] in pct_keys else f"{v:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="black" if abs(v) < 0.8 * mat.max() else "white")

    ax.set_title(title)
    fig.tight_layout()

    _savefig(fig, save_dir, "ablation_heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all plots from a results dict
# ---------------------------------------------------------------------------

def generate_all_plots(
    ablation_results: List[dict],   # list of run_single_ablation outputs
    fold_equity_map:  Dict[str, Dict[str, List[pd.Series]]],   # {fold_id: {abl_id: [series]}}
    save_dir:         str = "results_v2/plots",
    benchmark:        Optional[pd.Series] = None,
):
    """
    Generate the full visualisation suite from ablation results.

    This is a convenience wrapper that orchestrates all plot functions.
    """
    from evaluation.metrics import aggregate_seed_metrics

    # --- Aggregate metrics ---
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    returns_by_abl: Dict[str, List[float]] = {}

    for result in ablation_results:
        abl_id = result["ablation_id"]
        seed_metrics = result.get("per_seed_metrics", [])
        if not seed_metrics:
            continue
        scalar_metrics = [
            {k: v for k, v in sm.items() if isinstance(v, (int, float)) and k != "seed"}
            for sm in seed_metrics
        ]
        summary[abl_id] = aggregate_seed_metrics(scalar_metrics)
        returns_by_abl[abl_id] = [sm.get("total_return", 0.0) for sm in scalar_metrics]

    # --- Equity curves ---
    # Flatten across folds — take first fold for equity plots
    equity_by_abl: Dict[str, List[pd.Series]] = {}
    for fold_id, abl_map in fold_equity_map.items():
        for abl_id, curves in abl_map.items():
            equity_by_abl.setdefault(abl_id, []).extend(curves)

    if equity_by_abl:
        equity_curves_with_bands(equity_by_abl, benchmark=benchmark, save_dir=save_dir)
        drawdown_curves(equity_by_abl, save_dir=save_dir)

    if returns_by_abl:
        return_distributions(returns_by_abl, save_dir=save_dir)

    if summary:
        ablation_heatmap(summary, save_dir=save_dir)

    log.info(f"[visualizer] All plots saved to {save_dir}")
