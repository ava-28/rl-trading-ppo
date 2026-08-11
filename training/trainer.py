"""
training/trainer.py
-------------------
Single-seed training loop and multi-seed wrapper for the PPO agent.

Architecture
------------
  run_single_seed()
    → Creates env, agent, runs N training episodes, evaluates on test set.
    → Returns (agent, test_metrics, equity_curve, trade_log).

  run_multi_seed()
    → Calls run_single_seed() for each seed in config.training.seeds.
    → Collects metrics per seed, returns mean ± std summary.

Episode structure
-----------------
  Each training episode:
    1. Instantiate a fresh MultiAssetEnv (with randomised start if enabled).
    2. Roll out one full episode under the current policy.
    3. Buffer stores (obs, action, reward, log_prob, value, done).
    4. After the episode, call agent.update(last_value) — runs K PPO epochs.
    5. Every eval_freq episodes, run a deterministic greedy evaluation episode
       on the same training data (in-sample) to track convergence.

  Best checkpoint: the agent with the highest greedy in-sample return is
  saved; this checkpoint is loaded for out-of-sample evaluation.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from agent.ppo import PPOAgent
from env.multi_asset_env import MultiAssetEnv, compute_dims

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env:           MultiAssetEnv,
    agent:         PPOAgent,
    train:         bool = True,
    record_equity: bool = False,
) -> dict:
    """
    Run one complete episode.

    Parameters
    ----------
    train         : if True, push transitions to buffer and update at end
    record_equity : if True, log daily (date, portfolio_value) pairs

    Returns dict with keys:
      total_reward, total_return, num_trades, mean_loss,
      final_value, equity_curve (optional)
    """
    obs  = env.reset()
    done = False
    ep_reward  = 0.0
    loss_val   = 0.0
    equity_log: List[dict] = []

    while not done:
        action, log_prob, value = agent.select_action(obs, deterministic=not train)
        next_obs, reward, done, info = env.step(action)

        if train:
            agent.buffer.push(obs, action, reward, log_prob, value, done)

        obs        = next_obs
        ep_reward += reward

        if record_equity:
            equity_log.append({
                "date":            info["date"],
                "portfolio_value": info["portfolio"],
                "cash":            info["cash"],
            })

    if train:
        # Bootstrap the value of the last state (0.0 if done naturally)
        _, _, last_val = agent.select_action(obs, deterministic=False)
        last_bs  = 0.0 if done else last_val
        loss_val = agent.update(last_value=last_bs)

    result = {
        "total_reward": ep_reward,
        "total_return": env.total_return(),
        "num_trades":   len(env.get_trade_log()),
        "mean_loss":    loss_val,
        "final_value":  env.portfolio_value(),
    }
    if record_equity:
        result["equity_curve"] = equity_log
    return result


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def run_single_seed(
    seed:          int,
    fold_id:       str,
    train_feat:    Dict[str, pd.DataFrame],
    train_vix:     pd.Series,
    train_prices:  Dict[str, pd.DataFrame],
    test_feat:     Dict[str, pd.DataFrame],
    test_vix:      pd.Series,
    test_prices:   Dict[str, pd.DataFrame],
    config:        dict,
    ablation_cfg:  dict,
    checkpoint_dir: str,
) -> Tuple[PPOAgent, dict, List[dict]]:
    """
    Full training + evaluation for one seed on one walk-forward fold.

    Parameters
    ----------
    ablation_cfg : dict with keys use_sentiment, use_short_selling,
                   use_risk_reward, action_noise, shuffle_prices (opt.)

    Returns
    -------
    agent       : trained PPOAgent (best checkpoint loaded)
    test_metrics: dict of evaluation metrics
    history     : list of per-episode dicts {ep, total_return, num_trades, mean_loss}
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    tickers    = list(train_feat.keys())
    n          = len(tickers)
    state_dim, action_dim = compute_dims(n)

    train_cfg     = config.get("training", {})
    num_episodes  = int(train_cfg.get("num_episodes",  500))
    eval_freq     = int(train_cfg.get("eval_freq",      25))
    device        = train_cfg.get("device",           "auto")

    use_sentiment = ablation_cfg.get("use_sentiment",    True)
    use_short     = ablation_cfg.get("use_short_selling", True)
    use_risk_rew  = ablation_cfg.get("use_risk_reward",   True)
    action_noise  = ablation_cfg.get("action_noise",      True)
    shuffle_prices= ablation_cfg.get("shuffle_prices",    False)

    obs_noise_std = (
        float(config["features"].get("obs_noise_std", 0.01)) if action_noise else 0.0
    )

    # Reward config override for naive_reward ablation
    env_config = dict(config.get("env", {}))
    if not use_risk_rew:
        env_config["lambda_drawdown"]    = 0.0
        env_config["mu_turnover"]        = 0.0
        env_config["drawdown_threshold"] = 0.0
    else:
        reward_cfg = config.get("reward", {})
        env_config.update({
            "lambda_drawdown":    reward_cfg.get("lambda_drawdown",    0.02),
            "drawdown_threshold": reward_cfg.get("drawdown_threshold", 0.05),
            "mu_turnover":        reward_cfg.get("mu_turnover",        0.001),
        })

    # Optionally shuffle price series (sanity-check ablation)
    _train_prices = train_prices
    if shuffle_prices:
        _train_prices = _shuffle_price_series(train_prices, rng)
        log.warning(f"[trainer] shuffle_prices=True — temporal structure destroyed for seed {seed}")

    # ---- Agent ----
    agent = PPOAgent(
        state_dim = state_dim,
        n_assets  = n,
        config    = config,
        device    = device,
        seed      = seed,
    )

    ckpt_name = f"best_{fold_id}_seed{seed}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_return = -float("inf")
    history: List[dict] = []

    log.info(
        f"[trainer] fold={fold_id} seed={seed}  "
        f"episodes={num_episodes}  sentiment={use_sentiment}  "
        f"short={use_short}  risk_reward={use_risk_rew}"
    )

    # ---- Training loop ----
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    abl_id  = ablation_cfg.get("id", "run")   # label for tqdm

    ep_iter = range(1, num_episodes + 1)
    if _tqdm_available:
        pbar = tqdm(
            ep_iter,
            desc  = f"{fold_id}|{abl_id[:6]}|s{seed}",
            unit  = "ep",
            ncols = 88,
            leave = True,
        )
        ep_iter = pbar

    for ep in ep_iter:
        train_env = MultiAssetEnv(
            feat_panel    = train_feat,
            vix_norm      = train_vix,
            prices        = _train_prices,
            config        = env_config,
            mode          = "train",
            use_sentiment = use_sentiment,
            use_short     = use_short,
            obs_noise_std = obs_noise_std,
            seed          = seed + ep,
        )
        metrics = run_episode(train_env, agent, train=True)
        history.append({
            "ep":           ep,
            "total_return": metrics["total_return"],
            "num_trades":   metrics["num_trades"],
            "mean_loss":    metrics["mean_loss"],
        })

        # Update tqdm postfix every episode so the bar is always alive
        if _tqdm_available:
            pbar.set_postfix({
                "ret":    f"{metrics['total_return']:+.1%}",
                "loss":   f"{metrics['mean_loss']:.4f}",
                "trades": metrics["num_trades"],
            }, refresh=False)

        if ep % eval_freq == 0:
            eval_env = MultiAssetEnv(
                feat_panel    = train_feat,
                vix_norm      = train_vix,
                prices        = _train_prices,
                config        = env_config,
                mode          = "train",
                use_sentiment = use_sentiment,
                use_short     = use_short,
                obs_noise_std = 0.0,       # no noise during evaluation
                seed          = seed,
            )
            eval_metrics = run_episode(eval_env, agent, train=False)

            log.info(
                f"  Ep {ep:4d}/{num_episodes}  "
                f"train_ret={metrics['total_return']:+.2%}  "
                f"eval_ret={eval_metrics['total_return']:+.2%}  "
                f"loss={metrics['mean_loss']:.5f}  "
                f"trades={metrics['num_trades']}"
            )

            if eval_metrics["total_return"] > best_return:
                best_return = eval_metrics["total_return"]
                agent.save(ckpt_path)

    if _tqdm_available:
        pbar.close()

    log.info(f"[trainer] Training done. Best in-sample return: {best_return:+.2%}")

    # ---- Load best checkpoint ----
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)

    # ---- Out-of-sample evaluation ----
    test_env = MultiAssetEnv(
        feat_panel    = test_feat,
        vix_norm      = test_vix,
        prices        = test_prices,
        config        = env_config,
        mode          = "test",
        use_sentiment = use_sentiment,
        use_short     = use_short,
        obs_noise_std = 0.0,
        seed          = seed,
    )
    test_result = run_episode(test_env, agent, train=False, record_equity=True)
    test_result["trade_log"]   = test_env.get_trade_log()
    test_result["fold_id"]     = fold_id
    test_result["seed"]        = seed

    return agent, test_result, history


# ---------------------------------------------------------------------------
# Multi-seed wrapper
# ---------------------------------------------------------------------------

def run_multi_seed(
    fold_id:        str,
    train_feat:     Dict[str, pd.DataFrame],
    train_vix:      pd.Series,
    train_prices:   Dict[str, pd.DataFrame],
    test_feat:      Dict[str, pd.DataFrame],
    test_vix:       pd.Series,
    test_prices:    Dict[str, pd.DataFrame],
    config:         dict,
    ablation_cfg:   dict,
    checkpoint_dir: str,
) -> dict:
    """
    Run the same experiment for each seed in config.training.seeds.

    Returns a dict with:
      per_seed_metrics : list of test_result dicts (one per seed)
      mean_metrics     : dict of metric_name → mean value
      std_metrics      : dict of metric_name → std value
    """
    seeds         = config["training"].get("seeds", [42])
    per_seed: List[dict] = []

    for seed in seeds:
        _, test_result, _ = run_single_seed(
            seed           = seed,
            fold_id        = fold_id,
            train_feat     = train_feat,
            train_vix      = train_vix,
            train_prices   = train_prices,
            test_feat      = test_feat,
            test_vix       = test_vix,
            test_prices    = test_prices,
            config         = config,
            ablation_cfg   = ablation_cfg,
            checkpoint_dir = checkpoint_dir,
        )
        per_seed.append(test_result)
        log.info(
            f"[multi_seed] seed={seed}  "
            f"return={test_result['total_return']:+.2%}  "
            f"trades={test_result['num_trades']}"
        )

    # Aggregate scalar metrics
    scalar_keys = ["total_return", "final_value", "num_trades", "total_reward"]
    mean_m = {}
    std_m  = {}
    for k in scalar_keys:
        vals = [r[k] for r in per_seed if k in r]
        if vals:
            mean_m[k] = float(np.mean(vals))
            std_m[k]  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    log.info(
        f"[multi_seed] {fold_id}  "
        f"return={mean_m.get('total_return',0):+.2%} ± "
        f"{std_m.get('total_return',0):.2%}"
    )

    return {
        "fold_id":          fold_id,
        "per_seed_metrics": per_seed,
        "mean_metrics":     mean_m,
        "std_metrics":      std_m,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shuffle_price_series(
    prices: Dict[str, pd.DataFrame],
    rng:    np.random.Generator,
) -> Dict[str, pd.DataFrame]:
    """
    Randomly permute the rows of each price DataFrame (destroys all temporal
    structure).  Used in the random_prices sanity-check ablation.
    """
    shuffled = {}
    for ticker, df in prices.items():
        idx  = df.index
        vals = df.values.copy()
        rng.shuffle(vals)   # in-place row shuffle
        shuffled[ticker] = pd.DataFrame(vals, index=idx, columns=df.columns)
    return shuffled
