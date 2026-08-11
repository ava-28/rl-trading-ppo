"""
agent/ppo.py
------------
PPO agent for the multi-asset trading system.

Key choices
-----------
  * Factored per-asset policy (see network.py for motivation).
  * Clipped surrogate objective with shared value and entropy terms.
  * Gradient clipping (max_grad_norm) for training stability.
  * Checkpoint save/load via torch.save/load_state_dict.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.network import ActorCriticNet
from agent.buffer import RolloutBuffer

log = logging.getLogger(__name__)


class PPOAgent:
    """
    Proximal Policy Optimisation agent for multi-asset trading.

    Parameters
    ----------
    state_dim       : dimension of the observation vector
    n_assets        : number of assets (= N; action vector length)
    config          : dict with 'ppo' sub-tree from config.yaml
    device          : 'cpu' | 'cuda' | 'mps' | 'auto'
    seed            : random seed for the internal RNG
    """

    def __init__(
        self,
        state_dim: int,
        n_assets:  int,
        config:    dict,
        device:    str  = "auto",
        seed:      int  = 42,
    ):
        self.state_dim = state_dim
        self.n_assets  = n_assets
        self.cfg       = config.get("ppo", config)   # allow passing full config or just ppo sub-tree
        self.seed      = seed
        self.rng       = np.random.default_rng(seed)

        # Device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        log.info(f"[PPO] Device: {self.device}")

        # Hyperparameters
        self.lr              = float(self.cfg.get("lr",              3e-4))
        self.gamma           = float(self.cfg.get("gamma",           0.99))
        self.gae_lambda      = float(self.cfg.get("gae_lambda",      0.95))
        self.clip_eps        = float(self.cfg.get("clip_eps",        0.20))
        self.epochs          = int(self.cfg.get("epochs",            4))
        self.minibatch_size  = int(self.cfg.get("minibatch_size",    64))
        self.value_coef      = float(self.cfg.get("value_coef",      0.5))
        self.entropy_coef    = float(self.cfg.get("entropy_coef",    0.01))
        self.max_grad_norm   = float(self.cfg.get("max_grad_norm",   0.5))
        self.norm_advantages = bool(self.cfg.get("normalize_advantages", True))

        hidden_size = int(self.cfg.get("hidden_size", 256))
        num_layers  = int(self.cfg.get("num_layers",  2))
        dropout     = float(self.cfg.get("dropout",   0.10))

        # Network
        self.net = ActorCriticNet(
            state_dim   = state_dim,
            n_assets    = n_assets,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            state_dim  = state_dim,
            n_assets   = n_assets,
            gamma      = self.gamma,
            gae_lambda = self.gae_lambda,
            device     = str(self.device),
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        obs:          np.ndarray,   # (state_dim,)
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Sample an action from the current policy.

        Returns
        -------
        action   : np.ndarray shape (N,) int — per-asset actions
        log_prob : scalar float
        value    : scalar float
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.net.eval()
        act_t, logp_t, _, val_t = self.net.get_action_and_value(
            obs_t, deterministic=deterministic
        )
        self.net.train()

        action   = act_t.squeeze(0).cpu().numpy()   # (N,)
        log_prob = float(logp_t.item())
        value    = float(val_t.item())
        return action, log_prob, value

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0) -> float:
        """
        Run K epochs of PPO gradient updates on the current rollout buffer.

        Parameters
        ----------
        last_value : bootstrap value V(s_{T+1})

        Returns
        -------
        mean_loss : average total loss across all minibatch updates
        """
        self.buffer.compute_gae(last_value=last_value)
        self.net.train()

        total_loss = 0.0
        num_updates = 0

        for _ in range(self.epochs):
            for obs_b, act_b, logp_old_b, ret_b, adv_b in self.buffer.get_batches(
                minibatch_size       = self.minibatch_size,
                normalize_advantages = self.norm_advantages,
                rng                  = self.rng,
            ):
                # Evaluate current policy on the stored (obs, action) pairs
                logp_new_b, entropy_b, val_b = self.net.evaluate_actions(obs_b, act_b)

                # --- Policy loss (clipped surrogate) ---
                ratio = torch.exp(logp_new_b - logp_old_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value loss (clipped) ---
                value_loss = 0.5 * (val_b - ret_b).pow(2).mean()

                # --- Entropy bonus ---
                entropy_loss = -entropy_b.mean()

                # --- Total loss ---
                loss = (
                    policy_loss
                    + self.value_coef  * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss  += loss.item()
                num_updates += 1

        self.buffer.clear()
        return total_loss / max(num_updates, 1)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "net_state":       self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "state_dim":       self.state_dim,
            "n_assets":        self.n_assets,
            "config":          self.cfg,
        }, path)
        log.debug(f"[PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        log.info(f"[PPO] Checkpoint loaded ← {path}")
