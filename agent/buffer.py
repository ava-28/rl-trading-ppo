"""
agent/buffer.py
---------------
On-policy rollout buffer for PPO.

Stores one full episode of transitions and computes Generalised Advantage
Estimates (GAE) before each update.

GAE recap
---------
  δₜ  = rₜ + γ · V(sₜ₊₁) · (1 − doneₜ) − V(sₜ)
  Aₜ  = Σₖ (γλ)ᵏ · δₜ₊ₖ
  Rₜ  = Aₜ + V(sₜ)   (bootstrapped return)

The buffer is cleared after each PPO update (on-policy requirement).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """
    Stores a single rollout episode and supports GAE computation.

    Parameters
    ----------
    state_dim    : dimension of each observation
    n_assets     : number of assets (action vector length)
    gamma        : discount factor
    gae_lambda   : GAE lambda (bias-variance trade-off)
    device       : torch device string
    """

    def __init__(
        self,
        state_dim:  int,
        n_assets:   int,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        device:     str   = "cpu",
    ):
        self.state_dim  = state_dim
        self.n_assets   = n_assets
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = torch.device(device)

        self._clear()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _clear(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[np.ndarray] = []   # each shape (N,)
        self.rewards:   List[float]      = []
        self.log_probs: List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def push(
        self,
        obs:      np.ndarray,   # (state_dim,)
        action:   np.ndarray,   # (n_assets,) int
        reward:   float,
        log_prob: float,
        value:    float,
        done:     bool,
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.rewards)

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(self, last_value: float = 0.0) -> None:
        """
        Compute advantages (A) and bootstrapped returns (R) in-place
        using GAE(λ).

        Parameters
        ----------
        last_value : V(s_{T+1}) — bootstrap value for the state after the
                     final step.  Pass 0.0 if the episode ended naturally.
        """
        T         = len(self.rewards)
        values_np = np.array(self.values + [last_value], dtype=np.float32)
        dones_np  = np.array(self.dones,                 dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones_np[t]
            delta    = (self.rewards[t]
                        + self.gamma * values_np[t + 1] * next_non_terminal
                        - values_np[t])
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self._advantages = advantages
        self._returns    = advantages + np.array(self.values, dtype=np.float32)

    # ------------------------------------------------------------------
    # Mini-batch generator
    # ------------------------------------------------------------------

    def get_batches(
        self,
        minibatch_size:      int,
        normalize_advantages: bool = True,
        rng:                 Optional[np.random.Generator] = None,
    ):
        """
        Yield random mini-batches for PPO gradient updates.

        Yields tuples:
          (obs_b, act_b, logp_b, ret_b, adv_b)
          shapes: (B, state_dim), (B, N), (B,), (B,), (B,)

        Parameters
        ----------
        normalize_advantages : subtract mean and divide by std
        rng                  : numpy Generator for reproducible shuffling
        """
        T = len(self.rewards)
        if not hasattr(self, "_advantages"):
            raise RuntimeError("Call compute_gae() before get_batches().")

        obs_arr    = np.array(self.obs,       dtype=np.float32)   # (T, state_dim)
        act_arr    = np.array(self.actions,   dtype=np.int64)     # (T, N)
        logp_arr   = np.array(self.log_probs, dtype=np.float32)   # (T,)
        ret_arr    = self._returns.copy()                          # (T,)
        adv_arr    = self._advantages.copy()                       # (T,)

        if normalize_advantages:
            adv_mean = adv_arr.mean()
            adv_std  = adv_arr.std() + 1e-8
            adv_arr  = (adv_arr - adv_mean) / adv_std

        # Shuffle indices
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.permutation(T)

        for start in range(0, T, minibatch_size):
            batch_idx = idx[start : start + minibatch_size]

            yield (
                torch.tensor(obs_arr[batch_idx],  device=self.device),
                torch.tensor(act_arr[batch_idx],  device=self.device),
                torch.tensor(logp_arr[batch_idx], device=self.device),
                torch.tensor(ret_arr[batch_idx],  device=self.device),
                torch.tensor(adv_arr[batch_idx],  device=self.device),
            )

    def clear(self):
        self._clear()
        if hasattr(self, "_advantages"):
            del self._advantages
            del self._returns
