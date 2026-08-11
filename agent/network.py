"""
agent/network.py
----------------
Neural network architecture for the multi-asset PPO agent.

Design
------
  Shared trunk
    - Linear(state_dim → hidden) + LayerNorm + ReLU + Dropout
    - Linear(hidden → hidden) + LayerNorm + ReLU + Dropout
    (depth controlled by num_layers in config.yaml)

  Per-asset actor heads (N independent heads)
    - One Linear(hidden → 3) for each asset
    - Each outputs logits for {HOLD, BUY, SELL}
    - Factored policy: log π(a₁,…,aₙ | s) = Σᵢ log πᵢ(aᵢ | s)
    - This scales linearly with N (vs. exponentially for joint 3^N)

  Single critic head
    - Linear(hidden → 1) — scalar state value V(s)

Why per-asset heads?
--------------------
  With a joint action space of 3^N the number of output units explodes:
    N=7 → 3^7 = 2187 outputs, and the joint distribution becomes extremely
  sparse.  Independent per-asset heads give N×3 outputs (N=7 → 21) and
  allow each asset's policy gradient to propagate independently, making
  learning dramatically more sample-efficient.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCriticNet(nn.Module):
    """
    Shared-trunk actor-critic network for the multi-asset trading agent.

    Parameters
    ----------
    state_dim   : dimension of the observation vector (12N+2)
    n_assets    : number of assets (= N)
    hidden_size : neurons per hidden layer
    num_layers  : number of hidden layers in the shared trunk
    dropout     : dropout rate after each hidden layer
    """

    def __init__(
        self,
        state_dim:   int,
        n_assets:    int,
        hidden_size: int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.10,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.n_assets    = n_assets
        self.hidden_size = hidden_size

        # ---- Shared trunk ----
        layers: List[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)

        # ---- Per-asset actor heads ----
        # ModuleList so parameters are registered correctly
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_size, 3) for _ in range(n_assets)
        ])

        # ---- Single critic head ----
        self.critic_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for output layers (standard PPO practice)
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,   # (batch, state_dim)
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns
        -------
        logits_list : list of N tensors, each shape (batch, 3)
        value       : tensor shape (batch, 1)
        """
        h = self.trunk(obs)
        logits_list = [head(h) for head in self.actor_heads]
        value       = self.critic_head(h)
        return logits_list, value

    def get_action_and_value(
        self,
        obs:          torch.Tensor,   # (batch, state_dim)
        actions:      Optional[torch.Tensor] = None,  # (batch, N) — for computing log_prob
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate actions.

        Parameters
        ----------
        obs         : observation batch
        actions     : if provided, evaluate log_prob of these actions
        deterministic: use argmax instead of sampling

        Returns
        -------
        actions    : (batch, N) int64
        log_probs  : (batch,)  — sum of per-asset log probs
        entropy    : (batch,)  — sum of per-asset entropies
        value      : (batch,)
        """
        logits_list, value = self.forward(obs)
        value = value.squeeze(-1)

        all_actions:   List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []
        all_entropies: List[torch.Tensor] = []

        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)

            if actions is not None:
                a = actions[:, i]
            elif deterministic:
                a = logits.argmax(dim=-1)
            else:
                a = dist.sample()

            all_actions.append(a)
            all_log_probs.append(dist.log_prob(a))
            all_entropies.append(dist.entropy())

        # Stack along asset dimension
        act_tensor      = torch.stack(all_actions,   dim=-1)   # (batch, N)
        log_prob_tensor = torch.stack(all_log_probs, dim=-1).sum(dim=-1)  # (batch,)
        entropy_tensor  = torch.stack(all_entropies, dim=-1).sum(dim=-1)  # (batch,)

        return act_tensor, log_prob_tensor, entropy_tensor, value

    def evaluate_actions(
        self,
        obs:     torch.Tensor,   # (batch, state_dim)
        actions: torch.Tensor,   # (batch, N)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the log_prob, entropy, and value of given (obs, action) pairs.
        Used during the PPO update step.
        """
        _, log_probs, entropy, value = self.get_action_and_value(
            obs, actions=actions
        )
        return log_probs, entropy, value
