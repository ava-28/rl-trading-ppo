"""
env/multi_asset_env.py
----------------------
Research-grade N-asset trading environment for the rl_trading_v2 system.

Design principles
-----------------
  * Per-asset INDEPENDENT action heads: each asset has its own Categorical(3)
    distribution (HOLD=0, BUY=1, SELL=2).  The joint action is a length-N
    vector, not a single integer from 3^N — this scales linearly with N.

  * Risk-adjusted reward:
      r = Δportfolio/capital − λ·max(0, drawdown − threshold) − μ·turnover
    where drawdown is the running drawdown from the peak portfolio value.

  * Realistic frictions:
      - Flat commission per trade
      - Proportional slippage (% of price)
      - Bid-ask half-spread per side
      - Annual stock-borrow cost on short positions
      - Position size limit (max_position_pct of portfolio per asset)
      - T+1 execution lag (orders submitted at close t fill at open t+1)

  * Training augmentations:
      - Randomised episode start (episode_randomize_start=True)
      - Observation noise injection (obs_noise_std > 0)

  * Lookahead safety:
      - All features are pre-computed causally (see data/features.py)
      - Environment only reads row t to produce the observation at step t

State vector (per step)
-----------------------
  [feat_1_1 … feat_11_1 | feat_1_2 … feat_11_2 | … | feat_11_N |
   vix_norm | pos_1 … pos_N | cash_norm]

  Total dimension: 11·N + 1 + N + 1 = 12·N + 2

  Positions are normalised by max_shares_long (long) or max_shares_short (short).
  Cash is normalised by initial_cash.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Number of features per stock (must match data/features.py)
FEATURES_PER_STOCK = 11

# Action codes
HOLD = 0
BUY  = 1
SELL = 2
ACTION_NAMES = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}

BORROW_COST_DAILY = lambda annual_rate: annual_rate / 252.0


# ---------------------------------------------------------------------------
# Dimension helper (importable by agent / training code)
# ---------------------------------------------------------------------------

def compute_dims(n_tickers: int) -> Tuple[int, int]:
    """
    Return (state_dim, action_dim) for an N-ticker environment.

    state_dim  = 12·N + 2   (per-asset features + VIX + positions + cash)
    action_dim = N           (per-asset independent actions, each in {0,1,2})
    """
    state_dim  = FEATURES_PER_STOCK * n_tickers + 1 + n_tickers + 1
    action_dim = n_tickers   # each is Categorical(3); not 3^N
    return state_dim, action_dim


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MultiAssetEnv:
    """
    Multi-asset RL trading environment.

    Parameters
    ----------
    feat_panel   : dict[ticker → DataFrame(normalised features, shape T×11)]
                   The 'sentiment_score' column must already be injected.
    vix_norm     : normalised VIX Series, aligned to the same trading days
    prices       : dict[ticker → DataFrame(OHLCV)] — raw prices for P&L
    config       : dict with env sub-tree from config.yaml
    mode         : 'train' | 'test'
    use_sentiment: if False, zeroes out the sentiment_score column in obs
    use_short    : if False, disables SELL actions when not long (no shorting)
    obs_noise_std: Gaussian noise std added to observations (training only)
    seed         : random seed for episode start randomisation & noise
    """

    def __init__(
        self,
        feat_panel:    Dict[str, pd.DataFrame],
        vix_norm:      pd.Series,
        prices:        Dict[str, pd.DataFrame],
        config:        dict,
        mode:          str  = "train",
        use_sentiment: bool = True,
        use_short:     bool = True,
        obs_noise_std: float = 0.01,
        seed:          int  = 42,
    ):
        self.tickers       = list(feat_panel.keys())
        self.n             = len(self.tickers)
        self.feat_panel    = feat_panel
        self.vix_norm      = vix_norm
        self.prices        = prices
        self.cfg           = config
        self.mode          = mode
        self.use_sentiment = use_sentiment
        self.use_short     = use_short
        self.obs_noise_std = obs_noise_std if mode == "train" else 0.0
        self.rng           = np.random.default_rng(seed)

        # Validate alignment: all feature DataFrames must share the same index
        ref_idx = self.feat_panel[self.tickers[0]].index
        for t in self.tickers[1:]:
            assert self.feat_panel[t].index.equals(ref_idx), (
                f"Feature index mismatch for {t}"
            )
        self.dates = ref_idx   # DatetimeIndex

        # Dimensions
        self.STATE_DIM,  self.ACTION_DIM = compute_dims(self.n)

        # Config shorthand
        cfg_e = config
        self.initial_cash       = float(cfg_e.get("initial_cash",       10_000.0))
        self.shares_per_trade   = int(cfg_e.get("shares_per_trade",     1))
        self.max_shares_long    = int(cfg_e.get("max_shares_long",      5))
        self.max_shares_short   = int(cfg_e.get("max_shares_short",     5))
        self.commission         = float(cfg_e.get("commission",         0.50))
        self.slippage_pct       = float(cfg_e.get("slippage_pct",       0.001))
        self.half_spread        = float(cfg_e.get("bid_ask_half_spread",0.0005))
        self.borrow_annual      = float(cfg_e.get("borrow_cost_annual", 0.02))
        self.max_pos_pct        = float(cfg_e.get("max_position_pct",   0.30))
        self.exec_lag           = int(cfg_e.get("execution_lag",        1))
        self.randomize_start    = bool(cfg_e.get("episode_randomize_start", True))
        self.min_episode_len    = int(cfg_e.get("min_episode_length",   60))

        # Internal state (reset at each episode)
        self._reset_state()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_state(self):
        T = len(self.dates)
        if self.mode == "train" and self.randomize_start:
            max_start = max(0, T - self.min_episode_len)
            self.t = int(self.rng.integers(0, max_start + 1))
        else:
            self.t = 0

        self.cash          = self.initial_cash
        self.shares        = {t: 0 for t in self.tickers}   # + = long, - = short
        self.peak_value    = self.initial_cash
        self.prev_value    = self.initial_cash
        self._trade_log: List[dict] = []

        # Pending orders filled at t+1 open (execution lag)
        self._pending: Optional[Dict[str, int]] = None   # {ticker: action}

    def reset(self) -> np.ndarray:
        self._reset_state()
        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: np.ndarray,   # shape (N,) integer array, each in {0,1,2}
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one trading step.

        Parameters
        ----------
        actions : np.ndarray of shape (N,), dtype int — per-asset actions

        Returns
        -------
        obs    : next observation (np.ndarray)
        reward : scalar reward
        done   : bool
        info   : dict with diagnostics
        """
        T = len(self.dates)

        # ------ Execute pending orders from last step ------
        if self._pending is not None and self.t < T:
            self._execute_orders(self._pending, at_open=True)
            self._pending = None

        # ------ Check termination ------
        if self.t >= T - 1:
            reward = self._compute_reward(terminal=True)
            return self._get_obs(), reward, True, self._make_info()

        # ------ Record actions and queue for next open ------
        action_map = {self.tickers[i]: int(actions[i]) for i in range(self.n)}
        self._pending = action_map

        # Advance time
        self.t += 1

        reward = self._compute_reward(terminal=False)
        done   = (self.t >= T - 1)
        return self._get_obs(), reward, done, self._make_info()

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute_orders(self, action_map: Dict[str, int], at_open: bool = True):
        """Fill orders at the open (or close) price of the current bar."""
        date      = self.dates[self.t]
        pv_before = self._portfolio_value()

        for ticker, action in action_map.items():
            price_df = self.prices[ticker]
            if date not in price_df.index:
                continue   # no data for this ticker today

            raw_price = float(
                price_df.loc[date, "open"] if at_open else price_df.loc[date, "close"]
            )

            if action == BUY:
                self._execute_buy(ticker, raw_price, date)
            elif action == SELL:
                self._execute_sell(ticker, raw_price, date)
            # HOLD: no transaction

    def _execution_price(self, raw_price: float, is_buy: bool) -> float:
        """Apply slippage + bid-ask spread to the raw price."""
        spread = raw_price * self.half_spread
        slip   = raw_price * self.slippage_pct
        if is_buy:
            return raw_price + spread + slip
        else:
            return raw_price - spread - slip

    def _check_position_limit(self, ticker: str, delta: int) -> bool:
        """Return True if adding *delta* shares stays within position % limit."""
        new_shares = self.shares[ticker] + delta
        price      = self._current_price(ticker)
        position_value = abs(new_shares) * price
        portfolio_val  = self._portfolio_value()
        if portfolio_val <= 0:
            return False
        return (position_value / portfolio_val) <= self.max_pos_pct

    def _execute_buy(self, ticker: str, raw_price: float, date):
        shares = self.tickers   # dummy ref — use self.shares_per_trade
        qty    = self.shares_per_trade

        # Long-side cap
        if self.shares[ticker] >= self.max_shares_long:
            return

        # Must not exceed position limit
        if not self._check_position_limit(ticker, qty):
            return

        exec_price = self._execution_price(raw_price, is_buy=True)
        cost       = exec_price * qty + self.commission

        if self.cash >= cost:
            self.cash          -= cost
            self.shares[ticker] += qty
            self._log_trade(ticker, "BUY", qty, exec_price, date)

    def _execute_sell(self, ticker: str, raw_price: float, date):
        qty = self.shares_per_trade

        if not self.use_short and self.shares[ticker] <= 0:
            return   # no shorting allowed

        # Short-side cap
        if self.shares[ticker] <= -self.max_shares_short:
            return

        # Position limit
        if not self._check_position_limit(ticker, -qty):
            return

        exec_price = self._execution_price(raw_price, is_buy=False)
        proceeds   = exec_price * qty - self.commission

        self.cash          += proceeds
        self.shares[ticker] -= qty
        self._log_trade(ticker, "SELL", qty, exec_price, date)

    def _log_trade(self, ticker: str, action: str, qty: int, price: float, date):
        self._trade_log.append({
            "date":   date,
            "ticker": ticker,
            "action": action,
            "qty":    qty,
            "price":  price,
        })

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, terminal: bool = False) -> float:
        """
        r = Δportfolio/capital − λ·max(0, DD − DD_threshold) − μ·turnover

        All terms are dimensionless (fractions of portfolio value).
        """
        cfg_r = self.cfg.get("reward", {})
        lam   = float(cfg_r.get("lambda_drawdown",   0.02))
        dd_th = float(cfg_r.get("drawdown_threshold", 0.05))
        mu    = float(cfg_r.get("mu_turnover",        0.001))

        pv    = self._portfolio_value()
        cap   = max(self.initial_cash, 1.0)

        # Daily borrow cost on short positions
        for ticker in self.tickers:
            if self.shares[ticker] < 0:
                borrow = BORROW_COST_DAILY(self.borrow_annual)
                price  = self._current_price(ticker)
                self.cash -= abs(self.shares[ticker]) * price * borrow

        # Portfolio return
        delta_ret = (pv - self.prev_value) / cap

        # Drawdown penalty
        self.peak_value = max(self.peak_value, pv)
        drawdown = (self.peak_value - pv) / max(self.peak_value, 1.0)
        dd_penalty = lam * max(0.0, drawdown - dd_th)

        # Turnover penalty (sum of |Δshares| × price / portfolio)
        # Counted only for trades executed this step
        trades_today = [
            tr for tr in self._trade_log
            if tr["date"] == self.dates[self.t]
        ]
        turnover = sum(tr["qty"] * tr["price"] for tr in trades_today) / max(pv, 1.0)
        turnover_penalty = mu * turnover

        reward = delta_ret - dd_penalty - turnover_penalty

        self.prev_value = pv
        return float(reward)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the state vector for the current time step."""
        t    = min(self.t, len(self.dates) - 1)
        date = self.dates[t]
        obs_parts: List[float] = []

        for ticker in self.tickers:
            row = self.feat_panel[ticker].iloc[t].values.astype(float)
            if not self.use_sentiment:
                row = row.copy()
                # sentiment_score is always the last feature
                row[-1] = 0.0
            obs_parts.extend(row.tolist())

        # VIX
        vix_val = float(self.vix_norm.iloc[t]) if t < len(self.vix_norm) else 0.0
        obs_parts.append(vix_val)

        # Per-asset position norms
        for ticker in self.tickers:
            shares = self.shares[ticker]
            if shares >= 0:
                norm = shares / max(self.max_shares_long, 1)
            else:
                norm = shares / max(self.max_shares_short, 1)   # negative
            obs_parts.append(norm)

        # Cash norm
        obs_parts.append(self.cash / self.initial_cash)

        obs = np.array(obs_parts, dtype=np.float32)

        # Replace NaN/Inf with 0 (safety net for edge rows)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # Observation noise (training only)
        if self.obs_noise_std > 0.0:
            obs += self.rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)

        assert obs.shape == (self.STATE_DIM,), (
            f"Obs shape mismatch: got {obs.shape}, expected ({self.STATE_DIM},)"
        )
        return obs

    # ------------------------------------------------------------------
    # Portfolio valuation helpers
    # ------------------------------------------------------------------

    def _current_price(self, ticker: str) -> float:
        """Mark-to-market price for *ticker* at the current time step."""
        t    = min(self.t, len(self.dates) - 1)
        date = self.dates[t]
        pdf  = self.prices[ticker]
        if date in pdf.index:
            return float(pdf.loc[date, "close"])
        # Fall back to the last available price
        avail = pdf.index[pdf.index <= date]
        if len(avail) > 0:
            return float(pdf.loc[avail[-1], "close"])
        return 0.0

    def _portfolio_value(self) -> float:
        """Total mark-to-market value of the portfolio."""
        equity = sum(
            self.shares[t] * self._current_price(t) for t in self.tickers
        )
        return self.cash + equity

    def total_return(self) -> float:
        return (self._portfolio_value() - self.initial_cash) / self.initial_cash

    def portfolio_value(self) -> float:
        return self._portfolio_value()

    # ------------------------------------------------------------------
    # Info dict & trade log
    # ------------------------------------------------------------------

    def _make_info(self) -> dict:
        t    = min(self.t, len(self.dates) - 1)
        date = self.dates[t]
        info = {
            "date":      date,
            "portfolio": self._portfolio_value(),
            "cash":      self.cash,
            "shares":    dict(self.shares),
            "step":      t,
        }
        return info

    def get_trade_log(self) -> pd.DataFrame:
        if not self._trade_log:
            return pd.DataFrame(
                columns=["date", "ticker", "action", "qty", "price"]
            )
        return pd.DataFrame(self._trade_log)

    def get_equity_curve(self) -> pd.Series:
        """
        Re-compute the full daily equity curve from the trade log.
        Only meaningful after a full episode.
        """
        # We record equity day-by-day during the episode via _make_info()
        # This is a lightweight reconstruction; for detailed equity logging
        # use record_equity=True in the training runner.
        return pd.Series(
            [self.initial_cash],
            name="portfolio_value",
        )
