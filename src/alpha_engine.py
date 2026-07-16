# src/alpha_engine.py — Pairs Trading Alpha Module (current beta for exit)
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression

class PairConfig:
    """Stores configuration for a single pair."""
    def __init__(self, stock_a: str, stock_b: str, hedge_window: int = 60,
                 z_score_window: int = 20, entry_z: float = 2.0, exit_z: float = 0.0):
        self.stock_a = stock_a
        self.stock_b = stock_b
        self.hedge_window = hedge_window
        self.z_score_window = z_score_window
        self.entry_z = entry_z
        self.exit_z = exit_z

class PairsAlphaEngine:
    """
    Identifies cointegrated pairs and generates long/short spread signals
    based on rolling Z-score of the OLS-hedged spread.
    """
    def __init__(self, pairs: List[PairConfig]):
        self.pairs = pairs

    def _compute_beta(self, pair_cfg: PairConfig, raw_data: Dict[str, pd.DataFrame],
                      date: pd.Timestamp) -> Optional[float]:
        """Compute hedge ratio using trailing 'hedge_window' days up to 'date'."""
        a, b = pair_cfg.stock_a, pair_cfg.stock_b
        if a not in raw_data or b not in raw_data:
            return None
        prices_a = raw_data[a].loc[:date]['Close']
        prices_b = raw_data[b].loc[:date]['Close']
        if isinstance(prices_a, pd.DataFrame): prices_a = prices_a.iloc[:, 0]
        if isinstance(prices_b, pd.DataFrame): prices_b = prices_b.iloc[:, 0]
        common_idx = prices_a.index.intersection(prices_b.index)
        prices_a = prices_a.reindex(common_idx)
        prices_b = prices_b.reindex(common_idx)
        if len(prices_a) < pair_cfg.hedge_window + 5:
            return None
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)
        X = log_b.iloc[-pair_cfg.hedge_window:].values.reshape(-1, 1)
        y = log_a.iloc[-pair_cfg.hedge_window:].values
        model = LinearRegression()
        model.fit(X, y)
        return float(model.coef_[0])

    def compute_signal(self, pair_cfg: PairConfig, raw_data: Dict[str, pd.DataFrame],
                       date: pd.Timestamp) -> Optional[Dict]:
        """
        For a given pair and date, compute whether to enter a new position.
        Returns a signal dict with action and hedge ratio, or None.
        """
        a, b = pair_cfg.stock_a, pair_cfg.stock_b
        if a not in raw_data or b not in raw_data:
            return None

        prices_a = raw_data[a].loc[:date]['Close']
        prices_b = raw_data[b].loc[:date]['Close']
        if isinstance(prices_a, pd.DataFrame): prices_a = prices_a.iloc[:, 0]
        if isinstance(prices_b, pd.DataFrame): prices_b = prices_b.iloc[:, 0]

        common_idx = prices_a.index.intersection(prices_b.index)
        prices_a = prices_a.reindex(common_idx)
        prices_b = prices_b.reindex(common_idx)
        if len(prices_a) < max(pair_cfg.hedge_window, pair_cfg.z_score_window) + 5:
            return None

        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        beta = self._compute_beta(pair_cfg, raw_data, date)
        if beta is None:
            return None

        spread = log_a - beta * log_b
        spread_mean = spread.rolling(window=pair_cfg.z_score_window).mean()
        spread_std = spread.rolling(window=pair_cfg.z_score_window).std()

        current_spread = spread.iloc[-1].item() if hasattr(spread.iloc[-1], 'item') else float(spread.iloc[-1])
        current_mean = spread_mean.iloc[-1].item() if hasattr(spread_mean.iloc[-1], 'item') else float(spread_mean.iloc[-1])
        current_std = spread_std.iloc[-1].item() if hasattr(spread_std.iloc[-1], 'item') else float(spread_std.iloc[-1])

        if pd.isna(current_mean) or pd.isna(current_std) or current_std == 0:
            return None

        z_score = (current_spread - current_mean) / current_std

        if z_score < -pair_cfg.entry_z:
            return {
                'action': 'BUY_SPREAD',
                'z_score': z_score,
                'beta': beta,
                'entry_price_a': float(prices_a.iloc[-1]),
                'entry_price_b': float(prices_b.iloc[-1])
            }
        elif z_score > pair_cfg.entry_z:
            return {
                'action': 'SELL_SPREAD',
                'z_score': z_score,
                'beta': beta,
                'entry_price_a': float(prices_a.iloc[-1]),
                'entry_price_b': float(prices_b.iloc[-1])
            }
        return None

    def check_exit(self, pair_cfg: PairConfig, raw_data: Dict[str, pd.DataFrame],
                   date: pd.Timestamp, direction: str) -> Optional[str]:
        """
        Determine if an open pair position should be closed.
        Recomputes the hedge ratio from recent data (no stale beta).
        direction: 'LONG_SPREAD' or 'SHORT_SPREAD'
        Returns 'EXIT_Z_REVERSION' or None.
        """
        a, b = pair_cfg.stock_a, pair_cfg.stock_b
        if a not in raw_data or b not in raw_data or date not in raw_data[a].index or date not in raw_data[b].index:
            return None

        beta = self._compute_beta(pair_cfg, raw_data, date)
        if beta is None:
            return None

        prices_a = raw_data[a].loc[:date]['Close']
        prices_b = raw_data[b].loc[:date]['Close']
        if isinstance(prices_a, pd.DataFrame): prices_a = prices_a.iloc[:, 0]
        if isinstance(prices_b, pd.DataFrame): prices_b = prices_b.iloc[:, 0]

        common_idx = prices_a.index.intersection(prices_b.index)
        prices_a = prices_a.reindex(common_idx)
        prices_b = prices_b.reindex(common_idx)
        if len(prices_a) < pair_cfg.z_score_window + 5:
            return None

        log_a = np.log(prices_a)
        log_b = np.log(prices_b)
        spread = log_a - beta * log_b
        spread_mean = spread.rolling(window=pair_cfg.z_score_window).mean()
        spread_std = spread.rolling(window=pair_cfg.z_score_window).std()

        current_spread = spread.iloc[-1].item() if hasattr(spread.iloc[-1], 'item') else float(spread.iloc[-1])
        current_mean = spread_mean.iloc[-1].item() if hasattr(spread_mean.iloc[-1], 'item') else float(spread_mean.iloc[-1])
        current_std = spread_std.iloc[-1].item() if hasattr(spread_std.iloc[-1], 'item') else float(spread_std.iloc[-1])

        if pd.isna(current_mean) or pd.isna(current_std) or current_std == 0:
            return None

        z = (current_spread - current_mean) / current_std

        if direction == 'LONG_SPREAD' and z >= pair_cfg.exit_z:
            return 'EXIT_Z_REVERSION'
        elif direction == 'SHORT_SPREAD' and z <= pair_cfg.exit_z:
            return 'EXIT_Z_REVERSION'
        return None