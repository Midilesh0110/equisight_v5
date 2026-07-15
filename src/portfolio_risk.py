# src/portfolio_risk.py
import numpy as np
import pandas as pd

class PortfolioDefender:
    def __init__(self, max_kelly_cap=0.5, gap_threshold=0.05, vol_threshold=0.082, corr_window=20, corr_threshold=0.7):
        self.max_kelly = max_kelly_cap
        self.gap_threshold = gap_threshold
        self.vol_threshold = vol_threshold
        self.corr_window = corr_window
        self.corr_threshold = corr_threshold

    def calculate_half_kelly(self, win_trades: list, loss_trades: list) -> float:
        total_trades = len(win_trades) + len(loss_trades)
        if total_trades == 0:
            return 0.0
        win_prob = len(win_trades) / total_trades
        loss_prob = len(loss_trades) / total_trades
        avg_win = np.mean(win_trades) if win_trades else 0.0
        avg_loss = abs(np.mean(loss_trades)) if loss_trades else 0.0
        if avg_win == 0:
            return 0.0
        ev = (win_prob * avg_win) - (loss_prob * avg_loss)
        if ev <= 0:
            return 0.0
        full_kelly = ev / avg_win
        half_kelly = full_kelly * 0.5
        return min(half_kelly, self.max_kelly)

    def check_volatility_gatekeeper(self, range_pct: float) -> bool:
        return range_pct > self.vol_threshold

    def check_overnight_gap(self, close_t: float, open_t_plus_1: float) -> bool:
        gap = (open_t_plus_1 - close_t) / close_t
        return abs(gap) > self.gap_threshold

    def check_correlation_overlay(self, new_ticker: str, existing_tickers: list, returns_df: pd.DataFrame) -> float:
        """
        Returns a scaling factor for the new position based on correlation with existing holdings.
        - 1.0: no high correlation
        - 0.5: correlation > threshold (position size halved)
        - 0.0: skip trade if correlation > 0.9 or if asset is already in portfolio
        """
        if not existing_tickers:
            return 1.0
        
        # AUDIT FIX: De-duplicate list to prevent Pandas multi-index matrix crashes
        required_tickers = list(set([new_ticker] + existing_tickers))
        
        if not all(t in returns_df.columns for t in required_tickers):
            return 1.0
        
        recent_returns = returns_df[required_tickers].iloc[-self.corr_window:]
        if len(recent_returns) < 10:  
            return 1.0
        
        corr_matrix = recent_returns.corr()
        for et in existing_tickers:
            # Hard-block duplicate positions to prevent over-exposure
            if new_ticker == et:
                return 0.0
                
            # Safely extract scalar
            corr_val = float(corr_matrix.loc[new_ticker, et])
            
            if corr_val > self.corr_threshold:
                if corr_val > 0.9:  
                    return 0.0
                return 0.5
        return 1.0

if __name__ == "__main__":
    defender = PortfolioDefender()
    # Quick tests
    print("Half-Kelly:", defender.calculate_half_kelly([0.02,0.03],[0.01,0.015]))
    print("Vol gate 9%:", defender.check_volatility_gatekeeper(0.09))
    print("Overnight gap 6%:", defender.check_overnight_gap(100, 94))
    # Dummy returns for correlation test
    idx = pd.date_range('2025-01-01', periods=30, freq='B')
    dummy_ret = pd.DataFrame({
        'A.NS': np.random.randn(30)*0.01,
        'B.NS': np.random.randn(30)*0.01,
        'C.NS': np.random.randn(30)*0.01
    }, index=idx)
    print("Correlation scale (new 'A.NS', existing ['B.NS']):", defender.check_correlation_overlay('A.NS', ['B.NS'], dummy_ret))