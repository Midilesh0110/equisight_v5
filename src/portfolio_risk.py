import numpy as np

class PortfolioDefender:
    def __init__(self, max_kelly_cap=0.5, gap_threshold=0.05, vol_threshold=0.082):
        self.max_kelly = max_kelly_cap  # Half-Kelly cap for safety
        self.gap_threshold = gap_threshold  # +/- 5% overnight gap
        self.vol_threshold = vol_threshold  # 8.2% intraday range

    def calculate_half_kelly(self, win_trades: list, loss_trades: list) -> float:
        """
        Calculates the Expected Value (EV) based Kelly fraction.
        Formula: EV = (Win_Prob * Avg_Win) - (Loss_Prob * Avg_Loss)
        """
        total_trades = len(win_trades) + len(loss_trades)
        if total_trades == 0:
            return 0.0  # No edge proven yet, zero allocation

        win_prob = len(win_trades) / total_trades
        loss_prob = len(loss_trades) / total_trades

        avg_win = np.mean(win_trades) if win_trades else 0.0
        # Ensure average loss is a positive absolute value for the formula
        avg_loss = abs(np.mean(loss_trades)) if loss_trades else 0.0

        if avg_win == 0:
            return 0.0  # Cannot have a mathematical edge without wins

        # Calculate Expected Value (EV)
        ev = (win_prob * avg_win) - (loss_prob * avg_loss)

        if ev <= 0:
            return 0.0  # Negative expected value, do not trade

        # Kelly fraction = EV / Avg_Win_Size
        full_kelly = ev / avg_win
        
        # Apply the Half-Kelly multiplier for drawdown protection
        half_kelly = full_kelly * 0.5
        
        # Return the allocation, strictly capped at self.max_kelly (e.g., 50% max)
        return min(half_kelly, self.max_kelly)

    def check_volatility_gatekeeper(self, range_pct: float) -> bool:
        """
        Returns True if the asset is experiencing extreme intraday stress (>8.2%).
        """
        return range_pct > self.vol_threshold

    def check_overnight_gap(self, close_t: float, open_t_plus_1: float) -> bool:
        """
        Returns True if the gap exceeds the +/- 5% safety threshold.
        """
        gap = (open_t_plus_1 - close_t) / close_t
        return abs(gap) > self.gap_threshold

if __name__ == "__main__":
    print("[INFO] Testing Phase 4: Portfolio Defense Engine...")
    
    defender = PortfolioDefender()
    
    # 1. Test EV-Based Kelly Calculator
    # Simulating a strategy with a 57% win rate, avg win of 2.6%, avg loss of 1.5%
    wins = [0.02, 0.03, 0.015, 0.04] 
    losses = [-0.01, -0.015, -0.02] 
    kelly_fraction = defender.calculate_half_kelly(wins, losses)
    print(f"[SUCCESS] EV-based Half-Kelly Fraction calculated: {kelly_fraction:.4f}")
    
    # 2. Test Volatility Gatekeeper
    is_stressed = defender.check_volatility_gatekeeper(0.09) # 9% intraday range
    print(f"[SUCCESS] Volatility Gatekeeper triggered on 9% range: {is_stressed}")
    
    # 3. Test Overnight Gap Filter
    is_gapping = defender.check_overnight_gap(close_t=100.0, open_t_plus_1=94.0) # 6% gap down
    print(f"[SUCCESS] Overnight Gap Filter triggered on 6% gap down: {is_gapping}")