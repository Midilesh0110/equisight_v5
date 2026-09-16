# src/position_manager.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, db_path="database/equisight_v5.db", vol_threshold=0.082, gap_threshold=0.05, hold_days=5):
        self.db_path = db_path
        self.vol_threshold = vol_threshold
        self.gap_threshold = gap_threshold
        self.hold_days = hold_days
        # Reuse a single requests session for yfinance calls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def get_active_positions(self):
        with self._get_conn() as conn:
            return pd.read_sql_query("SELECT * FROM active_positions", conn)

    def _remove_position(self, ticker):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM active_positions WHERE ticker=?", (ticker,))
            conn.commit()

    def _log_close(self, ticker, entry_date, entry_price, exit_date, exit_price, allocated, reason):
        profit = (exit_price - entry_price) / entry_price  # raw return
        days_held = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO trade_outcomes (ticker, entry_date, exit_date, actual_profit, days_held, reason_for_exit)
                VALUES (?,?,?,?,?,?)
            """, (ticker, entry_date, exit_date, profit, days_held, reason))
            conn.commit()

    def update_positions(self, raw_data=None, alpha_engine=None):
        """
        Advances the holding day counter and checks for statistical reversion
        or maximum holding duration exits.
        """
        closed_count = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                active = pd.read_sql_query("SELECT * FROM active_positions", conn)
                if active.empty:
                    return 0

                today_str = datetime.today().strftime('%Y-%m-%d')
                
                for _, pos in active.iterrows():
                    ticker = pos['ticker']
                    current_day = int(pos['current_day'])
                    entry_price = float(pos['entry_price'])
                    allocated_cap = float(pos['allocated_capital'])
                    
                    exit_reason = None
                    exit_price = entry_price
                    calculated_pnl = 0.0

                    # 1. Fetch current price if pair exists in raw_data
                    if raw_data and '-' in ticker:
                        stock_a, stock_b = ticker.split('-')
                        if stock_a in raw_data and stock_b in raw_data:
                            c_a = float(raw_data[stock_a].iloc[-1]['Close'].iloc[0]) if isinstance(raw_data[stock_a].iloc[-1]['Close'], pd.Series) else float(raw_data[stock_a].iloc[-1]['Close'])
                            c_b = float(raw_data[stock_b].iloc[-1]['Close'].iloc[0]) if isinstance(raw_data[stock_b].iloc[-1]['Close'], pd.Series) else float(raw_data[stock_b].iloc[-1]['Close'])
                            exit_price = (c_a + c_b) / 2.0
                            calculated_pnl = ((exit_price - entry_price) / entry_price) * allocated_cap

                    # 2. Check Direction-Aware Statistical Exit via Alpha Engine
                    if alpha_engine and raw_data and hasattr(alpha_engine, 'check_exit'):
                        try:
                            should_exit, reason = alpha_engine.check_exit(ticker, raw_data)
                            if should_exit:
                                exit_reason = f"Statistical Exit ({reason})"
                        except Exception as ex_err:
                            logger.warning(f"Could not compute statistical exit for {ticker}: {ex_err}")

                    # 3. Check Time Stop (Defaults to max 10 completed trading sessions)
                    if not exit_reason and current_day >= 10:
                        exit_reason = f"Time Stop (Day {current_day})"

                    # 4. Process Exit or Increment Holding Days
                    if exit_reason:
                        # Move position to trade_outcomes
                        conn.execute("""
                            INSERT INTO trade_outcomes (ticker, entry_date, exit_date, actual_profit, days_held, reason_for_exit)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (ticker, pos['entry_date'], today_str, round(calculated_pnl, 2), current_day, exit_reason))
                        
                        conn.execute("DELETE FROM active_positions WHERE ticker = ?", (ticker,))
                        conn.commit()
                        closed_count += 1
                        logger.info(f"Closed {ticker} | Reason: {exit_reason} | PnL: ₹{calculated_pnl:.2f} | Days: {current_day}")
                    else:
                        # Increment holding days for continuing positions
                        conn.execute("UPDATE active_positions SET current_day = current_day + 1 WHERE ticker = ?", (ticker,))
                        conn.commit()

            return closed_count
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return 0

    def open_new_position(self, ticker, entry_date, entry_price, allocated_capital):
        """Insert a new position with current_day=1."""
        with self._get_conn() as conn:
            # Remove any existing position for same ticker (should not happen if we close first)
            conn.execute("DELETE FROM active_positions WHERE ticker=?", (ticker,))
            conn.execute("""
                INSERT INTO active_positions (ticker, entry_date, entry_price, current_day, allocated_capital)
                VALUES (?,?,?,1,?)
            """, (ticker, entry_date, entry_price, allocated_capital))
            conn.commit()
        logger.info(f"New position opened: {ticker} on {entry_date}")

if __name__ == "__main__":
    # Quick test (will need actual market data)
    pm = PositionManager()
    pm.open_new_position("TEST.NS", "2026-07-01", 100.0, 10000.0)
    print("Active before update:", pm.get_active_positions())
    closed = pm.update_positions()
    print("Closed:", closed)