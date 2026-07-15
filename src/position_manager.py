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

    def update_positions(self, today_date_str=None):
        """
        Run at the start of a trading day (or after market close).
        - Fetches current day's open/high/low/close from yfinance for each active position.
        - Checks overnight gap (using previous close from yesterday) and volatility gate.
        - If no force close, increments current_day and closes if >= hold_days.
        - Logs any closed trade.
        Returns list of tickers that were closed.
        """
        if today_date_str is None:
            today_date_str = datetime.today().strftime('%Y-%m-%d')
        
        active = self.get_active_positions()
        if active.empty:
            logger.info("No active positions to update.")
            return []

        closed_tickers = []
        for _, pos in active.iterrows():
            ticker = pos['ticker']
            entry_date = pos['entry_date']
            entry_price = pos['entry_price']
            current_day = pos['current_day']

            # 1. Download today's data and yesterday's close
            try:
                # Get last 2 days to have yesterday's close and today's full bar
                data = yf.download(ticker, period="5d", progress=False, session=self.session)
                if len(data) < 2:
                    logger.warning(f"Insufficient data for {ticker}, skipping update.")
                    continue
                # Yesterday is the row before today (today might be the last row if market is open)
                # We assume the latest row is today, previous is yesterday.
                # Better: use date filtering.
                today_data = data.loc[today_date_str] if today_date_str in data.index else None
                if today_data is None:
                    # Today not in data yet? Use last row as today if market is open, else we can't update.
                    logger.warning(f"No data for {today_date_str} for {ticker}, using latest.")
                    today_data = data.iloc[-1]
                    # Reassign date string
                    today_date_str_actual = today_data.name.strftime('%Y-%m-%d')
                else:
                    today_date_str_actual = today_date_str
                # Yesterday's close
                yesterday_close = data.iloc[-2]['Close']
                today_open = today_data['Open']
                today_high = today_data['High']
                today_low = today_data['Low']
                today_close = today_data['Close']
                today_range_pct = (today_high - today_low) / today_close
            except Exception as e:
                logger.error(f"Data fetch error for {ticker}: {e}")
                continue

            # 2. Overnight gap check
            gap = (today_open - yesterday_close) / yesterday_close
            if abs(gap) > self.gap_threshold:
                reason = f"Overnight Gap Filter ({gap*100:.1f}%)"
                self._log_close(ticker, entry_date, entry_price, today_date_str_actual, today_open, pos['allocated_capital'], reason)
                self._remove_position(ticker)
                closed_tickers.append(ticker)
                continue

            # 3. Volatility gatekeeper
            if today_range_pct > self.vol_threshold:
                reason = f"Volatility Gatekeeper (range {today_range_pct*100:.1f}%)"
                self._log_close(ticker, entry_date, entry_price, today_date_str_actual, today_close, pos['allocated_capital'], reason)
                self._remove_position(ticker)
                closed_tickers.append(ticker)
                continue

            # 4. Time decay (T+5 rule)
            if current_day >= self.hold_days:
                reason = "Time decay reached (T+5 Rule)"
                self._log_close(ticker, entry_date, entry_price, today_date_str_actual, today_close, pos['allocated_capital'], reason)
                self._remove_position(ticker)
                closed_tickers.append(ticker)
                continue

            # 5. Else, increment current_day
            with self._get_conn() as conn:
                conn.execute("UPDATE active_positions SET current_day = current_day + 1 WHERE ticker=?", (ticker,))
                conn.commit()

        logger.info(f"Position update complete. Closed: {closed_tickers}")
        return closed_tickers

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