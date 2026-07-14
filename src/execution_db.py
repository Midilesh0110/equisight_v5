import sqlite3
import os
import pandas as pd

class ExecutionDatabase:
    def __init__(self, db_path="database/equisight_v5.db"):
        self.db_path = db_path
        # Guarantee directory structure exists locally
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_db()

    def _get_connection(self):
        """Generates connection bound to asynchronous Write-Ahead Logging (WAL)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _initialize_db(self):
        """Initializes relational schemas for execution monitoring."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Active Position Table (Stateful 5-day lifecycle tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_positions (
                    ticker TEXT PRIMARY KEY,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_day INTEGER DEFAULT 1,
                    allocated_capital REAL NOT NULL
                )
            """)
            
            # Historical Outcomes Ledger (Feeds Alpha Decay Monitor & Kelly Engine)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    actual_profit REAL NOT NULL,
                    days_held INTEGER NOT NULL,
                    reason_for_exit TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_trade_outcome(self, ticker: str, entry_date: str, exit_date: str, 
                          actual_profit: float, days_held: int, reason: str):
        """Pushes finalized trade metadata into local history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trade_outcomes (ticker, entry_date, exit_date, actual_profit, days_held, reason_for_exit)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ticker, entry_date, exit_date, actual_profit, days_held, reason))
            conn.commit()

    def get_alpha_decay_metrics(self) -> float:
        """Computes current rolling average holding duration of winning positions."""
        with self._get_connection() as conn:
            query = "SELECT days_held FROM trade_outcomes WHERE actual_profit > 0"
            df = pd.read_sql_query(query, conn)
            if df.empty:
                return 0.0
            return float(df['days_held'].mean())

if __name__ == "__main__":
    print("[INFO] Core Relational Engine isolated successfully.")