# main.py — EquiSight V5 Live Production Orchestrator (Backtest Universe)
import yfinance as yf
import requests
import numpy as np
import pandas as pd
import sqlite3
import os
import warnings
from datetime import datetime
import logging

from src.data_pipeline import generate_v5_features
from src.execution_db import ExecutionDatabase
from src.portfolio_risk import PortfolioDefender
from src.position_manager import PositionManager
from src.alpha_engine import PairsAlphaEngine, PairConfig

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# STRICTLY BACKTESTED UNIVERSE – DO NOT ALTER WITHOUT RE-BACKTESTING
# -------------------------------------------------------------------
PAIRS = [
    PairConfig("TCS.NS", "INFY.NS"),
    PairConfig("ICICIBANK.NS", "SBIN.NS"),
    PairConfig("RELIANCE.NS", "HINDUNILVR.NS"),
    PairConfig("KOTAKBANK.NS", "BAJFINANCE.NS")
]

DB_PATH = "database/equisight_v5.db"
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE = 0.05
MAX_TOTAL_EXPOSURE = 0.25


def log_daily_equity(today_str, raw_data, position_mgr, initial_capital=INITIAL_CAPITAL):
    """Calculates total portfolio MTM and appends to daily_equity.csv."""
    active_positions = position_mgr.get_active_positions()
    
    allocated_cash = active_positions['allocated_capital'].sum() if not active_positions.empty else 0.0
    available_cash = initial_capital - allocated_cash
    
    mtm_positions_value = 0.0
    if not active_positions.empty:
        for _, pos in active_positions.iterrows():
            pair_name = pos['ticker']
            
            if '-' in pair_name:
                stock_a, stock_b = pair_name.split('-')
                if stock_a in raw_data and stock_b in raw_data:
                    close_a_raw = raw_data[stock_a].iloc[-1]['Close']
                    close_b_raw = raw_data[stock_b].iloc[-1]['Close']
                    
                    close_a = float(close_a_raw.iloc[0]) if isinstance(close_a_raw, pd.Series) else float(close_a_raw)
                    close_b = float(close_b_raw.iloc[0]) if isinstance(close_b_raw, pd.Series) else float(close_b_raw)
                    
                    entry_avg = float(pos['entry_price'])
                    current_avg = (close_a + close_b) / 2.0
                    
                    pos_return = (current_avg - entry_avg) / entry_avg if entry_avg > 0 else 0.0
                    mtm_positions_value += float(pos['allocated_capital']) * (1.0 + pos_return)
                else:
                    mtm_positions_value += float(pos['allocated_capital'])
            else:
                mtm_positions_value += float(pos['allocated_capital'])
                
    total_equity = available_cash + mtm_positions_value
    
    equity_row = pd.DataFrame([{
        'Date': today_str,
        'Total_Equity': round(total_equity, 2),
        'Open_Positions_Count': len(active_positions)
    }])
    
    equity_file = "daily_equity.csv"
    if os.path.exists(equity_file):
        equity_row.to_csv(equity_file, mode='a', header=False, index=False)
    else:
        equity_row.to_csv(equity_file, index=False)
    logger.info(f"Recorded Daily Equity: ₹{total_equity:,.2f} ({len(active_positions)} active positions)")


def run_production_loop():
    logger.info("=== EquiSight V5 Live Pairs Production Engine ===")
    os.makedirs("database", exist_ok=True)
    
    db = ExecutionDatabase(DB_PATH)
    position_mgr = PositionManager(DB_PATH)
    defender = PortfolioDefender()
    alpha_engine = PairsAlphaEngine(PAIRS)
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # 1. Update existing positions
    logger.info("Updating active positions...")
    closed_today = position_mgr.update_positions()
    if closed_today:
        logger.info(f"Closed positions today: {closed_today}")
    
    # 2. Download market data
    today_str = datetime.today().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H:%M:%S')
    logger.info("Downloading market data...")
    
    raw_data = {}
    tickers = set()
    for p in PAIRS:
        tickers.add(p.stock_a)
        tickers.add(p.stock_b)
        
    for t in tickers:
        try:
            raw = yf.download(t, period="1y", progress=False, session=session)
            if not raw.empty:
                raw_data[t] = raw
        except Exception as e:
            logger.error(f"Failed to download data for {t}: {e}")
    
    # 3. Evaluate pairs & generate signals
    active_positions = position_mgr.get_active_positions()
    current_exposure = (active_positions['allocated_capital'].sum() / INITIAL_CAPITAL) if not active_positions.empty else 0.0
    
    daily_ledger = []
    
    for pair_cfg in PAIRS:
        pair_name = f"{pair_cfg.stock_a}-{pair_cfg.stock_b}"
        if pair_cfg.stock_a not in raw_data or pair_cfg.stock_b not in raw_data:
            continue
        
        latest_date = raw_data[pair_cfg.stock_a].index[-1]
        
        close_a_raw = raw_data[pair_cfg.stock_a].loc[latest_date]['Close']
        close_b_raw = raw_data[pair_cfg.stock_b].loc[latest_date]['Close']
        open_a = float(close_a_raw.iloc[0]) if isinstance(close_a_raw, pd.Series) else float(close_a_raw)
        open_b = float(close_b_raw.iloc[0]) if isinstance(close_b_raw, pd.Series) else float(close_b_raw)
        
        a_open = pair_cfg.stock_a in (active_positions['ticker'].tolist() if not active_positions.empty else [])
        b_open = pair_cfg.stock_b in (active_positions['ticker'].tolist() if not active_positions.empty else [])
        
        signal = alpha_engine.compute_signal(pair_cfg, raw_data, latest_date)
        
        ledger_row = {
            'Date': today_str,
            'Time': current_time_str,
            'Pair': pair_name,
            'Price_A': round(open_a, 2),
            'Price_B': round(open_b, 2),
            'Master_Signal': 'HOLD',
            'Z_Score': 'N/A',
            'Beta': 'N/A',
            'Reason': 'Spread within normal bounds'
        }
        
        if signal:
            ledger_row['Z_Score'] = round(signal['z_score'], 3)
            ledger_row['Beta'] = round(signal['beta'], 3)
            
            if a_open or b_open:
                ledger_row['Reason'] = 'Leg already active in portfolio'
            elif current_exposure + POSITION_SIZE > MAX_TOTAL_EXPOSURE:
                ledger_row['Reason'] = 'Exposure cap reached'
            else:
                ledger_row['Master_Signal'] = signal['action']
                ledger_row['Reason'] = f"Statistical edge confirmed (Z={signal['z_score']:.2f})"
                
                allocated = INITIAL_CAPITAL * POSITION_SIZE
                logger.info(f"{pair_name}: {signal['action']} at Z={signal['z_score']:.2f}, allocating ₹{allocated:,.2f}")
                
                position_mgr.open_new_position(
                    ticker=pair_name,
                    entry_date=today_str,
                    entry_price=(open_a + open_b) / 2.0,
                    allocated_capital=allocated
                )
                current_exposure += POSITION_SIZE
                
        daily_ledger.append(ledger_row)

    # 4. Save Master Signal Ledger
    if daily_ledger:
        ledger_df = pd.DataFrame(daily_ledger)
        ledger_file = "equisight_v5_ledger.csv"
        if os.path.exists(ledger_file):
            ledger_df.to_csv(ledger_file, mode='a', header=False, index=False)
        else:
            ledger_df.to_csv(ledger_file, index=False)
        logger.info("Master signal ledger updated.")

    # 5. Log Daily MTM Equity Curve
    log_daily_equity(today_str, raw_data, position_mgr)

    # 6. Export Database Tables to CSVs
    logger.info("Exporting execution database tables to CSVs...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            pd.read_sql_query("SELECT * FROM active_positions", conn).to_csv("active_positions.csv", index=False)
            pd.read_sql_query("SELECT * FROM trade_outcomes", conn).to_csv("trade_history.csv", index=False)
        logger.info("Database export complete.")
    except Exception as e:
        logger.error(f"Failed to export SQLite database to CSV: {e}")

    logger.info("=== Daily Run Complete ===")


if __name__ == "__main__":
    run_production_loop()