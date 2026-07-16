# main.py — EquiSight V5 Live Multi‑Pair Statistical Arbitrage
import yfinance as yf
import requests
import numpy as np
import os
import warnings
from datetime import datetime
import pandas as pd
import logging
import os
os.makedirs("logs", exist_ok=True)
os.makedirs("database", exist_ok=True)

from src.data_pipeline import generate_v5_features
from src.execution_db import ExecutionDatabase
from src.portfolio_risk import PortfolioDefender
from src.position_manager import PositionManager
from src.alpha_engine import PairsAlphaEngine, PairConfig

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PAIRS = [
    PairConfig("TCS.NS", "INFY.NS"),
    PairConfig("ICICIBANK.NS", "SBIN.NS"),
    PairConfig("RELIANCE.NS", "HINDUNILVR.NS"),
    PairConfig("KOTAKBANK.NS", "BAJFINANCE.NS")
]
DB_PATH = "database/equisight_v5.db"
POSITION_SIZE = 0.05
MAX_TOTAL_EXPOSURE = 0.25

def run_production_loop():
    logger.info("=== EquiSight V5 Live Pairs Engine ===")
    
    db = ExecutionDatabase(DB_PATH)
    position_mgr = PositionManager(DB_PATH)
    defender = PortfolioDefender()
    alpha_engine = PairsAlphaEngine(PAIRS)
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # 1. Update existing positions
    logger.info("Updating active positions...")
    closed = position_mgr.update_positions()
    if closed:
        logger.info(f"Closed: {closed}")
    
    # 2. Download latest data
    today_str = datetime.today().strftime('%Y-%m-%d')
    logger.info("Downloading data...")
    raw_data = {}
    tickers = set()
    for p in PAIRS:
        tickers.add(p.stock_a)
        tickers.add(p.stock_b)
    for t in tickers:
        raw = yf.download(t, period="1y", progress=False, session=session)
        if not raw.empty:
            raw_data[t] = raw
    
    # 3. Generate pair signals
    active_positions = position_mgr.get_active_positions()
    existing_pairs = set()
    if not active_positions.empty:
        # We store pairs as "TICKER_A-TICKER_B" in active_positions; extract them
        for _, pos in active_positions.iterrows():
            # In live mode we store single‑ticker positions; for pairs we'd need a different table.
            # For now, we simply don't allow overlapping tickers.
            pass
    
    # Compute current exposure (approximate)
    current_exposure = active_positions['allocated_capital'].sum() / 1_000_000 if not active_positions.empty else 0
    
    for pair_cfg in PAIRS:
        pair_name = f"{pair_cfg.stock_a}-{pair_cfg.stock_b}"
        if pair_name in existing_pairs:
            continue
        
        # Check if either leg already has an open position
        a_open = pair_cfg.stock_a in (active_positions['ticker'].tolist() if not active_positions.empty else [])
        b_open = pair_cfg.stock_b in (active_positions['ticker'].tolist() if not active_positions.empty else [])
        if a_open or b_open:
            continue
        
        if pair_cfg.stock_a not in raw_data or pair_cfg.stock_b not in raw_data:
            continue
        
        latest_date = raw_data[pair_cfg.stock_a].index[-1]
        try:
            signal = alpha_engine.compute_signal(pair_cfg, raw_data, latest_date)
        except Exception as e:
            logger.warning(f"{pair_name}: signal computation failed – {e}")
            continue
        
        if signal:
            # ... rest unchanged (exposure cap, logging, etc.)
            # Apply exposure cap
            if current_exposure + POSITION_SIZE > MAX_TOTAL_EXPOSURE:
                logger.info(f"{pair_name}: signal suppressed (exposure cap)")
                continue
            
            # In live mode, we'd open both legs via the broker. For paper trading,
            # we log the pair as a single combined position.
            open_a = float(raw_data[pair_cfg.stock_a].loc[latest_date]['Close'])
            open_b = float(raw_data[pair_cfg.stock_b].loc[latest_date]['Close'])
            allocated = 1_000_000 * POSITION_SIZE
            
            logger.info(f"{pair_name}: {signal['action']} at Z={signal['z_score']:.2f}, allocating ₹{allocated:.2f}")
            
            # Log pair as a single entry (we store it as ticker_a-ticker_b)
            position_mgr.open_new_position(
                ticker=pair_name,
                entry_date=today_str,
                entry_price=(open_a + open_b) / 2,  # notional average for tracking
                allocated_capital=allocated
            )
            current_exposure += POSITION_SIZE
    
    logger.info("Daily run complete.")

if __name__ == "__main__":
    run_production_loop()