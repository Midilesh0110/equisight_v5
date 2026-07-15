# main.py (Enhanced Production Orchestrator)
import yfinance as yf
import requests
import numpy as np
import json
import os
import warnings
from datetime import datetime, timedelta
import pandas as pd
import logging

from src.data_pipeline import generate_v5_features
from src.macro_regime import MacroCircuitBreaker
from src.local_regime import LocalRegimeEngine
from src.q_agent import ContextQAgent
from src.execution_db import ExecutionDatabase
from src.portfolio_risk import PortfolioDefender
from src.position_manager import PositionManager

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Configuration (you can move to a config file later) ----------
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
MACRO_INDEX = "NIFTYBEES.NS"
DB_PATH = "database/equisight_v5.db"
Q_TABLE_PATH = "database/models/q_table.json"
MODEL_DIR_MACRO = "database/models"
MODEL_DIR_LOCAL = "database/models/local"
WEEKLY_RETRAIN_DAY = 6  # Sunday (0=Mon, 6=Sun)

def load_q_table(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            q_table = json.load(f)
        logger.info(f"Loaded Q-table with {len(q_table)} states.")
        return q_table
    else:
        logger.warning("Q-table not found. Agent will start fresh (no prior learning). Run trainer.py first!")
        return {}

def should_retrain_hmm():
    """Check if today is the weekly retrain day or if models don't exist."""
    if not os.path.exists(os.path.join(MODEL_DIR_MACRO, "macro_nifty_hmm.pkl")):
        return True
    today = datetime.today()
    if today.weekday() == WEEKLY_RETRAIN_DAY:
        return True
    return False

def run_production_loop():
    logger.info("=== EquiSight V5 Daily Orchestrator ===")
    
    # 1. Initialize components
    db = ExecutionDatabase(DB_PATH)
    defender = PortfolioDefender()
    position_mgr = PositionManager(DB_PATH, vol_threshold=defender.vol_threshold, gap_threshold=defender.gap_threshold)
    agent = ContextQAgent(epsilon=0.0)  # exploit only
    agent.q_table = load_q_table(Q_TABLE_PATH)
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # 2. Update existing positions (force close if needed)
    logger.info("Updating active positions...")
    closed_today = position_mgr.update_positions()
    if closed_today:
        logger.info(f"Positions closed today: {closed_today}")
    
    # 3. Download latest market data
    today_str = datetime.today().strftime('%Y-%m-%d')
    logger.info("Downloading market data...")
    try:
        macro_raw = yf.download(MACRO_INDEX, period="2y", progress=False, session=session)
        if macro_raw.empty:
            raise ValueError("No data for macro index")
        ticker_data = {}
        for t in TICKERS:
            raw = yf.download(t, period="2y", progress=False, session=session)
            if raw.empty:
                logger.warning(f"No data for {t}, skipping.")
                continue
            ticker_data[t] = raw
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        return

    # Generate features
    macro_features = generate_v5_features(macro_raw)
    ticker_features = {t: generate_v5_features(df) for t, df in ticker_data.items()}
    
    # 4. Handle HMM training / loading
    macro_breaker = MacroCircuitBreaker()
    local_engine = LocalRegimeEngine()
    
    if should_retrain_hmm():
        logger.info("Weekly retrain or first run. Training HMMs...")
        macro_breaker.train(macro_features)
        local_engine.train_all_parallel(ticker_features)
    else:
        # Models should exist; we'll load them via predict (which loads from disk)
        logger.info("Using existing HMM models from disk.")
        # We still need to instantiate to call predict; the predict method loads the saved model.
    
    # Predict current macro regime (latest day)
    # The predict method will load the saved model if available; else we'll train.
    try:
        current_macro_state = macro_breaker.predict(macro_features.tail(1)).iloc[0]
    except FileNotFoundError:
        logger.warning("Macro model missing, training now...")
        macro_breaker.train(macro_features)
        current_macro_state = macro_breaker.predict(macro_features.tail(1)).iloc[0]
    logger.info(f"Macro regime: {current_macro_state}")
    
    # 5. Get active positions list for correlation overlay
    active_positions = position_mgr.get_active_positions()
    existing_tickers = active_positions['ticker'].tolist() if not active_positions.empty else []
    
    # Build a DataFrame of log returns for correlation (last 30 days)
    # We need log returns for all tickers (new + existing)
    returns_dict = {}
    for t, df in ticker_features.items():
        returns_dict[t] = df['log_returns']
    # Add any active tickers that might not be in today's download? They should be in TICKERS, but just in case.
    for et in existing_tickers:
        if et not in returns_dict:
            # download minimal data for correlation if missing
            try:
                et_raw = yf.download(et, period="1mo", progress=False, session=session)
                et_feat = generate_v5_features(et_raw)
                returns_dict[et] = et_feat['log_returns']
            except:
                pass
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    # 6. Generate signals
    logger.info("\n--- TODAY'S SIGNALS ---")
    for ticker, df in ticker_features.items():
        try:
            local_state_series = local_engine.predict_ticker(ticker, df)
            local_state = local_state_series.iloc[-1]
            latest = df.iloc[-1]
            # Scalar extraction
            state_val = int(np.array(local_state).flatten()[0])
            ret_val = float(np.array(latest['log_returns']).flatten()[0])
            p10 = float(np.array(latest['P10_cone']).flatten()[0])
            p50 = float(np.array(latest['P50_cone']).flatten()[0])
            p90 = float(np.array(latest['P90_cone']).flatten()[0])
            
            state_str = agent._get_state(state_val, ret_val, p10, p50, p90)
            action = agent.get_action(state_str)
            reason = agent.generate_xai_log(state_str, action, current_macro_state)
            
            # Apply overrides and filters before entering
            if action == 1:
                # Macro breaker override (already in reason string, but double-check)
                macro_state_int = int(np.array(current_macro_state).flatten()[0])
                if macro_state_int == -1:
                    logger.info(f"{ticker}: BUY suppressed by macro breaker.")
                    continue
                
                # Volatility gatekeeper
                range_pct = float(np.array(latest['range_pct']).flatten()[0])
                if defender.check_volatility_gatekeeper(range_pct):
                    logger.info(f"{ticker}: BUY suppressed by volatility gatekeeper (range {range_pct*100:.1f}%).")
                    continue
                
                # Overnight gap check for tomorrow's open? We can't know open yet. 
                # The gap filter is applied the next day in position manager. We'll still log.
                
                # Correlation overlay
                scale = defender.check_correlation_overlay(ticker, existing_tickers, returns_df)
                if scale == 0.0:
                    logger.info(f"{ticker}: BUY skipped due to extreme correlation.")
                    continue
                elif scale < 1.0:
                    logger.info(f"{ticker}: Position scaled down (correlation).")
                
                # Kelly sizing: simplified fixed fraction (you can later implement the EV-based from trade history)
                # For now, use a fixed 10% of capital or half-kelly from historical stats.
                allocated_capital = 10000  # placeholder; you should compute based on portfolio equity
                
                # Open the position (will be executed at next day's open, but we record entry as today)
                # In a real system, you'd place a buy order for tomorrow's open.
                position_mgr.open_new_position(
                    ticker=ticker,
                    entry_date=today_str,  # will be T+1 open in practice; we can adjust later
                    entry_price=latest['Close'],  # using today's close as proxy for tomorrow's open
                    allocated_capital=allocated_capital
                )
                existing_tickers.append(ticker)  # update for next iterations
                
                # Log to dashboard
                db.log_trade_outcome(
                    ticker=ticker,
                    entry_date=today_str,
                    exit_date="PENDING",
                    actual_profit=0.0,
                    days_held=0,
                    reason=reason
                )
            
            logger.info(f"{ticker} -> {reason}")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
    
    logger.info("\n[SUCCESS] Daily run complete. Check dashboard.")

if __name__ == "__main__":
    run_production_loop()