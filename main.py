import yfinance as yf
import requests
import numpy as np
from datetime import datetime
import warnings

# Import all the core V5 modules you just built
from src.data_pipeline import generate_v5_features
from src.macro_regime import MacroCircuitBreaker
from src.local_regime import LocalRegimeEngine
from src.q_agent import ContextQAgent
from src.execution_db import ExecutionDatabase

warnings.filterwarnings("ignore")

def run_production_loop():
    print("=== EquiSight V5 Production Orchestrator ===")
    
    # 1. Initialize the ecosystem
    db = ExecutionDatabase()
    macro_breaker = MacroCircuitBreaker()
    local_engine = LocalRegimeEngine()
    q_agent = ContextQAgent()
    
    nifty_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    
    # Create a custom browser session to bypass Yahoo's anti-bot block
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    
    # 2. Ingest Real Market Data
    print("[INFO] Downloading Nifty 50 Baseline Data (NIFTYBEES.NS)...")
    nifty_macro = yf.download("NIFTYBEES.NS", period="2y", progress=False, session=session)
    
    if nifty_macro.empty:
        print("\n[CRITICAL ERROR] Yahoo is still blocking the connection.")
        return
        
    macro_features = generate_v5_features(nifty_macro)
    
    # 3. Train Global Macro Circuit Breaker
    print("[INFO] Training Macro Circuit Breaker...")
    macro_breaker.train(macro_features)
    current_macro_state = macro_breaker.predict(macro_features).iloc[-1]
    
    # 4. Ingest and Train Local Tickers
    ticker_data = {}
    print("[INFO] Downloading Asset Data...")
    for ticker in nifty_tickers:
        raw_df = yf.download(ticker, period="2y", progress=False, session=session)
        ticker_data[ticker] = generate_v5_features(raw_df)
        
    print("[INFO] Parallel Training Local HMMs on multiple CPU cores...")
    local_engine.train_all_parallel(ticker_data)
    
    print("\n--- TODAY'S LIVE SIGNALS ---")
    for ticker, df in ticker_data.items():
        # Get today's localized regime
        local_state = local_engine.predict_ticker(ticker, df).iloc[-1]
        
        # Get dynamic risk cones to build the 9-State Grid
        latest = df.iloc[-1]
        
        # THE FIX: Force strict Python scalar types to prevent Pandas comparison crashes.
        # np.array(...).flatten()[0] ensures we always pass a single number to the Q-Agent.
        state_val = int(np.array(local_state).flatten()[0])
        ret_val = float(np.array(latest['log_returns']).flatten()[0])
        p10_val = float(np.array(latest['P10_cone']).flatten()[0])
        p50_val = float(np.array(latest['P50_cone']).flatten()[0])
        p90_val = float(np.array(latest['P90_cone']).flatten()[0])
        
        state_str = q_agent._get_state(
            state_val, ret_val, p10_val, p50_val, p90_val
        )
        
        # Agent decides based on the real-world state
        action = q_agent.get_action(state_str)
        reason = q_agent.generate_xai_log(state_str, action, current_macro_state)
        
        print(f"{ticker} -> {reason}")
        
        # If the agent buys AND the market isn't crashing, log it to the dashboard
        # Convert state values to integers for the database check
        macro_state_int = int(np.array(current_macro_state).flatten()[0])
        
        if action == 1 and macro_state_int != -1:
            db.log_trade_outcome(
                ticker=ticker, 
                entry_date=datetime.today().strftime('%Y-%m-%d'), 
                exit_date="PENDING", 
                actual_profit=0.0, 
                days_held=0, 
                reason=reason
            )
            
    print("\n[SUCCESS] Execution loop complete. Refresh your Dashboard to see the logs.")

if __name__ == "__main__":
    run_production_loop()
    