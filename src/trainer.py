# src/trainer.py
import yfinance as yf
import requests
import numpy as np
import json
import os
import warnings
from tqdm import tqdm
import sys
import pandas as pd

# Make sure we can import src modules from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import generate_v5_features
from src.macro_regime import MacroCircuitBreaker
from src.local_regime import LocalRegimeEngine
from src.q_agent import ContextQAgent

warnings.filterwarnings("ignore")


def run_historical_training():
    print("=== EquiSight V5 Historical Training Gym (Walk-Forward HMMs) ===")
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # ------------------------------------------------------------------
    # 1. Download full 5-year data
    # ------------------------------------------------------------------
    print("[INFO] Downloading 5 Years of Historical Data...")
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS","HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS"]   # <-- your active universe
    
    # Nifty proxy
    nifty_raw = yf.download("NIFTYBEES.NS", period="5y", progress=False, session=session)
    if nifty_raw.empty:
        print("[ERROR] NIFTYBEES.NS download failed.")
        return
    macro_features = generate_v5_features(nifty_raw)
    print(f"[INFO] Macro features: {len(macro_features)} rows")

    ticker_data = {}
    for t in tickers:
        df_raw = yf.download(t, period="max", progress=False, session=session)
        if df_raw.empty:
            print(f"[ERROR] {t} download empty. Exiting.")
            return
        ticker_data[t] = generate_v5_features(df_raw)
        print(f"[INFO] {t}: {len(ticker_data[t])} rows after feature gen")
    
    # ------------------------------------------------------------------
    # 2. Align all data to the macro index (robust common mask)
    # ------------------------------------------------------------------
    master_idx = macro_features.index
    macro_aligned = macro_features  # keep as is
    
    # Reindex each ticker to master_idx; missing dates become NaN
    ticker_aligned = {}
    for t, df in ticker_data.items():
        ticker_aligned[t] = df.reindex(master_idx)
    
    # Build a DataFrame of log_returns to find valid (non‑NaN) dates for all tickers
    logret_df = pd.DataFrame({t: ticker_aligned[t]['log_returns'] for t in tickers})
    valid_mask = logret_df.notna().all(axis=1)
    common_days = valid_mask.sum()
    print(f"[INFO] Common valid days (no missing ticker data): {common_days}")

    # Apply the mask to all DataFrames
    macro_aligned = macro_aligned.loc[valid_mask]
    for t in tickers:
        ticker_aligned[t] = ticker_aligned[t].loc[valid_mask]
    
    n_days = len(macro_aligned)
    if n_days < 750:
        print(f"[ERROR] Not enough data for meaningful training: {n_days} common days. Need at least 1000.")
        print("Check if any ticker has large gaps in its history. You may need a shorter period or different tickers.")
        return
    print(f"[INFO] Total aligned days: {n_days}")

    # Split: last 252 days as hold-out test set
    test_start = n_days - 252
    train_macro = macro_aligned.iloc[:test_start]
    test_macro  = macro_aligned.iloc[test_start:]
    train_ticker = {t: ticker_aligned[t].iloc[:test_start] for t in tickers}
    test_ticker  = {t: ticker_aligned[t].iloc[test_start:] for t in tickers}
    
    print(f"[INFO] Training period: {train_macro.index[0].date()} to {train_macro.index[-1].date()} ({len(train_macro)} days)")
    print(f"[INFO] Test period:     {test_macro.index[0].date()} to {test_macro.index[-1].date()} ({len(test_macro)} days)")

    # ------------------------------------------------------------------
    # 3. Walk-forward HMM training & regime labelling
    # ------------------------------------------------------------------
    # We will retrain HMMs every 252 trading days using only data up to that point.
    retrain_freq = 126
    initial_min_days = 504  # need at least 2 years for first reliable training

    # Containers for regime labels (filled day-by-day)
    macro_regimes_train = np.full(len(train_macro), np.nan)
    local_regimes_train = {t: np.full(len(train_ticker[t]), np.nan) for t in tickers}

    macro_model = None
    local_models = {t: None for t in tickers}
    local_eng_instance = None

    def retrain_hmms(end_idx):
        nonlocal macro_model, local_eng_instance
        print(f"\n[RETRAIN] Training HMMs on data up to day {end_idx} ({train_macro.index[end_idx-1].date()})...")
        # Macro
        macro_breaker = MacroCircuitBreaker()
        macro_breaker.train(train_macro.iloc[:end_idx])
        macro_model = macro_breaker
        # Local tickers
        local_eng = LocalRegimeEngine()
        ticker_dict = {t: train_ticker[t].iloc[:end_idx] for t in tickers}
        local_eng.train_all_parallel(ticker_dict)
        local_eng_instance = local_eng
        return macro_breaker, local_eng

    # Walk forward through training days
    for i in tqdm(range(initial_min_days, len(train_macro)), desc="Walk-forward HMM labelling"):
        if i % retrain_freq == 0 or i == initial_min_days:
            macro_breaker, local_eng_instance = retrain_hmms(i)
        
        # Predict macro regime for day i
        macro_pred = macro_breaker.predict(train_macro.iloc[[i]])
        macro_regimes_train[i] = int(np.array(macro_pred.iloc[0]).flatten()[0])
        
        # Predict local regimes for each ticker for day i
        for t in tickers:
            local_pred = local_eng_instance.predict_ticker(t, train_ticker[t].iloc[[i]])
            local_regimes_train[t][i] = int(np.array(local_pred.iloc[0]).flatten()[0])
    
    # Slice off the initial burn-in period
    start_train = initial_min_days
    train_macro_trim = train_macro.iloc[start_train:]
    macro_regimes_trim = macro_regimes_train[start_train:]
    train_ticker_trim = {t: train_ticker[t].iloc[start_train:] for t in tickers}
    local_regimes_trim = {t: local_regimes_train[t][start_train:] for t in tickers}

    # ------------------------------------------------------------------
    # 4. Q-Learning on the walk-forward labelled data
    # ------------------------------------------------------------------
    agent = ContextQAgent(epsilon=1.0)
    epsilon_decay = 0.9995
    min_epsilon = 0.05

    print("\n[INFO] Q-Learning on Walk-Forward Labelled Data...")
    for ticker in tickers:
        print(f"\nTraining agent on {ticker}...")
        df = train_ticker_trim[ticker]
        local_states = local_regimes_trim[ticker]
        
                # Remove any remaining NaN rows
        valid_mask = ~np.isnan(local_states)
        df = df[valid_mask]
        local_states = local_states[valid_mask]
        
        # ADD THIS LINE to make local_states index-aware
        local_states = pd.Series(local_states, index=df.index)
        
        # Align with macro index
        common = df.index.intersection(train_macro_trim.index)
        df = df.loc[common]
        local_states = local_states[common]          # now works as a Series
        macro_states = pd.Series(macro_regimes_trim, index=train_macro_trim.index).loc[common]
        
        for i in tqdm(range(len(df) - 1), desc=f"{ticker} Q-learning"):
            current_local = int(local_states.iloc[i])
            row = df.iloc[i]
            ret_val = float(row['log_returns'])
            p10 = float(row['P10_cone'])
            p50 = float(row['P50_cone'])
            p90 = float(row['P90_cone'])
            state_str = agent._get_state(current_local, ret_val, p10, p50, p90)
            
            action = agent.get_action(state_str)
            
            next_row = df.iloc[i+1]
            actual_return = float(next_row['log_returns'])
            reward = agent.calculate_reward(action, actual_return)
            
            next_local = int(local_states.iloc[i+1])
            next_state_str = agent._get_state(
                next_local,
                float(next_row['log_returns']),
                float(next_row['P10_cone']),
                float(next_row['P50_cone']),
                float(next_row['P90_cone'])
            )
            
            agent.update_q_table(state_str, action, reward, next_state_str)
            
            if agent.epsilon > min_epsilon:
                agent.epsilon *= epsilon_decay
    
    # Save the trained Q-table
    os.makedirs("database/models", exist_ok=True)
    with open("database/models/q_table.json", "w") as f:
        json.dump(agent.q_table, f, indent=4)
    print(f"\n[SUCCESS] Q-table saved to database/models/q_table.json")
    print(f"[METRIC] Total Unique Market States Mastered: {len(agent.q_table)}")
    
    # ------------------------------------------------------------------
    # 5. Out-of-sample test on the last year
    # ------------------------------------------------------------------
    print("\n[INFO] Testing trained agent on the hold-out year (out-of-sample)...")
    agent.epsilon = 0.0  # pure exploitation
    
    # Use the last trained models (from the end of training period) to predict test data
    # These models saw data only up to the end of training, so no future info.
    test_macro_regimes = macro_breaker.predict(test_macro)
    test_macro_regimes = [int(np.array(x).flatten()[0]) for x in test_macro_regimes]
    
    test_local_regimes = {}
    for t in tickers:
        preds = local_eng_instance.predict_ticker(t, test_ticker[t])
        test_local_regimes[t] = [int(np.array(x).flatten()[0]) for x in preds]
    
    total_reward = 0
    trades = 0
    for ticker in tickers:
        df = test_ticker[ticker]
        local_states = test_local_regimes[ticker]
        for i in range(len(df) - 1):
            row = df.iloc[i]
            state_str = agent._get_state(
                local_states[i],
                float(row['log_returns']),
                float(row['P10_cone']),
                float(row['P50_cone']),
                float(row['P90_cone'])
            )
            action = agent.get_action(state_str)
            next_return = float(df.iloc[i+1]['log_returns'])
            reward = agent.calculate_reward(action, next_return)
            total_reward += reward
            if action == 1:
                trades += 1
    avg_reward = total_reward / (len(test_macro) * len(tickers))
    print(f"[METRIC] Out-of-sample: Total reward {total_reward:.4f}, Trades executed {trades}, Avg reward per step {avg_reward:.6f}")
    print("[INFO] If avg reward is positive after friction, the edge may be real.")


if __name__ == "__main__":
    run_historical_training()