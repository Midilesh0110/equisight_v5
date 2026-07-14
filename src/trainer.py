import yfinance as yf
import requests
import numpy as np
import json
import os
import warnings
from tqdm import tqdm
import sys

# Anchor to project root
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
    
    # ---- 1. Download full 5-year data ----
    print("[INFO] Downloading 5 Years of Historical Data...")
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    
    nifty_raw = yf.download("NIFTYBEES.NS", period="5y", progress=False, session=session)
    macro_features = generate_v5_features(nifty_raw)
    
    ticker_data = {}
    for t in tickers:
        df_raw = yf.download(t, period="5y", progress=False, session=session)
        ticker_data[t] = generate_v5_features(df_raw)
    
    # Ensure all DataFrames share the same date range (intersection)
    common_index = macro_features.index
    for t in tickers:
        common_index = common_index.intersection(ticker_data[t].index)
    macro_features = macro_features.loc[common_index]
    for t in tickers:
        ticker_data[t] = ticker_data[t].loc[common_index]
    
    n_days = len(common_index)
    if n_days < 1000:
        print("[ERROR] Not enough data for meaningful training.")
        return
    
    # Split: last 252 days as hold-out test set, rest for training
    test_start = n_days - 252
    train_macro = macro_features.iloc[:test_start]
    test_macro = macro_features.iloc[test_start:]
    train_ticker = {t: ticker_data[t].iloc[:test_start] for t in tickers}
    test_ticker = {t: ticker_data[t].iloc[test_start:] for t in tickers}
    
    print(f"[INFO] Training period: {train_macro.index[0].date()} to {train_macro.index[-1].date()}")
    print(f"[INFO] Test period:     {test_macro.index[0].date()} to {test_macro.index[-1].date()}")
    
    # ---- 2. Walk-forward HMM training & regime labelling ----
    # We will retrain the HMMs every 252 days using only data up to that point.
    # The regime for a given day is predicted by the model that was most recently trained.
    
    retrain_freq = 252
    initial_min_days = 504  # need at least 2 years before first retrain
    
    # Containers for regime labels (to be filled day-by-day)
    macro_regimes_train = np.full(len(train_macro), np.nan)
    local_regimes_train = {t: np.full(len(train_ticker[t]), np.nan) for t in tickers}
    
    last_retrain_idx = -1
    macro_model = None
    local_models = {t: None for t in tickers}
    
    # Helper to retrain HMMs using data up to index 'end_idx' (exclusive)
    def retrain_hmms(end_idx):
        nonlocal macro_model, local_models
        # Macro
        macro_breaker = MacroCircuitBreaker()
        macro_breaker.train(train_macro.iloc[:end_idx])
        macro_model = macro_breaker
        # Local tickers
        local_eng = LocalRegimeEngine()
        ticker_dict = {t: train_ticker[t].iloc[:end_idx] for t in tickers}
        local_eng.train_all_parallel(ticker_dict)
        for t in tickers:
            # load the model into memory for prediction
            local_models[t] = local_eng  # we'll use its predict_ticker method
        # Store the engine for prediction later
        return macro_breaker, local_eng
    
    # Walk forward through training days
    local_eng_instance = None
    for i in tqdm(range(initial_min_days, len(train_macro)), desc="Walk-forward HMM labelling"):
        # If we crossed a retrain boundary, retrain on all data up to i (i exclusive for training)
        if i % retrain_freq == 0 or i == initial_min_days:
            print(f"\n[RETRAIN] Retraining HMMs using data up to day {i} ({train_macro.index[i-1].date()})...")
            macro_breaker, local_eng_instance = retrain_hmms(i)
            last_retrain_idx = i
        
        # Predict macro regime for day i using current model
        # Use only data up to day i to predict (the model already knows only past data)
        macro_pred = macro_breaker.predict(train_macro.iloc[[i]])  # single row
        macro_regimes_train[i] = int(np.array(macro_pred.iloc[0]).flatten()[0])
        
        # Predict local regimes for each ticker for day i
        for t in tickers:
            local_pred = local_eng_instance.predict_ticker(t, train_ticker[t].iloc[[i]])
            local_regimes_train[t][i] = int(np.array(local_pred.iloc[0]).flatten()[0])
    
    # Now macro_regimes_train and local_regimes_train are filled from day 504 onward.
    # We'll slice off the first 504 days because we don't have reliable regime labels there.
    start_train = initial_min_days
    train_macro_trim = train_macro.iloc[start_train:]
    macro_regimes_trim = macro_regimes_train[start_train:]
    train_ticker_trim = {t: train_ticker[t].iloc[start_train:] for t in tickers}
    local_regimes_trim = {t: local_regimes_train[t][start_train:] for t in tickers}
    
    # ---- 3. Q-Learning on the training period ----
    agent = ContextQAgent(epsilon=1.0)
    epsilon_decay = 0.9995
    min_epsilon = 0.05
    
    print("\n[INFO] Q-Learning on Walk-Forward Labelled Data...")
    for ticker in tickers:
        print(f"\nTraining agent on {ticker}...")
        df = train_ticker_trim[ticker]
        local_states = local_regimes_trim[ticker]
        # local_states may contain NaN for days we couldn't predict; skip those
        valid_mask = ~np.isnan(local_states)
        df = df[valid_mask]
        local_states = local_states[valid_mask]
        
        # Re-index to align with macro
        common = df.index.intersection(train_macro_trim.index)
        df = df.loc[common]
        local_states = local_states[common]
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
    print(f"\n[SUCCESS] Q-table saved. States mastered: {len(agent.q_table)}")
    
    # ---- 4. Test on hold-out period (pure exploitation) ----
    print("\n[INFO] Testing trained agent on the last year (out-of-sample)...")
    agent.epsilon = 0.0  # no exploration
    
    # For the test period, we need to simulate HMM regime predictions using the
    # model trained on all training data (which we already have from the last retrain).
    # Actually, we can use the final models (macro_breaker, local_eng_instance) to predict
    # the test data. That's acceptable because those models were trained on data up to
    # the end of training, and the test period is after that — still no future info.
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
    avg_reward = total_reward / (len(test_macro) * len(tickers)) if len(test_macro) > 0 else 0
    print(f"[METRIC] Out-of-sample: Total reward {total_reward:.4f}, Trades executed {trades}, Avg reward per step {avg_reward:.6f}")
    print("[INFO] If avg reward is positive after friction, the edge may be real.")

if __name__ == "__main__":
    run_historical_training()