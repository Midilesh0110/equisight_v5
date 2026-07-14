import pandas as pd
import numpy as np

def generate_v5_features(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Ingests clean OHLCV data and appends standardized EquiSight V5 features.
    Strictly enforces trailing calculations to prevent data leakage.
    """
    data = df.copy()
    
    # 1. Calculate Stationary Log Returns
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 2. Continuous Intraday Volatility Proxy (Normalized Range Percentage)
    data['range_pct'] = (data['High'] - data['Low']) / data['Close']
    
    # 3. Dynamic Quantile Risk Cones (Probabilistic Return Boundaries)
    # CRITICAL FIX: .shift(1) ensures today's risk cone is calculated 
    # strictly using data up to yesterday's close. Zero look-ahead bias.
    data['P10_cone'] = data['log_returns'].shift(1).rolling(window=lookback).quantile(0.10)
    data['P50_cone'] = data['log_returns'].shift(1).rolling(window=lookback).quantile(0.50)
    data['P90_cone'] = data['log_returns'].shift(1).rolling(window=lookback).quantile(0.90)
    
    # Drop initial rolling NaNs cleanly
    return data.dropna()

if __name__ == "__main__":
    print("[INFO] Leak-Proof Data Pipeline initialized.")