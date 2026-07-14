import pandas as pd
import numpy as np
from typing import List, Tuple
import data_pipeline as dp 

def walk_forward_split(df: pd.DataFrame, train_size: int = 252, test_size: int = 63, purge_size: int = 30) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Slices the feature matrix into chronological rolling windows.
    Includes an Embedded Purge Window to allow HMM state probabilities to stabilize before testing.
    """
    splits = []
    total_len = len(df)
    
    # Slide the validation window forward without data leakage
    for i in range(0, total_len - train_size - test_size + 1, test_size):
        train_start = i
        train_end = i + train_size
        test_end = train_end + test_size
        
        train_set = df.iloc[train_start:train_end]
        test_set = df.iloc[train_end:test_end]
        
        # The evaluation set drops the first 'purge_size' days for HMM burn-in
        eval_set = test_set.iloc[purge_size:] 
        
        splits.append((train_set, eval_set))
        
    return splits

if __name__ == "__main__":
    print("[INFO] Testing Phase 1: Data Pipeline & Purged Walk-Forward Slicer...")
    
    # Generate dummy market data to test the mathematical logic
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="B")
    np.random.seed(42)
    dummy_data = pd.DataFrame({
        'Open': np.random.uniform(100, 105, 1000),
        'High': np.random.uniform(105, 110, 1000),
        'Low': np.random.uniform(95, 100, 1000),
        'Close': np.random.uniform(100, 105, 1000),
        'Volume': np.random.randint(1000, 5000, 1000)
    }, index=dates)
    
    # 1. Test Feature Engineering
    features_df = dp.generate_v5_features(dummy_data, lookback=252)
    print(f"[SUCCESS] Features Generated. Rows after dropping NaNs: {features_df.shape[0]}")
    
    # 2. Test Walk-Forward Slicer with Purging
    splits = walk_forward_split(features_df, train_size=252, test_size=63, purge_size=30)
    print(f"[SUCCESS] Purged Walk-Forward Splits created: {len(splits)} rolling windows.")
    
    if len(splits) > 0:
        print(f"[INFO] Window 1 -> Train shape: {splits[0][0].shape} | Eval shape (Test minus 30-day Purge): {splits[0][1].shape}")
        
    print("[INFO] Phase 1 is complete, purged, and error-free.")