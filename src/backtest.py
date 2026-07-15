# src/backtest.py — EquiSight V5 with Random Forest Adaptive Entry
import yfinance as yf
import requests
import numpy as np
import pandas as pd
import os
import warnings
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import generate_v5_features
from src.macro_regime import MacroCircuitBreaker
from src.local_regime import LocalRegimeEngine
from src.execution_db import ExecutionDatabase
from src.portfolio_risk import PortfolioDefender
from src.walk_forward import walk_forward_split

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS"
]
MACRO_INDEX = "NIFTYBEES.NS"
DB_PATH = "database/equisight_v5.db"
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE = 0.05            # 5% of equity per trade
HOLD_DAYS = 5
TRANSACTION_COST = 0.0015

# -------------------------------------------------------------------
# Feature builder for the ML model
# -------------------------------------------------------------------
def build_features(macro_regime, local_regime, log_return, p10, p50, p90, range_pct, momentum_5d):
    """
    Returns a 1D array of features used by the Random Forest.
    The features are exactly the context information available at decision time.
    """
    return np.array([
        macro_regime,
        local_regime,
        log_return,
        log_return - p10,   # distance to lower cone
        p50,
        p90,
        range_pct,
        momentum_5d
    ])

# -------------------------------------------------------------------
# Main backtest engine
# -------------------------------------------------------------------
def run_backtest():
    print("=== EquiSight V5 Adaptive Walk-Forward Backtest (Random Forest) ===")
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # 1. Download data
    print("[INFO] Downloading 5+ years of data...")
    nifty_raw = yf.download(MACRO_INDEX, period="5y", progress=False, session=session)
    macro_features = generate_v5_features(nifty_raw)
    print(f"Macro rows: {len(macro_features)}")
    
    ticker_data = {}
    for t in TICKERS:
        raw = yf.download(t, period="5y", progress=False, session=session)
        if raw.empty:
            print(f"WARNING: {t} empty, skipping.")
            continue
        ticker_data[t] = generate_v5_features(raw)
    
    # Align to macro index
    master_idx = macro_features.index
    macro_aligned = macro_features
    ticker_aligned = {}
    for t, df in ticker_data.items():
        ticker_aligned[t] = df.reindex(master_idx)
    valid_mask = pd.DataFrame({t: ticker_aligned[t]['log_returns'].notna() for t in ticker_aligned}).all(axis=1)
    macro_aligned = macro_aligned.loc[valid_mask]
    for t in ticker_aligned:
        ticker_aligned[t] = ticker_aligned[t].loc[valid_mask]
    
    if len(macro_aligned) < 750:
        print(f"ERROR: Only {len(macro_aligned)} common days, need at least 750.")
        return
    
    splits = walk_forward_split(macro_aligned, train_size=504, test_size=126, purge_size=30)
    print(f"[INFO] Using {len(splits)} walk-forward folds.")
    
    db = ExecutionDatabase(DB_PATH)
    with db._get_connection() as conn:
        conn.execute("DELETE FROM trade_outcomes")
        conn.execute("DELETE FROM active_positions")
        conn.commit()
    
    all_fold_metrics = []
    
    for fold_idx, (train_macro, test_macro) in enumerate(splits):
        print(f"\n{'='*40} FOLD {fold_idx+1}/{len(splits)} {'='*40}")
        print(f"Train: {train_macro.index[0].date()} to {train_macro.index[-1].date()} ({len(train_macro)} days)")
        print(f"Test : {test_macro.index[0].date()} to {test_macro.index[-1].date()} ({len(test_macro)} days)")
        
        train_ticker = {}
        for t in ticker_aligned:
            df = ticker_aligned[t].loc[train_macro.index.intersection(ticker_aligned[t].index)]
            if len(df) > 0:
                train_ticker[t] = df
        
        # Train HMMs on training data (for regime labels, not for the ML model directly)
        macro_breaker = MacroCircuitBreaker()
        local_engine = LocalRegimeEngine()
        print("[INFO] Training fold HMMs...")
        macro_breaker.train(train_macro)
        local_engine.train_all_parallel(train_ticker)
        
        # ------ Build training data for Random Forest ------
        print("[INFO] Building training data for Random Forest...")
        X_train = []
        y_train = []
        # We'll use the training portion of each ticker to generate examples
        for ticker in ticker_aligned:
            df = ticker_aligned[ticker].loc[train_macro.index.intersection(ticker_aligned[ticker].index)]
            if len(df) < 50:
                continue
            # Get HMM regime predictions for this ticker's training data
            local_regimes_train = local_engine.predict_ticker(ticker, df)
            macro_regimes_train = macro_breaker.predict(train_macro.reindex(df.index))
            for i in range(len(df) - HOLD_DAYS - 1):
                # Feature at day i
                row = df.iloc[i]
                if row.isna().any():
                    continue
                macro_reg = int(macro_regimes_train.iloc[i])
                local_reg = int(local_regimes_train.iloc[i])
                ret = float(row['log_returns'])
                p10 = float(row['P10_cone'])
                p50 = float(row['P50_cone'])
                p90 = float(row['P90_cone'])
                range_pct = float(row['range_pct'])
                # 5-day momentum
                mom = df['log_returns'].iloc[max(0,i-4):i+1].sum()
                # Target: profitability of a 5-day hold from T+1 open to T+5 close
                entry_price = float(row['Close'])  # today's close as proxy for tomorrow's open
                # Use the actual OHLCV raw data to simulate exit
                exit_idx = i + HOLD_DAYS
                if exit_idx >= len(df):
                    continue
                exit_row = df.iloc[exit_idx]
                exit_price = float(exit_row['Close'])
                profit = (exit_price - entry_price) / entry_price
                target = 1 if profit > TRANSACTION_COST else 0
                
                features = build_features(macro_reg, local_reg, ret, p10, p50, p90, range_pct, mom)
                X_train.append(features)
                y_train.append(target)
        
        if len(X_train) < 100:
            print("[WARNING] Not enough training samples for Random Forest; skipping fold.")
            continue
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(f"[INFO] Training Random Forest on {len(X_train)} samples...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # ------ Simulate trading on test period ------
        raw_data = {}
        for t in TICKERS:
            raw = yf.download(t, period="5y", progress=False, session=session)
            raw_data[t] = raw.reindex(master_idx)
        
        defender = PortfolioDefender()
        equity_curve = [INITIAL_CAPITAL]
        cash = INITIAL_CAPITAL
        active_positions = {}
        trade_log = []
        pending_entries = {}
        
        print("[INFO] Simulating trading on test period...")
        for day_idx, date in enumerate(tqdm(test_macro.index, desc="Simulating")):
            today_macro_feat = test_macro.loc[date]
            macro_reg = macro_breaker.predict(today_macro_feat.to_frame().T).iloc[0]
            macro_reg = int(macro_reg)
            
            # 1. Update active positions (same as before)
            closed_ticks = []
            for ticker, pos in list(active_positions.items()):
                if date not in raw_data[ticker].index:
                    continue
                bar = raw_data[ticker].loc[date]
                open_p, high, low, close = float(bar['Open']), float(bar['High']), float(bar['Low']), float(bar['Close'])
                
                prev_idx = raw_data[ticker].index.get_loc(date) - 1
                if prev_idx >= 0:
                    prev_date = raw_data[ticker].index[prev_idx]
                    prev_close = float(raw_data[ticker].loc[prev_date]['Close'])
                    gap = (open_p - prev_close) / prev_close
                    if abs(gap) > defender.gap_threshold:
                        profit = (open_p - pos['entry_price']) / pos['entry_price']
                        trade_log.append({
                            'ticker': ticker, 'entry_date': pos['entry_date'], 'exit_date': date,
                            'profit': profit, 'days_held': pos['days_held'], 'reason': f"Overnight Gap ({gap*100:.1f}%)"
                        })
                        closed_ticks.append(ticker)
                        cash += pos['allocated'] * (1 + profit - TRANSACTION_COST)
                        continue
                
                range_pct = (high - low) / close
                if range_pct > defender.vol_threshold:
                    profit = (close - pos['entry_price']) / pos['entry_price']
                    trade_log.append({
                        'ticker': ticker, 'entry_date': pos['entry_date'], 'exit_date': date,
                        'profit': profit, 'days_held': pos['days_held'], 'reason': f"Volatility Gate ({range_pct*100:.1f}%)"
                    })
                    closed_ticks.append(ticker)
                    cash += pos['allocated'] * (1 + profit - TRANSACTION_COST)
                    continue
                
                pos['days_held'] += 1
                if pos['days_held'] >= HOLD_DAYS:
                    profit = (close - pos['entry_price']) / pos['entry_price']
                    trade_log.append({
                        'ticker': ticker, 'entry_date': pos['entry_date'], 'exit_date': date,
                        'profit': profit, 'days_held': pos['days_held'], 'reason': "Time decay (T+5)"
                    })
                    closed_ticks.append(ticker)
                    cash += pos['allocated'] * (1 + profit - TRANSACTION_COST)
            
            for t in closed_ticks:
                del active_positions[t]
            
            # 2. Process pending entries
            if pending_entries:
                for ticker, signal_date in list(pending_entries.items()):
                    if date not in raw_data[ticker].index:
                        continue
                    bar_today = raw_data[ticker].loc[date]
                    open_price = float(bar_today['Open'])
                    
                    gap = 0.0
                    prev_idx = raw_data[ticker].index.get_loc(date) - 1
                    if prev_idx >= 0:
                        prev_date = raw_data[ticker].index[prev_idx]
                        prev_close = float(raw_data[ticker].loc[prev_date]['Close'])
                        gap = (open_price - prev_close) / prev_close
                    if abs(gap) > defender.gap_threshold:
                        del pending_entries[ticker]
                        continue
                    
                    active_tickers = list(active_positions.keys())
                    scale = 1.0
                    if active_tickers:
                        rets = {}
                        for t in active_tickers + [ticker]:
                            if t in ticker_aligned:
                                df_t = ticker_aligned[t].loc[:date].tail(30)
                                rets[t] = df_t['log_returns']
                        if rets:
                            ret_df = pd.DataFrame(rets).dropna()
                            scale = defender.check_correlation_overlay(ticker, active_tickers, ret_df)
                    if scale == 0.0:
                        del pending_entries[ticker]
                        continue
                    
                    allocated = equity_curve[-1] * POSITION_SIZE * scale
                    active_positions[ticker] = {
                        'entry_date': date,
                        'entry_price': open_price,
                        'allocated': allocated,
                        'days_held': 0
                    }
                    cash -= allocated
                    del pending_entries[ticker]
            
            # 3. Generate new signals using the trained Random Forest
            if macro_reg != -1:
                for ticker in TICKERS:
                    if ticker not in ticker_aligned or date not in ticker_aligned[ticker].index:
                        continue
                    row = ticker_aligned[ticker].loc[date]
                    if row.isna().any():
                        continue
                    
                    local_reg_series = local_engine.predict_ticker(ticker, row.to_frame().T)
                    local_reg = int(local_reg_series.iloc[0])
                    ret_val = float(row['log_returns'])
                    p10 = float(row['P10_cone'])
                    p50 = float(row['P50_cone'])
                    p90 = float(row['P90_cone'])
                    range_pct = float(row['range_pct'])
                    
                    # 5-day momentum
                    past_ret = ticker_aligned[ticker].loc[:date].tail(5)['log_returns']
                    mom_5d = past_ret.sum() if len(past_ret) == 5 else 0.0
                    
                    features = build_features(macro_reg, local_reg, ret_val, p10, p50, p90, range_pct, mom_5d)
                    prob = clf.predict_proba([features])[0][1]  # probability of class 1 (profitable)
                    
                    if prob > 0.6:
                        if ticker not in raw_data or date not in raw_data[ticker].index:
                            continue
                        bar = raw_data[ticker].loc[date]
                        close_val = float(bar['Close'].iloc[0]) if isinstance(bar['Close'], pd.Series) else float(bar['Close'])
                        high_val = float(bar['High'].iloc[0]) if isinstance(bar['High'], pd.Series) else float(bar['High'])
                        low_val = float(bar['Low'].iloc[0]) if isinstance(bar['Low'], pd.Series) else float(bar['Low'])
                        range_pct_day = (high_val - low_val) / close_val
                        if range_pct_day > defender.vol_threshold:
                            continue
                        pending_entries[ticker] = date
            
            portfolio_value = cash
            for t, pos in active_positions.items():
                if date in raw_data[t].index:
                    close_p = float(raw_data[t].loc[date]['Close'])
                    portfolio_value += pos['allocated'] * (1 + (close_p - pos['entry_price']) / pos['entry_price'])
            equity_curve.append(portfolio_value)
        
        last_date = test_macro.index[-1]
        for ticker, pos in list(active_positions.items()):
            if last_date in raw_data[ticker].index:
                exit_price = float(raw_data[ticker].loc[last_date]['Close'])
                profit = (exit_price - pos['entry_price']) / pos['entry_price']
                trade_log.append({
                    'ticker': ticker, 'entry_date': pos['entry_date'], 'exit_date': last_date,
                    'profit': profit, 'days_held': pos['days_held'], 'reason': "End of test"
                })
                cash += pos['allocated'] * (1 + profit - TRANSACTION_COST)
            del active_positions[ticker]
        
        fold_returns = pd.Series(equity_curve).pct_change().dropna()
        if len(fold_returns) > 0 and len(trade_log) > 0:
            total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
            sharpe = (fold_returns.mean() / fold_returns.std() * np.sqrt(252)) if fold_returns.std() != 0 else 0
            max_dd = ((pd.Series(equity_curve).cummax() - pd.Series(equity_curve)) / pd.Series(equity_curve).cummax()).max() * 100
            wins = [t['profit'] for t in trade_log if t['profit'] > 0]
            losses = [t['profit'] for t in trade_log if t['profit'] <= 0]
            profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')
            win_rate = len(wins) / len(trade_log) * 100
            print(f"\nFold {fold_idx+1} Results:")
            print(f"  Total Return: {total_return:.2f}%")
            print(f"  Sharpe Ratio: {sharpe:.2f}")
            print(f"  Max Drawdown: {max_dd:.2f}%")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Number of Trades: {len(trade_log)}")
            all_fold_metrics.append({
                'fold': fold_idx+1, 'return': total_return, 'sharpe': sharpe,
                'max_dd': max_dd, 'profit_factor': profit_factor, 'win_rate': win_rate, 'trades': len(trade_log)
            })
        else:
            print("  No trades executed in this fold.")
        
        for t in trade_log:
            db.log_trade_outcome(
                ticker=t['ticker'],
                entry_date=str(t['entry_date'])[:10],
                exit_date=str(t['exit_date'])[:10],
                actual_profit=t['profit'],
                days_held=t['days_held'],
                reason=t['reason']
            )
    
    if all_fold_metrics:
        summary = pd.DataFrame(all_fold_metrics)
        print("\n" + "="*60)
        print("                 OVERALL BACKTEST SUMMARY")
        print("="*60)
        print(f"Average Return per Fold: {summary['return'].mean():.2f}%")
        print(f"Average Sharpe:          {summary['sharpe'].mean():.2f}")
        print(f"Average Max Drawdown:    {summary['max_dd'].mean():.2f}%")
        print(f"Average Profit Factor:   {summary['profit_factor'].mean():.2f}")
        print(f"Average Win Rate:        {summary['win_rate'].mean():.1f}%")
        print(f"Total Trades:            {summary['trades'].sum()}")
        print("="*60)
        if summary['profit_factor'].mean() > 1.0:
            print("✅ PROFITABLE: The strategy beats breakeven. Ready for live testing.")
        else:
            print("❌ NOT PROFITABLE YET. Tune parameters or tickers to improve.")
    else:
        print("No valid folds completed.")

if __name__ == "__main__":
    run_backtest()