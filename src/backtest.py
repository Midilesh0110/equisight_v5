# src/backtest.py — EquiSight V5 Final Polished Mean Reversion
import yfinance as yf
import requests
import numpy as np
import pandas as pd
import os
import warnings
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import generate_v5_features
from src.macro_regime import MacroCircuitBreaker
from src.execution_db import ExecutionDatabase
from src.portfolio_risk import PortfolioDefender
from src.walk_forward import walk_forward_split

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "KOTAKBANK.NS", "BAJFINANCE.NS"
]
MACRO_INDEX = "NIFTYBEES.NS"
DB_PATH = "database/equisight_v5.db"
INITIAL_CAPITAL = 1_000_000
MAX_POSITION_FRACTION = 0.10
MAX_TOTAL_EXPOSURE = 0.25
MAX_HOLD_DAYS = 15
TRANSACTION_COST = 0.0015

def add_atr_features(raw_df, feat_df):
    high = raw_df['High'].iloc[:, 0] if isinstance(raw_df['High'], pd.DataFrame) else raw_df['High']
    low = raw_df['Low'].iloc[:, 0] if isinstance(raw_df['Low'], pd.DataFrame) else raw_df['Low']
    close = raw_df['Close'].iloc[:, 0] if isinstance(raw_df['Close'], pd.DataFrame) else raw_df['Close']
    true_range = pd.concat([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())], axis=1).max(axis=1)
    feat_df['ATR_20'] = true_range.rolling(window=20).mean()
    feat_df['ATR_50_ma'] = feat_df['ATR_20'].rolling(window=50).mean()
    return feat_df

# -------------------------------------------------------------------
# Main backtest engine
# -------------------------------------------------------------------
def run_backtest():
    print("=== EquiSight V5 Final Polished Mean Reversion ===")
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # 1. Download Data
    nifty_raw = yf.download(MACRO_INDEX, period="5y", progress=False, session=session)
    macro_features = generate_v5_features(nifty_raw)
    if isinstance(macro_features.columns, pd.MultiIndex):
        macro_features.columns = macro_features.columns.get_level_values(0)
    
    ticker_data = {}
    raw_data = {}
    for t in TICKERS:
        raw = yf.download(t, period="5y", progress=False, session=session)
        if raw.empty:
            continue
        feat_df = generate_v5_features(raw)
        if isinstance(feat_df.columns, pd.MultiIndex):
            feat_df.columns = feat_df.columns.get_level_values(0)
        feat_df = add_atr_features(raw, feat_df)
        # Compute 5-day momentum (sum of log returns over last 5 days)
        feat_df['momentum_5d'] = feat_df['log_returns'].rolling(window=5).sum()
        ticker_data[t] = feat_df
        raw_data[t] = raw
    
    master_idx = macro_features.index
    macro_aligned = macro_features
    ticker_aligned = {}
    for t, df in ticker_data.items():
        ticker_aligned[t] = df.reindex(master_idx)
    valid_mask = pd.DataFrame({t: ticker_aligned[t]['log_returns'].notna() for t in ticker_aligned}).all(axis=1)
    macro_aligned = macro_aligned.loc[valid_mask]
    for t in ticker_aligned:
        ticker_aligned[t] = ticker_aligned[t].loc[valid_mask]
    
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
        
        macro_breaker = MacroCircuitBreaker()
        macro_breaker.train(train_macro)
        # No local regime HMM needed, we removed local regime filter
        
        defender = PortfolioDefender()
        equity_curve = [INITIAL_CAPITAL]
        cash = INITIAL_CAPITAL
        active_positions = {}
        trade_log = []
        closed_trade_returns = []
        pending_entries = {}
        
        for date in tqdm(test_macro.index, desc="Simulating"):
            macro_reg = int(macro_breaker.predict(test_macro.loc[[date]]).iloc[0])
            
            # ------ 1. Update active positions (Dynamic Exits) ------
            closed_ticks = []
            for ticker, pos in list(active_positions.items()):
                if date not in raw_data[ticker].index:
                    continue
                
                bar = raw_data[ticker].loc[date]
                low_p = float(bar['Low'].iloc[0]) if isinstance(bar['Low'], pd.Series) else float(bar['Low'])
                high_p = float(bar['High'].iloc[0]) if isinstance(bar['High'], pd.Series) else float(bar['High'])
                close_p = float(bar['Close'].iloc[0]) if isinstance(bar['Close'], pd.Series) else float(bar['Close'])
                
                # Dynamic Stop Loss (1.5x P10) & Take Profit (2.0x P90)
                if low_p <= pos['stop_loss']:
                    profit = (pos['stop_loss'] - pos['entry_price']) / pos['entry_price']
                    trade_log.append({'ticker': ticker, 'profit': profit, 'reason': 'Stop Loss Hit'})
                    closed_ticks.append(ticker)
                elif high_p >= pos['take_profit']:
                    profit = (pos['take_profit'] - pos['entry_price']) / pos['entry_price']
                    trade_log.append({'ticker': ticker, 'profit': profit, 'reason': 'Take Profit Hit'})
                    closed_ticks.append(ticker)
                else:
                    pos['days_held'] += 1
                    if pos['days_held'] >= MAX_HOLD_DAYS:
                        profit = (close_p - pos['entry_price']) / pos['entry_price']
                        trade_log.append({'ticker': ticker, 'profit': profit, 'reason': 'Time Stop'})
                        closed_ticks.append(ticker)
            
            for t in closed_ticks:
                pos = active_positions.pop(t)
                last_trade = trade_log[-1]
                last_trade.update({'entry_date': pos['entry_date'], 'exit_date': date, 'days_held': pos['days_held']})
                cash += pos['allocated'] * (1 + last_trade['profit'] - TRANSACTION_COST)
                closed_trade_returns.append(last_trade['profit'])
            
            # ------ 2. Process pending entries ------
            for ticker, signal_data in list(pending_entries.items()):
                if date not in raw_data[ticker].index:
                    continue
                
                bar = raw_data[ticker].loc[date]
                open_price = float(bar['Open'].iloc[0]) if isinstance(bar['Open'], pd.Series) else float(bar['Open'])
                
                # Gap Filter
                prev_idx = raw_data[ticker].index.get_loc(date) - 1
                if prev_idx >= 0:
                    prev_close = float(raw_data[ticker].iloc[prev_idx]['Close'])
                    if (open_price - prev_close) / prev_close < -0.05:
                        del pending_entries[ticker]
                        continue
                
                kelly_frac = 0.05 if len(closed_trade_returns) < 5 else defender.calculate_half_kelly(
                    [r for r in closed_trade_returns if r > 0],
                    [r for r in closed_trade_returns if r <= 0]
                )
                position_frac = min(kelly_frac, MAX_POSITION_FRACTION)
                allocated = cash * position_frac
                
                current_exposure = sum(pos['allocated'] for pos in active_positions.values()) / equity_curve[-1]
                if current_exposure + position_frac > MAX_TOTAL_EXPOSURE:
                    del pending_entries[ticker]
                    continue
                
                if cash >= allocated and allocated > 0:
                    active_positions[ticker] = {
                        'entry_date': date,
                        'entry_price': open_price,
                        'allocated': allocated,
                        'stop_loss': open_price * (1 + (1.5 * signal_data['p10'])),
                        'take_profit': open_price * (1 + (2.0 * signal_data['p90'])),
                        'days_held': 0
                    }
                    cash -= allocated
                del pending_entries[ticker]
            
            # ------ 3. Generate new signals ------
            if macro_reg != -1:
                for ticker in TICKERS:
                    if ticker in active_positions or ticker in pending_entries:
                        continue
                    if date not in ticker_aligned[ticker].index:
                        continue
                    
                    row = ticker_aligned[ticker].loc[date]
                    if pd.isna(row['ATR_20']) or pd.isna(row['ATR_50_ma']) or pd.isna(row['momentum_5d']):
                        continue
                    
                    # 1. Relaxed ATR filter: allow trade if ATR_20 <= 1.5 * ATR_50_ma
                    if row['ATR_20'] > 1.5 * row['ATR_50_ma']:
                        continue
                    
                    # 2. Mean reversion entry: log_return <= P20_cone AND negative 5-day momentum
                    if float(row['log_returns']) <= float(row['P10_cone']) * 2.0 and float(row['momentum_5d']) < 0:
                        pending_entries[ticker] = {
                            'p10': float(row['P10_cone']),
                            'p90': float(row['P90_cone'])
                        }
            
            # Record Equity MTM
            portfolio_value = cash
            for t, pos in active_positions.items():
                if date in raw_data[t].index:
                    close_p = float(raw_data[t].loc[date]['Close'].iloc[0]) if isinstance(raw_data[t].loc[date]['Close'], pd.Series) else float(raw_data[t].loc[date]['Close'])
                    current_position_value = pos['allocated'] + (pos['allocated'] * ((close_p - pos['entry_price']) / pos['entry_price']))
                    portfolio_value += current_position_value
                else:
                    portfolio_value += pos['allocated']
            equity_curve.append(portfolio_value)
            
        # End of fold force close
        last_date = test_macro.index[-1]
        for t, pos in list(active_positions.items()):
            if last_date in raw_data[t].index:
                close_p = float(raw_data[t].loc[last_date]['Close'].iloc[0]) if isinstance(raw_data[t].loc[last_date]['Close'], pd.Series) else float(raw_data[t].loc[last_date]['Close'])
                profit = (close_p - pos['entry_price']) / pos['entry_price']
                cash += pos['allocated'] * (1 + profit - TRANSACTION_COST)
                trade_log.append({'ticker': t, 'profit': profit, 'reason': 'End of test'})
        
        # Metrics
        fold_returns = pd.Series(equity_curve).pct_change().dropna()
        if len(fold_returns) > 0 and len(trade_log) > 0:
            total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
            sharpe = (fold_returns.mean() / fold_returns.std() * np.sqrt(252)) if fold_returns.std() != 0 else 0
            max_dd = ((pd.Series(equity_curve).cummax() - pd.Series(equity_curve)) / pd.Series(equity_curve).cummax()).max() * 100
            wins = [t['profit'] for t in trade_log if t['profit'] > 0]
            losses = [t['profit'] for t in trade_log if t['profit'] <= 0]
            pf = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')
            win_rate = (len(wins) / len(trade_log)) * 100 if trade_log else 0
            
            print(f"\nFold {fold_idx+1} Results:")
            print(f"  Total Return: {total_return:.2f}% | Sharpe: {sharpe:.2f}")
            print(f"  Max DD: {max_dd:.2f}% | PF: {pf:.2f} | Trades: {len(trade_log)} | Win Rate: {win_rate:.1f}%")
            
            all_fold_metrics.append({
                'return': total_return, 'sharpe': sharpe, 'max_dd': max_dd,
                'pf': pf, 'trades': len(trade_log), 'win_rate': win_rate
            })

    if all_fold_metrics:
        summary = pd.DataFrame(all_fold_metrics)
        print("\n" + "="*60)
        print("                 OVERALL BACKTEST SUMMARY")
        print("="*60)
        print(f"Average Return: {summary['return'].mean():.2f}%")
        print(f"Average Sharpe: {summary['sharpe'].mean():.2f}")
        print(f"Average Max DD: {summary['max_dd'].mean():.2f}%")
        print(f"Avg Profit Factor: {summary['pf'].replace(np.inf, np.nan).mean():.2f}")
        print(f"Average Win Rate: {summary['win_rate'].mean():.1f}%")
        print(f"Total Trades: {summary['trades'].sum()}")
        print("="*60)
        if summary['win_rate'].mean() > 45.95 and summary['pf'].replace(np.inf, np.nan).mean() > 0.77:
            print("✅ V5 SUPERIOR: Beats V4 on both profit factor and win rate. Ready for live paper trading.")
        else:
            print("❌ Still not beating V4. Further tuning needed, but infrastructure is solid.")

if __name__ == "__main__":
    run_backtest()