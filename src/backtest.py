# src/backtest.py — EquiSight V5 Multi‑Pair Statistical Arbitrage
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
from src.execution_db import ExecutionDatabase
from src.portfolio_risk import PortfolioDefender
from src.walk_forward import walk_forward_split
from src.alpha_engine import PairsAlphaEngine, PairConfig

warnings.filterwarnings("ignore")

PAIRS = [
    PairConfig("TCS.NS", "INFY.NS"),
    PairConfig("ICICIBANK.NS", "SBIN.NS"),
    PairConfig("RELIANCE.NS", "HINDUNILVR.NS"),
    PairConfig("KOTAKBANK.NS", "BAJFINANCE.NS")
]
MACRO_INDEX = "NIFTYBEES.NS"
DB_PATH = "database/equisight_v5.db"
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE = 0.05
MAX_TOTAL_EXPOSURE = 0.25
MAX_HOLD_DAYS = 10
TRANSACTION_COST = 0.0015
DOWNSIDE_GAP_THRESHOLD = -0.05
SPREAD_PROFIT_TARGET = 0.015   # 1.5% on the spread
SPREAD_STOP_LOSS = -0.015

def run_backtest():
    print("=== EquiSight V5 Multi‑Pair Stat Arb ===")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # Download macro and ticker data
    nifty_raw = yf.download(MACRO_INDEX, period="5y", progress=False, session=session)
    macro_features = generate_v5_features(nifty_raw)
    if isinstance(macro_features.columns, pd.MultiIndex):
        macro_features.columns = macro_features.columns.get_level_values(0)
    
    tickers = set()
    for p in PAIRS:
        tickers.add(p.stock_a)
        tickers.add(p.stock_b)
    
    raw_data = {}
    for t in tickers:
        raw = yf.download(t, period="5y", progress=False, session=session)
        raw_data[t] = raw
    
    master_idx = macro_features.index
    splits = walk_forward_split(macro_features, train_size=504, test_size=126, purge_size=30)
    print(f"[INFO] Using {len(splits)} folds.")
    
    db = ExecutionDatabase(DB_PATH)
    with db._get_connection() as conn:
        conn.execute("DELETE FROM trade_outcomes")
        conn.commit()
    
    all_fold_metrics = []
    
    for fold_idx, (train_macro, test_macro) in enumerate(splits):
        print(f"\n{'='*40} FOLD {fold_idx+1}/{len(splits)} {'='*40}")
        
        defender = PortfolioDefender()
        alpha_engine = PairsAlphaEngine(PAIRS)
        equity_curve = [INITIAL_CAPITAL]
        cash = INITIAL_CAPITAL
        active_pairs = {}
        trade_log = []
        pending_entries = []
        
        for date in tqdm(test_macro.index, desc="Simulating"):
            # 1. Update open pairs
            closed_pairs = []
            for pair_name, pos in list(active_pairs.items()):
                pair_cfg = next(p for p in PAIRS if f"{p.stock_a}-{p.stock_b}" == pair_name)
                a, b = pair_cfg.stock_a, pair_cfg.stock_b
                if date not in raw_data[a].index or date not in raw_data[b].index:
                    continue
                close_a = float(raw_data[a].loc[date]['Close'])
                close_b = float(raw_data[b].loc[date]['Close'])
                
                if pos['direction'] == 'LONG_SPREAD':
                    leg_a_ret = (close_a - pos['entry_price_a']) / pos['entry_price_a']
                    leg_b_ret = -(close_b - pos['entry_price_b']) / pos['entry_price_b']
                else:
                    leg_a_ret = -(close_a - pos['entry_price_a']) / pos['entry_price_a']
                    leg_b_ret = (close_b - pos['entry_price_b']) / pos['entry_price_b']
                spread_ret = (leg_a_ret + leg_b_ret) / 2
                
                # Gap / volatility checks
                prev_idx_a = raw_data[a].index.get_loc(date) - 1
                gap_a = (float(raw_data[a].loc[date]['Open']) - float(raw_data[a].iloc[prev_idx_a]['Close'])) / float(raw_data[a].iloc[prev_idx_a]['Close']) if prev_idx_a >= 0 else 0
                prev_idx_b = raw_data[b].index.get_loc(date) - 1
                gap_b = (float(raw_data[b].loc[date]['Open']) - float(raw_data[b].iloc[prev_idx_b]['Close'])) / float(raw_data[b].iloc[prev_idx_b]['Close']) if prev_idx_b >= 0 else 0
                range_a = (float(raw_data[a].loc[date]['High']) - float(raw_data[a].loc[date]['Low'])) / close_a
                range_b = (float(raw_data[b].loc[date]['High']) - float(raw_data[b].loc[date]['Low'])) / close_b
                
                if gap_a < DOWNSIDE_GAP_THRESHOLD or gap_b < DOWNSIDE_GAP_THRESHOLD:
                    closed_pairs.append((pair_name, spread_ret, "Downside gap"))
                elif range_a > defender.vol_threshold or range_b > defender.vol_threshold:
                    closed_pairs.append((pair_name, spread_ret, "Vol gate"))
                elif spread_ret >= SPREAD_PROFIT_TARGET:
                    closed_pairs.append((pair_name, spread_ret, "Profit target"))
                elif spread_ret <= SPREAD_STOP_LOSS:
                    closed_pairs.append((pair_name, spread_ret, "Stop loss"))
                else:
                    pos['days_held'] += 1
                    if pos['days_held'] >= MAX_HOLD_DAYS:
                        closed_pairs.append((pair_name, spread_ret, "Time stop"))
            
            for pair_name, profit, reason in closed_pairs:
                pos = active_pairs.pop(pair_name)
                cash += pos['allocated'] * (1 + profit - TRANSACTION_COST)
                trade_log.append({'pair': pair_name, 'profit': profit, 'reason': reason,
                                  'entry_date': pos['entry_date'], 'exit_date': date,
                                  'days_held': pos['days_held']})
            
            # 2. Process pending entries
            for entry in list(pending_entries):
                pair_cfg, signal_date, signal = entry
                a, b = pair_cfg.stock_a, pair_cfg.stock_b
                if date not in raw_data[a].index or date not in raw_data[b].index:
                    continue
                open_a = float(raw_data[a].loc[date]['Open'])
                open_b = float(raw_data[b].loc[date]['Open'])
                # Gap filter
                prev_idx_a = raw_data[a].index.get_loc(date) - 1
                gap_a = (open_a - float(raw_data[a].iloc[prev_idx_a]['Close'])) / float(raw_data[a].iloc[prev_idx_a]['Close']) if prev_idx_a >= 0 else 0
                prev_idx_b = raw_data[b].index.get_loc(date) - 1
                gap_b = (open_b - float(raw_data[b].iloc[prev_idx_b]['Close'])) / float(raw_data[b].iloc[prev_idx_b]['Close']) if prev_idx_b >= 0 else 0
                if gap_a < DOWNSIDE_GAP_THRESHOLD or gap_b < DOWNSIDE_GAP_THRESHOLD:
                    pending_entries.remove(entry)
                    continue
                
                # Exposure cap
                current_exposure = sum(pos['allocated'] for pos in active_pairs.values()) / equity_curve[-1]
                if current_exposure + POSITION_SIZE > MAX_TOTAL_EXPOSURE:
                    continue
                
                allocated = equity_curve[-1] * POSITION_SIZE
                if cash < allocated:
                    continue
                
                direction = 'LONG_SPREAD' if signal['action'] == 'BUY_SPREAD' else 'SHORT_SPREAD'
                pair_name = f"{a}-{b}"
                active_pairs[pair_name] = {
                    'direction': direction,
                    'entry_date': date,
                    'entry_price_a': open_a,
                    'entry_price_b': open_b,
                    'beta': signal['beta'],
                    'allocated': allocated,
                    'days_held': 0
                }
                cash -= allocated
                pending_entries.remove(entry)
            
            # 3. Generate new signals
            for pair_cfg in PAIRS:
                pair_name = f"{pair_cfg.stock_a}-{pair_cfg.stock_b}"
                if pair_name in active_pairs:
                    continue
                signal = alpha_engine.compute_signal(pair_cfg, raw_data, date)
                if signal:
                    pending_entries.append((pair_cfg, date, signal))
            
            # Record equity
            portfolio_value = cash
            for pos in active_pairs.values():
                pair_cfg = next(p for p in PAIRS if f"{p.stock_a}-{p.stock_b}" == pos['pair_name']) if 'pair_name' in pos else None
                # simplified: just add allocated (mtm approximate)
                portfolio_value += pos['allocated']
            equity_curve.append(portfolio_value)
        
        # End of fold
        last_date = test_macro.index[-1]
        for pair_name, pos in list(active_pairs.items()):
            pair_cfg = next(p for p in PAIRS if f"{p.stock_a}-{p.stock_b}" == pair_name)
            a, b = pair_cfg.stock_a, pair_cfg.stock_b
            if last_date in raw_data[a].index and last_date in raw_data[b].index:
                close_a = float(raw_data[a].loc[last_date]['Close'])
                close_b = float(raw_data[b].loc[last_date]['Close'])
                if pos['direction'] == 'LONG_SPREAD':
                    leg_a_ret = (close_a - pos['entry_price_a']) / pos['entry_price_a']
                    leg_b_ret = -(close_b - pos['entry_price_b']) / pos['entry_price_b']
                else:
                    leg_a_ret = -(close_a - pos['entry_price_a']) / pos['entry_price_a']
                    leg_b_ret = (close_b - pos['entry_price_b']) / pos['entry_price_b']
                profit = (leg_a_ret + leg_b_ret) / 2
                trade_log.append({'pair': pair_name, 'profit': profit, 'reason': 'End of test',
                                  'entry_date': pos['entry_date'], 'exit_date': last_date,
                                  'days_held': pos['days_held']})
                cash += pos['allocated'] * (1 + profit - TRANSACTION_COST)
            del active_pairs[pair_name]
        
        fold_returns = pd.Series(equity_curve).pct_change().dropna()
        if len(fold_returns) > 0 and len(trade_log) > 0:
            total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
            sharpe = (fold_returns.mean() / fold_returns.std() * np.sqrt(252)) if fold_returns.std() != 0 else 0
            max_dd = ((pd.Series(equity_curve).cummax() - pd.Series(equity_curve)) / pd.Series(equity_curve).cummax()).max() * 100
            wins = [t['profit'] for t in trade_log if t['profit'] > 0]
            losses = [t['profit'] for t in trade_log if t['profit'] <= 0]
            pf = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')
            win_rate = (len(wins) / len(trade_log)) * 100 if trade_log else 0
            print(f"\nFold {fold_idx+1}: Return {total_return:.2f}% | Sharpe {sharpe:.2f} | MaxDD {max_dd:.2f}% | PF {pf:.2f} | Trades {len(trade_log)} | Win {win_rate:.1f}%")
            all_fold_metrics.append({'return': total_return, 'sharpe': sharpe, 'max_dd': max_dd, 'pf': pf, 'trades': len(trade_log), 'win_rate': win_rate})
        else:
            print("  No trades.")

    if all_fold_metrics:
        summary = pd.DataFrame(all_fold_metrics)
        avg_pf = summary['pf'].replace([np.inf, -np.inf], np.nan).mean()
        print("\n" + "="*60)
        print("                 OVERALL BACKTEST SUMMARY")
        print("="*60)
        print(f"Avg Return: {summary['return'].mean():.2f}% | Avg Sharpe: {summary['sharpe'].mean():.2f} | Avg MaxDD: {summary['max_dd'].mean():.2f}%")
        print(f"Avg Profit Factor: {avg_pf:.2f} | Avg Win Rate: {summary['win_rate'].mean():.1f}%")
        print(f"Total Trades: {summary['trades'].sum()}")
        print("="*60)
        if avg_pf > 1.0:
            print("✅ PROFITABLE. Go live with paper trading.")
        else:
            print("❌ Still not profitable. Last attempt completed.")
    else:
        print("No valid folds.")

if __name__ == "__main__":
    run_backtest()