"""
Main entry point for the ML trading strategy.
Orchestrates the entire pipeline: data loading → feature engineering → model training → 
backtesting → visualization. Run this file to execute the complete strategy.
"""

# - ML next-open strategy (leakage-safe, costs included)
# - Compact features (returns/gap, momentum, vol, RSI, SMAs, vol-adjusted momentum)
# - Rolling walk-forward GBM (retrain every N days on a rolling window)
# - Long-only baseline exposure + confidence tilts (optionally trend-gated)
# - Buy & Hold benchmark + side-by-side stats

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score

from config import (
    TICKER, START, H, MIN_TRAIN, RETRAIN_EVERY, MAX_TRAIN_WINDOW,
    BASE_EXPOSURE, THR_UP, THR_DN, MAX_ADD, MAX_SUB, MAX_GROSS,
    TREND_MODE, INITIAL_CAP, FAST_MODE,
)
from data import load_data, _col
from features import make_features, trend_series
from model import walk_forward_probs
from strategy import probs_to_weights, buy_and_hold
from backtest import backtest, Costs
from visualization import plot_equity_curves

# ===================== 7) Main =====================
def main() -> None:
    print("="*60)
    print("Simple ML Next-Open Strategy (leakage-safe, costs included)")
    print("="*60)
    print(f"FAST_MODE={FAST_MODE} | START={START} | H={H} | TREND={TREND_MODE}")

    # 1) Data
    df = load_data(TICKER, START)
    print(f"Loaded {len(df)} rows: {df.index[0].date()} → {df.index[-1].date()}")

    # 2) Features/Target
    X, y = make_features(df, H=H)
    print(f"After alignment: {len(X)} usable rows")
    print(f"X: {X.shape[0]}×{X.shape[1]} | y: {y.value_counts().to_dict()}")

    # Trend gate (optional) aligned to X
    trend = trend_series(_col(df, "Close"), TREND_MODE).reindex(X.index).fillna(1).astype(int)

    # 3) Walk-forward probabilities
    probs = walk_forward_probs(
        X, y,
        min_train=MIN_TRAIN,
        retrain_every=RETRAIN_EVERY,
        max_train_window=MAX_TRAIN_WINDOW,
        params=None,
    )

    print("\n=== Probability distribution check ===")
    print(probs.describe())
    print(probs.tail(20))
    
    print("======================================\n")
    # Ensure a fully populated probability series aligned to X
    if probs.empty:
        probs = pd.Series(0.5, index=X.index, name="p_up")
    else:
        probs = probs.reindex(X.index).ffill().bfill().fillna(0.5)

    # 4) Probs → weights (baseline + tilts, long-only)
    raw_w = probs_to_weights(
        probs, trend,
        base=BASE_EXPOSURE,
        thr_up=THR_UP, thr_dn=THR_DN,
        max_add=MAX_ADD, max_sub=MAX_SUB,
        max_gross=MAX_GROSS,
    )
    # Reindex to trading calendar; if NaN, fall back to baseline exposure
    w = pd.to_numeric(raw_w.reindex(df.index), errors="coerce").fillna(BASE_EXPOSURE)

    # 5) Backtests (model vs buy-and-hold)
    res_model = backtest(df, w, initial=INITIAL_CAP, costs=Costs())
    res_bh    = backtest(df, buy_and_hold(df.index), initial=INITIAL_CAP, costs=Costs())

    # direction accuracy where probs exist
    mask = probs.notna()
    if mask.any():
        acc = accuracy_score(y.loc[mask], (probs.loc[mask] >= 0.5).astype(int))
        print(f"Direction accuracy (H={H}, 0.5 cut): {acc:.3f}")

    # Report (side-by-side)
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    for name, R in [("Model (Baseline+Tilts)", res_model), ("Buy & Hold", res_bh)]:
        print(f"\n{name}:")
        for k in ["FinalEquity", "TotalReturn", "CAGR", "AnnVol", "Sharpe", "MaxDD", "Avg|w|"]:
            v = R["stats"][k]
            if k in ["TotalReturn", "CAGR"]:
                print(f"  {k:12s}: {v:8.2%}")
            else:
                print(f"  {k:12s}: {v:8.4f}")

    # Plot and save
    plot_equity_curves(res_bh, res_model, TICKER)

    print("="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
