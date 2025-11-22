"""
Backtesting engine module.
Simulates trading with next-open execution, transaction costs, and slippage.
Calculates performance metrics including Sharpe, drawdown, and CAGR.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

from data import _col
from config import FEE_BPS, SLIP_BPS

# ===================== 5) Backtest (next-open + costs) =====================
@dataclass
class Costs:
    fee_bps: float = FEE_BPS
    slip_bps: float = SLIP_BPS

def backtest(df: pd.DataFrame, w: pd.Series, initial: float, costs: Costs) -> Dict[str, object]:
    """
    Next-open execution backtest with proportional costs and slippage.
    """
    O = _col(df, "Open").to_numpy()
    idx = df.index
    n = len(idx)
    wv = pd.to_numeric(w.reindex(idx), errors="coerce").fillna(0.0).to_numpy()
    fee = float(costs.fee_bps) / 1e4
    slip = float(costs.slip_bps) / 1e4

    cash = np.zeros(n); sh = np.zeros(n); eq = np.zeros(n)
    cash[0] = eq[0] = float(initial)

    for i in range(1, n):
        cash[i], sh[i] = cash[i-1], sh[i-1]
        eq_prev  = float(cash[i-1] + sh[i-1] * O[i-1])
        desired  = float(wv[i] * eq_prev)     # target notional exposure
        current  = float(sh[i] * O[i])        # current notional exposure
        delta    = desired - current          # what we need to trade
        if abs(delta) > 1e-9:
            px = float(O[i] * (1 + (slip if delta > 0 else -slip)))  # pay slip in trade direction
            d_sh = float(delta / px)                                 # shares to buy/sell
            notional = float(d_sh * px)
            cash[i] -= notional + abs(notional) * fee
            sh[i]   += d_sh
        eq[i] = float(cash[i] + sh[i] * O[i])

    equity = pd.Series(eq, index=idx, name="Equity")
    rets   = equity.pct_change().fillna(0.0)
    dd     = equity / equity.cummax() - 1.0

    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_years   = len(equity) / 252
    cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_v     = float(rets.std(ddof=0) * np.sqrt(252))
    sharpe    = float((rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(252)) if rets.std(ddof=0) > 0 else np.nan

    stats = {
        "FinalEquity": float(equity.iloc[-1]),
        "TotalReturn": total_ret,
        "CAGR": float(cagr),
        "AnnVol": ann_v,
        "Sharpe": sharpe,
        "MaxDD": float(dd.min()),
        "Avg|w|": float(np.mean(np.abs(wv))),
    }
    return {"equity": equity, "returns": rets, "drawdown": dd, "stats": stats}