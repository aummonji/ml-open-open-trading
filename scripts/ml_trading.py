# - ML next-open strategy (leakage-safe, costs included)
# - Compact features (returns/gap, momentum, vol, RSI, SMAs, vol-adjusted momentum)
# - Rolling walk-forward GBM (retrain every N days on a rolling window)
# - Long-only baseline exposure + confidence tilts (optionally trend-gated)
# - Buy & Hold benchmark + side-by-side stats

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ===================== User toggles =====================
FAST_MODE = True  # quick run on shorter history; flip to False for longer history

# ===================== Config =====================
TICKER = "SPY"
START  = "2018-01-01" if FAST_MODE else "2010-01-01"
H = 5  # label horizon in trading days: predict whether Open[t+H] > Open[t]

# Walk-forward schedule 
if FAST_MODE:
    MIN_TRAIN        = 150   # minimum training rows before the first prediction
    RETRAIN_EVERY    = 10    # retrain cadence in trading days
    MAX_TRAIN_WINDOW = 600   # rolling window size (use the most recent N rows)
else:
    MIN_TRAIN        = 800
    RETRAIN_EVERY    = 10
    MAX_TRAIN_WINDOW = 2500

# Gradient Boosting settings 
GBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=42,
)

# Exposure policy: start from a baseline and tilt up/down with model confidence
BASE_EXPOSURE = 1.00  # always hold 1.0x baseline (long-only)
THR_UP        = 0.60  # if P(up) >= THR_UP, add long exposure
THR_DN        = 0.40  # if P(up) <= THR_DN, cut exposure
MAX_ADD       = 0.70  # max additional long exposure above baseline
MAX_SUB       = 0.50  # max reduction from baseline
MAX_GROSS     = 1.70  # cap final exposure for sanity (long-only floor applied later)

# Trend gate for tilts: "none", "loose"(20/100 SMA), "strict"(50/200 SMA)
TREND_MODE    = "loose"

FEE_BPS     = 0.5     # commissions per notional traded (0.5 bp = 0.005%)
SLIP_BPS    = 0.2     # slippage vs open price (0.2 bp = 0.002%)
INITIAL_CAP = 100_000.0

# Output folder for saved plots
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ===================== small helpers =====================
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Return a clean 1-D numeric Series for column `name` even if duplicates exist.
    """
    if name not in df.columns:
        raise ValueError(f"Column '{name}' not found.")
    s = df[name]
    if isinstance(s, pd.DataFrame):  # if duplicates created multi-column
        s = s.iloc[:, 0]
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return pd.to_numeric(s, errors="coerce")

# ===================== 1) Data =====================
def load_data(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Download OHLCV data with yfinance and standardize the DataFrame.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Missing yfinance. Install: pip install yfinance")

    print(f"Downloading {ticker} from {start}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    # Normalize and deduplicate columns
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.dropna(how="all")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df.sort_index()

# ===================== 2) Features + Target =====================
def _raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build compact, robust features:
    - ret_cc_1: log return Close[t]/Close[t-1]
    - gap_oc:   log gap Open[t]/Close[t-1] (overnight info)
    - mom_5:    log momentum Close[t]/Close[t-5]
    - vol_20:   realized volatility of daily returns over 20d
    - rsi_14:   14-day RSI (Wilder’s smoothing via EMA)
    - sma_10/50: simple moving averages for trend context
    - vam_20:   volatility-adjusted 20-day momentum (Sharpe-like ratio)
    """
    C = _col(df, "Close")
    O = _col(df, "Open")

    X = pd.DataFrame(index=df.index)
    X["ret_cc_1"] = np.log(C / C.shift(1))
    X["gap_oc"]   = np.log(O / C.shift(1))
    X["mom_5"]    = np.log(C / C.shift(5))
    X["vol_20"]   = C.pct_change().rolling(20, min_periods=2).std()

    # RSI using EMA of up/down moves (avoid division by zero)
    d  = C.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    X["rsi_14"]   = 100 - 100/(1 + up / (dn + 1e-12))

    X["sma_10"]   = C.rolling(10, min_periods=2).mean()
    X["sma_50"]   = C.rolling(50, min_periods=5).mean()

    # Volatility-adjusted momentum (akin to a short-horizon Sharpe)
    mom_20 = np.log(C / C.shift(20))
    vol_20 = X["vol_20"] * np.sqrt(20)   # scale vol to 20-day horizon
    X["vam_20"] = mom_20 / (vol_20 + 1e-12)

    return X

def make_features(df: pd.DataFrame, H: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Leakage-safe features (shift by 1 bar) + fixed H-day label (Open[t+H] > Open[t]).
    Returns:
        X (float DataFrame) and y (int Series) with identical indices.
    """
    O = _col(df, "Open")
    X_raw = _raw_features(df)

    # Shift by 1: only use information known by the prior close
    X = X_raw.shift(1).replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().astype(float)

    # Binary label: 1 if Open[t+H] > Open[t], else 0
    y = (O.shift(-H) > O).astype("Int64")  # nullable int keeps trailing NaNs

    # Keep only rows where the label exists
    valid = y.notna()
    if not valid.any():
        raise ValueError(f"Insufficient data for H={H}. Try a smaller horizon or an earlier START date.")

    X2 = X.loc[valid]
    y2 = y.loc[valid].astype(int)

    # if any NaNs remain in features, fill with column medians
    if X2.isna().any().any():
        X2 = X2.fillna(X2.median(numeric_only=True))

    return X2, y2

def trend_series(close: pd.Series, mode: str) -> pd.Series:
    """
    Optional trend gate for tilts:
    - "none": tilts always allowed
    - "loose": 20/100 SMA cross
    - "strict": 50/200 SMA cross
    Returns Series of 1 (uptrend) or 0 (downtrend), shifted by 1 to avoid look-ahead.
    """
    close = pd.to_numeric(close, errors="coerce")
    if mode == "none":
        return pd.Series(1, index=close.index, name="trend").astype(int)
    fast, slow = (20, 100) if mode == "loose" else (50, 200)
    s_f = close.rolling(fast, min_periods=max(2, fast // 5)).mean()
    s_s = close.rolling(slow, min_periods=max(2, slow // 5)).mean()
    t = (s_f > s_s).shift(1).fillna(0).astype(int)
    t.name = "trend"
    return t

# ===================== 3) Model + Walk-forward =====================
def build_model(params: Optional[Dict] = None) -> GradientBoostingClassifier:
    """
    Create a GradientBoostingClassifier with sane defaults.
    """
    cfg = GBM_PARAMS.copy()
    if params:
        cfg.update(params)
    return GradientBoostingClassifier(**cfg)

def walk_forward_probs(
    X: pd.DataFrame, y: pd.Series, *,
    min_train: int, retrain_every: int, max_train_window: int,
    params: Optional[Dict] = None,
) -> pd.Series:
    """
    Walk-forward training/prediction:
    - Start predicting once we have `min_train` rows.
    - Retrain every `retrain_every` days on a rolling window of size `max_train_window`.
    - Predict probabilities for the next block (from current anchor to next).
    """
    n = len(X)

    # If the available data window is smaller than MIN_TRAIN,
    # relax it so the loop can still run (unchanged behavior).
    if n <= min_train:
        min_train = max(50, int(0.6 * n))
        print(f"[Walk-forward] Relaxed MIN_TRAIN -> {min_train} (n={n})")

    out = pd.Series(np.nan, index=X.index, name="p_up")

    # Anchor points where we (re)train and then predict the next slice
    anchors = list(range(min_train, n, retrain_every))
    if not anchors or anchors[-1] != n:
        anchors.append(n)

    print(f"[Walk-forward] {n} obs, {len(anchors)-1} blocks, retrain every {retrain_every}d")

    for i in range(len(anchors) - 1):
        train_at, next_a = anchors[i], anchors[i + 1]
        start = max(0, train_at - max_train_window)  # rolling window

        # Progress logging at reasonable cadence
        if (i % 10 == 0) or (i == len(anchors) - 2):
            pct = 100 * (i + 1) / (len(anchors) - 1)
            print(f"  Block {i+1}/{len(anchors)-1} ({pct:.0f}%): train={train_at-start}, predict={next_a-train_at}")

        X_fit = X.iloc[start:train_at]
        y_fit = y.reindex(X_fit.index).astype(int).to_numpy().ravel()

        # Mild recency weighting; skip tiny windows
        n_fit = len(X_fit)
        if n_fit < 50:
            continue
        w = 0.995 ** np.arange(n_fit)[::-1]

        try:
            clf = build_model(params).fit(X_fit, y_fit, sample_weight=w)
            out.iloc[train_at:next_a] = clf.predict_proba(X.iloc[train_at:next_a])[:, 1]
        except Exception as e:
            # Keep going if an occasional block fails (data oddities, etc.)
            print(f"  [warn] skip block {i+1}: {e}")

    # Fill any initial NaNs (before first prediction) and sporadic gaps
    if not out.notna().any():
        out[:] = 0.5
    else:
        out = out.ffill().bfill().fillna(0.5)

    return out

# ===================== 4) Probs → weights (baseline + tilts, long-only) =====================
def probs_to_weights(
    probs: pd.Series, trend: Optional[pd.Series], *,
    base: float, thr_up: float, thr_dn: float,
    max_add: float, max_sub: float, max_gross: float,
) -> pd.Series:
    """
    Map predicted P(up) to a long-only exposure:
    - Start at `base` (e.g., 1.0x).
    - If p >= thr_up and trend allows, add up to `max_add` more exposure.
    - If p <= thr_dn and trend allows, cut up to `max_sub` from exposure.
    - Clamp to [0.0, max_gross] and shift by 1 day (next-open execution).
    """
    p = pd.Series(probs).astype(float)
    idx = p.index

    # Trend gating for when it's okay to add or cut exposure
    if trend is None or TREND_MODE == "none":
        add_gate = np.ones(len(p), dtype=bool)
        cut_gate = np.ones(len(p), dtype=bool)
    else:
        t = pd.Series(trend).reindex(idx)
        t = pd.to_numeric(t, errors="coerce").fillna(1).astype(int)
        add_gate = (t.values == 1)  # only add in uptrend
        cut_gate = (t.values == 0)  # only cut in downtrend

    # Start with the baseline weight everywhere
    w = np.full(len(p), float(base))
    pa = p.values

    # Add exposure when confident up
    long_mask = (pa >= thr_up) & add_gate
    if np.any(long_mask):
        strength = (pa[long_mask] - thr_up) / max(1e-12, (1.0 - thr_up))
        w[long_mask] += np.clip(strength, 0, 1) * max_add

    # Reduce exposure when confident down
    short_mask = (pa <= thr_dn) & cut_gate
    if np.any(short_mask):
        strength = (thr_dn - pa[short_mask]) / max(thr_dn, 1e-12)
        w[short_mask] -= np.clip(strength, 0, 1) * max_sub

    # Long-only floor and clamp
    w = np.maximum(w, 0.0)
    w = np.clip(w, 0.0, max_gross)

    # Shift by 1 to reflect trade at next open
    return pd.Series(w, index=idx, name="w").shift(1).fillna(base)

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

# ===================== 6) Baseline =====================
def buy_and_hold(index: pd.DatetimeIndex) -> pd.Series:
    """1.0x exposure every day (shifted by 1 to represent next-open execution)."""
    return pd.Series(1.0, index=index).shift(1).fillna(0.0)

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
    try:
        plt.figure(figsize=(12, 6))
        res_bh["equity"].plot(label="Buy & Hold", lw=2, alpha=0.85)
        res_model["equity"].plot(label="Model", lw=2)
        plt.title(f"{TICKER} — Equity (Next-Open, Costs)")
        plt.ylabel("Portfolio ($)")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(ARTIFACTS_DIR, f"equity_{TICKER}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"\nSaved plot: {out}")
    except Exception as e:
        # Plotting should never fail the whole run on a server/CI
        print(f"[plot warn] {e}")

    print("="*60)
    print("DONE")
    print("="*60)

if __name__ == "__main__":
    main()
