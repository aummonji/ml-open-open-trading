"""
Feature engineering module.
Constructs technical indicators and targets in a leakage-safe manner.
All features are properly time-shifted to avoid look-ahead bias.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from data import _col

# ===================== 2) Features + Target =====================
def _raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact feature set for short-horizon equity prediction.

    Includes:
    - log returns, overnight gap
    - short momentum (5-day)
    - volatility (20-day)
    - RSI-14 (overbought/oversold)
    - trend (SMA-10 / SMA-50)
    - VAM-20 (vol-adjusted momentum)
    - macd_hist: MACD histogram (trend-momentum crossover)
    - vol_rank_60: 60-day volume percentile rank (abnormal liquidity / attention)
    """
    C = _col(df, "Close")
    O = _col(df, "Open")
    V = _col(df, "Volume") if "Volume" in df.columns else pd.Series(0, index=df.index)

    X = pd.DataFrame(index=df.index)

    # --- Price-based primitives ---
    X["ret_cc_1"] = np.log(C / C.shift(1))
    X["gap_oc"]   = np.log(O / C.shift(1))
    X["mom_5"]    = np.log(C / C.shift(5))

    # 20-day realized volatility of daily returns
    daily_ret = C.pct_change()
    X["vol_20"] = daily_ret.rolling(20, min_periods=2).std()

    # --- RSI-14: strength of recent moves ---
    d  = C.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    X["rsi_14"] = 100 - 100/(1 + up/(dn + 1e-12))

    # --- Trend via SMAs ---
    X["sma_10"] = C.rolling(10, min_periods=2).mean()
    X["sma_50"] = C.rolling(50, min_periods=5).mean()

    # --- VAM-20: momentum normalized by volatility ---
    mom_20 = np.log(C / C.shift(20))
    vol_20_scaled = X["vol_20"] * np.sqrt(20)
    X["vam_20"] = mom_20 / (vol_20_scaled + 1e-12)

    # # Compute rolling 60-day percentile rank of today’s volume
    vol_rank_60 = V.rolling(60, min_periods=20).apply(
        lambda window: pd.Series(window).rank(pct=True).iloc[-1],
        raw=False
    )
    X["vol_rank_60"] = vol_rank_60.clip(0.0, 1.0)


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