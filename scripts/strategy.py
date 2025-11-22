"""
Strategy logic module.
Converts model probabilities into position weights using baseline + tilts approach.
Implements optional trend filtering and long-only constraints.
"""

from typing import Optional
import numpy as np
import pandas as pd

from config import TREND_MODE

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

# ===================== 6) Baseline =====================
def buy_and_hold(index: pd.DatetimeIndex) -> pd.Series:
    """1.0x exposure every day (shifted by 1 to represent next-open execution)."""
    return pd.Series(1.0, index=index).shift(1).fillna(0.0)