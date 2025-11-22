"""
Visualization module.
Generates and saves equity curve plots comparing strategy vs buy-and-hold.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ARTIFACTS_DIR

def plot_equity_curves(res_bh: dict, res_model: dict, ticker: str) -> None:
    """
    Plot and save equity curves comparing buy-and-hold vs model strategy.
    """
    # Plot and save
    try:
        plt.figure(figsize=(12, 6))
        res_bh["equity"].plot(label="Buy & Hold", lw=2, alpha=0.85)
        res_model["equity"].plot(label="Model", lw=2)
        plt.title(f"{ticker} — Equity (Next-Open, Costs)")
        plt.ylabel("Portfolio ($)")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(ARTIFACTS_DIR, f"equity_{ticker}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"\nSaved plot: {out}")
    except Exception as e:
        # Plotting should never fail the whole run on a server/CI
        print(f"[plot warn] {e}")