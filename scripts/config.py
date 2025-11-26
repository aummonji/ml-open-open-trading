"""
Configuration module for the ML trading strategy.
Contains all hyperparameters, toggles, and constants used throughout the project.
Modify settings here to experiment with different strategy parameters.
"""

from pathlib import Path

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
    MAX_TRAIN_WINDOW = 800   # rolling window size (use the most recent N rows)
else:
    MIN_TRAIN        = 800
    RETRAIN_EVERY    = 10
    MAX_TRAIN_WINDOW = 2500

# Gradient Boosting settings
GBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
   subsample=0.7,
    random_state=42,
    max_depth=2,
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

# ===================== Paths =====================
CONFIG_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = CONFIG_DIR.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
