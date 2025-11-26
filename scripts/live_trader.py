from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from config import (
    TICKER, START, H, MIN_TRAIN, RETRAIN_EVERY, MAX_TRAIN_WINDOW,
    BASE_EXPOSURE, THR_UP, THR_DN, MAX_ADD, MAX_SUB, MAX_GROSS,
    TREND_MODE,
)
from data import load_data, _col
from features import make_features, trend_series
from model import walk_forward_probs
from strategy import probs_to_weights

# ---- Alpaca imports (official SDK) ----
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


def compute_target_weight() -> float:
    """
    Run the SAME pipeline as the backtest, but only to get today's target weight.

    Returns:
        w_today: float
            Desired exposure today (e.g., 0.8 = 80% of equity in TICKER).
    """
    # 1) Load historical data up to now
    df = load_data(TICKER, START)

    # 2) Features & labels
    X, y = make_features(df, H=H)

    # 3) Trend series aligned to X
    trend = trend_series(_col(df, "Close"), TREND_MODE).reindex(X.index).fillna(1).astype(int)

    # 4) Walk-forward probabilities
    probs = walk_forward_probs(
        X, y,
        min_train=MIN_TRAIN,
        retrain_every=RETRAIN_EVERY,
        max_train_window=MAX_TRAIN_WINDOW,
        params=None,
    )

    if probs.empty:
        probs = pd.Series(0.5, index=X.index, name="p_up")
    else:
        probs = probs.reindex(X.index).ffill().bfill().fillna(0.5)

    # 5) Map probs → weights, same as backtest
    raw_w = probs_to_weights(
        probs, trend,
        base=BASE_EXPOSURE,
        thr_up=THR_UP, thr_dn=THR_DN,
        max_add=MAX_ADD, max_sub=MAX_SUB,
        max_gross=MAX_GROSS,
    )

    # We only care about the LAST weight: "what should my exposure be now?"
    w_today = float(raw_w.iloc[-1])
    return w_today


def get_alpaca_client() -> TradingClient:
    """
    Build an Alpaca TradingClient connected to PAPER environment.
    API keys are read from environment variables:
        ALPACA_API_KEY, ALPACA_API_SECRET
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_API_SECRET env vars")

    # paper=True ensures this is paper trading
    client = TradingClient(api_key, api_secret, paper=True)
    return client


def sync_position_with_weight(client: TradingClient, w_target: float) -> None:
    """
    Align actual Alpaca position in TICKER with the desired target weight.

    Steps:
    - Get account equity from Alpaca.
    - Get current position in TICKER (if any).
    - Compute desired notional = w_target * equity.
    - Convert to desired shares based on current price.
    - Compute delta_shares and send a market order if it's significant.
    """
    # 1) Get account info (paper account)
    account = client.get_account()
    equity = float(account.equity)

    # 2) Get current market price.
    # Simplest: use last close from our df. For "real" intraday, you'd use
    # Alpaca's data API, but this is enough to get the structure right.
    df = load_data(TICKER, START)
    current_price = float(_col(df, "Close").iloc[-1])

    # 3) Get current position size at Alpaca
    try:
        position = client.get_open_position(TICKER)
        current_shares = float(position.qty)
    except Exception:
        # No current position in this symbol
        current_shares = 0.0

    # 4) Compute target shares from target notional
    desired_notional = w_target * equity
    desired_shares = desired_notional / current_price

    delta_shares = desired_shares - current_shares

    print(f"[{datetime.now()}] Equity={equity:.2f}, price={current_price:.2f}")
    print(f"  Target weight={w_target:.3f}, target_shares={desired_shares:.2f}")
    print(f"  Current_shares={current_shares:.2f}, delta_shares={delta_shares:.2f}")

    # 5) If the change is tiny, don't bother trading
    if abs(delta_shares) < 0.1:
        print("Delta too small; no trade.")
        return

    side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
    qty = abs(int(round(delta_shares)))

    if qty == 0:
        print("Rounded delta to 0 shares; no trade.")
        return

    order = MarketOrderRequest(
        symbol=TICKER,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    print(f"Submitting {side.value} order for {qty} shares of {TICKER} (paper).")
    resp = client.submit_order(order)
    print("Order response:", resp)


def main() -> None:
    # Compute today's target exposure using the existing ML pipeline
    w_target = compute_target_weight()
    print(f"Computed target weight for {TICKER}: {w_target:.3f}")

    # Connect to Alpaca paper trading
    client = get_alpaca_client()

    # Align Alpaca position with model's target
    sync_position_with_weight(client, w_target)


if __name__ == "__main__":
    main()