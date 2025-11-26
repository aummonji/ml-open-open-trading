# scripts/run_tuning_once.py

"""
One-off hyperparameter tuning for the Gradient Boosting model.

- TimeSeriesSplit CV to avoid shuffling time.
- Small RandomizedSearchCV around current GBM_PARAMS.
- Prints a merged dict you can paste back into config.GBM_PARAMS
  IF it improves the walk-forward backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from config import (
    TICKER,
    START,
    H,
    FAST_MODE,
    GBM_PARAMS,
)
from data import load_data
from features import make_features


def main() -> None:
    print("=" * 60)
    print("Hyperparameter tuning (GradientBoostingClassifier, one-off helper)")
    print("=" * 60)
    print(f"TICKER={TICKER} | START={START} | H={H} | FAST_MODE={FAST_MODE}")

    # 1) Load data & build features
    df = load_data(TICKER, START)
    X, y = make_features(df, H=H)

    # Basic cleaning: drop rows with any NaNs in X or missing y
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    print(f"Usable rows after cleaning: {len(X)}")
    print(f"Feature shape: {X.shape[0]} x {X.shape[1]}")

    # 2) Base model using current GBM_PARAMS
    base_model = GradientBoostingClassifier(**GBM_PARAMS)

    # 3) Small search space around current params
    param_dist = {
        # Around current n_estimators=600
        "n_estimators": [500, 600, 700],
        # Around current learning_rate=0.03
        "learning_rate": [0.02, 0.03, 0.05],
        # Shallow trees only
        "max_depth": [2, 3],
        # Slightly different subsamples for robustness
        "subsample": [0.6, 0.7, 0.8, 0.9],
    }

    # 4) Time-series split (no shuffling)
    cv = TimeSeriesSplit(n_splits=3)

    # 5) RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,      # use all cores if possible
        verbose=1,      # print progress
        random_state=42,
    )

    print("\nFitting RandomizedSearchCV...")
    search.fit(X, y)

    print("\nBest ROC-AUC (CV):", f"{search.best_score_:.4f}")
    print("Best params found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # 6) Merge with existing GBM_PARAMS to show a drop-in config
    merged = GBM_PARAMS.copy()
    merged.update(search.best_params_)

    print("\nMerged GBM_PARAMS candidate (paste into config.py if you like):")
    print("GBM_PARAMS = dict(")
    for k, v in merged.items():
        print(f"    {k}={repr(v)},")
    print(")")


if __name__ == "__main__":
    main()
