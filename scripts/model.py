"""
Machine learning model module.
Implements walk-forward validation with rolling retraining.
Trains GradientBoostingClassifier and generates probability predictions.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.ensemble import GradientBoostingClassifier
from config import GBM_PARAMS


def build_model(params: Optional[Dict] = None) -> GradientBoostingClassifier:
    """
    Factory for GradientBoostingClassifier.
    Allows overriding defaults via params.
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