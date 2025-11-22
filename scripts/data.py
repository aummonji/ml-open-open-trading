"""Data loading and preprocessing utilities."""

from typing import Optional
import numpy as np
import pandas as pd


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