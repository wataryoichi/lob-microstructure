"""Load and parse LOB snapshot data from Parquet/CSV files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_lob_data(
    data_dir: str | Path,
    symbol: str = "BTCUSDT",
    max_files: int | None = None,
) -> pd.DataFrame:
    """Load LOB snapshots from parquet files in a directory.

    Args:
        data_dir: Directory containing parquet files
        symbol: Filter files by symbol prefix
        max_files: Maximum number of files to load

    Returns:
        DataFrame sorted by timestamp
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"{symbol}*.parquet"))

    if max_files is not None:
        files = files[:max_files]

    if not files:
        logger.warning(f"No parquet files found in {data_dir} for {symbol}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
        logger.info(f"Loaded {f.name}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Total: {len(combined)} snapshots")
    return combined


def filter_complete_snapshots(
    df: pd.DataFrame,
    depth: int = 40,
) -> pd.DataFrame:
    """Drop snapshots with missing entries for required depth levels.

    Paper: "Drop any snapshot missing one of the 40 bid-ask levels"
    """
    required_cols = []
    for i in range(1, depth + 1):
        required_cols.extend([f"bid_p{i}", f"bid_q{i}", f"ask_p{i}", f"ask_q{i}"])

    existing = [c for c in required_cols if c in df.columns]
    if len(existing) < len(required_cols):
        missing = set(required_cols) - set(existing)
        logger.warning(f"Missing columns for depth={depth}: {len(missing)} columns")
        return pd.DataFrame()

    before = len(df)
    df_clean = df.dropna(subset=existing).reset_index(drop=True)
    after = len(df_clean)
    logger.info(f"Filtered: {before} -> {after} ({after/before*100:.1f}% coverage)")
    return df_clean


def extract_prices_and_volumes(
    df: pd.DataFrame,
    depth: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract bid/ask prices and volumes as numpy arrays.

    Returns:
        (bid_prices, bid_volumes, ask_prices, ask_volumes)
        Each has shape (n_snapshots, depth)
    """
    bid_p = np.column_stack([df[f"bid_p{i}"].values for i in range(1, depth + 1)])
    bid_q = np.column_stack([df[f"bid_q{i}"].values for i in range(1, depth + 1)])
    ask_p = np.column_stack([df[f"ask_p{i}"].values for i in range(1, depth + 1)])
    ask_q = np.column_stack([df[f"ask_q{i}"].values for i in range(1, depth + 1)])

    return bid_p, bid_q, ask_p, ask_q


def get_mid_prices(df: pd.DataFrame) -> np.ndarray:
    """Compute mid-price series from level-1 bid/ask."""
    return (df["bid_p1"].values + df["ask_p1"].values) / 2.0


def train_test_split_temporal(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally (no shuffle) to avoid lookahead bias."""
    n = len(df)
    split_idx = int(n * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
