"""LOB feature engineering based on Wang (2025) Equations 1-4.

Features:
1. Mid-price (Eq. 1)
2. Level-1 order imbalance (Eq. 2)
3. N-level aggregate imbalance (Eq. 3)
4. Weighted mid-price change (Eq. 4)
5. Bid-ask spread
6. Raw LOB prices and volumes (top k levels)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import WEIGHTED_MID_WEIGHTS


def compute_mid_price(bid_p1: np.ndarray, ask_p1: np.ndarray) -> np.ndarray:
    """Eq. 1: m_t = (a^1_t + b^1_t) / 2"""
    return (ask_p1 + bid_p1) / 2.0


def compute_level1_imbalance(bid_q1: np.ndarray, ask_q1: np.ndarray) -> np.ndarray:
    """Eq. 2: I^1_t = (Q^b_{t,1} - Q^a_{t,1}) / (Q^b_{t,1} + Q^a_{t,1})"""
    total = bid_q1 + ask_q1
    # Avoid division by zero
    safe_total = np.where(total > 0, total, 1.0)
    return (bid_q1 - ask_q1) / safe_total


def compute_aggregate_imbalance(
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    depth: int = 5,
) -> np.ndarray:
    """Eq. 3: I^n_t = (sum_bid - sum_ask) / (sum_bid + sum_ask) for n levels."""
    sum_bid = bid_volumes[:, :depth].sum(axis=1)
    sum_ask = ask_volumes[:, :depth].sum(axis=1)
    total = sum_bid + sum_ask
    safe_total = np.where(total > 0, total, 1.0)
    return (sum_bid - sum_ask) / safe_total


def compute_weighted_mid_change(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    n_levels: int = 3,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Eq. 4: Delta_m^w_t = sum w_i * (m^i_t - m^i_{t-1})

    Args:
        bid_prices: (n_snapshots, depth) bid prices
        ask_prices: (n_snapshots, depth) ask prices
        n_levels: number of levels to use (default 3)
        weights: custom weights (default: 1/i normalized)

    Returns:
        (n_snapshots,) weighted mid-price change
    """
    if weights is None:
        weights = WEIGHTED_MID_WEIGHTS[:n_levels]

    # Compute mid-price at each level
    mids = (ask_prices[:, :n_levels] + bid_prices[:, :n_levels]) / 2.0

    # Compute changes
    mid_changes = np.diff(mids, axis=0)

    # Weighted sum
    w = np.array(weights[:n_levels])
    w = w / w.sum()  # ensure normalized
    weighted_change = (mid_changes * w).sum(axis=1)

    # Prepend NaN for first row
    return np.concatenate([[np.nan], weighted_change])


def compute_spread(bid_p1: np.ndarray, ask_p1: np.ndarray) -> np.ndarray:
    """Bid-ask spread in price units."""
    return ask_p1 - bid_p1


def compute_relative_spread(bid_p1: np.ndarray, ask_p1: np.ndarray) -> np.ndarray:
    """Bid-ask spread relative to mid-price."""
    mid = compute_mid_price(bid_p1, ask_p1)
    safe_mid = np.where(mid > 0, mid, 1.0)
    return (ask_p1 - bid_p1) / safe_mid


def compute_mid_return(mid_prices: np.ndarray, lag: int = 1) -> np.ndarray:
    """Log return of mid-price over lag steps."""
    safe_mid = np.where(mid_prices > 0, mid_prices, np.nan)
    ret = np.log(safe_mid[lag:] / safe_mid[:-lag])
    return np.concatenate([np.full(lag, np.nan), ret])


def compute_rolling_volatility(
    mid_prices: np.ndarray,
    window: int = 100,
) -> np.ndarray:
    """Rolling standard deviation of mid-price returns."""
    returns = compute_mid_return(mid_prices, lag=1)
    vol = pd.Series(returns).rolling(window=window, min_periods=window).std().values
    return vol


def build_feature_matrix(
    df: pd.DataFrame,
    depth: int = 5,
    include_raw_lob: bool = True,
    include_imbalance: bool = True,
    include_spread: bool = True,
    include_weighted_mid: bool = True,
) -> pd.DataFrame:
    """Build complete feature matrix from LOB snapshots.

    Args:
        df: Raw LOB DataFrame with bid_p1..bid_pN, bid_q1..bid_qN, ask_p1..ask_pN, ask_q1..ask_qN
        depth: Number of LOB levels to use
        include_raw_lob: Include raw price/volume features
        include_imbalance: Include order imbalance features
        include_spread: Include spread features
        include_weighted_mid: Include weighted mid-price change

    Returns:
        Feature DataFrame with named columns
    """
    features: dict[str, np.ndarray] = {}

    bid_p1 = df["bid_p1"].values
    ask_p1 = df["ask_p1"].values
    bid_q1 = df["bid_q1"].values
    ask_q1 = df["ask_q1"].values

    # Mid-price
    mid = compute_mid_price(bid_p1, ask_p1)
    features["mid_price"] = mid

    # Mid-price return
    features["mid_return_1"] = compute_mid_return(mid, lag=1)
    features["mid_return_5"] = compute_mid_return(mid, lag=5)
    features["mid_return_10"] = compute_mid_return(mid, lag=10)

    # Rolling volatility
    features["volatility_100"] = compute_rolling_volatility(mid, window=100)

    if include_imbalance:
        features["imbalance_l1"] = compute_level1_imbalance(bid_q1, ask_q1)

        # Aggregate imbalance at multiple depths
        bid_volumes = np.column_stack(
            [df[f"bid_q{i}"].values for i in range(1, min(depth, 5) + 1)]
        )
        ask_volumes = np.column_stack(
            [df[f"ask_q{i}"].values for i in range(1, min(depth, 5) + 1)]
        )
        features["imbalance_l5"] = compute_aggregate_imbalance(
            bid_volumes, ask_volumes, depth=min(depth, 5)
        )

        if depth >= 10:
            bid_v10 = np.column_stack(
                [df[f"bid_q{i}"].values for i in range(1, 11)]
            )
            ask_v10 = np.column_stack(
                [df[f"ask_q{i}"].values for i in range(1, 11)]
            )
            features["imbalance_l10"] = compute_aggregate_imbalance(
                bid_v10, ask_v10, depth=10
            )

    if include_spread:
        features["spread"] = compute_spread(bid_p1, ask_p1)
        features["relative_spread"] = compute_relative_spread(bid_p1, ask_p1)

    if include_weighted_mid:
        n_wm_levels = min(depth, 3)
        bid_prices = np.column_stack(
            [df[f"bid_p{i}"].values for i in range(1, n_wm_levels + 1)]
        )
        ask_prices = np.column_stack(
            [df[f"ask_p{i}"].values for i in range(1, n_wm_levels + 1)]
        )
        features["weighted_mid_change"] = compute_weighted_mid_change(
            bid_prices, ask_prices, n_levels=n_wm_levels
        )

    if include_raw_lob:
        for i in range(1, depth + 1):
            features[f"bid_p{i}"] = df[f"bid_p{i}"].values
            features[f"bid_q{i}"] = df[f"bid_q{i}"].values
            features[f"ask_p{i}"] = df[f"ask_p{i}"].values
            features[f"ask_q{i}"] = df[f"ask_q{i}"].values

    feat_df = pd.DataFrame(features)
    if "timestamp" in df.columns:
        feat_df["timestamp"] = df["timestamp"].values

    return feat_df


def normalize_features(
    df: pd.DataFrame,
    window: int = 1000,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Rolling z-score normalization using trailing window.

    Uses [t-window, ..., t-1] statistics to avoid lookahead bias.
    """
    if exclude_cols is None:
        exclude_cols = ["timestamp"]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    result = df.copy()

    for col in feature_cols:
        series = df[col].astype(float)
        shifted = series.shift(1)
        roll_mean = shifted.rolling(window=window, min_periods=window).mean()
        roll_std = shifted.rolling(window=window, min_periods=window).std(ddof=0)
        roll_std = roll_std.clip(lower=1e-8)
        result[col] = (series - roll_mean) / roll_std

    return result


def build_sequences(
    features: np.ndarray,
    seq_len: int = 10,
) -> np.ndarray:
    """Build sliding window sequences for deep learning models.

    Args:
        features: (n_samples, n_features) array
        seq_len: Number of timesteps per sequence

    Returns:
        (n_samples - seq_len + 1, seq_len, n_features) array
    """
    n = len(features)
    n_feat = features.shape[1]
    sequences = np.zeros((n - seq_len + 1, seq_len, n_feat))

    for i in range(n - seq_len + 1):
        sequences[i] = features[i : i + seq_len]

    return sequences
