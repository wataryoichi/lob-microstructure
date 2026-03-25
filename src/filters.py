"""Signal filtering / denoising: Savitzky-Golay and Kalman filters.

Based on Wang (2025) Equations 5-10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def apply_savitzky_golay(
    data: np.ndarray,
    window_size: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay filter to each column of data.

    Eq. 5-7: Fit cubic polynomial over sliding window, output smoothed value.

    Args:
        data: (n_samples,) or (n_samples, n_features) array
        window_size: Must be odd and > polyorder
        polyorder: Polynomial degree (paper uses 3)

    Returns:
        Filtered array (same shape)
    """
    if data.ndim == 1:
        return savgol_filter(data, window_length=window_size, polyorder=polyorder)

    result = np.empty_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.all(np.isnan(col)):
            result[:, j] = col
        else:
            # Fill NaNs temporarily for filtering
            mask = np.isnan(col)
            if mask.any():
                col_filled = pd.Series(col).ffill().bfill().values
            else:
                col_filled = col
            result[:, j] = savgol_filter(col_filled, window_length=window_size, polyorder=polyorder)
            result[mask, j] = np.nan

    return result


def apply_kalman_filter(
    data: np.ndarray,
    q: float = 0.001,
    r: float = 0.1,
) -> np.ndarray:
    """Apply 1D Kalman filter to each column of data.

    Eq. 8-10: Random-walk state model with noisy observations.

    State model: x_t = x_{t-1} + w, w ~ N(0, Q)
    Observation:  v_t = x_t + eps, eps ~ N(0, R)

    Args:
        data: (n_samples,) or (n_samples, n_features) array
        q: Process noise variance
        r: Measurement noise variance

    Returns:
        Filtered array (same shape)
    """
    if data.ndim == 1:
        return _kalman_1d(data, q, r)

    result = np.empty_like(data)
    for j in range(data.shape[1]):
        result[:, j] = _kalman_1d(data[:, j], q, r)

    return result


def _kalman_1d(series: np.ndarray, q: float, r: float) -> np.ndarray:
    """Single-column Kalman filter implementation."""
    n = len(series)
    x_hat = np.empty(n)
    p = np.empty(n)

    # Initialize with first observation
    first_valid = 0
    for i in range(n):
        if not np.isnan(series[i]):
            first_valid = i
            break

    x_hat[first_valid] = series[first_valid]
    p[first_valid] = 1.0

    # Fill before first valid
    x_hat[:first_valid] = np.nan

    for t in range(first_valid + 1, n):
        if np.isnan(series[t]):
            x_hat[t] = x_hat[t - 1]
            p[t] = p[t - 1] + q
            continue

        # Predict
        p_pred = p[t - 1] + q

        # Update (Eq. 9-10)
        k = p_pred / (p_pred + r)  # Kalman gain
        x_hat[t] = x_hat[t - 1] + k * (series[t] - x_hat[t - 1])
        p[t] = (1 - k) * p_pred

    return x_hat


def apply_filter(
    data: np.ndarray,
    filter_type: str = "savitzky_golay",
    sg_window: int = 21,
    sg_polyorder: int = 3,
    kalman_q: float = 0.001,
    kalman_r: float = 0.1,
) -> np.ndarray:
    """Apply the specified filter to data.

    Args:
        data: Input array
        filter_type: "raw" / "savitzky_golay" / "kalman"

    Returns:
        Filtered array (or original if filter_type="raw")
    """
    if filter_type == "raw":
        return data.copy()
    elif filter_type == "savitzky_golay":
        return apply_savitzky_golay(data, window_size=sg_window, polyorder=sg_polyorder)
    elif filter_type == "kalman":
        return apply_kalman_filter(data, q=kalman_q, r=kalman_r)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Use: raw, savitzky_golay, kalman")


def filter_feature_dataframe(
    df: pd.DataFrame,
    filter_type: str = "savitzky_golay",
    exclude_cols: list[str] | None = None,
    **filter_kwargs,
) -> pd.DataFrame:
    """Apply filter to all numeric columns in a DataFrame.

    Args:
        df: Feature DataFrame
        filter_type: Filter type
        exclude_cols: Columns to skip (e.g., timestamp)

    Returns:
        Filtered DataFrame
    """
    if exclude_cols is None:
        exclude_cols = ["timestamp"]

    result = df.copy()
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.float32, float]]

    if feature_cols and filter_type != "raw":
        data = df[feature_cols].values
        filtered = apply_filter(data, filter_type=filter_type, **filter_kwargs)
        result[feature_cols] = filtered

    return result
