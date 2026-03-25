"""Price direction labeling: binary and ternary schemes.

Labels are based on mid-price change over a prediction horizon.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_future_mid_change(
    mid_prices: np.ndarray,
    horizon: int = 5,
) -> np.ndarray:
    """Compute fractional mid-price change over horizon steps.

    Delta_p / p = (m_{t+h} - m_t) / m_t

    Returns:
        (n_samples,) array. Last `horizon` entries are NaN.
    """
    n = len(mid_prices)
    changes = np.full(n, np.nan)
    safe_mid = np.where(mid_prices > 0, mid_prices, np.nan)
    valid = n - horizon
    if valid > 0:
        changes[:valid] = (safe_mid[horizon:] - safe_mid[:valid]) / safe_mid[:valid]
    return changes


def label_binary(
    mid_prices: np.ndarray,
    horizon: int = 5,
) -> np.ndarray:
    """Binary labeling: Up (1) / Down (0).

    Label = 1 if mid-price increases over horizon, else 0.

    Returns:
        (n_samples,) integer array. NaN where label cannot be computed.
    """
    changes = compute_future_mid_change(mid_prices, horizon)
    labels = np.full(len(mid_prices), -1, dtype=np.int32)
    valid = ~np.isnan(changes)
    labels[valid & (changes > 0)] = 1
    labels[valid & (changes <= 0)] = 0
    labels[~valid] = -1  # invalid marker
    return labels


def label_ternary(
    mid_prices: np.ndarray,
    horizon: int = 5,
    epsilon: float | None = None,
) -> np.ndarray:
    """Ternary labeling: Up (2) / Flat (1) / Down (0).

    If epsilon is None, auto-tune to achieve roughly equal class frequencies.

    Returns:
        (n_samples,) integer array. -1 where label cannot be computed.
    """
    changes = compute_future_mid_change(mid_prices, horizon)
    valid_mask = ~np.isnan(changes)
    valid_changes = changes[valid_mask]

    if epsilon is None:
        epsilon = _tune_epsilon(valid_changes)
        logger.info(f"Auto-tuned ternary epsilon: {epsilon:.8f}")

    labels = np.full(len(mid_prices), -1, dtype=np.int32)
    labels[valid_mask & (changes > epsilon)] = 2   # Up
    labels[valid_mask & (changes < -epsilon)] = 0  # Down
    labels[valid_mask & (np.abs(changes) <= epsilon)] = 1  # Flat

    # Log class distribution
    valid_labels = labels[labels >= 0]
    if len(valid_labels) > 0:
        for cls in range(3):
            pct = (valid_labels == cls).mean() * 100
            logger.info(f"  Class {cls}: {pct:.1f}%")

    return labels


def _tune_epsilon(changes: np.ndarray, target_flat_pct: float = 0.333) -> float:
    """Find epsilon that gives roughly equal class frequencies.

    Binary search for epsilon where ~33% of changes fall in [-eps, +eps].
    """
    abs_changes = np.abs(changes)
    lo, hi = 0.0, np.percentile(abs_changes, 99)

    for _ in range(50):
        mid = (lo + hi) / 2
        flat_pct = (abs_changes <= mid).mean()
        if flat_pct < target_flat_pct:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def create_labels(
    mid_prices: np.ndarray,
    scheme: str = "binary",
    horizon: int = 5,
    ternary_epsilon: float | None = None,
) -> np.ndarray:
    """Create labels based on scheme.

    Args:
        mid_prices: Mid-price array
        scheme: "binary" or "ternary"
        horizon: Prediction horizon in timesteps
        ternary_epsilon: Threshold for ternary (None = auto-tune)

    Returns:
        Integer label array (-1 = invalid)
    """
    if scheme == "binary":
        return label_binary(mid_prices, horizon)
    elif scheme == "ternary":
        eps = ternary_epsilon if ternary_epsilon and ternary_epsilon > 0 else None
        return label_ternary(mid_prices, horizon, epsilon=eps)
    else:
        raise ValueError(f"Unknown labeling scheme: {scheme}. Use: binary, ternary")


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights for balanced training."""
    valid = labels[labels >= 0]
    classes = np.unique(valid)
    n_total = len(valid)
    weights = {}
    for cls in classes:
        n_cls = (valid == cls).sum()
        weights[int(cls)] = n_total / (len(classes) * n_cls) if n_cls > 0 else 1.0
    return weights
