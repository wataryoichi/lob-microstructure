"""ML as signal filter for the imbalance strategy (P1 task #5).

Architecture:
    Stage 1: imbalance_strategy.py generates candidate trades (extreme order imbalance).
    Stage 2: A trained ML model decides whether to ACCEPT or REJECT each candidate.

The key question: does adding an ML filter improve avg_net_bps even if n_trades decreases?
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import build_feature_matrix
from .imbalance_strategy import (
    ImbalanceResult,
    ImbalanceStrategyConfig,
    run_imbalance_strategy,
)

logger = logging.getLogger(__name__)

# Default feature columns used by the filter (lightweight set that avoids
# raw LOB prices/volumes to keep the filter simple and fast).
DEFAULT_FEATURE_COLS = [
    "mid_return_1",
    "mid_return_5",
    "mid_return_10",
    "volatility_100",
    "imbalance_l1",
    "imbalance_l5",
    "relative_spread",
    "weighted_mid_change",
]


# ---------------------------------------------------------------------------
# 1. Prepare training data
# ---------------------------------------------------------------------------

def prepare_filter_training_data(
    df: pd.DataFrame,
    trades: list[dict],
    feature_cols: list[str] | None = None,
    horizon: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build feature matrix and labels for the ML filter.

    For each candidate trade produced by the imbalance strategy, we extract
    the feature vector at the entry row and label it:
        1  if gross_bps > 0  (profitable)
        0  otherwise          (unprofitable)

    Args:
        df: Full LOB DataFrame (same one passed to run_imbalance_strategy).
        trades: List of trade dicts from ImbalanceResult.trades.
        feature_cols: Which columns from the feature matrix to use.
            Defaults to DEFAULT_FEATURE_COLS.
        horizon: Unused (kept for API symmetry / future forward-label logic).

    Returns:
        X: (n_trades, n_features) feature array.
        y: (n_trades,) binary label array.
        valid_indices: Row indices into *trades* that were kept (some may be
            dropped due to NaN features).
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    if not trades:
        return np.empty((0, len(feature_cols))), np.empty(0, dtype=int), []

    # Build the full feature matrix once
    feat_df = build_feature_matrix(df, depth=5, include_raw_lob=False)

    rows = []
    labels = []
    valid_indices: list[int] = []

    for idx, trade in enumerate(trades):
        row_idx = trade["index"]
        if row_idx < 0 or row_idx >= len(feat_df):
            continue

        feat_row = feat_df.iloc[row_idx]
        vals = feat_row[feature_cols].values.astype(float)

        # Skip if any feature is NaN
        if np.any(np.isnan(vals)):
            continue

        rows.append(vals)
        labels.append(1 if trade["gross_bps"] > 0 else 0)
        valid_indices.append(idx)

    if not rows:
        return np.empty((0, len(feature_cols))), np.empty(0, dtype=int), []

    X = np.vstack(rows)
    y = np.array(labels, dtype=int)
    return X, y, valid_indices


# ---------------------------------------------------------------------------
# 2. Train filter model
# ---------------------------------------------------------------------------

def train_filter_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "logistic_regression",
) -> tuple[object, StandardScaler]:
    """Train a binary classifier to predict trade profitability.

    Args:
        X_train: (n_samples, n_features) training features.
        y_train: (n_samples,) binary labels (1=profitable, 0=not).
        model_type: One of 'logistic_regression', 'xgboost'.

    Returns:
        (model, scaler) -- the fitted model and the StandardScaler used for
        feature normalization.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos = n_neg / max(n_pos, 1)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbosity=0,
        )
    else:
        raise ValueError(
            f"Unknown filter model_type: {model_type!r}. "
            "Choose 'logistic_regression' or 'xgboost'."
        )

    model.fit(X_scaled, y_train)
    logger.info(
        "Filter model trained (%s): %d samples, %.1f%% positive",
        model_type,
        len(y_train),
        y_train.mean() * 100,
    )
    return model, scaler


# ---------------------------------------------------------------------------
# 3. Apply filter to candidate trades
# ---------------------------------------------------------------------------

def apply_filter(
    model: object,
    scaler: StandardScaler,
    df: pd.DataFrame,
    trades: list[dict],
    feature_cols: list[str] | None = None,
    threshold: float = 0.5,
) -> list[dict]:
    """Filter candidate trades using the trained ML model.

    Args:
        model: Fitted sklearn-compatible classifier.
        scaler: Fitted StandardScaler (from train_filter_model).
        df: Full LOB DataFrame.
        trades: Candidate trade dicts from the imbalance strategy.
        feature_cols: Feature columns to extract (must match training).
        threshold: Probability threshold for ACCEPT (default 0.5).

    Returns:
        Subset of *trades* where the model predicts profitable (class 1).
    """
    if not trades:
        return []

    X, _y, valid_indices = prepare_filter_training_data(
        df, trades, feature_cols=feature_cols
    )

    if len(X) == 0:
        return []

    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)

    # predict_proba returns (n, 2) for binary; column 1 = P(profitable)
    prob_profitable = proba[:, 1] if proba.ndim == 2 else proba

    accepted = []
    for i, vi in enumerate(valid_indices):
        if prob_profitable[i] >= threshold:
            accepted.append(trades[vi])

    logger.info(
        "ML filter: %d / %d candidates accepted (threshold=%.2f)",
        len(accepted),
        len(trades),
        threshold,
    )
    return accepted


# ---------------------------------------------------------------------------
# 4. End-to-end filtered strategy
# ---------------------------------------------------------------------------

def _build_imbalance_result(trades: list[dict], config: dict) -> ImbalanceResult:
    """Compute summary statistics for a list of trade dicts."""
    if not trades:
        return ImbalanceResult(
            trades=[], n_trades=0, avg_gross_bps=0, avg_net_bps=0,
            gross_pnl_bps=0, net_pnl_bps=0, win_rate=0, win_rate_net=0,
            sharpe=0, max_drawdown_bps=0, profit_factor=0, config=config,
        )

    gross_arr = np.array([t["gross_bps"] for t in trades])
    net_arr = np.array([t["net_bps"] for t in trades])

    wins = gross_arr[gross_arr > 0].sum()
    losses = abs(gross_arr[gross_arr < 0].sum())
    pf = wins / losses if losses > 0 else float("inf")

    cumul_net = np.cumsum(net_arr)
    running_max = np.maximum.accumulate(cumul_net)
    drawdown = cumul_net - running_max
    max_dd = float(abs(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    if net_arr.std() > 1e-12:
        sharpe = float(np.mean(net_arr) / net_arr.std() * np.sqrt(len(net_arr)))
    else:
        sharpe = 0.0

    return ImbalanceResult(
        trades=trades,
        n_trades=len(trades),
        avg_gross_bps=float(np.mean(gross_arr)),
        avg_net_bps=float(np.mean(net_arr)),
        gross_pnl_bps=float(np.sum(gross_arr)),
        net_pnl_bps=float(np.sum(net_arr)),
        win_rate=float((gross_arr > 0).mean()),
        win_rate_net=float((net_arr > 0).mean()),
        sharpe=sharpe,
        max_drawdown_bps=max_dd,
        profit_factor=float(pf),
        config=config,
    )


def run_filtered_strategy(
    df: pd.DataFrame,
    imbalance_config: ImbalanceStrategyConfig | None = None,
    filter_model_type: str = "logistic_regression",
    train_ratio: float = 0.8,
    feature_cols: list[str] | None = None,
    filter_threshold: float = 0.5,
) -> ImbalanceResult:
    """End-to-end: generate candidates -> train filter -> apply on test set.

    Steps:
        1. Run imbalance strategy on *entire* DataFrame to get all candidates.
        2. Split candidates into train / test by chronological index.
        3. Train ML filter on train-period candidates.
        4. Apply filter to test-period candidates.
        5. Return ImbalanceResult for the filtered test trades.

    Args:
        df: Full LOB DataFrame.
        imbalance_config: Config for the imbalance strategy (default settings
            if None).
        filter_model_type: 'logistic_regression' or 'xgboost'.
        train_ratio: Fraction of data rows used for training the filter.
        feature_cols: Feature columns (default: DEFAULT_FEATURE_COLS).
        filter_threshold: Probability threshold for accepting a trade.

    Returns:
        ImbalanceResult for filtered trades on the test period.
    """
    if imbalance_config is None:
        imbalance_config = ImbalanceStrategyConfig()

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    # Step 1 -- run imbalance strategy on full data
    full_result = run_imbalance_strategy(df, imbalance_config)
    all_trades = full_result.trades

    if not all_trades:
        logger.warning("No candidate trades generated by imbalance strategy.")
        return _build_imbalance_result([], _config_dict(imbalance_config, filter_model_type))

    # Step 2 -- chronological train/test split based on row index
    split_row = int(len(df) * train_ratio)
    train_trades = [t for t in all_trades if t["index"] < split_row]
    test_trades = [t for t in all_trades if t["index"] >= split_row]

    if len(train_trades) < 10:
        logger.warning(
            "Too few training trades (%d). Returning unfiltered test trades.",
            len(train_trades),
        )
        return _build_imbalance_result(
            test_trades, _config_dict(imbalance_config, filter_model_type)
        )

    # Step 3 -- prepare training data and train filter
    X_train, y_train, _ = prepare_filter_training_data(
        df, train_trades, feature_cols=feature_cols
    )

    if len(X_train) < 10:
        logger.warning(
            "Too few valid training samples (%d). Returning unfiltered test trades.",
            len(X_train),
        )
        return _build_imbalance_result(
            test_trades, _config_dict(imbalance_config, filter_model_type)
        )

    model, scaler = train_filter_model(X_train, y_train, model_type=filter_model_type)

    # Step 4 -- apply filter to test-period candidates
    filtered_trades = apply_filter(
        model, scaler, df, test_trades,
        feature_cols=feature_cols, threshold=filter_threshold,
    )

    # Step 5 -- build result
    return _build_imbalance_result(
        filtered_trades, _config_dict(imbalance_config, filter_model_type)
    )


# ---------------------------------------------------------------------------
# 5. Compare filtered vs unfiltered
# ---------------------------------------------------------------------------

def compare_filtered_vs_unfiltered(
    df: pd.DataFrame,
    imbalance_config: ImbalanceStrategyConfig | None = None,
    filter_model_type: str = "logistic_regression",
    train_ratio: float = 0.8,
    feature_cols: list[str] | None = None,
    filter_threshold: float = 0.5,
) -> dict:
    """Compare imbalance-only vs imbalance+ML-filter on the same test set.

    Returns a dict with keys:
        unfiltered  -- ImbalanceResult for raw imbalance strategy on test period
        filtered    -- ImbalanceResult for ML-filtered strategy on test period
        improvement -- dict of deltas for key metrics
    """
    if imbalance_config is None:
        imbalance_config = ImbalanceStrategyConfig()

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    # Run imbalance strategy on full data
    full_result = run_imbalance_strategy(df, imbalance_config)
    all_trades = full_result.trades

    split_row = int(len(df) * train_ratio)
    test_trades = [t for t in all_trades if t["index"] >= split_row]

    # Unfiltered baseline on test set
    unfiltered = _build_imbalance_result(
        test_trades, _config_dict(imbalance_config, "none")
    )

    # Filtered result
    filtered = run_filtered_strategy(
        df,
        imbalance_config=imbalance_config,
        filter_model_type=filter_model_type,
        train_ratio=train_ratio,
        feature_cols=feature_cols,
        filter_threshold=filter_threshold,
    )

    # Compute improvement
    improvement = {
        "n_trades_delta": filtered.n_trades - unfiltered.n_trades,
        "avg_gross_bps_delta": filtered.avg_gross_bps - unfiltered.avg_gross_bps,
        "avg_net_bps_delta": filtered.avg_net_bps - unfiltered.avg_net_bps,
        "win_rate_delta": filtered.win_rate - unfiltered.win_rate,
        "win_rate_net_delta": filtered.win_rate_net - unfiltered.win_rate_net,
        "sharpe_delta": filtered.sharpe - unfiltered.sharpe,
        "profit_factor_delta": filtered.profit_factor - unfiltered.profit_factor,
        "net_pnl_bps_delta": filtered.net_pnl_bps - unfiltered.net_pnl_bps,
        "filter_kept_pct": (
            filtered.n_trades / unfiltered.n_trades * 100
            if unfiltered.n_trades > 0
            else 0.0
        ),
    }

    logger.info(
        "Filter comparison -- unfiltered: %d trades, avg_net=%.2f bps | "
        "filtered: %d trades, avg_net=%.2f bps | delta_avg_net=%.2f bps",
        unfiltered.n_trades,
        unfiltered.avg_net_bps,
        filtered.n_trades,
        filtered.avg_net_bps,
        improvement["avg_net_bps_delta"],
    )

    return {
        "unfiltered": unfiltered,
        "filtered": filtered,
        "improvement": improvement,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_dict(
    cfg: ImbalanceStrategyConfig,
    filter_model_type: str,
) -> dict:
    """Serialize config + filter info into a dict."""
    d = {
        "imbalance_depth": cfg.imbalance_depth,
        "apply_sg_filter": cfg.apply_sg_filter,
        "long_threshold_pct": cfg.long_threshold_pct,
        "short_threshold_pct": cfg.short_threshold_pct,
        "horizon_steps": cfg.horizon_steps,
        "cooldown_steps": cfg.cooldown_steps,
        "min_vol_percentile": cfg.min_vol_percentile,
        "max_spread_bps": cfg.max_spread_bps,
        "maker_fee_bps": cfg.maker_fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "filter_model_type": filter_model_type,
    }
    return d
