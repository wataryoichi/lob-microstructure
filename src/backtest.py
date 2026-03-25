"""Backtest engine for LOB prediction strategies.

Simulates trading based on model predictions with realistic costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_classification_metrics, compute_trading_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    classification_metrics: dict = field(default_factory=dict)
    trading_metrics: dict = field(default_factory=dict)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    mid_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    train_time_sec: float = 0.0
    model_name: str = ""


def run_single_backtest(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mid_prices_test: np.ndarray,
    timestamps_test: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    horizon: int = 5,
    maker_fee_bps: float = 1.0,
    taker_fee_bps: float = 5.5,
    slippage_bps: float = 1.0,
    model_name: str = "",
) -> BacktestResult:
    """Run a single model training + backtest evaluation.

    Args:
        model: BaseModel instance
        X_train/y_train: Training data
        X_test/y_test: Test data
        mid_prices_test: Mid-prices aligned with test set
        timestamps_test: Timestamps for test set
        X_val/y_val: Validation data (optional)
        horizon: Prediction horizon in timesteps
        maker_fee_bps/taker_fee_bps/slippage_bps: Cost parameters

    Returns:
        BacktestResult with all metrics
    """
    # Compute class weights
    from .labeling import compute_class_weights
    class_weights = compute_class_weights(y_train)
    sample_weight = np.array([class_weights.get(int(y), 1.0) for y in y_train])

    # Train and predict
    preds, probs, train_time = model.fit_predict(
        X_train, y_train, X_test, X_val, y_val, sample_weight
    )

    # Classification metrics
    labels = sorted(np.unique(y_test).tolist())
    cls_metrics = compute_classification_metrics(y_test, preds, labels=labels)

    # Trading metrics
    trade_metrics = compute_trading_metrics(
        preds, mid_prices_test, timestamps_test,
        horizon=horizon,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
    )

    logger.info(
        f"{model_name}: Acc={cls_metrics['accuracy']:.4f}, "
        f"F1={cls_metrics['f1_macro']:.4f}, "
        f"Trades={trade_metrics['n_trades']}, "
        f"Net PnL(maker)={trade_metrics['net_pnl_maker_bps']:.1f}bps, "
        f"Time={train_time:.1f}s"
    )

    return BacktestResult(
        classification_metrics=cls_metrics,
        trading_metrics=trade_metrics,
        predictions=preds,
        probabilities=probs,
        y_true=y_test,
        mid_prices=mid_prices_test,
        timestamps=timestamps_test if timestamps_test is not None else np.array([]),
        train_time_sec=train_time,
        model_name=model_name,
    )


def run_model_comparison(
    model_configs: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mid_prices_test: np.ndarray,
    timestamps_test: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    horizon: int = 5,
    maker_fee_bps: float = 1.0,
    taker_fee_bps: float = 5.5,
    slippage_bps: float = 1.0,
) -> list[BacktestResult]:
    """Compare multiple models on the same data split.

    Args:
        model_configs: List of dicts with 'name' and 'model' keys

    Returns:
        List of BacktestResult, one per model
    """
    results = []
    for cfg in model_configs:
        name = cfg["name"]
        model = cfg["model"]
        logger.info(f"--- Running {name} ---")

        result = run_single_backtest(
            model=model,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            mid_prices_test=mid_prices_test,
            timestamps_test=timestamps_test,
            X_val=X_val, y_val=y_val,
            horizon=horizon,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            model_name=name,
        )
        results.append(result)

    return results


def results_to_dataframe(results: list[BacktestResult]) -> pd.DataFrame:
    """Convert list of BacktestResults to a summary DataFrame."""
    rows = []
    for r in results:
        row = {"model": r.model_name, "train_time_sec": r.train_time_sec}
        row.update({f"cls_{k}": v for k, v in r.classification_metrics.items()})
        row.update({f"trade_{k}": v for k, v in r.trading_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)
