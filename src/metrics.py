"""Classification and trading performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> dict:
    """Compute comprehensive classification metrics.

    Returns dict with accuracy, f1 (per class and macro), precision, recall.
    """
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    n_classes = len(labels)
    avg = "binary" if n_classes == 2 else "macro"

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "n_samples": len(y_true),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    for i, cls in enumerate(labels):
        result[f"f1_class_{cls}"] = float(per_class_f1[i])
        result[f"support_class_{cls}"] = int((y_true == cls).sum())

    return result


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_trading_metrics(
    predictions: np.ndarray,
    mid_prices: np.ndarray,
    timestamps: np.ndarray | None = None,
    horizon: int = 5,
    maker_fee_bps: float = 1.0,
    taker_fee_bps: float = 5.5,
    slippage_bps: float = 1.0,
) -> dict:
    """Compute trading performance metrics from predictions.

    Assumes:
    - Prediction 1 (up) -> go long
    - Prediction 0 (down) -> go short (or flat)
    - Position held for `horizon` timesteps

    Returns dict with PnL, Sharpe, hit ratio, etc.
    """
    bps = 1e-4
    n = len(predictions)

    # Compute returns over horizon
    returns = np.full(n, np.nan)
    for i in range(n - horizon):
        if mid_prices[i] > 0:
            returns[i] = (mid_prices[i + horizon] - mid_prices[i]) / mid_prices[i]

    valid = ~np.isnan(returns) & (predictions >= 0)
    if valid.sum() == 0:
        return {"gross_pnl": 0, "net_pnl": 0, "n_trades": 0}

    # Directional returns: long if up prediction, short if down
    # For binary: pred=1 -> +return, pred=0 -> -return
    direction = np.where(predictions == 1, 1.0, -1.0)
    if np.max(predictions) == 2:  # ternary: flat=1
        direction = np.where(predictions == 2, 1.0, np.where(predictions == 0, -1.0, 0.0))

    trade_returns = direction[valid] * returns[valid]

    # Cost per trade (round-trip)
    cost_per_trade = (taker_fee_bps + slippage_bps) * 2 * bps  # round-trip taker
    maker_cost = (maker_fee_bps + slippage_bps) * 2 * bps  # round-trip maker

    n_trades = int(valid.sum())
    gross_pnl = float(trade_returns.sum())
    net_pnl_taker = float(trade_returns.sum() - n_trades * cost_per_trade)
    net_pnl_maker = float(trade_returns.sum() - n_trades * maker_cost)

    # Per-trade stats
    avg_return = float(trade_returns.mean()) if n_trades > 0 else 0.0
    hit_ratio = float((trade_returns > 0).mean()) if n_trades > 0 else 0.0

    # Sharpe-like metric (annualized assuming 100ms intervals)
    # ~86400 * 10 = 864000 intervals per day, ~365 days per year
    if trade_returns.std() > 1e-12:
        intervals_per_year = 864000 * 365
        sharpe = avg_return / trade_returns.std() * np.sqrt(intervals_per_year / horizon)
    else:
        sharpe = 0.0

    return {
        "n_trades": n_trades,
        "gross_pnl_bps": gross_pnl / bps,
        "net_pnl_taker_bps": net_pnl_taker / bps,
        "net_pnl_maker_bps": net_pnl_maker / bps,
        "avg_return_bps": avg_return / bps,
        "hit_ratio": hit_ratio,
        "sharpe": float(sharpe),
        "max_drawdown_bps": float(_max_drawdown(trade_returns) / bps),
        "cost_per_trade_taker_bps": cost_per_trade / bps,
        "cost_per_trade_maker_bps": maker_cost / bps,
    }


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a return series."""
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0


def format_metrics_table(
    results: list[dict],
    title: str = "Model Comparison",
) -> str:
    """Format metrics as a markdown table."""
    if not results:
        return "No results."

    df = pd.DataFrame(results)
    return f"## {title}\n\n{df.to_markdown(index=False)}"
