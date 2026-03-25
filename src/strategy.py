"""Trading strategy implementations for practical deployment.

Builds on paper's classification insights with cost-aware adaptations:
1. Longer horizons (1-5 min) where moves exceed costs
2. Volatility regime filtering (only trade high-vol periods)
3. Asymmetric position sizing based on confidence
4. Non-overlapping trade windows to avoid autocorrelation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cost_model import compute_round_trip_cost
from .labeling import compute_class_weights

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Single trade signal with metadata."""
    timestamp: int
    direction: int  # +1 = long, -1 = short, 0 = flat
    confidence: float
    horizon_steps: int
    entry_price: float


@dataclass
class StrategyResult:
    """Results from a strategy backtest."""
    trades: list[dict]
    gross_pnl_bps: float
    net_pnl_bps: float
    n_trades: int
    win_rate: float
    avg_gross_bps: float
    avg_net_bps: float
    sharpe: float
    max_drawdown_bps: float


def run_adaptive_strategy(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mid_prices: np.ndarray,
    timestamps: np.ndarray | None = None,
    horizon: int = 100,
    min_confidence: float = 0.55,
    vol_filter_percentile: float = 50.0,
    vol_window: int = 500,
    non_overlapping: bool = True,
    maker_fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
) -> StrategyResult:
    """Run adaptive strategy with volatility filtering and confidence thresholds.

    Args:
        model: Trained model with predict_proba
        X_test: Test features
        y_test: True labels
        mid_prices: Raw mid-prices aligned with test set
        timestamps: Timestamps (optional)
        horizon: Prediction horizon in timesteps
        min_confidence: Minimum probability to enter trade
        vol_filter_percentile: Only trade when vol > this percentile
        vol_window: Window for volatility computation
        non_overlapping: If True, wait horizon steps between trades
        maker_fee_bps: One-way maker fee
        slippage_bps: Estimated slippage

    Returns:
        StrategyResult with trade-by-trade analysis
    """
    rt_cost = compute_round_trip_cost(maker_fee_bps, 5.5, slippage_bps, is_maker=True)
    rt_cost_dec = rt_cost * 1e-4

    # Compute volatility
    returns = np.diff(np.log(np.maximum(mid_prices, 1e-10)))
    rolling_vol = pd.Series(returns).rolling(vol_window, min_periods=vol_window // 2).std().values
    rolling_vol = np.concatenate([[np.nan], rolling_vol])

    # Volatility threshold
    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    if len(valid_vol) > 0 and vol_filter_percentile > 0:
        vol_threshold = np.percentile(valid_vol, vol_filter_percentile)
    else:
        vol_threshold = 0.0

    # Get predictions
    probs = model.predict_proba(X_test)

    trades = []
    i = 0
    while i < len(X_test) - horizon:
        # Check volatility filter
        if np.isnan(rolling_vol[i]) or rolling_vol[i] < vol_threshold:
            i += 1
            continue

        # Check confidence
        max_prob = np.max(probs[i])
        if max_prob < min_confidence:
            i += 1
            continue

        # Generate signal
        pred_class = np.argmax(probs[i])
        direction = 1.0 if pred_class == 1 else -1.0

        # Compute return
        if mid_prices[i] > 0 and mid_prices[i + horizon] > 0:
            price_return = (mid_prices[i + horizon] - mid_prices[i]) / mid_prices[i]
            trade_return = direction * price_return
            net_return = trade_return - rt_cost_dec

            trades.append({
                "index": i,
                "timestamp": timestamps[i] if timestamps is not None else i,
                "direction": int(direction),
                "confidence": float(max_prob),
                "pred_class": int(pred_class),
                "true_class": int(y_test[i]),
                "correct": int(pred_class) == int(y_test[i]),
                "entry_price": float(mid_prices[i]),
                "exit_price": float(mid_prices[i + horizon]),
                "gross_return": float(trade_return),
                "net_return": float(net_return),
                "gross_bps": float(trade_return * 1e4),
                "net_bps": float(net_return * 1e4),
                "volatility": float(rolling_vol[i]),
            })

        # Skip forward
        if non_overlapping:
            i += horizon
        else:
            i += 1

    if not trades:
        return StrategyResult(
            trades=[], gross_pnl_bps=0, net_pnl_bps=0,
            n_trades=0, win_rate=0, avg_gross_bps=0, avg_net_bps=0,
            sharpe=0, max_drawdown_bps=0,
        )

    # Compute summary
    gross_rets = np.array([t["gross_return"] for t in trades])
    net_rets = np.array([t["net_return"] for t in trades])

    gross_pnl = np.sum(gross_rets) * 1e4
    net_pnl = np.sum(net_rets) * 1e4
    n_trades = len(trades)
    win_rate = (net_rets > 0).mean()
    avg_gross = np.mean(gross_rets) * 1e4
    avg_net = np.mean(net_rets) * 1e4

    # Sharpe (per-trade)
    if net_rets.std() > 1e-12:
        sharpe = np.mean(net_rets) / net_rets.std() * np.sqrt(n_trades)
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = np.cumsum(net_rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = abs(np.min(drawdown)) * 1e4 if len(drawdown) > 0 else 0.0

    return StrategyResult(
        trades=trades,
        gross_pnl_bps=float(gross_pnl),
        net_pnl_bps=float(net_pnl),
        n_trades=n_trades,
        win_rate=float(win_rate),
        avg_gross_bps=float(avg_gross),
        avg_net_bps=float(avg_net),
        sharpe=float(sharpe),
        max_drawdown_bps=float(max_dd),
    )


def sweep_strategy_params(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mid_prices: np.ndarray,
    timestamps: np.ndarray | None = None,
    horizons: list[int] | None = None,
    confidences: list[float] | None = None,
    vol_percentiles: list[float] | None = None,
    maker_fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
) -> pd.DataFrame:
    """Sweep strategy parameters to find optimal configuration."""
    if horizons is None:
        horizons = [50, 100, 200, 500, 1000]
    if confidences is None:
        confidences = [0.5, 0.55, 0.6, 0.7]
    if vol_percentiles is None:
        vol_percentiles = [0, 30, 50, 70]

    results = []
    for h in horizons:
        for conf in confidences:
            for vol_pct in vol_percentiles:
                result = run_adaptive_strategy(
                    model=model,
                    X_test=X_test, y_test=y_test,
                    mid_prices=mid_prices,
                    timestamps=timestamps,
                    horizon=h,
                    min_confidence=conf,
                    vol_filter_percentile=vol_pct,
                    non_overlapping=True,
                    maker_fee_bps=maker_fee_bps,
                    slippage_bps=slippage_bps,
                )

                if result.n_trades >= 3:
                    results.append({
                        "horizon_s": h * 0.1,
                        "confidence": conf,
                        "vol_pct": vol_pct,
                        "n_trades": result.n_trades,
                        "win_rate": result.win_rate,
                        "gross_pnl_bps": result.gross_pnl_bps,
                        "net_pnl_bps": result.net_pnl_bps,
                        "avg_gross_bps": result.avg_gross_bps,
                        "avg_net_bps": result.avg_net_bps,
                        "sharpe": result.sharpe,
                        "max_dd_bps": result.max_drawdown_bps,
                    })

    return pd.DataFrame(results)
