"""Rolling Walk-forward validation framework.

Splits data into rolling windows:
  Train(X hours) -> Test(Y hours) -> slide forward by Y hours -> repeat

At each step:
  1. Compute imbalance percentile thresholds on Train period only
  2. Train LR filter on Train period candidates
  3. Evaluate on Test period (strict OOS)
  4. Collect per-window metrics

No future information leakage:
  - Percentile thresholds: trailing window from Train only
  - Normalization: not applied (raw imbalance used)
  - LR filter: fit on Train trades, predict on Test trades
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import build_feature_matrix, compute_aggregate_imbalance, compute_mid_price
from .filters import apply_savitzky_golay
from .imbalance_strategy import ImbalanceStrategyConfig, ImbalanceResult, run_imbalance_strategy
from .ml_filter import (
    DEFAULT_FEATURE_COLS,
    prepare_filter_training_data,
    train_filter_model,
    apply_filter,
)

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result for a single walk-forward window."""
    window_idx: int
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int
    train_n_snaps: int
    test_n_snaps: int
    # Unfiltered
    unfilt_n_trades: int = 0
    unfilt_avg_gross_bps: float = 0.0
    unfilt_avg_net_bps: float = 0.0
    unfilt_win_rate: float = 0.0
    unfilt_pf: float = 0.0
    # Filtered
    filt_n_trades: int = 0
    filt_avg_gross_bps: float = 0.0
    filt_avg_net_bps: float = 0.0
    filt_win_rate: float = 0.0
    filt_pf: float = 0.0
    # All individual trades (for cumulative PnL)
    unfilt_trades: list = field(default_factory=list)
    filt_trades: list = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward result."""
    windows: list[WindowResult]
    # Aggregated unfiltered
    total_unfilt_trades: int = 0
    avg_unfilt_net_bps: float = 0.0
    unfilt_win_rate: float = 0.0
    unfilt_sharpe: float = 0.0
    unfilt_max_dd_bps: float = 0.0
    unfilt_pf: float = 0.0
    # Aggregated filtered
    total_filt_trades: int = 0
    avg_filt_net_bps: float = 0.0
    filt_win_rate: float = 0.0
    filt_sharpe: float = 0.0
    filt_max_dd_bps: float = 0.0
    filt_pf: float = 0.0
    # Per-window consistency
    n_windows: int = 0
    n_unfilt_positive_windows: int = 0
    n_filt_positive_windows: int = 0


def run_walk_forward(
    df: pd.DataFrame,
    imbalance_config: ImbalanceStrategyConfig,
    train_hours: float = 1.0,
    test_hours: float = 0.5,
    filter_model_type: str = "logistic_regression",
    filter_threshold: float = 0.55,
    feature_cols: list[str] | None = None,
    min_train_trades: int = 10,
) -> WalkForwardResult:
    """Run rolling walk-forward validation.

    Args:
        df: Full LOB DataFrame with valid timestamps, sorted by time
        imbalance_config: Strategy config
        train_hours: Training window duration
        test_hours: Test window duration
        filter_model_type: ML filter model
        filter_threshold: Probability threshold for filter
        feature_cols: Features for ML filter
        min_train_trades: Minimum trades in Train to train filter

    Returns:
        WalkForwardResult with per-window and aggregated metrics
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    train_ms = int(train_hours * 3600 * 1000)
    test_ms = int(test_hours * 3600 * 1000)

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    total_ms = ts_max - ts_min

    logger.info(f"Walk-forward: train={train_hours}h, test={test_hours}h, "
                f"total={total_ms/3600000:.2f}h")

    windows = []
    cursor = ts_min + train_ms  # First test window starts after first train window
    window_idx = 0

    while cursor + test_ms <= ts_max:
        train_start = cursor - train_ms
        train_end = cursor
        test_start = cursor
        test_end = cursor + test_ms

        train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
        test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] < test_end)

        train_df = df[train_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        if len(train_df) < 1000 or len(test_df) < 100:
            cursor += test_ms
            continue

        wr = _evaluate_window(
            window_idx, train_df, test_df,
            imbalance_config, filter_model_type, filter_threshold,
            feature_cols, min_train_trades,
        )
        wr.train_start_ts = int(train_start)
        wr.train_end_ts = int(train_end)
        wr.test_start_ts = int(test_start)
        wr.test_end_ts = int(test_end)

        windows.append(wr)
        window_idx += 1
        cursor += test_ms

        logger.info(f"Window {window_idx}: "
                     f"unfilt n={wr.unfilt_n_trades} net={wr.unfilt_avg_net_bps:+.3f} | "
                     f"filt n={wr.filt_n_trades} net={wr.filt_avg_net_bps:+.3f}")

    return _aggregate_results(windows)


def _evaluate_window(
    window_idx: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ImbalanceStrategyConfig,
    filter_model_type: str,
    filter_threshold: float,
    feature_cols: list[str],
    min_train_trades: int,
) -> WindowResult:
    """Evaluate a single train/test window."""
    wr = WindowResult(
        window_idx=window_idx,
        train_start_ts=0, train_end_ts=0,
        test_start_ts=0, test_end_ts=0,
        train_n_snaps=len(train_df),
        test_n_snaps=len(test_df),
    )

    # 1. Run imbalance strategy on test period (unfiltered)
    test_result = run_imbalance_strategy(test_df, config)
    wr.unfilt_n_trades = test_result.n_trades
    wr.unfilt_avg_gross_bps = test_result.avg_gross_bps
    wr.unfilt_avg_net_bps = test_result.avg_net_bps
    wr.unfilt_win_rate = test_result.win_rate
    wr.unfilt_pf = test_result.profit_factor
    wr.unfilt_trades = test_result.trades

    if test_result.n_trades == 0:
        return wr

    # 2. Train ML filter on train period, apply to test period
    train_result = run_imbalance_strategy(train_df, config)

    if train_result.n_trades < min_train_trades:
        # Not enough train trades for filter — use unfiltered for both
        wr.filt_n_trades = wr.unfilt_n_trades
        wr.filt_avg_gross_bps = wr.unfilt_avg_gross_bps
        wr.filt_avg_net_bps = wr.unfilt_avg_net_bps
        wr.filt_win_rate = wr.unfilt_win_rate
        wr.filt_pf = wr.unfilt_pf
        wr.filt_trades = wr.unfilt_trades
        return wr

    try:
        # Build features for train trades
        X_train, y_train, _ = prepare_filter_training_data(
            train_df, train_result.trades, feature_cols=feature_cols
        )

        if len(X_train) < min_train_trades:
            wr.filt_n_trades = wr.unfilt_n_trades
            wr.filt_avg_gross_bps = wr.unfilt_avg_gross_bps
            wr.filt_avg_net_bps = wr.unfilt_avg_net_bps
            wr.filt_win_rate = wr.unfilt_win_rate
            wr.filt_pf = wr.unfilt_pf
            wr.filt_trades = wr.unfilt_trades
            return wr

        # Train filter
        model, scaler = train_filter_model(X_train, y_train, model_type=filter_model_type)

        # Apply filter to test trades
        filtered_trades = apply_filter(
            model, scaler, test_df, test_result.trades,
            feature_cols=feature_cols, threshold=filter_threshold,
        )

        if filtered_trades:
            gross_arr = np.array([t["gross_bps"] for t in filtered_trades])
            net_arr = np.array([t["net_bps"] for t in filtered_trades])
            wins = gross_arr[gross_arr > 0].sum()
            losses = abs(gross_arr[gross_arr < 0].sum())

            wr.filt_n_trades = len(filtered_trades)
            wr.filt_avg_gross_bps = float(np.mean(gross_arr))
            wr.filt_avg_net_bps = float(np.mean(net_arr))
            wr.filt_win_rate = float((gross_arr > 0).mean())
            wr.filt_pf = float(wins / losses) if losses > 0 else float("inf")
            wr.filt_trades = filtered_trades
        else:
            wr.filt_n_trades = 0

    except Exception as e:
        logger.warning(f"Filter failed in window {window_idx}: {e}")
        wr.filt_n_trades = wr.unfilt_n_trades
        wr.filt_avg_net_bps = wr.unfilt_avg_net_bps
        wr.filt_win_rate = wr.unfilt_win_rate
        wr.filt_pf = wr.unfilt_pf
        wr.filt_trades = wr.unfilt_trades

    return wr


def _aggregate_results(windows: list[WindowResult]) -> WalkForwardResult:
    """Aggregate per-window results into a single WalkForwardResult."""
    if not windows:
        return WalkForwardResult(windows=[])

    # Collect all trades across all windows
    all_unfilt = []
    all_filt = []
    for w in windows:
        all_unfilt.extend(w.unfilt_trades)
        all_filt.extend(w.filt_trades)

    result = WalkForwardResult(windows=windows, n_windows=len(windows))

    # Unfiltered aggregation
    if all_unfilt:
        gross = np.array([t["gross_bps"] for t in all_unfilt])
        net = np.array([t["net_bps"] for t in all_unfilt])
        result.total_unfilt_trades = len(all_unfilt)
        result.avg_unfilt_net_bps = float(np.mean(net))
        result.unfilt_win_rate = float((gross > 0).mean())
        result.unfilt_sharpe = float(np.mean(net) / net.std() * np.sqrt(len(net))) if net.std() > 1e-12 else 0
        wins = gross[gross > 0].sum()
        losses = abs(gross[gross < 0].sum())
        result.unfilt_pf = float(wins / losses) if losses > 0 else 0
        # Max drawdown
        cumul = np.cumsum(net)
        dd = cumul - np.maximum.accumulate(cumul)
        result.unfilt_max_dd_bps = float(abs(np.min(dd))) if len(dd) > 0 else 0

    # Filtered aggregation
    if all_filt:
        gross = np.array([t["gross_bps"] for t in all_filt])
        net = np.array([t["net_bps"] for t in all_filt])
        result.total_filt_trades = len(all_filt)
        result.avg_filt_net_bps = float(np.mean(net))
        result.filt_win_rate = float((gross > 0).mean())
        result.filt_sharpe = float(np.mean(net) / net.std() * np.sqrt(len(net))) if net.std() > 1e-12 else 0
        wins = gross[gross > 0].sum()
        losses = abs(gross[gross < 0].sum())
        result.filt_pf = float(wins / losses) if losses > 0 else 0
        cumul = np.cumsum(net)
        dd = cumul - np.maximum.accumulate(cumul)
        result.filt_max_dd_bps = float(abs(np.min(dd))) if len(dd) > 0 else 0

    # Per-window consistency
    result.n_unfilt_positive_windows = sum(1 for w in windows if w.unfilt_avg_net_bps > 0 and w.unfilt_n_trades > 0)
    result.n_filt_positive_windows = sum(1 for w in windows if w.filt_avg_net_bps > 0 and w.filt_n_trades > 0)

    return result


def format_walk_forward_report(
    result: WalkForwardResult,
    symbol: str = "BTCUSDT",
    config_name: str = "",
) -> str:
    """Generate markdown report from walk-forward results."""
    lines = [
        f"# Walk-Forward Analysis: {symbol}",
        f"Config: {config_name}" if config_name else "",
        "",
        "## Summary",
        "",
        f"| Metric | Unfiltered | LR Filtered |",
        f"|--------|-----------|-------------|",
        f"| Total trades | {result.total_unfilt_trades} | {result.total_filt_trades} |",
        f"| Avg net bps | {result.avg_unfilt_net_bps:+.3f} | {result.avg_filt_net_bps:+.3f} |",
        f"| Win rate | {result.unfilt_win_rate:.1%} | {result.filt_win_rate:.1%} |",
        f"| Profit factor | {result.unfilt_pf:.2f} | {result.filt_pf:.2f} |",
        f"| Sharpe | {result.unfilt_sharpe:.2f} | {result.filt_sharpe:.2f} |",
        f"| Max drawdown | {result.unfilt_max_dd_bps:.1f} bps | {result.filt_max_dd_bps:.1f} bps |",
        f"| Positive windows | {result.n_unfilt_positive_windows}/{result.n_windows} | {result.n_filt_positive_windows}/{result.n_windows} |",
        "",
        "## Per-Window Detail",
        "",
        "| Window | Train snaps | Test snaps | Unfilt n | Unfilt net | Filt n | Filt net | Filt win |",
        "|--------|-----------|----------|---------|-----------|-------|---------|---------|",
    ]

    for w in result.windows:
        lines.append(
            f"| {w.window_idx} | {w.train_n_snaps:,} | {w.test_n_snaps:,} | "
            f"{w.unfilt_n_trades} | {w.unfilt_avg_net_bps:+.3f} | "
            f"{w.filt_n_trades} | {w.filt_avg_net_bps:+.3f} | {w.filt_win_rate:.1%} |"
        )

    lines.append("")

    # Cumulative PnL series
    all_filt_trades = []
    for w in result.windows:
        all_filt_trades.extend(w.filt_trades)

    if all_filt_trades:
        net_arr = [t["net_bps"] for t in all_filt_trades]
        cumul = np.cumsum(net_arr)
        lines.append("## Cumulative Net PnL (filtered)")
        lines.append(f"- Start: 0.000 bps")
        # Show every 10th trade or at window boundaries
        step = max(1, len(cumul) // 20)
        for i in range(0, len(cumul), step):
            lines.append(f"- Trade {i+1}: {cumul[i]:+.1f} bps")
        lines.append(f"- Final ({len(cumul)} trades): {cumul[-1]:+.1f} bps")
        lines.append("")

    return "\n".join(lines)
