"""Model-free imbalance direct strategy.

Trades on extreme order imbalance values without ML.
This is the primary trading strategy of this project.

Signal: 5-level aggregate order imbalance (SG-filtered)
Entry: imbalance > long_threshold_pct or < short_threshold_pct (rolling percentile)
Exit: fixed horizon (non-overlapping)
Filters: volatility, spread, cooldown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cost_model import compute_round_trip_cost
from .features import compute_aggregate_imbalance, compute_mid_price
from .filters import apply_savitzky_golay

logger = logging.getLogger(__name__)


@dataclass
class ImbalanceStrategyConfig:
    # Signal
    imbalance_depth: int = 5
    apply_sg_filter: bool = True
    sg_window: int = 21
    sg_polyorder: int = 3

    # Thresholds (percentile-based, computed on rolling train window)
    long_threshold_pct: float = 0.90   # go long when imb_rank > this
    short_threshold_pct: float = 0.10  # go short when imb_rank < this
    percentile_window: int = 2000      # lookback for percentile computation

    # Horizon & cooldown
    horizon_steps: int = 100  # hold for N steps (100 = 10s at 100ms)
    cooldown_steps: int = 0   # additional gap after horizon (0 = non-overlapping only)

    # Filters
    min_vol_percentile: float = 0.0   # 0 = no filter
    vol_window: int = 500
    max_spread_bps: float = 0.0       # 0 = no filter
    time_of_day_filter: list[int] = field(default_factory=list)  # UTC hours to exclude

    # Cost
    maker_fee_bps: float = 1.0
    slippage_bps: float = 0.5


@dataclass
class ImbalanceResult:
    """Result of imbalance strategy run."""
    trades: list[dict]
    n_trades: int
    avg_gross_bps: float
    avg_net_bps: float
    gross_pnl_bps: float
    net_pnl_bps: float
    win_rate: float
    win_rate_net: float
    sharpe: float
    max_drawdown_bps: float
    profit_factor: float
    config: dict


def run_imbalance_strategy(
    df: pd.DataFrame,
    config: ImbalanceStrategyConfig | None = None,
) -> ImbalanceResult:
    """Run model-free imbalance strategy on LOB data.

    Returns ImbalanceResult with trade-by-trade breakdown.
    """
    if config is None:
        config = ImbalanceStrategyConfig()

    depth = config.imbalance_depth
    horizon = config.horizon_steps
    gap = horizon + config.cooldown_steps

    rt_cost = compute_round_trip_cost(config.maker_fee_bps, 5.5, config.slippage_bps, is_maker=True) * 1e-4

    # Extract data
    bid_p1 = df["bid_p1"].values
    ask_p1 = df["ask_p1"].values
    mid = compute_mid_price(bid_p1, ask_p1)

    bid_q = np.column_stack([df[f"bid_q{i}"].values for i in range(1, depth + 1)])
    ask_q = np.column_stack([df[f"ask_q{i}"].values for i in range(1, depth + 1)])
    imbalance = compute_aggregate_imbalance(bid_q, ask_q, depth=depth)

    # Optional SG filter on imbalance
    if config.apply_sg_filter:
        imbalance = apply_savitzky_golay(imbalance, window_size=config.sg_window, polyorder=config.sg_polyorder)

    # Compute rolling percentile rank (no lookahead)
    n = len(imbalance)
    pct_rank = np.full(n, np.nan)
    pw = config.percentile_window
    for i in range(pw, n):
        window = imbalance[i - pw : i]
        valid = window[~np.isnan(window)]
        if len(valid) > 10:
            pct_rank[i] = (valid < imbalance[i]).mean()

    # Compute volatility (rolling std of log returns)
    safe_mid = np.where(mid > 0, mid, np.nan)
    log_ret = np.diff(np.log(safe_mid))
    vol = pd.Series(log_ret).rolling(config.vol_window, min_periods=config.vol_window // 2).std().values
    vol = np.concatenate([[np.nan], vol])

    # Volatility threshold
    valid_vol = vol[~np.isnan(vol)]
    vol_thresh = np.percentile(valid_vol, config.min_vol_percentile) if len(valid_vol) > 0 and config.min_vol_percentile > 0 else 0.0

    # Spread in bps
    spread_bps = (ask_p1 - bid_p1) / np.where(mid > 0, mid, 1.0) * 10000

    # Timestamps for time-of-day filter
    has_ts = "timestamp" in df.columns
    timestamps = df["timestamp"].values if has_ts else np.zeros(n)

    # Generate trades
    trades = []
    i = pw  # start after percentile window
    while i < n - horizon:
        # Skip if invalid
        if np.isnan(pct_rank[i]) or np.isnan(vol[i]):
            i += 1
            continue

        # Volatility filter
        if vol[i] < vol_thresh:
            i += 1
            continue

        # Spread filter
        if config.max_spread_bps > 0 and spread_bps[i] > config.max_spread_bps:
            i += 1
            continue

        # Time-of-day filter
        if config.time_of_day_filter and has_ts and timestamps[i] > 0:
            hour = int((timestamps[i] / 1000) % 86400 / 3600)
            if hour in config.time_of_day_filter:
                i += 1
                continue

        # Signal check
        direction = 0
        if pct_rank[i] >= config.long_threshold_pct:
            direction = 1  # long
        elif pct_rank[i] <= config.short_threshold_pct:
            direction = -1  # short
        else:
            i += 1
            continue

        # Compute return
        if mid[i] > 0 and mid[i + horizon] > 0:
            price_return = (mid[i + horizon] - mid[i]) / mid[i]
            gross = direction * price_return
            net = gross - rt_cost

            trades.append({
                "index": i,
                "timestamp": int(timestamps[i]) if has_ts else 0,
                "direction": direction,
                "imbalance": float(imbalance[i]),
                "imb_pct_rank": float(pct_rank[i]),
                "volatility": float(vol[i]),
                "spread_bps": float(spread_bps[i]),
                "entry_price": float(mid[i]),
                "exit_price": float(mid[i + horizon]),
                "gross_bps": float(gross * 1e4),
                "net_bps": float(net * 1e4),
            })

        # Non-overlapping: skip ahead
        i += gap

    # Compute summary
    if not trades:
        return ImbalanceResult(
            trades=[], n_trades=0, avg_gross_bps=0, avg_net_bps=0,
            gross_pnl_bps=0, net_pnl_bps=0, win_rate=0, win_rate_net=0,
            sharpe=0, max_drawdown_bps=0, profit_factor=0,
            config=_config_to_dict(config),
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
        config=_config_to_dict(config),
    )


def sweep_imbalance_params(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    horizons: list[int] | None = None,
    vol_percentiles: list[float] | None = None,
    maker_fees: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep imbalance strategy parameters.

    Returns DataFrame with one row per config, sorted by avg_net_bps.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20]
    if horizons is None:
        horizons = [50, 100, 300, 600]
    if vol_percentiles is None:
        vol_percentiles = [0, 50, 70]
    if maker_fees is None:
        maker_fees = [1.0, 0.0]  # standard, VIP

    results = []
    total = len(thresholds) * len(horizons) * len(vol_percentiles) * len(maker_fees)
    count = 0

    for thresh in thresholds:
        for horizon in horizons:
            for vol_pct in vol_percentiles:
                for maker_fee in maker_fees:
                    count += 1
                    cfg = ImbalanceStrategyConfig(
                        long_threshold_pct=1.0 - thresh,
                        short_threshold_pct=thresh,
                        horizon_steps=horizon,
                        min_vol_percentile=vol_pct,
                        maker_fee_bps=maker_fee,
                    )

                    result = run_imbalance_strategy(df, cfg)

                    if result.n_trades >= 5:
                        fee_label = "VIP" if maker_fee == 0 else "std"
                        results.append({
                            "threshold": thresh,
                            "horizon_s": horizon * 0.1,
                            "vol_pct": vol_pct,
                            "fee_tier": fee_label,
                            "n_trades": result.n_trades,
                            "avg_gross_bps": result.avg_gross_bps,
                            "avg_net_bps": result.avg_net_bps,
                            "total_net_bps": result.net_pnl_bps,
                            "win_rate": result.win_rate,
                            "win_rate_net": result.win_rate_net,
                            "sharpe": result.sharpe,
                            "max_dd_bps": result.max_drawdown_bps,
                            "profit_factor": result.profit_factor,
                        })

                    if count % 20 == 0:
                        logger.info(f"Sweep progress: {count}/{total}")

    df_out = pd.DataFrame(results)
    if len(df_out) > 0:
        df_out = df_out.sort_values("avg_net_bps", ascending=False)
    return df_out


def _config_to_dict(cfg: ImbalanceStrategyConfig) -> dict:
    """Convert config to serializable dict."""
    return {
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
    }
