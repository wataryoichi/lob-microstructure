"""Regime analysis: slice results by volatility, spread, time-of-day, symbol.

Answers: "Is the edge stable, or only present in specific conditions?"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_regime_labels(
    df: pd.DataFrame,
    vol_window: int = 500,
    vol_quantiles: int = 3,
    spread_quantiles: int = 3,
) -> pd.DataFrame:
    """Add regime columns to a LOB DataFrame.

    Adds: vol_regime, spread_regime, hour_utc, hour_bucket
    All computed without lookahead (rolling/trailing).
    """
    result = df.copy()
    n = len(df)

    # Mid-price and spread
    mid = (df["bid_p1"].values + df["ask_p1"].values) / 2
    spread = df["ask_p1"].values - df["bid_p1"].values
    rel_spread_bps = spread / np.where(mid > 0, mid, 1.0) * 10000

    # Rolling volatility (trailing, no lookahead)
    safe_mid = np.where(mid > 0, mid, np.nan)
    log_ret = np.concatenate([[np.nan], np.diff(np.log(safe_mid))])
    vol = pd.Series(log_ret).rolling(vol_window, min_periods=vol_window // 2).std().values

    # Volatility regime: tertiles on trailing window
    vol_labels = np.full(n, "unknown", dtype=object)
    for i in range(vol_window, n):
        window_vol = vol[max(0, i - vol_window) : i]
        valid = window_vol[~np.isnan(window_vol)]
        if len(valid) < 10:
            continue
        p33 = np.percentile(valid, 33)
        p67 = np.percentile(valid, 67)
        if np.isnan(vol[i]):
            continue
        elif vol[i] <= p33:
            vol_labels[i] = "low"
        elif vol[i] <= p67:
            vol_labels[i] = "med"
        else:
            vol_labels[i] = "high"

    result["vol_regime"] = vol_labels

    # Spread regime: tertiles
    spread_labels = np.full(n, "unknown", dtype=object)
    for i in range(vol_window, n):
        window_sp = rel_spread_bps[max(0, i - vol_window) : i]
        valid = window_sp[~np.isnan(window_sp)]
        if len(valid) < 10:
            continue
        p33 = np.percentile(valid, 33)
        p67 = np.percentile(valid, 67)
        if np.isnan(rel_spread_bps[i]):
            continue
        elif rel_spread_bps[i] <= p33:
            spread_labels[i] = "tight"
        elif rel_spread_bps[i] <= p67:
            spread_labels[i] = "med"
        else:
            spread_labels[i] = "wide"

    result["spread_regime"] = spread_labels

    # Time of day (UTC)
    if "timestamp" in df.columns:
        ts_sec = df["timestamp"].values / 1000.0
        hours = ((ts_sec % 86400) / 3600).astype(int)
        result["hour_utc"] = hours
        # 4-hour buckets
        buckets = (hours // 4) * 4
        result["hour_bucket"] = [f"{b:02d}-{b+4:02d}" for b in buckets]
    else:
        result["hour_utc"] = 0
        result["hour_bucket"] = "unknown"

    return result


def analyze_trades_by_regime(
    trades: list[dict],
    regime_cols: list[str] | None = None,
    min_trades: int = 5,
) -> pd.DataFrame:
    """Analyze trades across regime slices.

    Args:
        trades: List of trade dicts (from ImbalanceResult.trades or similar)
        regime_cols: Columns to group by (default: all regime columns)
        min_trades: Minimum trades per group to report

    Returns:
        DataFrame with metrics per regime slice
    """
    if not trades:
        return pd.DataFrame()

    tdf = pd.DataFrame(trades)

    if regime_cols is None:
        # Auto-detect regime columns
        regime_cols = [c for c in tdf.columns if c in [
            "vol_regime", "spread_regime", "hour_bucket", "symbol"
        ]]
        if not regime_cols:
            regime_cols = ["all"]
            tdf["all"] = "all"

    results = []

    # Per-dimension analysis
    for col in regime_cols:
        if col not in tdf.columns:
            continue
        for group_val, group_df in tdf.groupby(col):
            if len(group_df) < min_trades:
                continue
            gross = group_df["gross_bps"].values
            net = group_df["net_bps"].values

            wins_gross = gross[gross > 0].sum()
            losses_gross = abs(gross[gross < 0].sum())
            pf = wins_gross / losses_gross if losses_gross > 0 else float("inf")

            results.append({
                "dimension": col,
                "value": str(group_val),
                "n_trades": len(group_df),
                "avg_gross_bps": float(np.mean(gross)),
                "avg_net_bps": float(np.mean(net)),
                "total_net_bps": float(np.sum(net)),
                "win_rate": float((gross > 0).mean()),
                "win_rate_net": float((net > 0).mean()),
                "profit_factor": float(pf),
                "avg_imbalance": float(group_df["imbalance"].mean()) if "imbalance" in group_df.columns else 0,
            })

    return pd.DataFrame(results)


def generate_regime_report(
    trades: list[dict],
    title: str = "Regime Analysis",
) -> str:
    """Generate markdown regime analysis report."""
    regime_df = analyze_trades_by_regime(trades)
    if regime_df.empty:
        return f"# {title}\n\nNo trades to analyze."

    lines = [f"# {title}", ""]

    for dim in regime_df["dimension"].unique():
        sub = regime_df[regime_df["dimension"] == dim].copy()
        sub = sub.sort_values("avg_net_bps", ascending=False)

        lines.append(f"## By {dim}")
        lines.append("")
        cols = ["value", "n_trades", "avg_gross_bps", "avg_net_bps", "win_rate", "profit_factor"]
        lines.append(sub[cols].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

        # Highlight best/worst
        best = sub.iloc[0]
        worst = sub.iloc[-1]
        lines.append(f"**Best**: {best['value']} (net {best['avg_net_bps']:.3f} bps/trade)")
        lines.append(f"**Worst**: {worst['value']} (net {worst['avg_net_bps']:.3f} bps/trade)")
        lines.append("")

    return "\n".join(lines)
