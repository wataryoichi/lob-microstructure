"""Execution analytics for Paper Trading.

Reads SQLite logs and computes execution quality metrics:
- Maker fill rate (entry/exit)
- Taker fallback rate and cost impact
- Time-to-fill distribution
- Spread at entry analysis
- PnL decomposition (gross / maker cost / taker penalty)
- What-if scenarios (all-maker vs all-taker)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = "results/paper_trades.db"


def _connect_readonly(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open SQLite in read-only mode (WAL-safe with running writer)."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)


# --------------------------------------------------------------------------
# Task 1: Execution quality metrics
# --------------------------------------------------------------------------

@dataclass
class ExecutionMetrics:
    """Execution quality summary."""
    # Fill rates
    n_trades: int = 0
    entry_maker_rate: float = 0.0
    exit_maker_rate: float = 0.0
    exit_taker_rate: float = 0.0
    # Time-to-fill (entry)
    avg_time_to_fill_ms: float = 0.0
    median_time_to_fill_ms: float = 0.0
    p90_time_to_fill_ms: float = 0.0
    # Spread at entry
    avg_spread_at_entry_bps: float = 0.0
    median_spread_at_entry_bps: float = 0.0
    max_spread_at_entry_bps: float = 0.0
    # Signal acceptance
    n_signals: int = 0
    signal_accept_rate: float = 0.0
    reject_spread_count: int = 0
    # Orders
    n_orders: int = 0
    order_fill_rate: float = 0.0
    order_cancel_rate: float = 0.0


def compute_execution_metrics(db_path: str = DB_PATH) -> dict[str, ExecutionMetrics]:
    """Compute execution metrics per config from SQLite data."""
    conn = _connect_readonly(db_path)
    results: dict[str, ExecutionMetrics] = {}

    # Trade-level metrics
    trades = pd.read_sql("SELECT * FROM trades", conn)
    signals = pd.read_sql("SELECT * FROM signals", conn)
    orders = pd.read_sql("SELECT * FROM orders", conn)

    for config_name in trades["config_name"].unique():
        m = ExecutionMetrics()
        t = trades[trades["config_name"] == config_name]
        s = signals[signals["config_name"] == config_name]
        o = orders[orders["config_name"] == config_name]

        m.n_trades = len(t)

        if m.n_trades > 0:
            m.entry_maker_rate = (t["entry_type"] == "maker").mean()
            m.exit_maker_rate = (t["exit_type"] == "maker").mean()
            m.exit_taker_rate = (t["exit_type"] == "taker").mean()

        # Time-to-fill: from order placed_ts to fill_ts
        filled = o[o["status"] == "filled"]
        if len(filled) > 0 and "fill_ts" in filled.columns:
            ttf = (filled["fill_ts"] - filled["ts"]).values
            ttf = ttf[ttf >= 0]
            if len(ttf) > 0:
                m.avg_time_to_fill_ms = float(np.mean(ttf))
                m.median_time_to_fill_ms = float(np.median(ttf))
                m.p90_time_to_fill_ms = float(np.percentile(ttf, 90))

        # Spread at entry (from signals)
        accepted = s[s["accepted"] == 1]
        if len(accepted) > 0 and "spread_bps" in accepted.columns:
            sp = accepted["spread_bps"].values
            m.avg_spread_at_entry_bps = float(np.mean(sp))
            m.median_spread_at_entry_bps = float(np.median(sp))
            m.max_spread_at_entry_bps = float(np.max(sp))

        # Signal acceptance
        m.n_signals = len(s)
        m.signal_accept_rate = s["accepted"].mean() if len(s) > 0 else 0.0
        m.reject_spread_count = int((s["reject_reason"] == "spread_too_wide").sum())

        # Order fill/cancel rates
        m.n_orders = len(o)
        if len(o) > 0:
            m.order_fill_rate = (o["status"] == "filled").mean()
            m.order_cancel_rate = (o["status"] == "cancelled").mean()

        results[config_name] = m

    conn.close()
    return results


# --------------------------------------------------------------------------
# Task 2: PnL decomposition & what-if scenarios
# --------------------------------------------------------------------------

@dataclass
class PnLDecomposition:
    """PnL breakdown per config."""
    n_trades: int = 0
    total_gross_bps: float = 0.0
    total_net_bps: float = 0.0
    avg_gross_bps: float = 0.0
    avg_net_bps: float = 0.0
    # Cost breakdown
    total_maker_cost_bps: float = 0.0
    total_taker_penalty_bps: float = 0.0
    total_slippage_bps: float = 0.0
    # What-if
    net_if_all_maker_bps: float = 0.0   # best case
    net_if_all_taker_bps: float = 0.0   # worst case
    # Per-trade stats
    win_rate: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    profit_factor: float = 0.0
    max_win_bps: float = 0.0
    max_loss_bps: float = 0.0
    sharpe: float = 0.0
    max_dd_bps: float = 0.0


def compute_pnl_decomposition(
    db_path: str = DB_PATH,
    maker_fee_bps: float = 0.0,
    taker_fee_bps: float = 5.5,
    slippage_bps: float = 0.5,
) -> dict[str, PnLDecomposition]:
    """Decompose PnL into gross, maker cost, taker penalty, slippage."""
    conn = _connect_readonly(db_path)
    trades = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()

    results: dict[str, PnLDecomposition] = {}

    for config_name in trades["config_name"].unique():
        t = trades[trades["config_name"] == config_name].copy()
        d = PnLDecomposition()
        d.n_trades = len(t)

        if d.n_trades == 0:
            results[config_name] = d
            continue

        gross = t["gross_bps"].values
        net = t["net_bps"].values

        d.total_gross_bps = float(np.sum(gross))
        d.total_net_bps = float(np.sum(net))
        d.avg_gross_bps = float(np.mean(gross))
        d.avg_net_bps = float(np.mean(net))

        # Cost decomposition per trade
        bps = 1e-4
        for _, row in t.iterrows():
            entry_cost = maker_fee_bps if row["entry_type"] == "maker" else taker_fee_bps
            exit_cost = maker_fee_bps if row["exit_type"] == "maker" else taker_fee_bps

            d.total_maker_cost_bps += (entry_cost + exit_cost) if row["entry_type"] == "maker" and row["exit_type"] == "maker" else 0
            if row["exit_type"] == "taker":
                d.total_taker_penalty_bps += (taker_fee_bps - maker_fee_bps)
            d.total_slippage_bps += 2 * slippage_bps

        # What-if: all maker (best case)
        all_maker_cost = d.n_trades * 2 * (maker_fee_bps + slippage_bps)
        d.net_if_all_maker_bps = d.total_gross_bps - all_maker_cost

        # What-if: all taker (worst case)
        all_taker_cost = d.n_trades * 2 * (taker_fee_bps + slippage_bps)
        d.net_if_all_taker_bps = d.total_gross_bps - all_taker_cost

        # Per-trade stats
        d.win_rate = float((net > 0).mean())
        winners = net[net > 0]
        losers = net[net < 0]
        d.avg_win_bps = float(np.mean(winners)) if len(winners) > 0 else 0.0
        d.avg_loss_bps = float(np.mean(losers)) if len(losers) > 0 else 0.0
        d.max_win_bps = float(np.max(net))
        d.max_loss_bps = float(np.min(net))

        win_sum = float(np.sum(winners)) if len(winners) > 0 else 0.0
        loss_sum = float(abs(np.sum(losers))) if len(losers) > 0 else 0.0
        d.profit_factor = win_sum / loss_sum if loss_sum > 0 else float("inf")

        if net.std() > 1e-12:
            d.sharpe = float(np.mean(net) / net.std() * np.sqrt(len(net)))
        else:
            d.sharpe = 0.0

        cumul = np.cumsum(net)
        dd = cumul - np.maximum.accumulate(cumul)
        d.max_dd_bps = float(abs(np.min(dd))) if len(dd) > 0 else 0.0

        results[config_name] = d

    return results


# --------------------------------------------------------------------------
# Task 3: Report generation
# --------------------------------------------------------------------------

def generate_execution_report(db_path: str = DB_PATH) -> str:
    """Generate full execution analysis report as markdown."""
    exec_metrics = compute_execution_metrics(db_path)
    pnl_decomp = compute_pnl_decomposition(db_path)

    lines = [
        "# Execution Analysis Report",
        "",
        f"Database: `{db_path}`",
        "",
    ]

    for config_name in sorted(set(list(exec_metrics.keys()) + list(pnl_decomp.keys()))):
        m = exec_metrics.get(config_name, ExecutionMetrics())
        p = pnl_decomp.get(config_name, PnLDecomposition())

        lines.append(f"## {config_name}")
        lines.append("")

        # Execution quality
        lines.append("### Execution Quality")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Trades | {m.n_trades} |")
        lines.append(f"| Entry Maker rate | {m.entry_maker_rate:.1%} |")
        lines.append(f"| Exit Maker rate | {m.exit_maker_rate:.1%} |")
        lines.append(f"| Exit Taker fallback | {m.exit_taker_rate:.1%} |")
        lines.append(f"| Avg time-to-fill | {m.avg_time_to_fill_ms:.0f} ms |")
        lines.append(f"| Median time-to-fill | {m.median_time_to_fill_ms:.0f} ms |")
        lines.append(f"| P90 time-to-fill | {m.p90_time_to_fill_ms:.0f} ms |")
        lines.append(f"| Avg spread at entry | {m.avg_spread_at_entry_bps:.4f} bps |")
        lines.append(f"| Signals total | {m.n_signals} |")
        lines.append(f"| Signal accept rate | {m.signal_accept_rate:.1%} |")
        lines.append(f"| Rejected (spread) | {m.reject_spread_count} |")
        lines.append(f"| Order fill rate | {m.order_fill_rate:.1%} |")
        lines.append(f"| Order cancel rate | {m.order_cancel_rate:.1%} |")
        lines.append("")

        # PnL decomposition
        lines.append("### PnL Decomposition")
        lines.append("")
        lines.append("| Component | Value (bps) |")
        lines.append("|-----------|------------|")
        lines.append(f"| Total Gross PnL | {p.total_gross_bps:+.2f} |")
        lines.append(f"| Maker cost (entry+exit) | {p.total_maker_cost_bps:-.2f} |")
        lines.append(f"| Taker penalty (fallback) | {p.total_taker_penalty_bps:-.2f} |")
        lines.append(f"| Slippage (total) | {p.total_slippage_bps:-.2f} |")
        lines.append(f"| **Total Net PnL** | **{p.total_net_bps:+.2f}** |")
        lines.append("")

        # What-if
        lines.append("### What-If Scenarios")
        lines.append("")
        lines.append("| Scenario | Net PnL (bps) |")
        lines.append("|----------|--------------|")
        lines.append(f"| Actual | {p.total_net_bps:+.2f} |")
        lines.append(f"| All Maker (best case) | {p.net_if_all_maker_bps:+.2f} |")
        lines.append(f"| All Taker (worst case) | {p.net_if_all_taker_bps:+.2f} |")
        lines.append("")

        # Trade stats
        lines.append("### Trade Statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Avg gross bps/trade | {p.avg_gross_bps:+.3f} |")
        lines.append(f"| Avg net bps/trade | {p.avg_net_bps:+.3f} |")
        lines.append(f"| Win rate | {p.win_rate:.1%} |")
        lines.append(f"| Avg win | {p.avg_win_bps:+.3f} bps |")
        lines.append(f"| Avg loss | {p.avg_loss_bps:+.3f} bps |")
        lines.append(f"| Max win | {p.max_win_bps:+.3f} bps |")
        lines.append(f"| Max loss | {p.max_loss_bps:+.3f} bps |")
        lines.append(f"| Profit factor | {p.profit_factor:.2f} |")
        lines.append(f"| Sharpe | {p.sharpe:.2f} |")
        lines.append(f"| Max drawdown | {p.max_dd_bps:.1f} bps |")
        lines.append("")

    return "\n".join(lines)
