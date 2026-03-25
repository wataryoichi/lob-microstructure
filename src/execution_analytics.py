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

# --------------------------------------------------------------------------
# Adverse Selection Analysis (Task 1 - Phase 11)
# --------------------------------------------------------------------------

@dataclass
class AdverseSelectionResult:
    """Adverse selection analysis for unfilled orders."""
    n_cancelled: int = 0
    n_filled: int = 0
    # If we had chased unfilled orders with Taker
    taker_chase_avg_gross_bps: float = 0.0
    taker_chase_avg_net_bps: float = 0.0
    taker_chase_win_rate: float = 0.0
    taker_chase_total_net_bps: float = 0.0
    # Comparison
    filled_avg_net_bps: float = 0.0
    unfilled_would_have_been_bps: float = 0.0
    # Adverse selection indicator: unfilled > filled means we're getting picked off
    adverse_selection_present: bool = False


def compute_adverse_selection(
    db_path: str = DB_PATH,
    taker_fee_bps: float = 5.5,
    slippage_bps: float = 0.5,
) -> dict[str, AdverseSelectionResult]:
    """Analyze what happened to signals where our Maker order was NOT filled.

    For each cancelled order:
    - Look up the mid-price at order time (from signals table)
    - Compute what the price did over the horizon period
    - Calculate PnL if we had entered via Taker at signal time

    This answers: "Are we systematically missing the best trades?"
    """
    conn = _connect_readonly(db_path)

    orders = pd.read_sql("SELECT * FROM orders", conn)
    trades = pd.read_sql("SELECT * FROM trades", conn)
    signals = pd.read_sql("SELECT * FROM signals", conn)

    # We need WS data to look up prices at future timestamps.
    # Since paper_trader doesn't store the full LOB history, we use the
    # signal's mid_price and the NEXT signal/trade's mid_price as a proxy
    # for the price at horizon-end.
    # Better approach: join cancelled orders with subsequent status log prices.

    # Build a price timeline from all signals (they contain mid_price)
    price_timeline = signals[["ts", "mid_price"]].dropna().sort_values("ts").reset_index(drop=True)

    results: dict[str, AdverseSelectionResult] = {}

    for config_name in orders["config_name"].unique():
        r = AdverseSelectionResult()

        cfg_orders = orders[orders["config_name"] == config_name]
        cfg_trades = trades[trades["config_name"] == config_name]
        cfg_signals = signals[signals["config_name"] == config_name]

        cancelled = cfg_orders[cfg_orders["status"] == "cancelled"]
        filled = cfg_orders[cfg_orders["status"] == "filled"]
        r.n_cancelled = len(cancelled)
        r.n_filled = len(filled)

        # Filled trades' avg net
        r.filled_avg_net_bps = float(cfg_trades["net_bps"].mean()) if len(cfg_trades) > 0 else 0.0

        # Determine horizon from config name
        if "120s" in config_name:
            horizon_ms = 120_000
        elif "60s" in config_name:
            horizon_ms = 60_000
        else:
            horizon_ms = 60_000

        # For each cancelled order, find what price was at signal_ts + horizon
        taker_pnls = []
        taker_rt_cost = 2 * (taker_fee_bps + slippage_bps) * 1e-4  # decimal

        for _, order in cancelled.iterrows():
            signal_ts = order["ts"]
            order_price = order["price"]
            side = order["side"]

            # Find the signal's mid_price at entry
            sig_match = cfg_signals[cfg_signals["ts"] == signal_ts]
            if len(sig_match) == 0:
                continue
            entry_mid = sig_match.iloc[0]["mid_price"]
            if entry_mid <= 0:
                continue

            # Taker entry: buy at ask (mid + half_spread) or sell at bid (mid - half_spread)
            # Approximate: entry at mid_price (conservative)
            taker_entry = entry_mid

            # Find price at horizon end
            target_ts = signal_ts + horizon_ms
            future = price_timeline[price_timeline["ts"] >= target_ts]
            if len(future) == 0:
                continue
            exit_mid = future.iloc[0]["mid_price"]

            # PnL
            if side == "buy":  # would have gone long
                gross = (exit_mid - taker_entry) / taker_entry
            else:  # would have gone short
                gross = (taker_entry - exit_mid) / taker_entry

            net = gross - taker_rt_cost
            taker_pnls.append(net * 1e4)

        if taker_pnls:
            pnl_arr = np.array(taker_pnls)
            r.taker_chase_avg_gross_bps = float(np.mean(pnl_arr + taker_rt_cost * 1e4))
            r.taker_chase_avg_net_bps = float(np.mean(pnl_arr))
            r.taker_chase_win_rate = float((pnl_arr > 0).mean())
            r.taker_chase_total_net_bps = float(np.sum(pnl_arr))
            r.unfilled_would_have_been_bps = float(np.mean(pnl_arr))

            # Adverse selection: unfilled signals would have been more profitable
            r.adverse_selection_present = (
                r.unfilled_would_have_been_bps > r.filled_avg_net_bps
                and r.taker_chase_avg_net_bps > 0
            )

        results[config_name] = r

    conn.close()
    return results


def generate_execution_report(db_path: str = DB_PATH) -> str:
    """Generate full execution analysis report as markdown."""
    exec_metrics = compute_execution_metrics(db_path)
    pnl_decomp = compute_pnl_decomposition(db_path)
    adverse = compute_adverse_selection(db_path)

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

        # Adverse selection
        a = adverse.get(config_name)
        if a and (a.n_cancelled > 0):
            lines.append("### Adverse Selection Analysis")
            lines.append("")
            lines.append(f"Unfilled (cancelled) orders: **{a.n_cancelled}** vs filled: **{a.n_filled}**")
            lines.append("")
            lines.append("What if we had chased unfilled signals with Taker entry?")
            lines.append("")
            lines.append("| Metric | Filled (actual) | Unfilled (Taker chase) |")
            lines.append("|--------|----------------|----------------------|")
            lines.append(f"| Avg net bps | {a.filled_avg_net_bps:+.3f} | {a.taker_chase_avg_net_bps:+.3f} |")
            lines.append(f"| Win rate | {p.win_rate:.1%} | {a.taker_chase_win_rate:.1%} |")
            lines.append(f"| Total net bps | {p.total_net_bps:+.1f} | {a.taker_chase_total_net_bps:+.1f} |")
            lines.append("")
            if a.adverse_selection_present:
                lines.append("**WARNING: Adverse selection detected.** Unfilled signals would have been")
                lines.append("more profitable than filled ones. This suggests the Maker strategy is")
                lines.append("systematically missing the best moves (price runs away before fill).")
            else:
                lines.append("No adverse selection detected. Filled trades are comparable to or better")
                lines.append("than what unfilled signals would have produced.")
            lines.append("")

    return "\n".join(lines)
