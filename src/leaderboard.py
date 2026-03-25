"""Auto-generated leaderboard for strategy comparison.

Trading-first: sorted by avg_net_bps by default.
F1/accuracy are secondary columns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LEADERBOARD_PATH = "results/leaderboard.json"


@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard."""
    run_id: str
    strategy_type: str  # "imbalance_direct", "model_based", "imbalance_ml_filter"
    symbol: str
    data_source: str    # "synthetic_100k", "real_rest_5k", "real_ws_24h"
    timestamp: str      # ISO format

    # Config
    horizon_s: float = 0.0
    preprocessing: str = ""
    config_summary: str = ""

    # PRIMARY: Trading metrics
    n_trades: int = 0
    avg_gross_bps: float = 0.0
    avg_net_bps: float = 0.0
    total_net_bps: float = 0.0
    win_rate: float = 0.0
    win_rate_net: float = 0.0
    sharpe: float = 0.0
    max_drawdown_bps: float = 0.0
    profit_factor: float = 0.0
    breakeven_gap_bps: float = 0.0  # how far from net=0

    # SECONDARY: Classification metrics (optional)
    f1_macro: float | None = None
    accuracy: float | None = None

    # Cost assumptions
    fee_tier: str = "standard_maker"
    rt_cost_bps: float = 3.0


def load_leaderboard(path: str = LEADERBOARD_PATH) -> list[LeaderboardEntry]:
    """Load leaderboard from JSON."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    entries = []
    for d in data:
        # Handle optional fields
        entry = LeaderboardEntry(
            run_id=d.get("run_id", ""),
            strategy_type=d.get("strategy_type", ""),
            symbol=d.get("symbol", ""),
            data_source=d.get("data_source", ""),
            timestamp=d.get("timestamp", ""),
        )
        for k, v in d.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        entries.append(entry)
    return entries


def save_leaderboard(entries: list[LeaderboardEntry], path: str = LEADERBOARD_PATH) -> None:
    """Save leaderboard to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(e) for e in entries]
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def add_entry(entry: LeaderboardEntry, path: str = LEADERBOARD_PATH) -> None:
    """Add or update an entry in the leaderboard."""
    entries = load_leaderboard(path)
    # Replace if same run_id exists
    entries = [e for e in entries if e.run_id != entry.run_id]
    entries.append(entry)
    save_leaderboard(entries, path)


def entry_from_imbalance_result(
    result,
    symbol: str = "BTCUSDT",
    data_source: str = "real_rest_5k",
) -> LeaderboardEntry:
    """Create LeaderboardEntry from ImbalanceResult."""
    cfg = result.config
    thresh = cfg.get("short_threshold_pct", 0.1)
    horizon = cfg.get("horizon_steps", 100)
    fee = cfg.get("maker_fee_bps", 1.0)
    fee_label = "vip_maker" if fee == 0 else "standard_maker"
    rt = (2 * (fee + cfg.get("slippage_bps", 0.5)))

    run_id = f"imb_t{thresh:.2f}_h{horizon}_f{fee_label}_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    return LeaderboardEntry(
        run_id=run_id,
        strategy_type="imbalance_direct",
        symbol=symbol,
        data_source=data_source,
        timestamp=datetime.now(timezone.utc).isoformat(),
        horizon_s=horizon * 0.1,
        preprocessing="savitzky_golay" if cfg.get("apply_sg_filter") else "raw",
        config_summary=f"imb_depth={cfg.get('imbalance_depth')}, thresh={thresh}, vol_pct={cfg.get('min_vol_percentile')}",
        n_trades=result.n_trades,
        avg_gross_bps=result.avg_gross_bps,
        avg_net_bps=result.avg_net_bps,
        total_net_bps=result.net_pnl_bps,
        win_rate=result.win_rate,
        win_rate_net=result.win_rate_net,
        sharpe=result.sharpe,
        max_drawdown_bps=result.max_drawdown_bps,
        profit_factor=result.profit_factor,
        breakeven_gap_bps=abs(min(0, result.avg_net_bps)),
        fee_tier=fee_label,
        rt_cost_bps=rt,
    )


def render_leaderboard(
    entries: list[LeaderboardEntry],
    sort_by: str = "avg_net_bps",
    top_n: int = 20,
) -> str:
    """Render leaderboard as markdown, sorted by trading metric."""
    if not entries:
        return "# Leaderboard\n\nNo entries."

    df = pd.DataFrame([asdict(e) for e in entries])

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    df = df.head(top_n)

    # Primary columns (trading-first)
    primary_cols = [
        "run_id", "strategy_type", "symbol", "horizon_s", "fee_tier",
        "n_trades", "avg_gross_bps", "avg_net_bps", "win_rate",
        "profit_factor", "sharpe", "max_drawdown_bps", "breakeven_gap_bps",
    ]
    # Only include columns that exist
    cols = [c for c in primary_cols if c in df.columns]

    lines = [
        "# Strategy Leaderboard",
        "",
        f"Sorted by: **{sort_by}** (trading-first)",
        f"Total entries: {len(entries)}, showing top {min(top_n, len(df))}",
        "",
        df[cols].to_markdown(index=False, floatfmt=".3f"),
        "",
    ]

    # Highlight best
    if len(df) > 0:
        best = df.iloc[0]
        lines.append(f"**Top strategy**: {best.get('run_id', 'N/A')}")
        lines.append(f"  Avg net: {best.get('avg_net_bps', 0):.3f} bps/trade, "
                      f"Win rate: {best.get('win_rate', 0):.1%}, "
                      f"Trades: {best.get('n_trades', 0)}")

    return "\n".join(lines)
