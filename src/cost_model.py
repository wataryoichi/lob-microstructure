"""Crypto exchange cost model for LOB strategies.

Cost components for Bybit:
1. Maker fee: 0.01% (1 bps)
2. Taker fee: 0.055% (5.5 bps)
3. Slippage: estimated from LOB depth
"""

from __future__ import annotations

import numpy as np


def compute_round_trip_cost(
    maker_fee_bps: float = 1.0,
    taker_fee_bps: float = 5.5,
    slippage_bps: float = 1.0,
    is_maker: bool = True,
) -> float:
    """Compute round-trip cost in basis points.

    Round-trip = 2 * (fee + slippage)
    """
    fee = maker_fee_bps if is_maker else taker_fee_bps
    return 2 * (fee + slippage_bps)


def compute_breakeven_accuracy(
    avg_move_bps: float,
    cost_bps: float,
) -> float:
    """Minimum accuracy needed to break even.

    If expected absolute price move is `avg_move_bps` and cost is `cost_bps`:
    breakeven_acc = 0.5 + cost_bps / (2 * avg_move_bps)

    E.g., avg_move=5bps, cost=4bps -> need 90% accuracy
    """
    if avg_move_bps <= 0:
        return 1.0
    return 0.5 + cost_bps / (2 * avg_move_bps)


def estimate_slippage_from_lob(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    trade_size: float = 0.01,
) -> float:
    """Estimate average slippage for a given trade size using LOB data.

    Walks through LOB levels until trade_size is filled.
    Returns average slippage in basis points.
    """
    mid = (bid_prices[0] + ask_prices[0]) / 2
    if mid <= 0:
        return 0.0

    # Buy side: walk through ask levels
    remaining = trade_size
    cost = 0.0
    for i in range(len(ask_prices)):
        fill = min(remaining, ask_volumes[i])
        cost += fill * ask_prices[i]
        remaining -= fill
        if remaining <= 0:
            break

    if trade_size > 0:
        avg_fill_price = cost / trade_size
        slippage_bps = (avg_fill_price - mid) / mid * 10000
        return float(slippage_bps)
    return 0.0
