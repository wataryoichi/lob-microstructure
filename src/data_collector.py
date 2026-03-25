"""Bybit LOB data collection via REST and WebSocket APIs.

Collects limit order book snapshots at configurable intervals.
Stores data in Parquet format for efficient I/O.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .constants import BYBIT_REST_BASE, MAX_LOB_DEPTH

logger = logging.getLogger(__name__)


def fetch_orderbook_snapshot(
    symbol: str = "BTCUSDT",
    depth: int = 200,
) -> dict | None:
    """Fetch a single LOB snapshot from Bybit REST API.

    Args:
        symbol: Trading pair symbol
        depth: Number of levels (max 200 for Bybit linear)

    Returns:
        Dict with keys: timestamp, bids, asks
        Each bid/ask is list of [price, qty] pairs
    """
    url = f"{BYBIT_REST_BASE}/v5/market/orderbook"
    params = {"category": "linear", "symbol": symbol, "limit": min(depth, MAX_LOB_DEPTH)}

    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            logger.error(f"Bybit API error: {data.get('retMsg')}")
            return None

        result = data["result"]
        return {
            "timestamp": int(result["ts"]),
            "symbol": result["s"],
            "bids": [[float(p), float(q)] for p, q in result["b"]],
            "asks": [[float(p), float(q)] for p, q in result["a"]],
        }
    except Exception as e:
        logger.error(f"Failed to fetch orderbook: {e}")
        return None


def snapshot_to_flat_row(snapshot: dict, depth: int = 40) -> dict:
    """Convert a LOB snapshot to a flat dict for DataFrame construction.

    Columns: timestamp, bid_p1..bid_pN, bid_q1..bid_qN, ask_p1..ask_pN, ask_q1..ask_qN
    """
    row: dict = {"timestamp": snapshot["timestamp"]}

    bids = snapshot["bids"][:depth]
    asks = snapshot["asks"][:depth]

    for i in range(depth):
        if i < len(bids):
            row[f"bid_p{i+1}"] = bids[i][0]
            row[f"bid_q{i+1}"] = bids[i][1]
        else:
            row[f"bid_p{i+1}"] = np.nan
            row[f"bid_q{i+1}"] = np.nan

        if i < len(asks):
            row[f"ask_p{i+1}"] = asks[i][0]
            row[f"ask_q{i+1}"] = asks[i][1]
        else:
            row[f"ask_p{i+1}"] = np.nan
            row[f"ask_q{i+1}"] = np.nan

    return row


def collect_snapshots(
    symbol: str = "BTCUSDT",
    n_snapshots: int = 1000,
    interval_ms: int = 100,
    depth: int = 40,
    output_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Collect multiple LOB snapshots at regular intervals.

    Args:
        symbol: Trading pair
        n_snapshots: Number of snapshots to collect
        interval_ms: Time between snapshots in milliseconds
        depth: LOB depth levels
        output_dir: Directory to save the parquet file

    Returns:
        DataFrame with all snapshots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    interval_sec = interval_ms / 1000.0

    logger.info(f"Collecting {n_snapshots} snapshots of {symbol} at {interval_ms}ms intervals...")

    for i in range(n_snapshots):
        start = time.monotonic()

        snapshot = fetch_orderbook_snapshot(symbol, depth)
        if snapshot is not None:
            row = snapshot_to_flat_row(snapshot, depth)
            rows.append(row)

        if (i + 1) % 1000 == 0:
            logger.info(f"Collected {i + 1}/{n_snapshots} snapshots")

        # Sleep for remaining interval
        elapsed = time.monotonic() - start
        sleep_time = max(0, interval_sec - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{ts}_{len(df)}snaps.parquet"
        path = output_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} snapshots to {path}")

    return df


def load_sample_data(
    n_snapshots: int = 100000,
    depth: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic LOB data for development/testing.

    Creates realistic-looking LOB snapshots with:
    - Mid-price random walk around 100,000
    - Exponentially decaying volume away from best bid/ask
    - Realistic bid-ask spread
    """
    rng = np.random.default_rng(seed)

    # Simulate mid-price random walk
    mid_price = 100000.0
    returns = rng.normal(0, 0.00005, n_snapshots)
    mid_prices = mid_price * np.exp(np.cumsum(returns))

    rows: list[dict] = []
    base_ts = int(datetime(2025, 1, 30, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    for i in range(n_snapshots):
        row: dict = {"timestamp": base_ts + i * 100}
        mp = mid_prices[i]
        spread = mp * rng.uniform(0.00001, 0.0001)

        for lvl in range(1, depth + 1):
            # Price: decreasing for bids, increasing for asks
            bid_offset = spread / 2 + mp * 0.00001 * (lvl - 1) * rng.uniform(0.8, 1.2)
            ask_offset = spread / 2 + mp * 0.00001 * (lvl - 1) * rng.uniform(0.8, 1.2)

            row[f"bid_p{lvl}"] = mp - bid_offset
            row[f"ask_p{lvl}"] = mp + ask_offset

            # Volume: exponential decay with noise
            base_vol = rng.exponential(0.5) * np.exp(-0.05 * (lvl - 1))
            row[f"bid_q{lvl}"] = max(0.001, base_vol)
            row[f"ask_q{lvl}"] = max(0.001, rng.exponential(0.5) * np.exp(-0.05 * (lvl - 1)))

        rows.append(row)

    return pd.DataFrame(rows)
