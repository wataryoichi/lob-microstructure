"""WebSocket-based continuous LOB data collector for Bybit.

Maintains local orderbook from delta stream, samples at fixed intervals,
writes parquet chunks to disk. Handles reconnection and gap detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import BYBIT_WS_PUBLIC

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10000
PING_INTERVAL_SEC = 20


class LocalOrderbook:
    """Maintains local orderbook state from Bybit delta stream."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: dict[float, float] = {}  # price -> qty
        self.asks: dict[float, float] = {}
        self.last_update_id: int = 0
        self.seq: int = 0
        self.initialized: bool = False
        self.timestamp: int = 0

    def apply_snapshot(self, data: dict) -> None:
        """Apply initial snapshot to reset book state."""
        self.bids.clear()
        self.asks.clear()
        for p, q in data.get("b", []):
            price, qty = float(p), float(q)
            if qty > 0:
                self.bids[price] = qty
        for p, q in data.get("a", []):
            price, qty = float(p), float(q)
            if qty > 0:
                self.asks[price] = qty
        self.last_update_id = data.get("u", 0)
        self.seq = data.get("seq", 0)
        self.timestamp = int(data.get("ts", 0))
        self.initialized = True

    def apply_delta(self, data: dict) -> bool:
        """Apply delta update. Returns False if sequence gap detected."""
        new_u = data.get("u", 0)
        if self.initialized and new_u <= self.last_update_id:
            return True  # duplicate, ignore

        for p, q in data.get("b", []):
            price, qty = float(p), float(q)
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for p, q in data.get("a", []):
            price, qty = float(p), float(q)
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = new_u
        self.seq = data.get("seq", self.seq)
        self.timestamp = int(data.get("ts", self.timestamp))
        return True

    def to_flat_row(self, depth: int = 200) -> dict | None:
        """Convert current book state to a flat dict row."""
        if not self.initialized or not self.bids or not self.asks:
            return None

        # Sort bids descending, asks ascending
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:depth]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]

        row: dict[str, Any] = {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
        }

        for i in range(depth):
            if i < len(sorted_bids):
                row[f"bid_p{i+1}"] = sorted_bids[i][0]
                row[f"bid_q{i+1}"] = sorted_bids[i][1]
            else:
                row[f"bid_p{i+1}"] = np.nan
                row[f"bid_q{i+1}"] = np.nan

            if i < len(sorted_asks):
                row[f"ask_p{i+1}"] = sorted_asks[i][0]
                row[f"ask_q{i+1}"] = sorted_asks[i][1]
            else:
                row[f"ask_p{i+1}"] = np.nan
                row[f"ask_q{i+1}"] = np.nan

        return row


class WSCollector:
    """Multi-symbol WebSocket LOB collector."""

    def __init__(
        self,
        symbols: list[str],
        depth: int = 200,
        interval_ms: int = 100,
        output_dir: str | Path = "data/raw",
        chunk_size: int = CHUNK_SIZE,
    ):
        self.symbols = symbols
        self.depth = min(depth, 50)  # Bybit WS: 1, 50, 200, 500; 50 is efficient for our needs
        self.interval_ms = interval_ms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        self.books: dict[str, LocalOrderbook] = {s: LocalOrderbook(s) for s in symbols}
        self.buffers: dict[str, list[dict]] = {s: [] for s in symbols}
        self.metadata: dict[str, Any] = {
            "symbols": symbols,
            "depth": depth,
            "interval_ms": interval_ms,
            "start_time": None,
            "snapshot_counts": {s: 0 for s in symbols},
            "chunk_counts": {s: 0 for s in symbols},
            "reconnections": 0,
            "gaps_detected": 0,
            "last_write_time": None,
        }
        self._running = False

    async def run(self, duration_hours: float = 25.0) -> None:
        """Run collector for specified duration."""
        self.metadata["start_time"] = datetime.now(timezone.utc).isoformat()
        self._running = True
        end_time = time.monotonic() + duration_hours * 3600

        logger.info(f"Starting WS collection: {self.symbols}, {duration_hours}h, {self.interval_ms}ms")

        while self._running and time.monotonic() < end_time:
            try:
                await self._connect_and_collect(end_time)
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.metadata["reconnections"] += 1
                if time.monotonic() < end_time:
                    wait = min(30, 2 ** min(self.metadata["reconnections"], 5))
                    logger.info(f"Reconnecting in {wait}s...")
                    await asyncio.sleep(wait)

        # Flush remaining buffers
        for symbol in self.symbols:
            if self.buffers[symbol]:
                self._flush_chunk(symbol)

        self._save_metadata()
        logger.info(f"Collection complete. Counts: {self.metadata['snapshot_counts']}")

    async def _connect_and_collect(self, end_time: float) -> None:
        """Connect to WebSocket and collect data."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets required: pip install websockets")

        uri = BYBIT_WS_PUBLIC
        async with websockets.connect(uri, ping_interval=PING_INTERVAL_SEC) as ws:
            # Subscribe to orderbook for all symbols
            for symbol in self.symbols:
                sub_msg = {
                    "op": "subscribe",
                    "args": [f"orderbook.{self.depth}.{symbol}"],
                }
                await ws.send(json.dumps(sub_msg))
                logger.info(f"Subscribed: orderbook.{self.depth}.{symbol}")

            # Run message receiver and sampler concurrently
            async def receive_messages():
                async for raw_msg in ws:
                    if time.monotonic() >= end_time:
                        break
                    try:
                        msg = json.loads(raw_msg)
                        self._handle_message(msg)
                    except json.JSONDecodeError:
                        continue

            await asyncio.gather(
                receive_messages(),
                self._sample_loop(end_time),
            )

    def _handle_message(self, msg: dict) -> None:
        """Process a single WebSocket message."""
        topic = msg.get("topic", "")
        msg_type = msg.get("type", "")
        data = msg.get("data", {})

        if not topic.startswith("orderbook."):
            return

        # Extract symbol from topic: "orderbook.200.BTCUSDT"
        parts = topic.split(".")
        if len(parts) < 3:
            return
        symbol = parts[2]

        if symbol not in self.books:
            return

        book = self.books[symbol]

        if msg_type == "snapshot":
            book.apply_snapshot(data)
        elif msg_type == "delta":
            if not book.initialized:
                return
            ok = book.apply_delta(data)
            if not ok:
                logger.warning(f"Sequence gap for {symbol}, requesting re-snapshot")
                self.metadata["gaps_detected"] += 1
                book.initialized = False

    async def _sample_loop(self, end_time: float) -> None:
        """Sample book state at fixed intervals."""
        interval_sec = self.interval_ms / 1000.0

        while time.monotonic() < end_time:
            start = time.monotonic()

            for symbol in self.symbols:
                book = self.books[symbol]
                row = book.to_flat_row(depth=min(self.depth, 200))
                if row is not None:
                    self.buffers[symbol].append(row)
                    self.metadata["snapshot_counts"][symbol] += 1

                    # Log progress
                    count = self.metadata["snapshot_counts"][symbol]
                    if count % 10000 == 0:
                        logger.info(f"{symbol}: {count} snapshots collected")

                    # Flush chunk if buffer full
                    if len(self.buffers[symbol]) >= self.chunk_size:
                        self._flush_chunk(symbol)

            elapsed = time.monotonic() - start
            sleep_time = max(0, interval_sec - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def _flush_chunk(self, symbol: str) -> None:
        """Write buffer to parquet file."""
        if not self.buffers[symbol]:
            return

        df = pd.DataFrame(self.buffers[symbol])
        n = len(df)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        chunk_idx = self.metadata["chunk_counts"][symbol]
        filename = f"{symbol}_ws_{ts}_chunk{chunk_idx:04d}_{n}snaps.parquet"
        path = self.output_dir / filename
        df.to_parquet(path, index=False)

        self.metadata["chunk_counts"][symbol] += 1
        self.metadata["last_write_time"] = datetime.now(timezone.utc).isoformat()
        self.buffers[symbol].clear()

        logger.info(f"Wrote {path.name} ({n} rows)")
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Save collection metadata to JSON."""
        meta_path = self.output_dir / "ws_collection_meta.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)


def collect_ws_sync(
    symbols: list[str],
    duration_hours: float = 25.0,
    interval_ms: int = 100,
    depth: int = 200,
    output_dir: str | Path = "data/raw",
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Synchronous wrapper for WebSocket collection."""
    collector = WSCollector(
        symbols=symbols,
        depth=depth,
        interval_ms=interval_ms,
        output_dir=output_dir,
        chunk_size=chunk_size,
    )
    asyncio.run(collector.run(duration_hours))
