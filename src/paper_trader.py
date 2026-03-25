"""Paper Trading engine with Maker execution simulation.

Connects to Bybit WebSocket, generates signals in real-time,
simulates Maker order fills, and logs all activity to SQLite.

No real orders are placed. This is a forward-test only.

Architecture:
  1. WS receives LOB deltas -> local orderbook maintained
  2. Every 100ms: snapshot -> rolling feature buffer
  3. Signal check: imbalance percentile + spread filter
  4. If signal fires: place virtual Maker order at best bid/ask
  5. Fill simulation: order fills when price crosses our level
  6. Horizon exit: Maker limit at target, or Taker if timeout
  7. All events logged to SQLite
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .constants import BYBIT_WS_PUBLIC
from .cost_model import compute_round_trip_cost
from .features import compute_aggregate_imbalance
from .filters import apply_savitzky_golay
from .ws_collector import LocalOrderbook

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class PaperTraderConfig:
    """Paper trader configuration."""
    symbol: str = "BTCUSDT"
    ws_depth: int = 50

    # Strategy configs to run in parallel
    configs: list[dict] = field(default_factory=lambda: [
        {"name": "A_5pct_60s", "threshold": 0.05, "horizon_s": 60},
        {"name": "B_10pct_120s", "threshold": 0.10, "horizon_s": 120},
    ])

    # Common parameters
    imbalance_depth: int = 5
    sg_window: int = 21
    percentile_window: int = 2000  # lookback for rolling percentile
    max_spread_ticks: float = 1.5  # max spread in ticks (0.1 USDT for BTC)
    tick_size: float = 0.1         # BTC/USDT tick size

    # Maker execution
    maker_fee_bps: float = 0.0     # VIP maker
    taker_fee_bps: float = 5.5
    slippage_bps: float = 0.5
    fill_timeout_s: float = 5.0    # max wait for maker fill
    exit_timeout_s: float = 10.0   # max wait for exit maker fill

    # Feature buffer
    buffer_size: int = 5000        # max snapshots in memory
    sample_interval_ms: int = 100

    # Logging
    db_path: str = "results/paper_trades.db"
    log_interval_s: float = 60.0   # status log interval


# --------------------------------------------------------------------------
# Virtual Order & Position
# --------------------------------------------------------------------------

@dataclass
class VirtualOrder:
    """A virtual Maker order waiting for fill."""
    order_id: str
    config_name: str
    side: str         # "buy" or "sell"
    price: float
    qty: float = 1.0
    placed_at: float = 0.0  # monotonic time
    placed_ts: int = 0      # exchange timestamp
    status: str = "pending"  # pending / filled / cancelled / taker_filled
    fill_price: float = 0.0
    fill_ts: int = 0


@dataclass
class VirtualPosition:
    """An open virtual position."""
    position_id: str
    config_name: str
    side: str          # "long" or "short"
    entry_price: float
    entry_ts: int
    entry_type: str    # "maker" or "taker"
    horizon_s: float
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_type: str = ""
    gross_bps: float = 0.0
    net_bps: float = 0.0
    status: str = "open"  # open / closed


# --------------------------------------------------------------------------
# SQLite Logger
# --------------------------------------------------------------------------

class TradeLogger:
    """Logs signals, orders, and trades to SQLite."""

    def __init__(self, db_path: str = "results/paper_trades.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER, config_name TEXT, symbol TEXT,
                direction TEXT, imbalance REAL, pct_rank REAL,
                spread_bps REAL, mid_price REAL,
                accepted INTEGER, reject_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT, ts INTEGER, config_name TEXT, symbol TEXT,
                side TEXT, price REAL, status TEXT,
                fill_price REAL, fill_ts INTEGER
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT, config_name TEXT, symbol TEXT,
                side TEXT,
                entry_price REAL, entry_ts INTEGER, entry_type TEXT,
                exit_price REAL, exit_ts INTEGER, exit_type TEXT,
                gross_bps REAL, net_bps REAL,
                duration_s REAL
            );
            CREATE TABLE IF NOT EXISTS status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER, config_name TEXT,
                total_trades INTEGER, total_net_bps REAL,
                win_rate REAL, open_positions INTEGER
            );
        """)
        self.conn.commit()

    def log_signal(self, ts, config_name, symbol, direction, imbalance,
                   pct_rank, spread_bps, mid_price, accepted, reject_reason=""):
        self.conn.execute(
            "INSERT INTO signals (ts,config_name,symbol,direction,imbalance,"
            "pct_rank,spread_bps,mid_price,accepted,reject_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ts, config_name, symbol, direction, imbalance,
             pct_rank, spread_bps, mid_price, int(accepted), reject_reason))
        self.conn.commit()

    def log_order(self, order: VirtualOrder, symbol: str):
        self.conn.execute(
            "INSERT INTO orders (order_id,ts,config_name,symbol,side,price,"
            "status,fill_price,fill_ts) VALUES (?,?,?,?,?,?,?,?,?)",
            (order.order_id, order.placed_ts, order.config_name, symbol,
             order.side, order.price, order.status, order.fill_price, order.fill_ts))
        self.conn.commit()

    def log_trade(self, pos: VirtualPosition, symbol: str):
        duration = (pos.exit_ts - pos.entry_ts) / 1000 if pos.exit_ts > 0 else 0
        self.conn.execute(
            "INSERT INTO trades (position_id,config_name,symbol,side,"
            "entry_price,entry_ts,entry_type,exit_price,exit_ts,exit_type,"
            "gross_bps,net_bps,duration_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pos.position_id, pos.config_name, symbol, pos.side,
             pos.entry_price, pos.entry_ts, pos.entry_type,
             pos.exit_price, pos.exit_ts, pos.exit_type,
             pos.gross_bps, pos.net_bps, duration))
        self.conn.commit()

    def log_status(self, ts, config_name, total_trades, total_net, win_rate, open_pos):
        self.conn.execute(
            "INSERT INTO status (ts,config_name,total_trades,total_net_bps,"
            "win_rate,open_positions) VALUES (?,?,?,?,?,?)",
            (ts, config_name, total_trades, total_net, win_rate, open_pos))
        self.conn.commit()

    def get_summary(self, config_name: str) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(AVG(net_bps),0), "
            "COALESCE(SUM(CASE WHEN net_bps>0 THEN 1.0 ELSE 0 END)/NULLIF(COUNT(*),0),0) "
            "FROM trades WHERE config_name=?", (config_name,)).fetchone()
        return {"total_trades": row[0], "avg_net_bps": row[1], "win_rate": row[2]}

    def close(self):
        self.conn.close()


# --------------------------------------------------------------------------
# Paper Trading Engine
# --------------------------------------------------------------------------

class PaperTrader:
    """Real-time paper trading engine."""

    def __init__(self, config: PaperTraderConfig):
        self.cfg = config
        self.book = LocalOrderbook(config.symbol)
        self.trade_logger = TradeLogger(config.db_path)

        # Feature buffer: rolling window of snapshots
        self._imb_buffer: deque[float] = deque(maxlen=config.buffer_size)
        self._mid_buffer: deque[float] = deque(maxlen=config.buffer_size)
        self._ts_buffer: deque[int] = deque(maxlen=config.buffer_size)

        # Per-config state
        self._orders: dict[str, VirtualOrder | None] = {}
        self._positions: dict[str, VirtualPosition | None] = {}
        self._cooldown_until: dict[str, float] = {}
        self._trade_count: dict[str, int] = {}
        self._total_net: dict[str, float] = {}

        for c in config.configs:
            name = c["name"]
            self._orders[name] = None
            self._positions[name] = None
            self._cooldown_until[name] = 0
            self._trade_count[name] = 0
            self._total_net[name] = 0.0

        self._order_counter = 0
        self._running = False

    async def run(self, duration_hours: float = 168.0) -> None:
        """Run paper trader for specified duration (default 7 days)."""
        self._running = True
        end_time = time.monotonic() + duration_hours * 3600

        logger.info(f"Paper trader starting: {self.cfg.symbol}, "
                     f"configs={[c['name'] for c in self.cfg.configs]}")

        while self._running and time.monotonic() < end_time:
            try:
                await self._connect_and_trade(end_time)
            except Exception as e:
                logger.error(f"Connection error: {e}")
                if time.monotonic() < end_time:
                    await asyncio.sleep(5)

        self.trade_logger.close()
        logger.info("Paper trader stopped.")

    async def _connect_and_trade(self, end_time: float) -> None:
        import websockets
        uri = BYBIT_WS_PUBLIC
        async with websockets.connect(uri, ping_interval=20) as ws:
            sub = {"op": "subscribe",
                   "args": [f"orderbook.{self.cfg.ws_depth}.{self.cfg.symbol}"]}
            await ws.send(json.dumps(sub))
            logger.info(f"Subscribed: orderbook.{self.cfg.ws_depth}.{self.cfg.symbol}")

            async def receive():
                async for raw in ws:
                    if time.monotonic() >= end_time:
                        break
                    try:
                        msg = json.loads(raw)
                        self._handle_ws_message(msg)
                    except json.JSONDecodeError:
                        pass

            async def sample_and_trade():
                interval = self.cfg.sample_interval_ms / 1000
                last_status = time.monotonic()
                while time.monotonic() < end_time:
                    t0 = time.monotonic()
                    self._on_tick()

                    # Periodic status log
                    if t0 - last_status >= self.cfg.log_interval_s:
                        self._log_status()
                        last_status = t0

                    elapsed = time.monotonic() - t0
                    await asyncio.sleep(max(0, interval - elapsed))

            await asyncio.gather(receive(), sample_and_trade())

    def _handle_ws_message(self, msg: dict) -> None:
        topic = msg.get("topic", "")
        if not topic.startswith("orderbook."):
            return
        msg_type = msg.get("type", "")
        data = msg.get("data", {})
        msg_ts = int(msg.get("ts", 0))

        if msg_type == "snapshot":
            self.book.apply_snapshot(data)
            if msg_ts > 0:
                self.book.timestamp = msg_ts
        elif msg_type == "delta":
            if self.book.initialized:
                self.book.apply_delta(data)
                if msg_ts > 0:
                    self.book.timestamp = msg_ts

    def _on_tick(self) -> None:
        """Called every sample interval. Updates features and checks signals."""
        if not self.book.initialized or not self.book.bids or not self.book.asks:
            return

        # Get current book state
        sorted_bids = sorted(self.book.bids.items(), key=lambda x: -x[0])
        sorted_asks = sorted(self.book.asks.items(), key=lambda x: x[0])
        if not sorted_bids or not sorted_asks:
            return

        best_bid = sorted_bids[0][0]
        best_ask = sorted_asks[0][0]
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = spread / mid * 1e4
        ts = self.book.timestamp

        # Compute 5-level imbalance
        depth = min(self.cfg.imbalance_depth, len(sorted_bids), len(sorted_asks))
        if depth < 1:
            return
        bid_vol = sum(sorted_bids[i][1] for i in range(depth))
        ask_vol = sum(sorted_asks[i][1] for i in range(depth))
        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0

        # Update buffers
        self._imb_buffer.append(imbalance)
        self._mid_buffer.append(mid)
        self._ts_buffer.append(ts)

        # Need enough history for percentile
        if len(self._imb_buffer) < self.cfg.percentile_window:
            return

        # Compute SG-filtered imbalance (on recent window)
        window = list(self._imb_buffer)[-self.cfg.sg_window * 2:]
        if len(window) >= self.cfg.sg_window:
            sg = apply_savitzky_golay(np.array(window),
                                      window_size=self.cfg.sg_window, polyorder=3)
            current_imb = sg[-1]
        else:
            current_imb = imbalance

        # Compute percentile rank (trailing window, no lookahead)
        lookback = list(self._imb_buffer)[-self.cfg.percentile_window:]
        pct_rank = sum(1 for v in lookback if v < current_imb) / len(lookback)

        # Check pending orders and positions
        self._check_fills(best_bid, best_ask, ts)
        self._check_exits(best_bid, best_ask, mid, ts)

        # Check signals for each config
        for cfg_dict in self.cfg.configs:
            name = cfg_dict["name"]
            threshold = cfg_dict["threshold"]
            horizon_s = cfg_dict["horizon_s"]

            # Skip if in cooldown or has open position/order
            if self._positions[name] is not None:
                continue
            if self._orders[name] is not None:
                continue
            if time.monotonic() < self._cooldown_until.get(name, 0):
                continue

            # Signal check
            direction = None
            if pct_rank >= (1.0 - threshold):
                direction = "long"
            elif pct_rank <= threshold:
                direction = "short"

            if direction is None:
                continue

            # Spread filter
            max_spread = self.cfg.max_spread_ticks * self.cfg.tick_size
            if spread > max_spread:
                self.trade_logger.log_signal(
                    ts, name, self.cfg.symbol, direction, current_imb,
                    pct_rank, spread_bps, mid, False, "spread_too_wide")
                continue

            # Place virtual Maker order
            self.trade_logger.log_signal(
                ts, name, self.cfg.symbol, direction, current_imb,
                pct_rank, spread_bps, mid, True)

            self._order_counter += 1
            oid = f"{name}_{self._order_counter}"

            if direction == "long":
                order = VirtualOrder(
                    order_id=oid, config_name=name, side="buy",
                    price=best_bid, placed_at=time.monotonic(), placed_ts=ts)
            else:
                order = VirtualOrder(
                    order_id=oid, config_name=name, side="sell",
                    price=best_ask, placed_at=time.monotonic(), placed_ts=ts)

            self._orders[name] = order
            logger.info(f"[{name}] Order placed: {order.side} @ {order.price:.2f} "
                         f"(imb={current_imb:.4f}, rank={pct_rank:.3f}, spread={spread:.2f})")

    def _check_fills(self, best_bid: float, best_ask: float, ts: int) -> None:
        """Check if pending Maker orders are filled."""
        for cfg_dict in self.cfg.configs:
            name = cfg_dict["name"]
            order = self._orders[name]
            if order is None or order.status != "pending":
                continue

            elapsed = time.monotonic() - order.placed_at
            filled = False

            if order.side == "buy":
                # Buy order fills when best ask drops to or below our bid price
                # (i.e., someone is selling at our price)
                if best_ask <= order.price:
                    filled = True
                    order.fill_price = order.price
            else:
                # Sell order fills when best bid rises to or above our ask price
                if best_bid >= order.price:
                    filled = True
                    order.fill_price = order.price

            if filled:
                order.status = "filled"
                order.fill_ts = ts
                self.trade_logger.log_order(order, self.cfg.symbol)

                # Create position
                horizon_s = cfg_dict["horizon_s"]
                side = "long" if order.side == "buy" else "short"
                pos = VirtualPosition(
                    position_id=order.order_id, config_name=name,
                    side=side, entry_price=order.fill_price,
                    entry_ts=ts, entry_type="maker", horizon_s=horizon_s)
                self._positions[name] = pos
                self._orders[name] = None

                logger.info(f"[{name}] FILL: {side} @ {order.fill_price:.2f}")

            elif elapsed > self.cfg.fill_timeout_s:
                # Timeout: cancel order
                order.status = "cancelled"
                self.trade_logger.log_order(order, self.cfg.symbol)
                self._orders[name] = None
                # Cooldown
                self._cooldown_until[name] = time.monotonic() + cfg_dict["horizon_s"]

    def _check_exits(self, best_bid: float, best_ask: float,
                     mid: float, ts: int) -> None:
        """Check if open positions should be closed."""
        for cfg_dict in self.cfg.configs:
            name = cfg_dict["name"]
            pos = self._positions[name]
            if pos is None or pos.status != "open":
                continue

            elapsed_s = (ts - pos.entry_ts) / 1000
            if elapsed_s < pos.horizon_s:
                continue

            # Horizon reached: attempt Maker exit, fallback to Taker
            if pos.side == "long":
                # Try to sell at best ask (Maker)
                exit_price = best_ask
                exit_type = "maker"
            else:
                # Try to buy at best bid (Maker)
                exit_price = best_bid
                exit_type = "maker"

            # For simplicity in paper trading: immediate Maker exit at best price
            # (conservative: in reality, Maker exit might not fill immediately)
            # If elapsed > horizon + exit_timeout: force Taker exit
            if elapsed_s > pos.horizon_s + self.cfg.exit_timeout_s:
                # Taker exit: pay taker fee
                if pos.side == "long":
                    exit_price = best_bid  # sell at bid (taker)
                else:
                    exit_price = best_ask  # buy at ask (taker)
                exit_type = "taker"

            self._close_position(pos, exit_price, ts, exit_type)

    def _close_position(self, pos: VirtualPosition, exit_price: float,
                        ts: int, exit_type: str) -> None:
        """Close a virtual position and log the trade."""
        pos.exit_price = exit_price
        pos.exit_ts = ts
        pos.exit_type = exit_type
        pos.status = "closed"

        # PnL calculation
        if pos.side == "long":
            gross = (exit_price - pos.entry_price) / pos.entry_price
        else:
            gross = (pos.entry_price - exit_price) / pos.entry_price

        pos.gross_bps = gross * 1e4

        # Cost: entry is always Maker, exit depends
        entry_cost = self.cfg.maker_fee_bps * 1e-4
        if exit_type == "maker":
            exit_cost = self.cfg.maker_fee_bps * 1e-4
        else:
            exit_cost = self.cfg.taker_fee_bps * 1e-4
        slippage = self.cfg.slippage_bps * 1e-4

        total_cost = entry_cost + exit_cost + 2 * slippage
        pos.net_bps = pos.gross_bps - total_cost * 1e4

        name = pos.config_name
        self._trade_count[name] = self._trade_count.get(name, 0) + 1
        self._total_net[name] = self._total_net.get(name, 0) + pos.net_bps

        self.trade_logger.log_trade(pos, self.cfg.symbol)
        self._positions[name] = None

        # Cooldown
        for c in self.cfg.configs:
            if c["name"] == name:
                self._cooldown_until[name] = time.monotonic() + c["horizon_s"]
                break

        logger.info(
            f"[{name}] CLOSE: {pos.side} entry={pos.entry_price:.2f} "
            f"exit={exit_price:.2f} gross={pos.gross_bps:+.2f} "
            f"net={pos.net_bps:+.2f} ({exit_type}) "
            f"[total: {self._trade_count[name]} trades, "
            f"net={self._total_net[name]:+.1f}bps]")

    def _log_status(self) -> None:
        """Periodic status logging."""
        ts = self.book.timestamp
        for cfg_dict in self.cfg.configs:
            name = cfg_dict["name"]
            n = self._trade_count[name]
            net = self._total_net[name]
            summary = self.trade_logger.get_summary(name)
            open_pos = 1 if self._positions[name] is not None else 0

            self.trade_logger.log_status(
                ts, name, n, net, summary["win_rate"], open_pos)

            logger.info(
                f"[{name}] Status: {n} trades, net={net:+.1f}bps, "
                f"win={summary['win_rate']:.1%}, open={open_pos}")

    def stop(self):
        self._running = False


# --------------------------------------------------------------------------
# Sync wrapper
# --------------------------------------------------------------------------

def run_paper_trader(
    symbol: str = "BTCUSDT",
    duration_hours: float = 168.0,
    db_path: str = "results/paper_trades.db",
    max_spread_ticks: float = 1.5,
) -> None:
    """Run paper trader (sync entry point for CLI)."""
    config = PaperTraderConfig(
        symbol=symbol,
        db_path=db_path,
        max_spread_ticks=max_spread_ticks,
    )
    trader = PaperTrader(config)
    asyncio.run(trader.run(duration_hours))
