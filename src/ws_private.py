"""Bybit V5 Private WebSocket for order/execution notifications.

Receives real-time updates on:
- Order status changes (New, Filled, Cancelled, Rejected)
- Execution details (fill price, qty, fee)

Used by Live Trader to confirm fills instead of simulating them.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

WS_MAINNET_PRIVATE = "wss://stream.bybit.com/v5/private"
WS_TESTNET_PRIVATE = "wss://stream-testnet.bybit.com/v5/private"


@dataclass
class OrderUpdate:
    """Standardized order update from private WS."""
    order_id: str
    order_link_id: str
    symbol: str
    side: str           # "Buy" or "Sell"
    status: str         # "New", "Filled", "PartiallyFilled", "Cancelled", "Rejected"
    price: float
    qty: float
    filled_qty: float
    avg_price: float
    fee: float
    timestamp: int      # ms


@dataclass
class ExecutionUpdate:
    """Standardized execution (fill) notification."""
    exec_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    qty: float
    fee: float
    fee_currency: str
    is_maker: bool
    timestamp: int


def _load_keys(env_path: str = ".env") -> tuple[str, str]:
    """Load API keys from .env or environment."""
    env = {}
    p = Path(env_path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")

    key = env.get("BYBIT_API_KEY", "") or os.environ.get("BYBIT_API_KEY", "")
    secret = env.get("BYBIT_API_SECRET", "") or os.environ.get("BYBIT_API_SECRET", "")
    return key, secret


class BybitPrivateWS:
    """Bybit V5 Private WebSocket client.

    Usage:
        ws = BybitPrivateWS(testnet=True)
        ws.on_order = my_order_handler
        ws.on_execution = my_execution_handler
        await ws.run()
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        env_path: str = ".env",
    ):
        if not api_key or not api_secret:
            api_key, api_secret = _load_keys(env_path)

        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.ws_url = WS_TESTNET_PRIVATE if testnet else WS_MAINNET_PRIVATE

        # Callbacks
        self.on_order: Callable[[OrderUpdate], None] | None = None
        self.on_execution: Callable[[ExecutionUpdate], None] | None = None

        self._running = False

    def _auth_msg(self) -> dict:
        """Generate authentication message."""
        expires = int(time.time() * 1000) + 10_000
        sign_str = f"GET/realtime{expires}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "op": "auth",
            "args": [self.api_key, expires, signature],
        }

    async def run(self, duration_hours: float = 168.0) -> None:
        """Connect, authenticate, subscribe, and process messages."""
        self._running = True
        end_time = time.monotonic() + duration_hours * 3600

        while self._running and time.monotonic() < end_time:
            try:
                await self._connect(end_time)
            except Exception as e:
                logger.error(f"Private WS error: {e}")
                if time.monotonic() < end_time:
                    await asyncio.sleep(5)

    async def _connect(self, end_time: float) -> None:
        import websockets

        if not self.api_key:
            logger.error("No API key. Cannot connect to private WS.")
            self._running = False
            return

        async with websockets.connect(self.ws_url, ping_interval=20) as ws:
            # Authenticate
            await ws.send(json.dumps(self._auth_msg()))
            auth_resp = json.loads(await ws.recv())
            if not auth_resp.get("success"):
                logger.error(f"Auth failed: {auth_resp}")
                return
            logger.info("Private WS authenticated")

            # Subscribe to order and execution topics
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": ["order", "execution"],
            }))
            logger.info("Subscribed to order + execution")

            async for raw in ws:
                if time.monotonic() >= end_time:
                    break
                try:
                    msg = json.loads(raw)
                    self._handle_message(msg)
                except json.JSONDecodeError:
                    pass

    def _handle_message(self, msg: dict) -> None:
        """Route messages to appropriate handlers."""
        topic = msg.get("topic", "")
        data_list = msg.get("data", [])

        if topic == "order":
            for data in data_list:
                update = self._parse_order(data)
                if update and self.on_order:
                    self.on_order(update)

        elif topic == "execution":
            for data in data_list:
                update = self._parse_execution(data)
                if update and self.on_execution:
                    self.on_execution(update)

    def _parse_order(self, data: dict) -> OrderUpdate | None:
        try:
            return OrderUpdate(
                order_id=data.get("orderId", ""),
                order_link_id=data.get("orderLinkId", ""),
                symbol=data.get("symbol", ""),
                side=data.get("side", ""),
                status=data.get("orderStatus", ""),
                price=float(data.get("price", 0)),
                qty=float(data.get("qty", 0)),
                filled_qty=float(data.get("cumExecQty", 0)),
                avg_price=float(data.get("avgPrice", 0)),
                fee=float(data.get("cumExecFee", 0)),
                timestamp=int(data.get("updatedTime", 0)),
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse order: {e}")
            return None

    def _parse_execution(self, data: dict) -> ExecutionUpdate | None:
        try:
            return ExecutionUpdate(
                exec_id=data.get("execId", ""),
                order_id=data.get("orderId", ""),
                symbol=data.get("symbol", ""),
                side=data.get("side", ""),
                price=float(data.get("execPrice", 0)),
                qty=float(data.get("execQty", 0)),
                fee=float(data.get("execFee", 0)),
                fee_currency=data.get("feeCurrency", ""),
                is_maker=data.get("isMaker", False),
                timestamp=int(data.get("execTime", 0)),
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse execution: {e}")
            return None

    def stop(self):
        self._running = False
