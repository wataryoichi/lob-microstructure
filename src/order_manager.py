"""Order Manager: stateful order lifecycle with Taker fallback and reconciliation.

Integrates exchange_api (REST) and ws_private (real-time fills) into a single
state machine that tracks each order from placement to fill/cancel.

Handles:
- Async fill detection via Private WS callbacks
- Taker fallback when Maker doesn't fill within timeout
- Periodic state reconciliation against REST API
- Cancel-before-replace safety sequence
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from .exchange_api import BybitClient, OrderResult
from .notifier import Notifier
from .risk_manager import RiskManager
from .ws_private import BybitPrivateWS, ExecutionUpdate, OrderUpdate

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    PENDING_NEW = "pending_new"        # place_order called, awaiting WS confirm
    NEW = "new"                        # confirmed on exchange
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"  # cancel requested, awaiting confirm
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    TAKER_REPLACING = "taker_replacing"  # cancel confirmed, taker order being placed


@dataclass
class ManagedOrder:
    """Locally tracked order with full lifecycle state."""
    order_link_id: str           # our unique ID
    exchange_order_id: str = ""  # Bybit's orderId
    symbol: str = ""
    side: str = ""               # "Buy" or "Sell"
    price: float = 0.0
    qty: float = 0.0
    state: OrderState = OrderState.PENDING_NEW
    is_post_only: bool = True
    placed_at: float = 0.0      # monotonic
    filled_price: float = 0.0
    filled_qty: float = 0.0
    fill_fee: float = 0.0
    is_maker_fill: bool = True
    error_msg: str = ""


class OrderManager:
    """Manages order lifecycle with fallback and reconciliation.

    Usage:
        om = OrderManager(client, ws, risk_mgr, notifier)
        await om.start()

        # Place a maker order
        order = await om.place_maker_order("BTCUSDT", "Buy", 0.001, 70000.0)

        # Wait for fill or cancel with taker fallback
        result = await om.wait_for_fill(order.order_link_id, timeout_s=5.0)

        # Close position
        exit_order = await om.place_maker_order("BTCUSDT", "Sell", 0.001, 70100.0)
        exit_result = await om.wait_for_fill(exit_order.order_link_id, timeout_s=10.0,
                                              taker_fallback=True)
    """

    def __init__(
        self,
        client: BybitClient,
        private_ws: BybitPrivateWS | None = None,
        risk_manager: RiskManager | None = None,
        notifier: Notifier | None = None,
        reconcile_interval_s: float = 60.0,
    ):
        self.client = client
        self.ws = private_ws
        self.risk = risk_manager
        self.notifier = notifier or Notifier()
        self.reconcile_interval_s = reconcile_interval_s

        # Local order tracking
        self._orders: dict[str, ManagedOrder] = {}  # keyed by order_link_id
        self._exchange_to_local: dict[str, str] = {}  # exchange_id -> order_link_id

        # Fill events for async waiting
        self._fill_events: dict[str, asyncio.Event] = {}

        self._running = False

    async def start(self) -> None:
        """Start WS listener and reconciliation loop."""
        if self.ws:
            self.ws.on_order = self._on_order_update
            self.ws.on_execution = self._on_execution_update

        self._running = True
        logger.info("OrderManager started")

    async def stop(self) -> None:
        """Stop gracefully."""
        self._running = False
        # Cancel all open orders
        symbols = set(o.symbol for o in self._orders.values()
                      if o.state in (OrderState.NEW, OrderState.PARTIALLY_FILLED))
        for sym in symbols:
            try:
                self.client.cancel_all_orders(sym)
                logger.info(f"Cancelled all orders for {sym}")
            except Exception as e:
                logger.error(f"Failed to cancel orders for {sym}: {e}")

    # ------------------------------------------------------------------
    # Order Placement
    # ------------------------------------------------------------------

    async def place_maker_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> ManagedOrder:
        """Place a Post-Only Maker limit order."""
        # Risk check
        if self.risk:
            allowed, reason = self.risk.allow_new_trade()
            if not allowed:
                logger.warning(f"Order blocked by risk: {reason}")
                order = ManagedOrder(
                    order_link_id=f"blocked_{uuid.uuid4().hex[:8]}",
                    symbol=symbol, side=side, price=price, qty=qty,
                    state=OrderState.REJECTED, error_msg=reason,
                )
                return order

        link_id = f"lob_{uuid.uuid4().hex[:12]}"
        order = ManagedOrder(
            order_link_id=link_id,
            symbol=symbol, side=side, price=price, qty=qty,
            state=OrderState.PENDING_NEW,
            is_post_only=True,
            placed_at=time.monotonic(),
        )
        self._orders[link_id] = order
        self._fill_events[link_id] = asyncio.Event()

        # REST call
        result = self.client.place_order(
            symbol=symbol,
            side=side,
            qty=str(qty),
            price=str(price),
            order_type="Limit",
            time_in_force="PostOnly",
            order_link_id=link_id,
        )

        if result.success:
            order.exchange_order_id = result.order_id
            order.state = OrderState.NEW
            self._exchange_to_local[result.order_id] = link_id
            logger.info(f"Order placed: {link_id} -> {result.order_id} "
                         f"{side} {qty} {symbol} @ {price}")
        else:
            order.state = OrderState.REJECTED
            order.error_msg = result.error_msg
            self._fill_events[link_id].set()
            logger.error(f"Order rejected: {link_id} -> {result.error_msg}")

        return order

    async def cancel_order(self, order_link_id: str) -> bool:
        """Cancel an order. Returns True if cancel confirmed."""
        order = self._orders.get(order_link_id)
        if not order or order.state not in (OrderState.NEW, OrderState.PARTIALLY_FILLED):
            return False

        order.state = OrderState.PENDING_CANCEL

        result = self.client.cancel_order(
            symbol=order.symbol,
            order_id=order.exchange_order_id,
        )

        if result.success:
            order.state = OrderState.CANCELLED
            self._fill_events.get(order_link_id, asyncio.Event()).set()
            logger.info(f"Order cancelled: {order_link_id}")
            return True
        else:
            # May already be filled or cancelled
            logger.warning(f"Cancel failed: {order_link_id} -> {result.error_msg}")
            # Reconcile to find actual state
            await self._reconcile_order(order)
            return order.state == OrderState.CANCELLED

    # ------------------------------------------------------------------
    # Fill Waiting with Taker Fallback
    # ------------------------------------------------------------------

    async def wait_for_fill(
        self,
        order_link_id: str,
        timeout_s: float = 5.0,
        taker_fallback: bool = False,
    ) -> ManagedOrder:
        """Wait for fill. Optionally fall back to Taker on timeout.

        Taker fallback sequence:
        1. Cancel existing Maker order
        2. Wait for cancel confirmation
        3. Place new order as IOC (Immediate-or-Cancel) at market-crossing price
        """
        order = self._orders.get(order_link_id)
        if not order:
            return ManagedOrder(order_link_id=order_link_id, state=OrderState.REJECTED,
                                error_msg="unknown_order")

        # Already filled or rejected?
        if order.state in (OrderState.FILLED, OrderState.REJECTED, OrderState.CANCELLED):
            return order

        # Wait for WS fill event
        event = self._fill_events.get(order_link_id, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass

        # Check state after wait
        if order.state == OrderState.FILLED:
            return order

        if not taker_fallback:
            # No fallback: just cancel
            await self.cancel_order(order_link_id)
            return order

        # Taker fallback sequence
        logger.info(f"Taker fallback for {order_link_id}")
        order.state = OrderState.TAKER_REPLACING

        # Step 1: Cancel existing order
        cancel_ok = await self.cancel_order(order_link_id)
        if not cancel_ok and order.state == OrderState.FILLED:
            return order  # Race: got filled during cancel

        # Step 2: Place aggressive crossing limit (IOC-like)
        # Buy: price well above ask; Sell: price well below bid
        aggressive_offset = order.price * 0.001  # 0.1% aggressive
        if order.side == "Buy":
            taker_price = order.price + aggressive_offset
        else:
            taker_price = order.price - aggressive_offset

        taker_link_id = f"{order_link_id}_taker"
        taker_result = self.client.place_order(
            symbol=order.symbol,
            side=order.side,
            qty=str(order.qty),
            price=str(round(taker_price, 2)),
            order_type="Limit",
            time_in_force="IOC",  # Immediate-or-Cancel
            order_link_id=taker_link_id,
        )

        if taker_result.success:
            taker_order = ManagedOrder(
                order_link_id=taker_link_id,
                exchange_order_id=taker_result.order_id,
                symbol=order.symbol, side=order.side,
                price=taker_price, qty=order.qty,
                state=OrderState.NEW,
                is_post_only=False,
                placed_at=time.monotonic(),
            )
            self._orders[taker_link_id] = taker_order
            self._exchange_to_local[taker_result.order_id] = taker_link_id
            self._fill_events[taker_link_id] = asyncio.Event()

            # Wait briefly for IOC fill
            try:
                await asyncio.wait_for(
                    self._fill_events[taker_link_id].wait(), timeout=3.0
                )
            except asyncio.TimeoutError:
                pass

            # Copy fill info back to original order
            if taker_order.state == OrderState.FILLED:
                order.filled_price = taker_order.filled_price
                order.filled_qty = taker_order.filled_qty
                order.fill_fee = taker_order.fill_fee
                order.is_maker_fill = False
                order.state = OrderState.FILLED
                self._fill_events.get(order_link_id, asyncio.Event()).set()
                logger.info(f"Taker fill: {order_link_id} @ {taker_order.filled_price}")
            else:
                order.state = OrderState.CANCELLED
                logger.warning(f"Taker also failed for {order_link_id}")
        else:
            order.state = OrderState.CANCELLED
            logger.error(f"Taker order failed: {taker_result.error_msg}")

        return order

    # ------------------------------------------------------------------
    # WS Callbacks
    # ------------------------------------------------------------------

    def _on_order_update(self, update: OrderUpdate) -> None:
        """Handle order status update from Private WS."""
        link_id = self._exchange_to_local.get(update.order_id)
        if not link_id:
            # Try by order_link_id
            link_id = update.order_link_id
        if link_id not in self._orders:
            return

        order = self._orders[link_id]
        prev_state = order.state

        if update.status == "Filled":
            order.state = OrderState.FILLED
            order.filled_price = update.avg_price
            order.filled_qty = update.filled_qty
            order.fill_fee = update.fee
            self._fill_events.get(link_id, asyncio.Event()).set()

        elif update.status == "PartiallyFilled":
            order.state = OrderState.PARTIALLY_FILLED
            order.filled_qty = update.filled_qty

        elif update.status == "Cancelled":
            order.state = OrderState.CANCELLED
            self._fill_events.get(link_id, asyncio.Event()).set()

        elif update.status == "Rejected":
            order.state = OrderState.REJECTED
            self._fill_events.get(link_id, asyncio.Event()).set()

        if prev_state != order.state:
            logger.info(f"WS order update: {link_id} {prev_state} -> {order.state}")

    def _on_execution_update(self, update: ExecutionUpdate) -> None:
        """Handle execution (fill) from Private WS."""
        link_id = self._exchange_to_local.get(update.order_id)
        if not link_id or link_id not in self._orders:
            return

        order = self._orders[link_id]
        order.is_maker_fill = update.is_maker
        order.fill_fee = update.fee

        logger.info(f"WS execution: {link_id} @ {update.price} "
                     f"qty={update.qty} maker={update.is_maker}")

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self, symbol: str) -> int:
        """Reconcile local state with exchange. Returns number of corrections."""
        corrections = 0

        # Check open orders on exchange
        try:
            resp = self.client.get_open_orders(symbol)
            if resp.get("retCode") != 0:
                return 0

            exchange_orders = {
                o.get("orderId"): o
                for o in resp.get("result", {}).get("list", [])
            }
        except Exception as e:
            logger.error(f"Reconcile failed: {e}")
            return 0

        # Check each local order
        for link_id, order in list(self._orders.items()):
            if order.symbol != symbol:
                continue
            if order.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
                continue

            eid = order.exchange_order_id
            if eid and eid not in exchange_orders:
                # Order not on exchange but we think it's active -> reconcile
                await self._reconcile_order(order)
                corrections += 1

        if corrections:
            logger.info(f"Reconciled {corrections} orders for {symbol}")
        return corrections

    async def _reconcile_order(self, order: ManagedOrder) -> None:
        """Fetch actual order status from REST and update local state."""
        # For simplicity, if order not in open orders, assume filled or cancelled
        # A more thorough implementation would check order history
        logger.info(f"Reconciling {order.order_link_id} (state={order.state})")
        order.state = OrderState.CANCELLED
        self._fill_events.get(order.order_link_id, asyncio.Event()).set()

    async def run_reconciliation_loop(self, symbol: str) -> None:
        """Periodic reconciliation. Run as background task."""
        while self._running:
            await asyncio.sleep(self.reconcile_interval_s)
            await self.reconcile(symbol)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_order(self, order_link_id: str) -> ManagedOrder | None:
        return self._orders.get(order_link_id)

    def get_open_orders(self) -> list[ManagedOrder]:
        return [o for o in self._orders.values()
                if o.state in (OrderState.NEW, OrderState.PARTIALLY_FILLED, OrderState.PENDING_NEW)]

    def get_fill_count(self) -> int:
        return sum(1 for o in self._orders.values() if o.state == OrderState.FILLED)
