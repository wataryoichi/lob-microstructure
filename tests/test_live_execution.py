"""End-to-End Testnet execution tests.

These tests require BYBIT_TESTNET_API_KEY and BYBIT_TESTNET_API_SECRET
in .env or environment. They are skipped if keys are not available.

IMPORTANT: Uses Testnet ONLY. Mainnet keys are NEVER used.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

# Skip all tests if no testnet keys
_has_testnet_keys = bool(
    os.environ.get("BYBIT_TESTNET_API_KEY")
    or _check_env_file()
    if False else False
)


def _check_env_file() -> bool:
    """Check .env for testnet keys."""
    from pathlib import Path
    p = Path(".env")
    if not p.exists():
        return False
    content = p.read_text()
    return "BYBIT_TESTNET_API_KEY" in content and "BYBIT_TESTNET_API_SECRET" in content


def _get_testnet_client():
    """Create a Testnet-only client with safety assertions."""
    from src.exchange_api import BybitClient, TESTNET_REST

    key = os.environ.get("BYBIT_TESTNET_API_KEY", "")
    secret = os.environ.get("BYBIT_TESTNET_API_SECRET", "")

    # Try .env if not in environment
    if not key:
        from src.exchange_api import _load_env
        env = _load_env(".env")
        key = env.get("BYBIT_TESTNET_API_KEY", "")
        secret = env.get("BYBIT_TESTNET_API_SECRET", "")

    client = BybitClient(api_key=key, api_secret=secret, testnet=True)

    # SAFETY: assert we're on testnet
    assert client.testnet is True, "MUST be testnet"
    assert "testnet" in client.base_url, "MUST use testnet URL"
    assert client.base_url == TESTNET_REST, f"Expected {TESTNET_REST}, got {client.base_url}"

    return client


# ---------------------------------------------------------------
# Unit tests (always run, no API keys needed)
# ---------------------------------------------------------------

class TestOrderManagerUnit:
    """Unit tests for OrderManager (no API calls)."""

    def test_managed_order_states(self):
        from src.order_manager import ManagedOrder, OrderState
        order = ManagedOrder(order_link_id="test1")
        assert order.state == OrderState.PENDING_NEW
        order.state = OrderState.NEW
        assert order.state == OrderState.NEW

    def test_order_state_enum(self):
        from src.order_manager import OrderState
        assert OrderState.FILLED == "filled"
        assert OrderState.CANCELLED == "cancelled"

    def test_order_manager_creation(self):
        from src.exchange_api import BybitClient
        from src.order_manager import OrderManager

        client = BybitClient(api_key="test", api_secret="test", testnet=True)
        om = OrderManager(client)
        assert om.get_fill_count() == 0
        assert len(om.get_open_orders()) == 0

    def test_ws_order_update_handling(self):
        """Simulate a WS order update."""
        from src.exchange_api import BybitClient
        from src.order_manager import ManagedOrder, OrderManager, OrderState
        from src.ws_private import OrderUpdate

        client = BybitClient(api_key="test", api_secret="test", testnet=True)
        om = OrderManager(client)

        # Simulate an order in our tracking
        order = ManagedOrder(
            order_link_id="lob_test123",
            exchange_order_id="exch_456",
            symbol="BTCUSDT", side="Buy",
            price=70000, qty=0.001,
            state=OrderState.NEW,
        )
        om._orders["lob_test123"] = order
        om._exchange_to_local["exch_456"] = "lob_test123"
        om._fill_events["lob_test123"] = asyncio.Event()

        # Simulate WS fill
        update = OrderUpdate(
            order_id="exch_456", order_link_id="lob_test123",
            symbol="BTCUSDT", side="Buy", status="Filled",
            price=70000, qty=0.001, filled_qty=0.001,
            avg_price=70000, fee=0.01, timestamp=int(time.time() * 1000),
        )
        om._on_order_update(update)

        assert order.state == OrderState.FILLED
        assert order.filled_price == 70000
        assert om.get_fill_count() == 1

    def test_ws_cancel_update_handling(self):
        from src.exchange_api import BybitClient
        from src.order_manager import ManagedOrder, OrderManager, OrderState
        from src.ws_private import OrderUpdate

        client = BybitClient(api_key="test", api_secret="test", testnet=True)
        om = OrderManager(client)

        order = ManagedOrder(
            order_link_id="lob_cancel1",
            exchange_order_id="exch_789",
            symbol="BTCUSDT", side="Sell",
            state=OrderState.NEW,
        )
        om._orders["lob_cancel1"] = order
        om._exchange_to_local["exch_789"] = "lob_cancel1"
        om._fill_events["lob_cancel1"] = asyncio.Event()

        update = OrderUpdate(
            order_id="exch_789", order_link_id="lob_cancel1",
            symbol="BTCUSDT", side="Sell", status="Cancelled",
            price=0, qty=0, filled_qty=0, avg_price=0, fee=0,
            timestamp=int(time.time() * 1000),
        )
        om._on_order_update(update)

        assert order.state == OrderState.CANCELLED


# ---------------------------------------------------------------
# Integration tests (require testnet keys, skipped otherwise)
# ---------------------------------------------------------------

@pytest.mark.skipif(
    not (os.environ.get("BYBIT_TESTNET_API_KEY") or _check_env_file()),
    reason="No testnet API keys available"
)
class TestTestnetIntegration:
    """Live Testnet tests. Only run when keys are configured."""

    def test_testnet_connectivity(self):
        client = _get_testnet_client()
        resp = client.get_server_time()
        assert resp.get("retCode") == 0

    def test_testnet_balance(self):
        client = _get_testnet_client()
        resp = client.get_wallet_balance()
        assert resp.get("retCode") == 0

    def test_testnet_place_and_cancel(self):
        """Place a far-from-market limit order and cancel it."""
        client = _get_testnet_client()

        # Get current price
        import requests
        price_resp = requests.get(
            f"{client.base_url}/v5/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            timeout=5,
        ).json()
        last_price = float(price_resp["result"]["list"][0]["lastPrice"])

        # Place buy limit well below market (won't fill)
        far_price = round(last_price * 0.9, 2)  # 10% below
        result = client.place_order(
            symbol="BTCUSDT",
            side="Buy",
            qty="0.001",
            price=str(far_price),
        )
        assert result.success, f"Place failed: {result.error_msg}"
        assert result.order_id

        # Cancel it
        cancel = client.cancel_order("BTCUSDT", order_id=result.order_id)
        assert cancel.success, f"Cancel failed: {cancel.error_msg}"
