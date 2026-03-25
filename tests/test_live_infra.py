"""Tests for live trading infrastructure (Phase 12)."""

from __future__ import annotations

import pytest


class TestExchangeAPI:
    """Test Bybit API wrapper (no actual API calls)."""

    def test_client_creation_testnet(self):
        from src.exchange_api import BybitClient
        client = BybitClient(api_key="test", api_secret="test", testnet=True)
        assert client.testnet is True
        assert "testnet" in client.base_url

    def test_client_creation_mainnet(self):
        from src.exchange_api import BybitClient
        client = BybitClient(api_key="test", api_secret="test", testnet=False)
        assert client.testnet is False
        assert "testnet" not in client.base_url

    def test_server_time_mainnet(self):
        """Connectivity check (no auth needed)."""
        from src.exchange_api import BybitClient
        client = BybitClient(testnet=False)
        resp = client.get_server_time()
        assert resp.get("retCode") == 0

    def test_order_result_dataclass(self):
        from src.exchange_api import OrderResult
        r = OrderResult(success=True, order_id="123")
        assert r.success
        assert r.order_id == "123"

    def test_place_order_no_key(self):
        """Should fail gracefully without API key."""
        from src.exchange_api import BybitClient
        client = BybitClient(api_key="", api_secret="", testnet=True)
        result = client.place_order("BTCUSDT", "Buy", "0.001", "50000")
        # Will fail auth but shouldn't crash
        assert isinstance(result.success, bool)


class TestPrivateWS:
    def test_order_update_parsing(self):
        from src.ws_private import BybitPrivateWS

        ws = BybitPrivateWS(api_key="test", api_secret="test")
        parsed = ws._parse_order({
            "orderId": "abc123",
            "orderLinkId": "link1",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderStatus": "Filled",
            "price": "70000",
            "qty": "0.001",
            "cumExecQty": "0.001",
            "avgPrice": "70000",
            "cumExecFee": "0.01",
            "updatedTime": "1700000000000",
        })
        assert parsed is not None
        assert parsed.order_id == "abc123"
        assert parsed.status == "Filled"
        assert parsed.price == 70000.0

    def test_execution_update_parsing(self):
        from src.ws_private import BybitPrivateWS

        ws = BybitPrivateWS(api_key="test", api_secret="test")
        parsed = ws._parse_execution({
            "execId": "exec1",
            "orderId": "abc123",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "execPrice": "70001.5",
            "execQty": "0.001",
            "execFee": "0.01",
            "feeCurrency": "USDT",
            "isMaker": True,
            "execTime": "1700000000000",
        })
        assert parsed is not None
        assert parsed.exec_id == "exec1"
        assert parsed.is_maker is True
        assert parsed.price == 70001.5


class TestNotifier:
    def test_console_notifier(self):
        """Console-only notifier (no external deps)."""
        from src.notifier import Notifier
        n = Notifier()  # no webhook/telegram configured
        assert "console" in n._channels
        n.send("test message")  # should not raise

    def test_notify_trade(self):
        from src.notifier import Notifier
        n = Notifier()
        # Should not raise
        n.notify_trade("A", "long", "BTCUSDT", 70000, 70100, 14.3, 13.3, 5, 25.0)

    def test_notify_risk_halt(self):
        from src.notifier import Notifier
        n = Notifier()
        n.notify_risk_halt("max_daily_drawdown=100", 3600)

    def test_notify_system_event(self):
        from src.notifier import Notifier
        n = Notifier()
        n.notify_system_event("started", "Paper Trading mode")
