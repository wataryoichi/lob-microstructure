"""Tests for Risk Manager."""

from __future__ import annotations

import pytest


class TestRiskManager:
    def test_allows_initial_trade(self):
        from src.risk_manager import RiskManager
        rm = RiskManager()
        ok, reason = rm.allow_new_trade()
        assert ok is True
        assert reason == "ok"

    def test_blocks_on_max_open(self):
        from src.risk_manager import RiskLimits, RiskManager
        rm = RiskManager(RiskLimits(max_open_positions=1))
        rm.on_trade_open()
        ok, reason = rm.allow_new_trade()
        assert ok is False
        assert "max_open" in reason

    def test_blocks_on_consecutive_losses(self):
        from src.risk_manager import RiskLimits, RiskManager
        rm = RiskManager(RiskLimits(consecutive_losses_limit=3))
        for _ in range(3):
            rm.on_trade_close(net_bps=-5.0)
        ok, reason = rm.allow_new_trade()
        assert ok is False
        assert "consecutive" in reason.lower() or "HALT" in reason

    def test_resets_consecutive_on_win(self):
        from src.risk_manager import RiskManager
        rm = RiskManager()
        rm.on_trade_close(net_bps=-5.0)
        rm.on_trade_close(net_bps=-5.0)
        assert rm.state.consecutive_losses == 2
        rm.on_trade_close(net_bps=+10.0)
        assert rm.state.consecutive_losses == 0

    def test_daily_drawdown_halt(self):
        from src.risk_manager import RiskLimits, RiskManager
        rm = RiskManager(RiskLimits(max_daily_drawdown_bps=50))
        # Win then big loss = drawdown
        rm.on_trade_close(net_bps=+30.0)
        rm.on_trade_close(net_bps=-80.0)  # peak=30, now=-50, dd=80
        ok, _ = rm.allow_new_trade()
        assert ok is False

    def test_get_status(self):
        from src.risk_manager import RiskManager
        rm = RiskManager()
        rm.on_trade_close(net_bps=+5.0)
        status = rm.get_status()
        assert status["daily_pnl_bps"] == 5.0
        assert status["halted"] is False
