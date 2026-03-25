"""Tests for paper trading engine."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


class TestTradeLogger:
    def test_create_and_log(self):
        from src.paper_trader import TradeLogger, VirtualPosition
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            logger = TradeLogger(path)
            pos = VirtualPosition(
                position_id="test1", config_name="A", side="long",
                entry_price=100.0, entry_ts=1000, entry_type="maker",
                horizon_s=60, exit_price=101.0, exit_ts=2000,
                exit_type="maker", gross_bps=10.0, net_bps=9.0, status="closed")
            logger.log_trade(pos, "BTCUSDT")
            summary = logger.get_summary("A")
            assert summary["total_trades"] == 1
            assert summary["avg_net_bps"] == 9.0
            logger.close()
        finally:
            os.unlink(path)

    def test_signal_logging(self):
        from src.paper_trader import TradeLogger
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            logger = TradeLogger(path)
            logger.log_signal(1000, "A", "BTCUSDT", "long", 0.5, 0.95,
                              0.01, 70000.0, True)
            logger.log_signal(2000, "A", "BTCUSDT", "short", -0.3, 0.05,
                              0.5, 70000.0, False, "spread_too_wide")
            logger.close()
        finally:
            os.unlink(path)


class TestPaperTraderConfig:
    def test_default_configs(self):
        from src.paper_trader import PaperTraderConfig
        cfg = PaperTraderConfig()
        assert len(cfg.configs) == 2
        assert cfg.configs[0]["name"] == "A_5pct_60s"
        assert cfg.configs[1]["name"] == "B_10pct_120s"
        assert cfg.symbol == "BTCUSDT"
        assert cfg.max_spread_ticks == 1.5
